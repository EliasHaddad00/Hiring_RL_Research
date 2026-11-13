"""Exact optimum of the small instance by dynamic programming on belief states.

Port of the prototype notes/prototypes/small.py to our conventions. With binned signals a
candidate's state is (type, days waited), where type = (resume bin, interview outcome counts),
and the pool is a sorted tuple of candidate states. Days are numbered 1..T here (the environment
uses 0..T-1). Four value functions, all memoised on (day, pool):

  W   start of day, before the arrival
  U   offer phase (offers one at a time; an acceptance ends the episode, a decliner leaves)
  I   interview phase (one slot; no interview on the last day)
  Pc  overnight poaching, then the next day

Because nothing happens between the single interview and the night, updating the type at once
(as here) is equivalent to the environment's "result visible next morning".

overflow = "choose"     the decision maker picks whom to drop when the pool exceeds its cap
                        (the prototype's convention; reproduces its optimum 0.6335)
overflow = "automatic"  the lowest naive estimate is dropped by pool.overflow_drop_index
                        (the environment's convention; the optimum PPO and rules are scored against)
"""
from __future__ import annotations

import itertools
import math
import sys

from .belief import BinnedBelief
from .config import Scenario
from .pool import overflow_drop_index

TIE = 1e-12


class ExactSolver:
    def __init__(self, sc: Scenario, overflow: str = "automatic"):
        assert overflow in ("automatic", "choose")
        assert sc.signals == "binned" and sc.roles == 1 and sc.interviews_per_day == 1
        assert sc.arrival == "bernoulli" and sc.pool_cap is not None
        self.sc, self.overflow = sc, overflow
        b = self.b = BinnedBelief(sc)
        self.T, self.P, self.pa, self.c = sc.horizon, sc.pool_cap, sc.arrival_rate, sc.interview_cost
        self.S, self.M, self.PB, self.nx = b.S.tolist(), b.M.tolist(), b.PB.tolist(), b.nxt.tolist()
        self.prior_r = b.prior_r.tolist()
        self.fresh = [b.fresh_type(r) for r in range(b.R)]
        self.naive = [b.naive_mean(i) for i in range(b.n_types)]
        self.memo = {"W": {}, "U": {}, "I": {}, "Pc": {}}
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))

    def p_accept(self, d: int) -> float:
        return self.sc.accept_base * math.exp(-self.sc.accept_decay * d)

    # ---- transitions ------------------------------------------------------------------------------
    def after_arrival(self, t, pool, r):
        """Pool after a newcomer with resume bin r joins and any overflow is resolved."""
        new = tuple(sorted(pool + ((self.fresh[r], 0),)))
        if len(new) <= self.P:
            return new
        if self.overflow == "choose":
            return max((new[:j] + new[j + 1:] for j in range(len(new))), key=lambda p: self.U(t, p))
        j = overflow_drop_index([self.naive[ty] for ty, _ in new], [d for _, d in new], [ty for ty, _ in new])
        return new[:j] + new[j + 1:]

    def _interview_value(self, t, pool, j):
        ty, d = pool[j]
        e = -self.c
        for b_, nt in enumerate(self.nx[ty]):
            e += self.PB[ty][b_][d] * self.Pc(t, tuple(sorted(pool[:j] + ((nt, d),) + pool[j + 1:])))
        return e

    def _offer_value(self, t, pool, j):
        ty, d = pool[j]
        p = self.p_accept(d)
        return p * self.M[ty][d] + (1 - p) * self.U(t, pool[:j] + pool[j + 1:])

    # ---- value functions --------------------------------------------------------------------------
    def W(self, t, pool):
        if t > self.T:
            return 0.0
        key = (t, pool)
        v = self.memo["W"].get(key)
        if v is None:
            v = (1 - self.pa) * self.U(t, pool)
            for r, pr in enumerate(self.prior_r):
                v += self.pa * pr * self.U(t, self.after_arrival(t, pool, r))
            self.memo["W"][key] = v
        return v

    def U(self, t, pool):
        key = (t, pool)
        v = self.memo["U"].get(key)
        if v is None:
            v = self.I(t, pool)
            for j in range(len(pool)):
                v = max(v, self._offer_value(t, pool, j))
            self.memo["U"][key] = v
        return v

    def I(self, t, pool):
        key = (t, pool)
        v = self.memo["I"].get(key)
        if v is None:
            v = self.Pc(t, pool)
            if t < self.T:
                for j, (ty, _) in enumerate(pool):
                    if self.nx[ty][0] >= 0:
                        v = max(v, self._interview_value(t, pool, j))
            self.memo["I"][key] = v
        return v

    def Pc(self, t, pool):
        key = (t, pool)
        v = self.memo["Pc"].get(key)
        if v is None:
            v = 0.0
            if t < self.T:
                for stay in itertools.product((0, 1), repeat=len(pool)):
                    pr, new = 1.0, []
                    for keep, (ty, d) in zip(stay, pool):
                        s = self.S[ty][d]
                        if keep:
                            pr *= s
                            new.append((ty, d + 1))
                        else:
                            pr *= 1 - s
                    if pr > 0:
                        v += pr * self.W(t + 1, tuple(new))
            self.memo["Pc"][key] = v
        return v

    def value(self) -> float:
        """Optimal expected payoff from an empty pool on day 1."""
        return self.W(1, ())

    # ---- optimal actions ----------------------------------------------------------------------------
    def offer_choice(self, t, pool):
        """Index in `pool` to offer to, or None to move on to the interview phase."""
        best, a = self.I(t, pool), None
        for j in range(len(pool)):
            e = self._offer_value(t, pool, j)
            if e > best + TIE:
                best, a = e, j
        return a

    def interview_choice(self, t, pool):
        """Index in `pool` to interview, or None to end the day."""
        best, a = self.Pc(t, pool), None
        if t < self.T:
            for j, (ty, _) in enumerate(pool):
                if self.nx[ty][0] >= 0:
                    e = self._interview_value(t, pool, j)
                    if e > best + TIE:
                        best, a = e, j
        return a


class ExactPolicy:
    """Plays the solver's optimal policy inside HiringEnv (small instance)."""
    name = "exact_optimal"

    def __init__(self, solver: ExactSolver):
        self.solver = solver

    def reset(self, env):
        assert env.binned and env.N >= env.sc.pool_cap, "the whole pool must be visible"

    def act(self, env) -> int:
        v, N = env.view(), env.N
        if not v.offers_open or len(v.ids) == 0:
            return 2 * N                    # after the day's single interview, or with an empty pool
        states = [(int(ty), int(d)) for ty, d in zip(v.types, v.days)]
        order = sorted(range(len(states)), key=lambda i: states[i])
        pool, t = tuple(states[i] for i in order), v.t + 1
        j = self.solver.offer_choice(t, pool)
        if j is not None:
            return N + order[j]
        j = self.solver.interview_choice(t, pool)
        return order[j] if j is not None else 2 * N
