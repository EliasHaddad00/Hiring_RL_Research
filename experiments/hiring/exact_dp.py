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


# ---- exact policy evaluation ------------------------------------------------------------------------
#
# The same backward pass as the solver, with a policy's action in place of the maximum. A policy
# is a function of the solver state: `choose(t, pool)` returns [(probability, kind, j), ...] with
# kind in {"offer", "interview", "end"} and j an index into `pool`. It is asked only in the offer
# phase. After an interview the environment allows nothing but "end day" (one slot per day), and
# "interview" below already includes that forced step.

class PolicyEvaluator:
    def __init__(self, solver: ExactSolver, choose):
        assert solver.overflow == "automatic", "policies play the environment, which drops automatically"
        self.s, self.choose = solver, choose
        self.memo = {"W": {}, "U": {}, "Pc": {}}

    def W(self, t, pool):
        s = self.s
        if t > s.T:
            return 0.0
        key = (t, pool)
        v = self.memo["W"].get(key)
        if v is None:
            v = (1 - s.pa) * self.U(t, pool)
            for r, pr in enumerate(s.prior_r):
                v += s.pa * pr * self.U(t, s.after_arrival(t, pool, r))
            self.memo["W"][key] = v
        return v

    def U(self, t, pool):
        s = self.s
        key = (t, pool)
        v = self.memo["U"].get(key)
        if v is None:
            v, total = 0.0, 0.0
            for prob, kind, j in self.choose(t, pool):
                total += prob
                if prob == 0.0:
                    continue
                if kind == "end":
                    q = self.Pc(t, pool)
                elif kind == "offer":
                    ty, d = pool[j]
                    p = s.p_accept(d)
                    q = p * s.M[ty][d] + (1 - p) * self.U(t, pool[:j] + pool[j + 1:])
                else:
                    ty, d = pool[j]
                    assert t < s.T and s.nx[ty][0] >= 0, "the policy chose an interview that is not allowed"
                    q = -s.c
                    for b_, nt in enumerate(s.nx[ty]):
                        q += s.PB[ty][b_][d] * self.Pc(t, tuple(sorted(pool[:j] + ((nt, d),) + pool[j + 1:])))
                v += prob * q
            assert abs(total - 1.0) < 1e-9, "action probabilities must sum to 1"
            self.memo["U"][key] = v
        return v

    def Pc(self, t, pool):
        s = self.s
        key = (t, pool)
        v = self.memo["Pc"].get(key)
        if v is None:
            v = 0.0
            if t < s.T:
                for stay in itertools.product((0, 1), repeat=len(pool)):
                    pr, new = 1.0, []
                    for keep, (ty, d) in zip(stay, pool):
                        sv = s.S[ty][d]
                        if keep:
                            pr *= sv
                            new.append((ty, d + 1))
                        else:
                            pr *= 1 - sv
                    if pr > 0:
                        v += pr * self.W(t + 1, tuple(new))
            self.memo["Pc"][key] = v
        return v

    def value(self) -> float:
        """Expected payoff of the policy from an empty pool on day 1."""
        return self.W(1, ())


def valid_actions(solver: ExactSolver, t, pool):
    """[(kind, j), ...] allowed in the offer phase of state (t, pool); "end" is always allowed."""
    acts = [("offer", j) for j in range(len(pool))]
    if t < solver.T:
        acts += [("interview", j) for j, (ty, _) in enumerate(pool) if solver.nx[ty][0] >= 0]
    return acts + [("end", None)]


def optimal_choice(solver: ExactSolver):
    def choose(t, pool):
        j = solver.offer_choice(t, pool)
        if j is not None:
            return [(1.0, "offer", j)]
        j = solver.interview_choice(t, pool)
        return [(1.0, "interview", j)] if j is not None else [(1.0, "end", None)]
    return choose


def uniform_choice(solver: ExactSolver):
    """The random policy: uniform over the valid actions."""
    def choose(t, pool):
        acts = valid_actions(solver, t, pool)
        return [(1.0 / len(acts), kind, j) for kind, j in acts]
    return choose


# ---- the environment's view of a solver state ------------------------------------------------------

def slot_order(solver: ExactSolver, pool):
    """Pool indices in the environment's slot order: highest naive estimate first, ties by arrival
    (the candidate who has waited longer arrived earlier)."""
    return sorted(range(len(pool)), key=lambda i: (-solver.naive[pool[i][0]], -pool[i][1]))


def state_observation(solver: ExactSolver, t, pool):
    """Observation and action mask the environment shows in the offer phase of (t, pool).

    Used to evaluate a learned agent exactly. tests/test_exact_dp.py checks it against env.py.
    """
    import numpy as np
    from .env import MEAN_SCALE, N_GLOBAL

    sc, b = solver.sc, solver.b
    N, F = sc.n_slots, 5 + 1 + b.B
    obs = np.zeros(N * F + N_GLOBAL, np.float32)
    mask = np.zeros(2 * N + 1, bool)
    mask[2 * N] = True
    order = slot_order(solver, pool)
    assert len(order) <= N
    for slot, i in enumerate(order):
        ty, d = pool[i]
        r, counts = b.types[ty]
        est = b.M[ty, d] if sc.obs_belief == "survival" else b.M[ty, 0]
        row = [1.0, est / MEAN_SCALE, sum(counts) / sc.max_interviews, d / sc.horizon, 0.0, r / (b.R - 1)]
        obs[slot * F:(slot + 1) * F] = row + [c / sc.max_interviews for c in counts]
        mask[N + slot] = True
        mask[slot] = t < solver.T and solver.nx[ty][0] >= 0
    g = N * F
    obs[g:] = [(sc.horizon - (t - 1)) / sc.horizon, 1.0, 1.0, min(len(pool), 2 * N) / (2 * N), 1.0]
    return obs, mask, order


def greedy_choice(solver: ExactSolver, model):
    """The greedy (most probable valid action) policy of a MaskablePPO model, as a function of the state."""
    import numpy as np

    solver.value()                                   # fills the memo with every reachable offer-phase state
    known = set(solver.memo["U"])
    N = solver.sc.n_slots
    table = {}

    def fill(keys):
        for i in range(0, len(keys), 8192):
            chunk = keys[i:i + 8192]
            views = [state_observation(solver, t, pool) for t, pool in chunk]
            actions, _ = model.predict(np.stack([v[0] for v in views]), action_masks=np.stack([v[1] for v in views]),
                                       deterministic=True)
            for key, (_, _, order), a in zip(chunk, views, actions):
                a = int(a)
                table[key] = ("end", None) if a == 2 * N else (("offer", order[a - N]) if a >= N else ("interview", order[a]))

    fill(list(known))

    def choose(t, pool):
        if (t, pool) not in table:                   # a state the solver never visited (should not happen)
            fill([(t, pool)])
        kind, j = table[(t, pool)]
        return [(1.0, kind, j)]
    return choose
