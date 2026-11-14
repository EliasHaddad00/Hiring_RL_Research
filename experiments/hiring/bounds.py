"""Lagrangian upper bound on the optimal payoff, and the one-candidate tables behind it.

IDEA. Two constraints tie candidates together: at most `roles` hires in the episode, and at most
`interviews_per_day` interviews on each day. Put a price z >= 0 on each hire and a price lam >= 0
on each interview, and drop both constraints. Any policy of the real problem is still allowed and
earns at least as much in the priced problem (it pays for at most `roles` hires and at most K
interviews per day, and gets that money back as the constant terms below). Without the
constraints, candidates do not interact, so the priced problem splits into one-candidate problems:

    optimum  <=  roles * z  +  lam * K * (T - 1)  +  sum over days t of  E[arrivals on t] * F_t(z, lam)

for every z, lam >= 0, where F_t is the optimal value of ONE fresh candidate arriving on day t who
is paid q - z on acceptance and charged cost + lam per interview. The bound is the minimum over
(z, lam). (T - 1 because no interview is allowed on the last day.) This is the Lagrangian
relaxation for dynamic selection problems (Brown and Smith 2020), applied to our model; the
construction follows the prototype notes/prototypes/lag.py.

ONE-CANDIDATE PROBLEM, with the conventions of env.py and exact_dp.py:
    state     naive mean, interview results seen n, days waited d, days remaining after today r
    pass      0, the candidate is ignored from now on
    offer     acceptance(d) * (E[q | state] - z); allowed on the arrival day; the candidate leaves
    interview only if r >= 1 (never on the last day) and n < max_interviews; costs cost + lam;
              the result is seen next morning and is lost if the candidate is poached tonight;
              so at most one interview per day and an offer tomorrow at the earliest
    wait      keep the candidate; poached tonight with the belief-averaged hazard
The belief is the survival-aware posterior of belief.py, so E[q | state], the chance of surviving
tonight and the law of the next interview result are the ones the environment generates.

WHAT IS RELAXED (each can only raise the bound, so it stays valid): the role cap and the daily
interview limit (priced), strict offer-then-interview phases, the N-slot view, the pool cap of the
small instance (a dropped candidate is a passed candidate), and the cap on the per-episode draw table.

TWO DEVIATIONS FROM THE PROTOTYPE, both keeping validity: a "wait" action is included (the
prototype has none), and the numerical grid is wider and finer.

ACCURACY. On the small instance (binned signals) the tables are exact. With Gaussian signals the
naive mean lives on a grid with linear interpolation, so the bound is exact only up to grid and
quadrature error; `lagrangian_bound` is rerun on a finer grid to measure that error.

belief="naive" builds the tables a decision maker would compute who ignores survival: E[q] is the
naive mean, and tonight's loss risk is the hazard averaged over the naive belief. Those tables
are for the naive price-directed rule only. They are NOT a bound.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import norm

from .belief import BinnedBelief, GaussianBelief, acceptance
from .config import Scenario

PASS, OFFER, INTERVIEW, WAIT = 0, 1, 2, 3
ACTION_NAMES = ("pass", "offer", "interview", "wait")


class GaussianTables:
    """One-candidate dynamic program for Gaussian signals (the full problem)."""

    def __init__(self, sc: Scenario, belief: str = "survival", n_grid: int = 181,
                 mu_range=(-4.5, 5.5), n_nodes: int = 80, n_noise: int = 15):
        assert sc.signals == "gaussian" and belief in ("survival", "naive")
        t0 = time.perf_counter()
        self.sc, self.belief_kind = sc, belief
        b = self.b = GaussianBelief(sc, n_nodes)
        T, nmax = sc.horizon, sc.max_interviews
        self.T, self.nmax, self.G = T, nmax, n_grid
        self.grid = grid = np.linspace(mu_range[0], mu_range[1], n_grid)
        self.dx = dx = grid[1] - grid[0]
        tilt = belief == "survival"
        e, we = np.polynomial.hermite_e.hermegauss(n_noise)
        we = we / we.sum()
        d_all = np.arange(T, dtype=float)[:, None]

        self.M = np.zeros((nmax + 1, T, n_grid))          # E[q | naive mean, n, d]
        self.S = np.zeros((nmax + 1, T, n_grid))          # P(not poached tonight | ...)
        self.W = np.zeros((nmax, T, n_grid, n_grid))      # interview: weight on tomorrow's grid point
        gi = np.arange(n_grid)[:, None, None]
        ji = np.arange(n_nodes)[None, :, None]
        for n in range(nmax + 1):
            q, w, keep = b.nodes(grid[None, :], n, d_all, tilt=tilt)        # (T, G, J)
            self.M[n] = (w * q).sum(-1)
            self.S[n] = (w * keep).sum(-1)
            if n < nmax:
                # tomorrow's naive mean after the result y = q + noise, by linear interpolation
                mu2 = b.updated_mean(grid[:, None, None], n, q[0][:, :, None] + sc.interview_sd * e)
                pos = np.clip((mu2 - grid[0]) / dx, 0, n_grid - 1 - 1e-9)
                i = np.floor(pos).astype(int)
                f = pos - i
                P = np.zeros((n_grid, n_nodes, n_grid))
                np.add.at(P, (gi, ji, i), we * (1 - f))
                np.add.at(P, (gi, ji, i + 1), we * f)
                self.W[n] = np.einsum("dgj,gjh->dgh", w * keep, P, optimize=True)

        # law of a fresh arrival's naive mean, N(0, 1 - posterior variance), as weights on the grid
        sd0 = np.sqrt(1.0 - b.variance(0))
        fine = np.linspace(grid[0], grid[-1], (n_grid - 1) * 20 + 1)
        pw = norm.pdf(fine / sd0)
        pw /= pw.sum()
        pos = np.clip((fine - grid[0]) / dx, 0, n_grid - 1 - 1e-9)
        i = np.floor(pos).astype(int)
        f = pos - i
        self.w0 = np.bincount(i, pw * (1 - f), n_grid) + np.bincount(i + 1, pw * f, n_grid)
        self.acc = acceptance(np.arange(T), sc)
        self.state_count = n_grid * sum(max(0, T - r - n) for r in range(T) for n in range(nmax + 1))
        self.build_seconds = time.perf_counter() - t0

    def solve(self, z, lam: float = 0.0, keep=False):
        """Value of a fresh arrival, F[r, k] for r days remaining and price z[k].

        keep=True also returns Q[r] with shape (d, n, action, price, grid): the value of offer,
        interview and wait (index action - 1) in every state; -inf where an action is not allowed.
        keep="policy" returns instead (F, V, A): per r, the value V[r] (float32) and the best
        action A[r] (int8; ties go to offer, then interview, then wait), shape (d, n, price, grid).
        """
        sc, T, nmax, G = self.sc, self.T, self.nmax, self.G
        z = np.atleast_1d(np.asarray(z, float))
        Z = len(z)
        cost = sc.interview_cost + lam
        F = np.zeros((T, Z))
        Q, V, A = [], [], []
        prev = None
        for r in range(T):
            dmax = T - 1 - r
            cur = np.zeros((dmax + 1, nmax + 1, Z, G))
            q_r = np.full((dmax + 1, nmax + 1, 3, Z, G), -np.inf, np.float32) if keep is True else None
            a_r = np.zeros((dmax + 1, nmax + 1, Z, G), np.int8) if keep == "policy" else None
            for n in range(min(nmax, dmax) + 1):
                ds = np.arange(n, dmax + 1)
                offer = self.acc[ds][:, None, None] * (self.M[n][ds][:, None, :] - z[None, :, None])
                best = np.maximum(offer, 0.0)
                act = (offer > 0).astype(np.int8) * OFFER if a_r is not None else None
                if q_r is not None:
                    q_r[ds, n, OFFER - 1] = offer
                if r >= 1:
                    if n < nmax:
                        cont = prev[ds + 1, n + 1] @ self.W[n][ds].transpose(0, 2, 1) - cost
                        if act is not None:
                            act[cont > best] = INTERVIEW
                        best = np.maximum(best, cont)
                        if q_r is not None:
                            q_r[ds, n, INTERVIEW - 1] = cont
                    wait = self.S[n][ds][:, None, :] * prev[ds + 1, n]
                    if act is not None:
                        act[wait > best] = WAIT
                    best = np.maximum(best, wait)
                    if q_r is not None:
                        q_r[ds, n, WAIT - 1] = wait
                cur[ds, n] = best
                if a_r is not None:
                    a_r[ds, n] = act
            F[r] = cur[0, 0] @ self.w0
            prev = cur
            if q_r is not None:
                Q.append(q_r)
            if a_r is not None:
                V.append(cur.astype(np.float32))
                A.append(a_r)
        if keep == "policy":
            return F, V, A
        return (F, Q) if keep else F


class BinnedTables:
    """One-candidate dynamic program for binned signals (the exact small instance). Exact."""

    def __init__(self, sc: Scenario):
        assert sc.signals == "binned"
        t0 = time.perf_counter()
        self.sc = sc
        b = self.b = BinnedBelief(sc)
        self.T = sc.horizon
        self.acc = acceptance(np.arange(self.T), sc)
        self.fresh = np.array([b.fresh_type(r) for r in range(b.R)])
        self.can = b.nxt[:, 0] >= 0
        self.state_count = b.n_types * self.T * (self.T + 1) // 2
        self.build_seconds = time.perf_counter() - t0

    def solve(self, z, lam: float = 0.0):
        sc, b, T = self.sc, self.b, self.T
        z = np.atleast_1d(np.asarray(z, float))
        cost = sc.interview_cost + lam
        F = np.zeros((T, len(z)))
        nxt = np.maximum(b.nxt, 0)
        prev = None
        for r in range(T):
            dmax = T - 1 - r
            cur = np.zeros((dmax + 1, b.n_types, len(z)))
            for d in range(dmax + 1):
                best = np.maximum(self.acc[d] * (b.M[:, d][:, None] - z[None, :]), 0.0)
                if r >= 1:
                    nx = prev[d + 1]                                            # (types, Z)
                    best = np.maximum(best, b.S[:, d][:, None] * nx)            # wait
                    # interview: outcome b, then the new type must survive the night (as in exact_dp.Pc)
                    cont = (b.PB[:, :, d] * b.S[nxt, d])[:, :, None] * nx[nxt]  # (types, B, Z)
                    cont = cont.sum(1) - cost
                    best = np.where(self.can[:, None], np.maximum(best, cont), best)
                cur[d] = best
            F[r] = b.prior_r @ cur[0][self.fresh]
            prev = cur
        return F


def make_tables(sc: Scenario, **kw):
    return BinnedTables(sc) if sc.signals == "binned" else GaussianTables(sc, **kw)


def relaxed_value(sc: Scenario, F: np.ndarray, z, lam: float = 0.0) -> np.ndarray:
    """roles * z + lam * K * (T - 1) + expected arrivals per day * sum over arrival days of F."""
    z = np.atleast_1d(np.asarray(z, float))
    return sc.roles * z + lam * sc.interviews_per_day * (sc.horizon - 1) + sc.arrival_rate * F.sum(0)


def lagrangian_bound(sc: Scenario, lam_grid=(0.0,), z_max: float = 3.0, z_points: int = 31,
                     tables=None, **table_kw) -> dict:
    """Upper bound on the optimal expected payoff of scenario `sc`.

    The relaxed value is convex in z, so a coarse grid on [0, z_max] is followed by two finer
    grids around the best point. Every (z, lam) gives a valid bound; the search only tightens it.
    """
    tb = tables if tables is not None else make_tables(sc, **table_kw)
    t0 = time.perf_counter()
    per_lam = []
    for lam in lam_grid:
        zs = np.linspace(0.0, z_max, z_points)
        for _ in range(3):
            ub = relaxed_value(sc, tb.solve(zs, lam), zs, lam)
            j = int(ub.argmin())
            step = zs[1] - zs[0]
            lo, hi = max(0.0, zs[j] - step), zs[j] + step
            best = (float(ub[j]), float(zs[j]))
            zs = np.linspace(lo, hi, 11)
        per_lam.append(dict(lam=float(lam), bound=best[0], z=best[1]))
    top = min(per_lam, key=lambda x: x["bound"])
    return dict(bound=top["bound"], z=top["z"], lam=top["lam"], per_lam=per_lam,
                state_count=int(tb.state_count), build_seconds=float(tb.build_seconds),
                solve_seconds=time.perf_counter() - t0,
                grid_points=getattr(tb, "G", None), scenario=sc.name, h0=sc.h0, accept_decay=sc.accept_decay)


# ---- simulation of the relaxed policy (a check of the tables, not a policy of the real problem) -----

def simulate_relaxed(sc: Scenario, tables: GaussianTables, z: float, seeds, lam: float = 0.0) -> np.ndarray:
    """Per-episode value of the priced, unconstrained problem under the tables' own policy.

    Every candidate follows the one-candidate policy at price z, on the same per-candidate draws
    as the environment (seeding.make_draws). Returns roles * z + lam * K * (T - 1) + sum over
    candidates of [accepted * (q - z) - (cost + lam) * interviews] for each episode; its mean
    estimates the relaxed value that `relaxed_value` computes.
    """
    from .belief import hazard
    from .seeding import make_draws

    _, Q = tables.solve([z], lam, keep=True)
    b, grid, dx, G, T = tables.b, tables.grid, tables.dx, tables.G, sc.horizon
    cost = sc.interview_cost + lam
    const = sc.roles * z + lam * sc.interviews_per_day * (T - 1)
    out = []
    for seed in seeds:
        dr = make_draws(int(seed), sc)
        haz = hazard(dr.quality, sc.h0, sc.quality_slope)
        total = 0.0
        for c in range(dr.n):
            t0, q = int(dr.arrival_day[c]), float(dr.quality[c])
            resume = q + sc.resume_sd * dr.resume_z[c]
            isum, n = 0.0, 0
            for t in range(t0, T):
                d, r = t - t0, T - 1 - t
                mu = float(b.naive_mean(resume, isum, n))
                pos = min(max((mu - grid[0]) / dx, 0.0), G - 1 - 1e-9)
                i = int(pos)
                f = pos - i
                vals = Q[r][d, n, :, 0, i] * (1 - f) + Q[r][d, n, :, 0, i + 1] * f
                a = int(np.argmax(vals)) + 1 if np.max(vals) > 0 else PASS
                if a == PASS:
                    break
                if a == OFFER:
                    if dr.u_accept[c] < sc.accept_base * np.exp(-sc.accept_decay * d):
                        total += q - z
                    break
                signal = None
                if a == INTERVIEW:
                    total -= cost
                    signal = q + sc.interview_sd * dr.interview_z[c, n]
                if dr.u_poach[c, t] < haz[c]:
                    break                                   # poached tonight; a pending result is lost
                if signal is not None:
                    isum, n = isum + signal, n + 1
        out.append(const + total)
    return np.array(out)


# ---- tables for the price-directed rule ---------------------------------------------------------------

RULE_GRID = dict(n_grid=121, mu_range=(-3.5, 4.5))     # coarser than the bound's grid: 33 prices are stored
RULE_PRICES = np.linspace(0.0, 2.4, 33)


class PriceTables:
    """One-candidate values and actions on a grid of hire prices, for the price-directed rule.

    V[r][d, n, k, g] and A[r][...] for r days remaining; Fcum[r, k] is the value of all arrivals
    still to come after today (expected arrivals per day times the sum of F over later days).
    """

    def __init__(self, sc: Scenario, belief: str = "survival", interview_price: float = 0.0):
        tb = GaussianTables(sc, belief=belief, **RULE_GRID)
        self.sc, self.belief, self.z = sc, belief, RULE_PRICES
        self.grid, self.dx, self.G = tb.grid, tb.dx, tb.G
        F, self.V, self.A = tb.solve(self.z, interview_price, keep="policy")
        self.Fcum = sc.arrival_rate * np.concatenate([np.zeros((1, len(self.z))), np.cumsum(F, 0)])[:-1]
        self.megabytes = sum(v.nbytes + a.nbytes for v, a in zip(self.V, self.A)) / 1e6

    def locate(self, mu):
        """Grid cell and interpolation weight of naive means `mu`."""
        pos = np.clip((np.asarray(mu, float) - self.grid[0]) / self.dx, 0, self.G - 1 - 1e-9)
        i = np.floor(pos).astype(int)
        return i, pos - i


_PRICE_TABLES: dict = {}


def price_tables(sc: Scenario, belief: str = "survival", interview_price: float = 0.0) -> PriceTables:
    """Cached per process. Fields that do not enter the one-candidate problem are ignored in the key."""
    key = (sc.with_(name="", n_slots=0, obs_belief="survival", max_candidates=0, roles=0,
                    interviews_per_day=0), belief, round(float(interview_price), 6))
    if key not in _PRICE_TABLES:
        if len(_PRICE_TABLES) >= 4:                     # each entry is about 150 MB
            _PRICE_TABLES.pop(next(iter(_PRICE_TABLES)))
        _PRICE_TABLES[key] = PriceTables(sc, belief, interview_price)
    return _PRICE_TABLES[key]
