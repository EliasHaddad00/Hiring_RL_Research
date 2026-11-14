"""Hand-written policies. Each acts through the same N-slot view and action set as PPO.

A policy has reset(env), called after env.reset(), and act(env) -> action index.

ThresholdRule is the generic rule of the earlier simulator (notes/hiring_sim_v3.py). With n
interview results seen, estimate e, f = day / horizon and `hired` roles already filled:

    offer      if day >= start_day and e >= o[n] - so[n] * f + r * hired
    interview  if n < kmax           and e >= i[n] - si * f

Offers go to the highest estimate first; interviews too. The estimate is the naive or the
survival-aware mean (`belief`). The rule families below restrict this rule.

v3_compat reproduces two details of the earlier simulator and is used only for the G1
reproduction: the offer bar uses `hired` as it stood at the start of the day, and at most 12
offers are made per day. By default the bar is re-checked after every offer and there is no cap.
"""
from __future__ import annotations

import numpy as np

INF = 99.0
V3_OFFERS_PER_DAY = 12


class RandomPolicy:
    """Uniform over valid actions. The normalisation anchor."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def reset(self, env):
        self.rng = np.random.default_rng([self.seed, env.episode_seed or 0])

    def act(self, env) -> int:
        return int(self.rng.choice(np.flatnonzero(env.action_masks())))


class ThresholdRule:
    def __init__(self, kmax, o, so, i, si, r=0.0, start_day=0, belief="naive", v3_compat=False, name="rule"):
        assert belief in ("naive", "survival")
        self.kmax, self.si, self.r, self.start_day = int(kmax), float(si), float(r), int(start_day)
        self.o, self.so, self.i = np.asarray(o, float), np.asarray(so, float), np.asarray(i, float)
        assert len(self.o) == len(self.so) >= self.kmax + 1
        self.belief, self.v3_compat, self.name = belief, v3_compat, name

    def reset(self, env):
        self._day = -1

    def offer_bar(self, n, f, hired):
        return self.o[n] - self.so[n] * f + self.r * hired

    def interview_bar(self, n, f):
        return self.i[np.minimum(n, len(self.i) - 1)] - self.si * f

    def act(self, env) -> int:
        v, N = env.view(), env.N
        if v.t != self._day:
            self._day, self._hired_at_start, self._offers_today = v.t, v.hired, 0
        k = len(v.ids)
        if k == 0:
            return 2 * N
        f = v.t / v.horizon
        est = v.naive if self.belief == "naive" else v.survival
        if v.offers_open and v.t >= self.start_day:
            hired = self._hired_at_start if self.v3_compat else v.hired
            if not (self.v3_compat and self._offers_today >= V3_OFFERS_PER_DAY):
                n = np.minimum(v.n_obs, len(self.o) - 1)
                eligible = est >= self.offer_bar(n, f, hired)
                if eligible.any():
                    self._offers_today += 1
                    return N + int(np.argmax(np.where(eligible, est, -np.inf)))
        if self.kmax > 0:
            valid = env.action_masks()[:k]
            want = valid & (v.n_done < self.kmax) & (est >= self.interview_bar(v.n_obs, f))
            if want.any():
                return int(np.argmax(np.where(want, est, -np.inf)))
        return 2 * N


# ---- rule families (parameter vectors as in hiring_sim_v3.FAM) ----------------------------------------

def fixed_threshold(tau, **kw):
    """Offer to anyone whose estimate is at least tau; no interviews."""
    return ThresholdRule(0, [tau], [0.0], [INF], 0.0, name="fixed_threshold", **kw)


def falling_bar(v, **kw):
    """v = (bar, fall, role term). No interviews; the bar falls linearly toward the deadline."""
    return ThresholdRule(0, [v[0]], [v[1]], [INF], 0.0, r=v[2], name="falling_bar", **kw)


def one_interview(v, **kw):
    """v = (fast-track bar, post-interview bar, their two falls, interview bar, its fall, role term)."""
    return ThresholdRule(1, [v[0], v[1]], [v[2], v[3]], [v[4]], v[5], r=v[6], name="one_interview", **kw)


def up_to_two_interviews(v, **kw):
    return ThresholdRule(2, [v[0], v[1], v[2]], [v[3], v[4], v[4]], [v[5], v[6]], v[7], r=v[8],
                         name="up_to_two_interviews", **kw)


FAMILIES = {
    "static_threshold": lambda v, **kw: fixed_threshold(v[0], **kw),
    "resume_falling_bar": falling_bar,
    "one_interview": one_interview,
    "up_to_two_interviews": up_to_two_interviews,
}


# ---- price-directed rule (from the Lagrangian relaxation, hiring/bounds.py) ---------------------------

class PriceDirectedRule:
    """Follow each candidate's one-candidate policy at the current price of a hire.

    At every decision the rule picks the price z that minimises
        open roles * z + value of future arrivals at z + sum of the shown candidates' values at z,
    multiplies it by `price_scale`, and then takes each candidate's own best action at that price:
    offers first (highest one-candidate value first), then interviews (highest value first), up to
    the daily limit. The price is recomputed after every offer. This is the index-style rule the
    relaxation prescribes; `price_scale` and `interview_price` are its two tuned parameters
    (1 and 0 are the untuned, theory values).

    It sees only the N shown candidates, like every other policy. belief="naive" uses tables
    solved under the naive belief (bounds.py); the table state is the naive mean in both versions.
    """

    def __init__(self, price_scale: float = 1.0, interview_price: float = 0.0, belief: str = "survival",
                 name: str = "price_directed"):
        assert belief in ("naive", "survival") and price_scale > 0 and interview_price >= 0
        self.price_scale, self.interview_price = float(price_scale), float(interview_price)
        self.belief, self.name = belief, name

    def reset(self, env):
        from .bounds import price_tables
        assert not env.binned, "the price-directed rule is built for Gaussian signals"
        self.tb = price_tables(env.sc, self.belief, self.interview_price)

    def price_index(self, v, values) -> int:
        """Index of the price in force. `values` is (shown candidates, prices)."""
        tb = self.tb
        r = v.horizon - 1 - v.t
        objective = v.roles_left * tb.z + tb.Fcum[r] + values.sum(0)
        z = self.price_scale * tb.z[int(np.argmin(objective))]
        return int(np.argmin(np.abs(tb.z - z)))

    def act(self, env) -> int:
        from .bounds import INTERVIEW, OFFER
        v, N, tb = env.view(), env.N, self.tb
        k = len(v.ids)
        if k == 0:
            return 2 * N
        r = v.horizon - 1 - v.t
        i, f = tb.locate(v.naive)
        V = tb.V[r][v.days, v.n_obs]                                   # (k, prices, grid)
        rows = np.arange(k)
        values = V[rows, :, i] * (1 - f)[:, None] + V[rows, :, i + 1] * f[:, None]
        iz = self.price_index(v, values)
        action = tb.A[r][v.days, v.n_obs, iz, np.rint(i + f).astype(int)]
        value = values[:, iz]
        if v.offers_open:
            want = action == OFFER
            if want.any():
                return N + int(np.argmax(np.where(want, value, -np.inf)))
        want = (action == INTERVIEW) & env.action_masks()[:k]
        if want.any():
            return int(np.argmax(np.where(want, value, -np.inf)))
        return 2 * N


# FAMILIES stays the four threshold families of the earlier simulator (the G1 scripts loop over it).
# RULE_FAMILIES is every tuned rule: parameter vector for price_directed = (price_scale, interview_price).
RULE_FAMILIES = dict(FAMILIES, price_directed=lambda v, **kw: PriceDirectedRule(v[0], v[1], **kw))
