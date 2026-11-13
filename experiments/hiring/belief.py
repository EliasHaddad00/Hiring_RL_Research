"""Quality estimates: naive and survival-aware.

A candidate's quality q is N(0, 1). The naive estimate is the usual Gaussian posterior given the
resume and interview signals. It ignores that strong candidates are poached faster, so a candidate
who is still available after d poaching rounds is weaker on average than the naive estimate says.

The survival-aware (corrected) posterior multiplies the naive posterior by the probability of
having survived d rounds:

    posterior(q | signals, still here after d rounds)  is proportional to
        naive_posterior(q | signals) * (1 - h(q))^d,      h(q) = min(1, h0 * exp(slope * q)).

The survival factor depends only on d, so the exact state of a candidate is
(naive mean, number of interviews seen, days waited). Expectations are one-dimensional integrals.

GaussianBelief: continuous signals (full problem), Gauss-Hermite quadrature.
BinnedBelief:   coarsened signals (exact small instance), tables on a fine quality grid.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import norm

from .config import Scenario


def hazard(q, h0: float, slope: float):
    """Daily probability that a candidate of quality q is poached."""
    return np.minimum(1.0, h0 * np.exp(slope * np.asarray(q, dtype=float)))


def acceptance(days_waited, sc: Scenario):
    """Probability that an offer is accepted after waiting this many days."""
    return sc.accept_base * np.exp(-sc.accept_decay * np.asarray(days_waited, dtype=float))


class GaussianBelief:
    def __init__(self, sc: Scenario, n_nodes: int = 80):
        self.sc = sc
        self.prec0 = 1.0 + 1.0 / sc.resume_sd ** 2
        self.iprec = 1.0 / sc.interview_sd ** 2
        x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
        self._x, self._w = x, w / w.sum()

    # ---- naive Gaussian posterior -------------------------------------------------------------
    def variance(self, n):
        """Naive posterior variance after n interviews."""
        return 1.0 / (self.prec0 + np.asarray(n, dtype=float) * self.iprec)

    def naive_mean(self, resume, interview_sum, n):
        """Naive posterior mean from the resume signal and the sum of n interview signals."""
        return (np.asarray(resume, dtype=float) / self.sc.resume_sd ** 2
                + np.asarray(interview_sum, dtype=float) * self.iprec) * self.variance(n)

    # ---- survival-aware posterior -------------------------------------------------------------
    def _tilt(self, mu, n, d):
        mu, n, d = np.broadcast_arrays(np.asarray(mu, float), np.asarray(n, float), np.asarray(d, float))
        q = mu[..., None] + np.sqrt(self.variance(n))[..., None] * self._x
        keep = 1.0 - hazard(q, self.sc.h0, self.sc.quality_slope)
        w = self._w * keep ** d[..., None]
        return q, w, keep

    def survival_moments(self, mu, n, d):
        """Mean and variance of q given the signals and survival of d poaching rounds."""
        q, w, _ = self._tilt(mu, n, d)
        z = w.sum(-1)
        mean = (w * q).sum(-1) / z
        var = (w * q * q).sum(-1) / z - mean ** 2
        return mean, var

    def survival_mean(self, mu, n, d):
        return self.survival_moments(mu, n, d)[0]

    def survive_tonight(self, mu, n, d):
        """Probability of not being poached tonight, given survival so far."""
        _, w, keep = self._tilt(mu, n, d)
        return (w * keep).sum(-1) / w.sum(-1)

    def predictive_interview(self, mu, n, d, n_noise: int = 15):
        """Nodes and weights for the next interview signal y = q + noise, given survival so far.

        Returns (y, weights) with a trailing axis of n_nodes * n_noise points; weights sum to 1.
        """
        q, w, _ = self._tilt(mu, n, d)
        e, we = np.polynomial.hermite_e.hermegauss(n_noise)
        we = we / we.sum()
        y = q[..., :, None] + self.sc.interview_sd * e
        wy = (w / w.sum(-1, keepdims=True))[..., :, None] * we
        return y.reshape(*y.shape[:-2], -1), wy.reshape(*wy.shape[:-2], -1)

    def updated_mean(self, mu, n, y):
        """Naive mean after one more interview signal y."""
        prec = self.prec0 + np.asarray(n, float) * self.iprec
        return (np.asarray(mu, float) * prec + np.asarray(y, float) * self.iprec) / (prec + self.iprec)

    def first_order_mean(self, mu, n, d):
        """Closed-form approximation for hand-written rules only; poor when d * h0 is large."""
        var = self.variance(n)
        a, h0 = self.sc.quality_slope, self.sc.h0
        return mu - a * np.asarray(d, float) * h0 * var * np.exp(a * np.asarray(mu, float) + 0.5 * a * a * var)


class BinnedBelief:
    """Exact tables for the small instance, where signals are reported in bins.

    A candidate's type is (resume bin, counts of interview outcomes per bin). With days waited d:
      Z[type, d]  probability of this signal history and of surviving d rounds
      M[type, d]  survival-aware posterior mean of q        (M[type, 0] is the naive estimate)
      S[type, d]  probability of surviving tonight           = Z[type, d+1] / Z[type, d]
      PB[type, b, d]  probability that the next interview falls in bin b
      nxt[type, b]    type after an interview outcome b (-1 if no interview is left)
      prior_r[r]      probability that a new arrival's resume falls in bin r
    """

    def __init__(self, sc: Scenario, grid_points: int = 4001):
        assert sc.signals == "binned"
        self.sc = sc
        rcuts, icuts = list(sc.resume_cuts), list(sc.interview_cuts)
        self.R, self.B, self.NI = len(rcuts) + 1, len(icuts) + 1, sc.max_interviews
        qg = np.linspace(-8, 8, grid_points)
        w = norm.pdf(qg)
        w /= w.sum()

        def bin_probs(cuts, sd):
            e = [-np.inf] + cuts + [np.inf]
            return np.array([norm.cdf((e[i + 1] - qg) / sd) - norm.cdf((e[i] - qg) / sd)
                             for i in range(len(e) - 1)])

        Lr, Li = bin_probs(rcuts, sc.resume_sd), bin_probs(icuts, sc.interview_sd)
        keep = 1.0 - hazard(qg, sc.h0, sc.quality_slope)

        self.counts = [c for c in itertools.product(range(self.NI + 1), repeat=self.B) if sum(c) <= self.NI]
        self.types = [(r, c) for r in range(self.R) for c in self.counts]
        self.tid = {t: i for i, t in enumerate(self.types)}
        self.n_types = len(self.types)
        self.max_age = sc.horizon + 1
        D = self.max_age + 1

        self.Z = np.zeros((self.n_types, D))
        self.M = np.zeros((self.n_types, D))
        for i, (r, cnt) in enumerate(self.types):
            like = w * Lr[r] * np.prod([Li[b] ** cnt[b] for b in range(self.B)], axis=0)
            for d in range(D):
                ww = like * keep ** d
                self.Z[i, d] = ww.sum()
                self.M[i, d] = (ww * qg).sum() / ww.sum()

        self.nxt = -np.ones((self.n_types, self.B), dtype=int)
        for i, (r, cnt) in enumerate(self.types):
            if sum(cnt) < self.NI:
                for b in range(self.B):
                    c2 = list(cnt)
                    c2[b] += 1
                    self.nxt[i, b] = self.tid[(r, tuple(c2))]

        self.prior_r = np.array([self.Z[self.fresh_type(r), 0] for r in range(self.R)])
        self.S = self.Z[:, 1:] / self.Z[:, :-1]
        self.PB = np.where(self.nxt[:, :, None] >= 0,
                           self.Z[np.maximum(self.nxt, 0), :] / self.Z[:, None, :], 0.0)

    def fresh_type(self, resume_bin: int) -> int:
        return self.tid[(resume_bin, (0,) * self.B)]

    def resume_bin(self, signal: float) -> int:
        return int(np.searchsorted(self.sc.resume_cuts, signal))

    def interview_bin(self, signal: float) -> int:
        return int(np.searchsorted(self.sc.interview_cuts, signal))

    def naive_mean(self, type_id: int) -> float:
        return float(self.M[type_id, 0])

    def survival_mean(self, type_id: int, d: int) -> float:
        return float(self.M[type_id, d])
