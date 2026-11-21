"""The pre-registered decision rule (PREREGISTRATION.md, "Decision rule") as plain functions.

    Take the 95% interval of the mean paired difference and a margin of 2% of the comparator's
    payoff. Interval entirely inside the margin: equivalent. Otherwise, entirely above zero:
    better; entirely below zero: worse. Anything else: inconclusive. Equivalence takes precedence.

Scores are arrays of shape (units, episodes): one row per training seed, each row the payoffs on
the same episodes in the same order. A rule, or an exact value, is a single row. Where the
pre-registration leaves a choice open (what is resampled, mean or interquartile mean, whether
seeds of two actors are paired), the choice is an argument here and is read from
configs/decision_rule.yaml by scripts/apply_decision_rule.py; nothing is chosen in this file.
"""
from __future__ import annotations

import numpy as np

BOOT = 10_000          # bootstrap draws, as in scripts/check_plateau.py
MARGIN = 0.02


def iqm(x) -> float:
    """Interquartile mean: the mean of the middle half (with three values, the plain mean)."""
    x = np.sort(np.asarray(x, float))
    cut = int(np.floor(0.25 * len(x)))
    return float(x[cut:len(x) - cut].mean())


STATS = {"mean": lambda x: float(np.mean(x)), "iqm": iqm}


def verdict(lo: float, hi: float, margin: float) -> str:
    if -margin < lo and hi < margin:
        return "equivalent"
    if lo > 0:
        return "better"
    if hi < 0:
        return "worse"
    return "inconclusive"


def boot_stat(values, stat: str = "iqm", level: float = 0.95, seed: int = 0, draws: int = BOOT) -> dict:
    """A statistic of per-seed scores with its percentile bootstrap interval over seeds."""
    values, rng, f = np.asarray(values, float), np.random.default_rng(seed), STATS[stat]
    out = [f(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)]
    lo, hi = np.percentile(out, [50 * (1 - level), 100 - 50 * (1 - level)])
    return dict(value=f(values), lo=float(lo), hi=float(hi), n=len(values))


def compare(a, b, *, resample: str = "seeds", pair_seeds: bool = False, stat: str = "mean",
            level: float = 0.95, margin: float = MARGIN, seed: int = 0, draws: int = BOOT) -> dict:
    """Policy A against comparator B on the same episodes, and the verdict.

    a, b       (units, episodes); a group with one unit (a rule, a constant) is never resampled.
    resample   "seeds": rows only. "seeds_and_episodes": rows and episode columns.
    pair_seeds True: row i of A is paired with row i of B (needs equal counts) and the statistic
               is taken over the per-seed differences. False: the statistic of A's seed scores
               minus the statistic of B's, the groups resampled independently.
    """
    a, b = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float))
    assert a.shape[1] == b.shape[1], "A and B must be scored on the same episodes"
    assert resample in ("seeds", "seeds_and_episodes")
    assert not pair_seeds or len(a) == len(b), "paired seeds need the same number of seeds"
    f, rng = STATS[stat], np.random.default_rng(seed)

    def difference(sa, sb):                     # sa, sb: per-seed scores
        return f(sa - sb) if pair_seeds else f(sa) - f(sb)

    am, bm = a.mean(axis=1), b.mean(axis=1)
    out = np.empty(draws)
    for k in range(draws):
        ra = rng.integers(0, len(a), len(a))
        rb = ra if pair_seeds else rng.integers(0, len(b), len(b))
        if resample == "seeds":
            out[k] = difference(am[ra], bm[rb])
        else:
            e = rng.integers(0, a.shape[1], a.shape[1])
            out[k] = difference(a[np.ix_(ra, e)].mean(axis=1), b[np.ix_(rb, e)].mean(axis=1))
    lo, hi = (float(x) for x in np.percentile(out, [50 * (1 - level), 100 - 50 * (1 - level)]))
    m = margin * abs(f(bm))
    return dict(diff=difference(am, bm), lo=lo, hi=hi, margin=m, verdict=verdict(lo, hi, m), level=level,
                a=f(am), b=f(bm), seeds_a=len(a), seeds_b=len(b), episodes=a.shape[1])


def welch(a_scores, b_scores) -> float:
    """Two-sided p-value of Welch's t-test on per-seed scores. A comparator with a single score
    (a rule) has no variance of its own, so the test is then the one-sample t-test against it."""
    from scipy import stats
    a, b = np.asarray(a_scores, float).ravel(), np.asarray(b_scores, float).ravel()
    if len(b) == 1:
        return float(stats.ttest_1samp(a, b[0]).pvalue)
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def adjust(pvalues, method: str) -> list:
    """Multiplicity-adjusted p-values: "holm" or "bonferroni"."""
    p = np.asarray(pvalues, float)
    n = len(p)
    if method == "bonferroni":
        return [float(min(1.0, x * n)) for x in p]
    assert method == "holm"
    order, out, running = np.argsort(p), np.empty(n), 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        out[i] = min(1.0, running)
    return [float(x) for x in out]
