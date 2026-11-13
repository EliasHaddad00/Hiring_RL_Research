"""Belief checks: quadrature targets, an independent integration, and simulated survivors."""
import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from hiring.belief import BinnedBelief, GaussianBelief, hazard
from hiring.config import load_scenario

STRESS = load_scenario("stress")


def belief(h0):
    return GaussianBelief(STRESS.with_(h0=h0))


def test_naive_posterior_closed_form():
    b = belief(0.05)
    assert b.variance(0) == pytest.approx(1 / (1 + 1 / 2.25))        # 0.6923
    assert b.variance(1) == pytest.approx(1 / (1 + 1 / 2.25 + 1))    # 0.4091
    # resume 2.0 alone, then one interview of 1.0
    assert b.naive_mean(2.0, 0.0, 0) == pytest.approx((2.0 / 2.25) / (1 + 1 / 2.25))
    assert b.naive_mean(2.0, 1.0, 1) == pytest.approx((2.0 / 2.25 + 1.0) / (2 + 1 / 2.25))
    assert b.updated_mean(b.naive_mean(2.0, 0.0, 0), 0, 1.0) == pytest.approx(b.naive_mean(2.0, 1.0, 1))


@pytest.mark.parametrize("h0,target", [(0.02, 0.883), (0.05, 0.721), (0.10, 0.482)])
def test_corrected_figures_after_a_week(h0, target):
    """Estimate 1.0 from the resume alone, still available after 7 days."""
    assert belief(h0).survival_mean(1.0, 0, 7) == pytest.approx(target, abs=0.002)


def test_prototype_check_values():
    """Values printed by notes/prototypes/a.py (prototype, reproduced here)."""
    b3, b5 = belief(0.03), belief(0.05)
    assert b3.survival_mean(1.5, 1, 10) == pytest.approx(1.312, abs=0.002)
    assert b3.first_order_mean(1.5, 1, 10) == pytest.approx(1.305, abs=0.002)
    assert b5.survival_mean(1.5, 1, 30) == pytest.approx(0.806, abs=0.002)
    assert b5.first_order_mean(1.5, 1, 30) == pytest.approx(0.525, abs=0.002)


def test_no_poaching_means_no_correction():
    b = belief(0.0)
    mean, var = b.survival_moments(0.7, 1, 20)
    assert mean == pytest.approx(0.7) and var == pytest.approx(b.variance(1))
    assert b.survive_tonight(0.7, 1, 20) == pytest.approx(1.0)


@pytest.mark.parametrize("h0,mu,n,d", [(0.02, 0.3, 0, 3), (0.05, 1.0, 0, 7), (0.05, 1.5, 2, 30),
                                       (0.10, -0.5, 1, 15), (0.10, 2.0, 3, 59)])
def test_quadrature_against_independent_integration(h0, mu, n, d):
    b = belief(h0)
    sd = np.sqrt(b.variance(n))
    f = lambda q, k: q ** k * norm.pdf(q, mu, sd) * (1 - hazard(q, h0, 0.6)) ** d
    pts = [np.log(1 / h0) / 0.6]   # kink where the hazard reaches 1
    z0 = integrate.quad(f, mu - 12 * sd, mu + 12 * sd, args=(0,), points=pts, limit=200)[0]
    z1 = integrate.quad(f, mu - 12 * sd, mu + 12 * sd, args=(1,), points=pts, limit=200)[0]
    g = lambda q: norm.pdf(q, mu, sd) * (1 - hazard(q, h0, 0.6)) ** (d + 1)
    zn = integrate.quad(g, mu - 12 * sd, mu + 12 * sd, points=pts, limit=200)[0]
    assert b.survival_mean(mu, n, d) == pytest.approx(z1 / z0, abs=1e-4)
    assert b.survive_tonight(mu, n, d) == pytest.approx(zn / z0, abs=1e-4)


def test_predictive_interview_moments():
    b = belief(0.05)
    mean, var = b.survival_moments(1.0, 1, 10)
    y, w = b.predictive_interview(1.0, 1, 10)
    assert w.sum() == pytest.approx(1.0)
    m = (w * y).sum()
    assert m == pytest.approx(mean, abs=1e-8)
    assert (w * y * y).sum() - m * m == pytest.approx(var + STRESS.interview_sd ** 2, abs=1e-6)


def test_vectorised_matches_scalar():
    b = belief(0.05)
    mu, n, d = np.array([0.2, 1.0, -0.4]), np.array([0, 1, 3]), np.array([0, 7, 22])
    vec = b.survival_mean(mu, n, d)
    assert np.allclose(vec, [b.survival_mean(*t) for t in zip(mu, n, d)])


@pytest.mark.parametrize("h0", [0.02, 0.05, 0.10])
def test_calibration_against_simulated_survivors(h0):
    """Simulate candidates with 0, 1 or 2 interviews, poach them day by day, and check that among
    survivors grouped by naive estimate and age the average true quality equals the corrected mean."""
    rng = np.random.default_rng(20261112)
    sc = STRESS.with_(h0=h0)
    b = GaussianBelief(sc)
    n_sim = 1_500_000
    q = rng.standard_normal(n_sim)
    n_int = rng.integers(0, 3, n_sim)
    resume = q + sc.resume_sd * rng.standard_normal(n_sim)
    isum = np.where(n_int >= 1, q + rng.standard_normal(n_sim), 0.0) + np.where(n_int >= 2, q + rng.standard_normal(n_sim), 0.0)
    mu = b.naive_mean(resume, isum, n_int)
    alive = np.ones(n_sim, bool)
    haz = hazard(q, h0, sc.quality_slope)
    edges = np.array([-1.0, -0.25, 0.25, 0.75, 1.25, 2.0])
    zs, bias = [], []
    for d in range(1, 15):
        alive &= rng.random(n_sim) >= haz
        if d not in (3, 7, 14):
            continue
        corrected = b.survival_mean(mu, n_int, d)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = alive & (mu >= lo) & (mu < hi)
            if sel.sum() < 2000:
                continue
            diff = q[sel] - corrected[sel]
            zs.append(diff.mean() / (diff.std() / np.sqrt(sel.sum())))
            bias.append((q[sel] - mu[sel]).mean())
    zs = np.array(zs)
    assert len(zs) >= 12
    assert np.abs(zs).max() < 4.0, zs            # corrected mean is calibrated in every cell
    assert abs(zs.mean()) < 3 / np.sqrt(len(zs))   # and not biased overall
    assert min(bias) < -0.05                      # while the naive mean is visibly too high


# ---- binned tables (small instance) ---------------------------------------------------------------

SMALL = load_scenario("small_exact")


def test_binned_table_structure():
    b = BinnedBelief(SMALL)
    assert b.n_types == 30 and b.R == 3 and b.B == 3
    assert b.prior_r.sum() == pytest.approx(1.0)
    fresh = b.fresh_type(1)
    assert np.allclose(b.PB[fresh].sum(0), 1.0)
    done = b.tid[(1, (2, 0, 0))]
    assert np.all(b.nxt[done] == -1)
    assert np.all((b.S > 0) & (b.S <= 1))


def test_binned_tables_against_independent_integration():
    b = BinnedBelief(SMALL)
    h0, sl = SMALL.h0, SMALL.quality_slope
    # resume in the top bin, one interview in the middle bin, waited 4 days
    t = b.tid[(2, (0, 1, 0))]
    like = lambda q: (norm.sf((0.8 - q) / 1.5) * (norm.cdf(1.0 - q) - norm.cdf(0.0 - q))
                      * norm.pdf(q) * (1 - hazard(q, h0, sl)) ** 4)
    z0 = integrate.quad(like, -8, 8, limit=200)[0]
    z1 = integrate.quad(lambda q: q * like(q), -8, 8, limit=200)[0]
    zs = integrate.quad(lambda q: like(q) * (1 - hazard(q, h0, sl)), -8, 8, limit=200)[0]
    assert b.M[t, 4] == pytest.approx(z1 / z0, abs=1e-6)
    assert b.S[t, 4] == pytest.approx(zs / z0, abs=1e-6)
    assert b.naive_mean(t) > b.survival_mean(t, 4)


def test_binned_calibration_against_simulation():
    rng = np.random.default_rng(7)
    b = BinnedBelief(SMALL)
    n = 800_000
    q = rng.standard_normal(n)
    rbin = np.searchsorted(SMALL.resume_cuts, q + 1.5 * rng.standard_normal(n))
    ibin = np.searchsorted(SMALL.interview_cuts, q + rng.standard_normal(n))
    alive = np.ones(n, bool)
    for _ in range(5):
        alive &= rng.random(n) >= hazard(q, SMALL.h0, SMALL.quality_slope)
    for r in range(3):
        for ib in range(3):
            sel = alive & (rbin == r) & (ibin == ib)
            cnt = [0, 0, 0]
            cnt[ib] = 1
            t = b.tid[(r, tuple(cnt))]
            se = q[sel].std() / np.sqrt(sel.sum())
            assert abs(q[sel].mean() - b.M[t, 5]) < 4 * se
