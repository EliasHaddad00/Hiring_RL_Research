"""Lagrangian upper bound: validity against known optima, known answers, stability, simulation."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from hiring.belief import GaussianBelief
from hiring.bounds import (BinnedTables, GaussianTables, lagrangian_bound, relaxed_value,
                           simulate_relaxed)
from hiring.config import episode_seeds, load_scenario
from hiring.evaluation import evaluate
from hiring.exact_dp import ExactSolver
from hiring.rules import FAMILIES

SMALL = load_scenario("small_exact")
STRESS = load_scenario("stress")                      # h0 = 0.05, decay 0.03: the primary setting
COARSE = dict(n_grid=91, z_points=16)                 # for shape tests, where only the order matters
G1 = Path(__file__).resolve().parents[2] / "results" / "g1"


@pytest.fixture(scope="module")
def primary():
    tb = GaussianTables(STRESS)
    return tb, lagrangian_bound(STRESS, tables=tb)


# ---- 1. the bound is above the exact optimum on the small instance -------------------------------

@pytest.mark.parametrize("T", [10, 12])
def test_small_instance_bound_is_above_both_exact_optima(T):
    sc = SMALL.with_(horizon=T, max_candidates=T)
    bound = lagrangian_bound(sc, lam_grid=(0.0, 0.01, 0.02, 0.05))["bound"]
    choose, auto = ExactSolver(sc, "choose").value(), ExactSolver(sc, "automatic").value()
    assert bound >= choose >= auto
    if T == 12:
        assert choose == pytest.approx(0.63348, abs=1e-5) and auto == pytest.approx(0.63333, abs=1e-5)


@pytest.mark.slow
def test_small_instance_bound_is_above_the_exact_optimum_T15():
    sc = SMALL.with_(horizon=15, max_candidates=15)
    assert lagrangian_bound(sc)["bound"] >= ExactSolver(sc, "choose").value()


def test_small_instance_bound_holds_for_every_price():
    """Any z, lam >= 0 gives a valid bound, not only the minimiser."""
    tb = BinnedTables(SMALL)
    optimum = ExactSolver(SMALL, "choose").value()
    for lam in (0.0, 0.03):
        zs = np.linspace(0.0, 2.0, 21)
        assert (relaxed_value(SMALL, tb.solve(zs, lam), zs, lam) >= optimum).all()


def test_one_candidate_table_equals_the_exact_solver_with_a_single_candidate():
    """With one candidate and price zero nothing is relaxed, so the two recursions must agree.

    The exact solver is started on day 1 with one fresh candidate in the pool and no later
    arrivals. This checks the offer, interview, wait and overnight conventions against exact_dp.
    """
    sc = SMALL.with_(horizon=6, max_candidates=6)
    tb = BinnedTables(sc)
    one_candidate_value = float(tb.solve([0.0])[sc.horizon - 1, 0])       # arrives with T - 1 days remaining
    solver = ExactSolver(sc, "automatic")
    exact = sum(solver.prior_r[r] * solver.U(1, ((solver.fresh[r], 0),)) for r in range(len(solver.prior_r)))
    assert exact >= one_candidate_value - 1e-12                           # later arrivals can only help
    lonely = ExactSolver(sc.with_(arrival_rate=0.0), "automatic")
    exact_alone = sum(lonely.prior_r[r] * lonely.U(1, ((lonely.fresh[r], 0),)) for r in range(len(lonely.prior_r)))
    assert one_candidate_value == pytest.approx(exact_alone, abs=1e-12)


# ---- 2. closed form -------------------------------------------------------------------------------

def test_closed_form_without_interviews_or_pressure():
    """No interviews, no decay, no poaching: each arrival is worth 0.9 * E[(mu0 - z)+]."""
    sc = STRESS.with_(max_interviews=0, h0=0.0, accept_decay=0.0)
    sd = np.sqrt(1.0 - GaussianBelief(sc).variance(0))
    arrivals = sc.arrival_rate * sc.horizon

    def value(z):
        u = z / sd
        return sc.roles * z + arrivals * sc.accept_base * sd * (norm.pdf(u) - u * norm.sf(u))

    exact = minimize_scalar(value, bounds=(0.0, 3.0), method="bounded").fun
    assert lagrangian_bound(sc)["bound"] == pytest.approx(exact, abs=0.002)


# ---- 3. shape --------------------------------------------------------------------------------------

def test_bound_does_not_rise_with_poaching():
    b = [lagrangian_bound(STRESS.with_(h0=h), **COARSE)["bound"] for h in (0.0, 0.02, 0.05, 0.10)]
    assert all(x >= y for x, y in zip(b, b[1:])), b


def test_bound_does_not_rise_with_decay_or_interview_cost():
    base = lagrangian_bound(STRESS, **COARSE)["bound"]
    assert lagrangian_bound(STRESS.with_(accept_decay=0.0), **COARSE)["bound"] >= base
    assert lagrangian_bound(STRESS.with_(interview_cost=0.10), **COARSE)["bound"] <= base
    assert lagrangian_bound(STRESS.with_(interview_cost=0.0), **COARSE)["bound"] >= base


def test_bound_does_not_fall_with_roles_or_interview_cap():
    b = [lagrangian_bound(STRESS.with_(roles=m), **COARSE)["bound"] for m in (1, 2, 3)]
    assert b[0] <= b[1] <= b[2]
    three = lagrangian_bound(STRESS, **COARSE)["bound"]
    six = lagrangian_bound(STRESS.with_(max_interviews=6), **COARSE)["bound"]
    assert six >= three - 1e-9


def test_interview_price_of_zero_is_best_when_slots_are_slack(primary):
    tb, _ = primary
    r = lagrangian_bound(STRESS, lam_grid=(0.0, 0.005, 0.02), tables=tb)
    assert r["lam"] == 0.0 and all(x["bound"] >= r["bound"] for x in r["per_lam"])


# ---- 4. numerical stability ------------------------------------------------------------------------

def test_finer_grid_changes_the_bound_by_less_than_0_01(primary):
    _, coarse = primary
    fine = lagrangian_bound(STRESS, n_grid=361, n_nodes=160, n_noise=30)
    assert abs(coarse["bound"] - fine["bound"]) < 0.01
    assert fine["bound"] <= coarse["bound"] + 1e-6       # interpolation error is on the safe side


def test_wider_grid_changes_the_bound_by_less_than_0_01(primary):
    _, base = primary
    wide = lagrangian_bound(STRESS, n_grid=261, mu_range=(-6.5, 8.0))
    assert abs(base["bound"] - wide["bound"]) < 0.01


# ---- 5. the tables agree with the belief module and with simulation --------------------------------

def test_tables_use_the_belief_module(primary):
    tb, _ = primary
    b = GaussianBelief(STRESS)
    for n, d in [(0, 0), (0, 7), (1, 3), (3, 20)]:
        assert np.allclose(tb.M[n, d], b.survival_mean(tb.grid, n, d), atol=1e-10)
        assert np.allclose(tb.S[n, d], b.survive_tonight(tb.grid, n, d), atol=1e-10)
    # the interview operator spreads exactly the mass that survives the night
    assert np.allclose(tb.W.sum(-1), tb.S[:tb.nmax], atol=1e-10)
    assert tb.w0.sum() == pytest.approx(1.0) and (tb.w0 >= 0).all()
    assert tb.w0 @ tb.grid ** 2 == pytest.approx(1.0 - b.variance(0), rel=0.01)


def test_naive_tables_ignore_survival():
    tb = GaussianTables(STRESS, belief="naive", n_grid=91)
    assert np.allclose(tb.M[0, 0], tb.grid, atol=1e-8) and np.allclose(tb.M[0, 30], tb.grid, atol=1e-8)
    assert np.allclose(tb.S[1, 0], tb.S[1, 30])           # tonight's risk does not depend on days waited
    assert (tb.S < 1).all()                               # but the risk of loss is still there


def test_simulated_relaxed_policy_matches_the_computed_value(primary):
    """The priced, unconstrained problem is simulated on the environment's own draws."""
    tb, res = primary
    z = res["z"]
    computed = float(relaxed_value(STRESS, tb.solve([z]), [z])[0])
    sim = simulate_relaxed(STRESS, tb, z, episode_seeds("dev", 3000))
    se = sim.std(ddof=1) / np.sqrt(len(sim))
    assert abs(sim.mean() - computed) < 3 * se, (sim.mean(), computed, se)


# ---- 6. the bound is above rules played in the environment -----------------------------------------

@pytest.mark.parametrize("h0", [0.02, 0.05, 0.10])
def test_bound_is_above_the_earlier_tuned_rules(h0):
    params = json.loads((G1 / "v3_fullprec.json").read_text())[f"h0={h0},b=0.03"]
    sc = STRESS.with_(h0=h0)
    bound = lagrangian_bound(sc, **COARSE)["bound"]
    for family in ("one_interview", "up_to_two_interviews"):
        res = evaluate(sc, FAMILIES[family](params[family]["par_full"]), seeds=episode_seeds("dev", 400))["ret"]
        assert bound > res.mean() + 3 * res.std(ddof=1) / np.sqrt(len(res))
