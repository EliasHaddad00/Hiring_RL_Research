"""Exact small instance: prototype optima, overflow rule agreement, and simulation of the optimal policy."""
import itertools

import numpy as np
import pytest

import hiring.env as env_module
from helpers import random_valid_action
from hiring.config import episode_seeds, load_scenario
from hiring.env import HiringEnv
from hiring.evaluation import evaluate
from hiring.exact_dp import ExactPolicy, ExactSolver
from hiring.pool import overflow_drop_index
from hiring.rules import RandomPolicy

SMALL = load_scenario("small_exact")
TOL = 0.0005          # the prototype values are printed to four decimals
PROTOTYPE = {10: (0.5838, 86194), 12: (0.6335, 165558), 15: (0.6947, 357279)}   # optimum, start-of-day states


def solver(T, mode):
    return ExactSolver(SMALL.with_(horizon=T, max_candidates=T), mode)


@pytest.fixture(scope="module")
def auto12():
    s = ExactSolver(SMALL, "automatic")
    s.value()
    return s


@pytest.mark.parametrize("T", [10, 12])
def test_choose_mode_reproduces_prototype_optimum(T):
    s = solver(T, "choose")
    assert s.value() == pytest.approx(PROTOTYPE[T][0], abs=TOL)
    assert len(s.memo["W"]) == PROTOTYPE[T][1]


@pytest.mark.slow
def test_choose_mode_reproduces_prototype_optimum_T15():
    s = solver(15, "choose")
    assert s.value() == pytest.approx(PROTOTYPE[15][0], abs=TOL)
    assert len(s.memo["W"]) == PROTOTYPE[15][1]


def test_automatic_drop_optimum_is_close_to_and_not_above_choose(auto12):
    choose, auto = solver(12, "choose").value(), auto12.value()
    assert auto <= choose + 1e-12                    # losing a choice cannot help
    assert (choose - auto) / choose < 0.01           # agreed stop condition: 1%


def test_solver_overflow_matches_shared_rule(auto12):
    b = auto12.b
    rng = np.random.default_rng(0)
    for t1, t2 in itertools.combinations_with_replacement(range(b.n_types), 2):
        d1, d2 = sorted(rng.choice(np.arange(1, 10), 2, replace=False))
        pool = tuple(sorted([(t1, int(d1)), (t2, int(d2))]))
        for r in range(b.R):
            new = tuple(sorted(pool + ((b.fresh_type(r), 0),)))
            j = overflow_drop_index([b.naive_mean(ty) for ty, _ in new], [d for _, d in new], [ty for ty, _ in new])
            assert auto12.after_arrival(5, pool, r) == new[:j] + new[j + 1:]


def test_environment_and_solver_drop_the_same_candidate(auto12, monkeypatch):
    """Record every overflow the environment resolves and replay it through the solver."""
    calls = []

    def recording(means, days, types=None):
        j = overflow_drop_index(means, days, types)
        calls.append((list(types), list(days), j))
        return j

    monkeypatch.setattr(env_module, "overflow_drop_index", recording)
    env, rng = HiringEnv(SMALL), np.random.default_rng(1)
    for seed in episode_seeds("dev", 3000):
        env.reset(seed=seed)
        done = False
        while not done:
            a = random_valid_action(env, rng) if rng.random() < 0.3 else 2 * env.N
            *_, done, _, _ = env.step(a)
    assert len(calls) > 300
    b = auto12.b
    for types, days, j in calls:
        states = list(zip(types, days))
        kept_env = tuple(sorted(states[:j] + states[j + 1:]))
        new = next(i for i, (_, d) in enumerate(states) if d == 0)
        r = b.types[states[new][0]][0]
        pool = tuple(sorted(states[:new] + states[new + 1:]))
        assert auto12.after_arrival(5, pool, r) == kept_env


def test_optimal_policy_shape(auto12):
    b = auto12.b
    top, low = b.fresh_type(2), b.fresh_type(0)
    assert auto12.offer_choice(12, ((top, 0),)) == 0                 # last day: offer any positive estimate
    assert auto12.offer_choice(12, ((low, 0),)) is None              # but not a negative one
    assert auto12.interview_choice(12, ((top, 0),)) is None          # no interviews on the last day
    assert auto12.value() > auto12.U(1, ()) - 1e-12


def _mc(auto12, n):
    res = evaluate(SMALL, ExactPolicy(auto12), seeds=episode_seeds("dev", n))
    return res["ret"].mean(), res["ret"].std(ddof=1) / np.sqrt(n)


def test_optimal_policy_simulated_in_environment_quick(auto12):
    mean, se = _mc(auto12, 4000)
    assert abs(mean - auto12.value()) < 3 * se


@pytest.mark.slow
def test_optimal_policy_simulated_in_environment_40000(auto12):
    """G1 acceptance: 40,000 episodes in our own environment, within three standard errors."""
    mean, se = _mc(auto12, 40000)
    assert se < 0.006
    assert abs(mean - auto12.value()) < 3 * se, (mean, se, auto12.value())


def test_no_policy_beats_the_optimum_in_simulation(auto12):
    n = 4000
    rand = evaluate(SMALL, RandomPolicy(), seeds=episode_seeds("dev", n))["ret"]
    assert rand.mean() + 3 * rand.std(ddof=1) / np.sqrt(n) < auto12.value()
