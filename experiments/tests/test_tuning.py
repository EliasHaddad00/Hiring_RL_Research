"""Rule tuning: pool lock, repeatability, the budget, and the last-day rule by estimate."""
import numpy as np
import pytest

from helpers import end_day, hand_draws
from hiring.config import load_scenario
from hiring.env import HiringEnv
from hiring.evaluation import evaluate
from hiring.rules import FAMILIES, falling_bar, fixed_threshold
from hiring.tuning import SCALES, clean, earlier_key, start_vector, tune

SC = load_scenario("stress")
SHORT = SC.with_(horizon=6, max_candidates=20)


# ---- tuning ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("pool", ["dev", "evaluation"])
def test_tuning_refuses_any_pool_but_the_tuning_pool(pool):
    with pytest.raises(AssertionError):
        tune(SC, "static_threshold", "naive", budget=2, pool=pool, episodes=10)


def test_tuning_is_repeatable_and_uses_exactly_the_budget():
    a = tune(SC, "resume_falling_bar", "naive", budget=6, episodes=60)
    b = tune(SC, "resume_falling_bar", "naive", budget=6, episodes=60)
    assert a["history"] == b["history"]
    assert len(a["history"]) == 6 and [h["configuration"] for h in a["history"]] == list(range(1, 7))
    assert len({tuple(h["params"]) for h in a["history"]}) == 6            # no configuration is spent twice
    c = tune(SC, "resume_falling_bar", "naive", budget=6, episodes=60, search_seed=1)
    assert c["history"][0] == a["history"][0] and c["history"][1:] != a["history"][1:]


def test_best_is_the_highest_scoring_configuration_and_never_below_the_start():
    r = tune(SC, "one_interview", "survival", budget=8, episodes=80)
    means = [h["mean"] for h in r["history"]]
    assert r["best"]["mean"] == pytest.approx(max(means)) and r["best"]["mean"] >= means[0]
    assert r["best_params"] == r["history"][r["best_configuration"] - 1]["params"]


def test_start_vectors_and_scales_match_every_family():
    for family, scale in SCALES.items():
        for sc in (SC, SC.with_(h0=0.0, accept_decay=0.0), load_scenario("calibrated")):
            assert len(start_vector(family, sc)) == len(scale)
    assert set(SCALES) == set(FAMILIES) | {"price_directed"}
    assert earlier_key(SC) == "h0=0.05,b=0.03"
    assert earlier_key(SC.with_(h0=0.0, accept_decay=0.0)) == "h0=0.0,b=0.0"
    assert earlier_key(SC.with_(h0=0.0)) == "h0=0.0,b=0.03"
    assert earlier_key(SC.with_(h0=0.10)) == "h0=0.1,b=0.03"
    assert start_vector("price_directed", SC) == [1.0, 0.0]


def test_price_directed_parameters_stay_valid():
    assert clean("price_directed", [-1.0, -0.3]) == [0.3, 0.0]
    assert clean("price_directed", [1.234, 0.0123])[1] == pytest.approx(0.01)


def test_calibrated_scenario_values():
    c = load_scenario("calibrated")
    assert (c.resume_sd, c.interview_sd, c.accept_base, c.accept_decay, c.h0) == (2.4, 2.2, 0.85, 0.003, 0.02)
    assert (c.horizon, c.roles, c.arrival_rate, c.interviews_per_day, c.max_interviews) == (60, 3, 1.5, 3, 3)


# ---- last-day rule ---------------------------------------------------------------------------------

def last_day_action(rule, quality):
    """One candidate arriving on the last day with a noise-free resume."""
    env = HiringEnv(SHORT)
    env.reset(options={"draws": hand_draws(SHORT, [SHORT.horizon - 1], [quality])})
    rule.reset(env)
    for _ in range(SHORT.horizon - 1):
        end_day(env)
    assert env.t == SHORT.horizon - 1 and len(env.view().ids) == 1
    return rule.act(env), env


def test_survival_rules_offer_to_any_positive_estimate_on_the_last_day():
    a, env = last_day_action(fixed_threshold(1.5, belief="survival"), quality=0.4)
    assert 0 < env.view().survival[0] < 1.5 and a == env.N            # below the tuned bar, still offered
    a, env = last_day_action(fixed_threshold(1.5, belief="survival"), quality=-0.4)
    assert env.view().survival[0] < 0 and a == 2 * env.N              # negative estimate: no offer


def test_naive_rules_keep_their_tuned_bar_on_the_last_day():
    a, env = last_day_action(fixed_threshold(1.5, belief="naive"), quality=0.4)
    assert a == 2 * env.N
    a, env = last_day_action(falling_bar([1.5, 1.4, 0.0], belief="naive"), quality=0.4)
    assert a == 2 * env.N                                             # bar on the last day is about 0.33


def test_last_day_rule_only_acts_on_the_last_day_and_can_be_switched_off():
    rule = fixed_threshold(1.5, belief="survival")
    env = HiringEnv(SHORT)
    env.reset(options={"draws": hand_draws(SHORT, [0], [0.4])})
    rule.reset(env)
    assert rule.act(env) == 2 * env.N                                 # day 0: the bar applies
    a, env = last_day_action(fixed_threshold(1.5, belief="survival", last_day_zero=False), quality=0.4)
    assert a == 2 * env.N
    assert fixed_threshold(1.0, belief="naive").last_day_zero is False
    assert fixed_threshold(1.0, belief="survival").last_day_zero is True
    assert fixed_threshold(1.0, belief="naive", v3_compat=True).last_day_zero is False


def test_last_day_rule_never_lowers_a_survival_rule_on_average():
    seeds = range(1_000_000, 1_000_400)
    for h0 in (0.02, 0.10):
        sc = SC.with_(h0=h0)
        on = evaluate(sc, falling_bar([1.2, 0.3, 0.1], belief="survival"), seeds=seeds)["ret"]
        off = evaluate(sc, falling_bar([1.2, 0.3, 0.1], belief="survival", last_day_zero=False), seeds=seeds)["ret"]
        assert not np.array_equal(on, off)
        assert on.mean() > off.mean() - 2 * (on - off).std(ddof=1) / np.sqrt(len(on))
