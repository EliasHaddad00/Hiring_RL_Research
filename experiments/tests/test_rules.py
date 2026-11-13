"""Hand-written rules and the random policy."""
import numpy as np
import pytest

from helpers import end_day, hand_draws
from hiring.config import episode_seeds, load_scenario
from hiring.env import HiringEnv
from hiring.evaluation import evaluate, run_episode, summarize
from hiring.rules import (FAMILIES, RandomPolicy, ThresholdRule, falling_bar, fixed_threshold,
                          one_interview)

STRESS = load_scenario("stress")
SEEDS = list(episode_seeds("dev", 200))
EXAMPLE = {  # parameter vectors of the right length for each family
    "static_threshold": [0.8],
    "resume_falling_bar": [1.0, 0.6, 0.15],
    "one_interview": [1.5, 0.9, 0.0, 0.25, 0.6, 0.1, 0.03],
    "up_to_two_interviews": [1.4, 1.3, 1.0, 0.8, 0.7, 0.7, 0.9, 0.4, 0.1],
}


def start(rule, sc, quality, arrival=None, **kw):
    env = HiringEnv(sc)
    env.reset(options={"draws": hand_draws(sc, arrival or [0] * len(quality), quality, **kw)})
    rule.reset(env)
    return env


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("belief", ["naive", "survival"])
@pytest.mark.parametrize("compat", [False, True])
def test_rules_only_take_valid_actions(family, belief, compat):
    rule = FAMILIES[family](EXAMPLE[family], belief=belief, v3_compat=compat)
    res = evaluate(STRESS, rule, seeds=SEEDS[:40])       # strict environment: an invalid action raises
    assert np.all(res["hires"] <= STRESS.roles)
    if rule.kmax == 0:
        assert res["interviews"].sum() == 0
    else:
        assert res["interviews"].sum() > 0


def test_random_policy_is_valid_and_reproducible():
    a = evaluate(STRESS, RandomPolicy(seed=1), seeds=SEEDS[:30])
    b = evaluate(STRESS, RandomPolicy(seed=1), seeds=SEEDS[:30])
    c = evaluate(STRESS, RandomPolicy(seed=2), seeds=SEEDS[:30])
    assert np.array_equal(a["ret"], b["ret"]) and not np.array_equal(a["ret"], c["ret"])


def test_fixed_threshold_offers_exactly_above_the_bar():
    # naive estimate = resume * 0.3077; resumes 3.0 and 2.0 give 0.923 and 0.615
    rule = fixed_threshold(0.8)
    env = start(rule, STRESS, [3.0, 2.0])
    assert rule.act(env) == env.N            # offer slot 0
    env.step(env.N)
    assert rule.act(env) == 2 * env.N        # 0.615 < 0.8: pass, never interview


def test_one_interview_rule_three_zones():
    # fast-track bar 1.5, interview bar 0.6, post-interview bar 0.9 (no falls, no role term)
    rule = one_interview([1.5, 0.9, 0.0, 0.0, 0.6, 0.0, 0.0])
    env = start(rule, STRESS, [6.0, 3.0, 1.0])            # estimates 1.85, 0.92, 0.31
    assert rule.act(env) == env.N                         # fast-track the top candidate
    env.step(env.N)
    a = rule.act(env)
    assert a == 0                                         # interview the middle one
    env.step(a)
    assert rule.act(env) == 2 * env.N                     # the low one is passed; day ends
    end_day(env)
    # interview read q = 3.0, estimate now (3/2.25 + 3) / 2.444 = 1.77 >= 0.9: offer
    assert env.view().n_obs[0] == 1 and rule.act(env) == env.N


def test_rule_never_interviews_twice_when_kmax_is_one():
    rule = one_interview([9.0, 9.0, 0.0, 0.0, -9.0, 0.0, 0.0])   # never offers, always wants to interview
    env = start(rule, STRESS, [1.0])
    for _ in range(5):
        while (a := rule.act(env)) != 2 * env.N:
            env.step(a)
        end_day(env)
    assert env.n_done[0] == 1


def test_bar_falls_with_time_and_rises_with_hires():
    rule = falling_bar([1.0, 0.6, 0.2])
    assert rule.offer_bar(0, 0.5, 0) == pytest.approx(0.7)
    assert rule.offer_bar(0, 0.5, 2) == pytest.approx(1.1)        # fewer open roles: higher bar
    assert rule.offer_bar(0, 0.0, 0) > rule.offer_bar(0, 0.9, 0)


def test_default_rule_rechecks_bar_after_each_hire_but_v3_compat_does_not():
    # estimates 0.923 and 0.862; bar 0.8 + 0.1 * hired. After one hire the bar is 0.9.
    quality = [3.0, 2.8]
    for compat, expected_hires in [(False, 1), (True, 2)]:
        rule = ThresholdRule(0, [0.8], [0.0], [99.0], 0.0, r=0.1, v3_compat=compat)
        env = start(rule, STRESS, quality)
        while (a := rule.act(env)) != 2 * env.N:
            env.step(a)
        assert env.stats["hires"] == expected_hires


def test_v3_compat_caps_offers_per_day():
    sc = STRESS.with_(n_slots=20, roles=50)
    for compat, expected in [(True, 12), (False, 15)]:
        rule = fixed_threshold(0.0, v3_compat=compat)
        env = start(rule, sc, [3.0] * 15, u_accept=1.0)           # everyone declines
        n = 0
        while (a := rule.act(env)) != 2 * env.N:
            env.step(a); n += 1
        assert n == expected


def test_belief_switch_changes_a_decision():
    """A candidate whose naive estimate clears the bar but who has waited a week under heavy poaching."""
    sc = STRESS.with_(h0=0.10)
    decisions = {}
    for belief in ("naive", "survival"):
        rule = ThresholdRule(0, [0.8], [0.0], [99.0], 0.0, start_day=7, belief=belief)
        env = start(rule, sc, [3.25])                              # naive estimate 1.0
        for _ in range(7):
            end_day(env)
        decisions[belief] = rule.act(env)
        assert env.view().naive[0] == pytest.approx(1.0)
        assert env.view().survival[0] == pytest.approx(0.482, abs=0.002)
    assert decisions["naive"] == env.N and decisions["survival"] == 2 * env.N


def test_random_is_far_below_a_sensible_rule():
    rand = summarize(evaluate(STRESS, RandomPolicy(), seeds=SEEDS))
    rule = summarize(evaluate(STRESS, fixed_threshold(0.8), seeds=SEEDS))
    assert rule["mean"] > rand["mean"] + 1.0
    assert run_episode(HiringEnv(STRESS), RandomPolicy(), seed=SEEDS[0])["ret"] == pytest.approx(
        run_episode(HiringEnv(STRESS), RandomPolicy(), seed=SEEDS[0])["ret"])
