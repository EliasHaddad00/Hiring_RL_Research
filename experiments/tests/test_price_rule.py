"""Price-directed rule: it follows the one-candidate tables, through the same view as every policy."""
import numpy as np
import pytest

from helpers import hand_draws
from hiring.bounds import INTERVIEW, OFFER, PASS, WAIT, price_tables
from hiring.config import episode_seeds, load_scenario
from hiring.env import HiringEnv
from hiring.evaluation import evaluate
from hiring.rules import FAMILIES, RULE_FAMILIES, PriceDirectedRule

SC = load_scenario("stress")                          # h0 = 0.05
SEEDS = list(episode_seeds("dev", 150))


@pytest.fixture(scope="module")
def tb():
    return price_tables(SC, "survival", 0.0)


def table_action(tb, rule, env):
    """What the tables say for the single shown candidate, at the price the rule computes."""
    v = env.view()
    r = v.horizon - 1 - v.t
    i, f = tb.locate(v.naive)
    values = tb.V[r][v.days, v.n_obs][np.arange(1), :, i] * (1 - f)[:, None] \
        + tb.V[r][v.days, v.n_obs][np.arange(1), :, i + 1] * f[:, None]
    iz = rule.price_index(v, values)
    return int(tb.A[r][v.days[0], v.n_obs[0], iz, int(np.rint(i[0] + f[0]))])


@pytest.mark.parametrize("quality", [-1.0, 0.3, 0.9, 1.4, 2.5])
def test_with_one_candidate_the_rule_takes_the_table_action(tb, quality):
    env, rule = HiringEnv(SC), PriceDirectedRule()
    env.reset(options={"draws": hand_draws(SC, [0], [quality])})
    rule.reset(env)
    N, seen = env.N, set()
    done = False
    while not done and len(env.view().ids) == 1:
        expected = table_action(tb, rule, env)
        a = rule.act(env)
        seen.add(expected)
        v = env.view()
        if expected == OFFER and v.offers_open:
            assert a == N
        elif expected == INTERVIEW and env.action_masks()[0]:
            assert a == 0
        else:
            assert a == 2 * N                         # pass, wait, or an action the phase does not allow
        _, _, done, _, _ = env.step(a)
    assert seen <= {PASS, OFFER, INTERVIEW, WAIT}


def test_no_invalid_action_and_no_last_day_interview():
    """The strict environment raises on an invalid action; interviews on the last day are masked."""
    env = HiringEnv(SC)
    for belief in ("survival", "naive"):
        rule = PriceDirectedRule(belief=belief)
        for s in SEEDS[:60]:
            env.reset(seed=s)
            rule.reset(env)
            done = False
            while not done:
                a = rule.act(env)
                assert env.action_masks()[a]
                assert not (a < env.N and env.t == SC.horizon - 1)
                _, _, done, _, _ = env.step(a)


def test_same_seed_gives_the_same_episode():
    a = evaluate(SC, PriceDirectedRule(), seeds=SEEDS[:40])
    b = evaluate(SC, PriceDirectedRule(), seeds=SEEDS[:40])
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_rule_interviews_and_hires():
    res = evaluate(SC, PriceDirectedRule(), seeds=SEEDS)
    assert res["hires"].mean() > 2.0 and 2.0 < res["interviews"].mean() < 60.0
    assert res["ret"].mean() > 3.0                    # far above the random policy (-0.48)


def test_price_falls_with_fewer_days_left_and_parameters_act(tb):
    """More open roles or fewer days left make a hire cheaper; the two tuned parameters bite."""
    assert (np.diff(tb.Fcum, axis=0) >= -1e-9).all()               # more days left: future arrivals worth more
    base = evaluate(SC, PriceDirectedRule(), seeds=SEEDS)
    cheap = evaluate(SC, PriceDirectedRule(price_scale=0.5), seeds=SEEDS)
    dear = evaluate(SC, PriceDirectedRule(interview_price=0.5), seeds=SEEDS)
    assert cheap["ret"].mean() != base["ret"].mean()
    assert dear["interviews"].mean() < base["interviews"].mean()


def test_naive_version_uses_its_own_tables():
    naive, surv = price_tables(SC, "naive", 0.0), price_tables(SC, "survival", 0.0)
    assert naive is not surv and naive.belief == "naive"
    assert not np.allclose(naive.Fcum, surv.Fcum)
    a = evaluate(SC, PriceDirectedRule(belief="naive"), seeds=SEEDS)["ret"]
    b = evaluate(SC, PriceDirectedRule(belief="survival"), seeds=SEEDS)["ret"]
    assert not np.array_equal(a, b)


def test_tables_are_cached_and_ignore_the_slot_count():
    assert price_tables(SC.with_(n_slots=20), "survival", 0.0) is price_tables(SC, "survival", 0.0)
    assert price_tables(SC.with_(h0=0.02), "survival", 0.0) is not price_tables(SC, "survival", 0.0)


def test_rule_family_lists():
    assert set(RULE_FAMILIES) == set(FAMILIES) | {"price_directed"}
    assert "price_directed" not in FAMILIES           # the G1 scripts loop over the four earlier families
    rule = RULE_FAMILIES["price_directed"]([0.9, 0.01], belief="naive")
    assert rule.price_scale == 0.9 and rule.interview_price == 0.01 and rule.belief == "naive"
