"""The pre-registered decision rule on made-up inputs with known verdicts, and the guards of the
G4 scripts (evaluation pool locked, open wording points undecided, resume after an interruption)."""
import sys
from pathlib import Path

import numpy as np
import pytest

from hiring.config import load_scenario
from hiring.decision import adjust, boot_stat, compare, iqm, verdict, welch
from hiring.evaluation import evaluate
from hiring.rules import RULE_FAMILIES

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))
import apply_decision_rule as adr  # noqa: E402
import score_final  # noqa: E402

SEEDS, EPISODES = 15, 400


def made_up(shift_per_seed, payoff=3.5, noise=0.5):
    """A rule (one row) and 15 'training seeds' on the same episodes: seed i is the rule plus shift i."""
    rng = np.random.default_rng(1)
    rule = payoff + noise * rng.standard_normal(EPISODES)
    agents = rule[None, :] + np.asarray(shift_per_seed, float)[:, None] + 0.01 * rng.standard_normal((SEEDS, EPISODES))
    return agents, rule[None, :]


def test_verdict_on_intervals_with_known_answers():
    assert verdict(-0.01, 0.02, 0.07) == "equivalent"
    assert verdict(0.10, 0.20, 0.07) == "better"
    assert verdict(-0.20, -0.10, 0.07) == "worse"
    assert verdict(-0.10, 0.10, 0.07) == "inconclusive"      # crosses zero and leaves the margin
    assert verdict(0.01, 0.10, 0.07) == "better"             # above zero, not inside the margin
    assert verdict(0.01, 0.05, 0.07) == "equivalent"         # above zero AND inside: equivalence takes precedence
    assert verdict(-0.07, 0.01, 0.07) == "inconclusive"      # touching the margin is not "entirely inside"


@pytest.mark.parametrize("shift, expected", [
    (np.linspace(-0.01, 0.01, SEEDS), "equivalent"),
    (np.linspace(0.20, 0.30, SEEDS), "better"),
    (np.linspace(-0.30, -0.20, SEEDS), "worse"),
    (np.tile([-0.4, 0.4, 0.0], 5), "inconclusive"),
    (np.full(SEEDS, 0.03), "equivalent"),                    # a sure gain smaller than the 2% margin (0.07)
])
@pytest.mark.parametrize("resample", ["seeds", "seeds_and_episodes"])
@pytest.mark.parametrize("stat", ["mean", "iqm"])
def test_decision_rule_on_made_up_scores(shift, expected, resample, stat):
    agents, rule = made_up(shift)
    r = compare(agents, rule, resample=resample, stat=stat, draws=400)
    assert r["verdict"] == expected
    assert r["lo"] <= r["diff"] <= r["hi"] and abs(r["margin"] - 0.02 * rule.mean()) < 1e-9


def test_actor_against_actor_paired_or_independent_seeds():
    a, rule = made_up(np.linspace(0.20, 0.30, SEEDS))
    b, _ = made_up(np.linspace(-0.01, 0.01, SEEDS))
    for pair in (False, True):
        assert compare(a, b, pair_seeds=pair, draws=400)["verdict"] == "better"
        assert compare(b, a, pair_seeds=pair, draws=400)["verdict"] == "worse"
    with pytest.raises(AssertionError):
        compare(a[:10], b, pair_seeds=True)                  # paired seeds need equal counts
    with pytest.raises(AssertionError):
        compare(a[:, :100], b)                               # not the same episodes


def test_exact_values_against_the_optimum():
    """H2a: 15 exact values (one column) against a constant. 99% of the optimum is equivalent, 96% is worse."""
    opt = 0.63333
    near, far = opt * np.linspace(0.989, 0.993, SEEDS), opt * np.linspace(0.955, 0.965, SEEDS)
    assert compare(near[:, None], [[opt]], draws=400)["verdict"] == "equivalent"
    assert compare(far[:, None], [[opt]], draws=400)["verdict"] == "worse"


def test_statistics_helpers():
    assert iqm([0, 1, 2, 3, 4, 5, 6, 100]) == 3.5            # the middle half of eight values
    assert iqm([1.0, 2.0, 6.0]) == 3.0                       # three values: the plain mean
    s = boot_stat(np.arange(15.0), "iqm", draws=300)
    assert s["lo"] < s["value"] == 7.0 < s["hi"]
    assert np.allclose(adjust([0.01, 0.04, 0.03], "holm"), [0.03, 0.06, 0.06])
    assert np.allclose(adjust([0.01, 0.04, 0.03], "bonferroni"), [0.03, 0.12, 0.09])
    x = np.array([3.50, 3.52, 3.49, 3.51, 3.50])
    assert welch(x, x + 1.0) < 0.001 < 0.5 < welch(x, x[::-1] + 0.001)
    assert welch(x, [2.0]) < 0.001                            # one comparator score: one-sample test


class FakeScores:
    """Made-up scores in place of the files score_final.py writes."""
    def __init__(self, shift):
        self.m, self.r = made_up(shift)

    def agents(self, group, setting):
        return {"index": self.m}

    def rule(self, setting, label):
        return self.r[0]


@pytest.mark.parametrize("shift, by_rule, hypothesis", [
    (np.linspace(-0.01, 0.01, SEEDS), "equivalent", "holds"),
    (np.linspace(0.20, 0.30, SEEDS), "better", "holds"),
    (np.linspace(-0.30, -0.20, SEEDS), "worse", "does not hold"),
    (np.tile([-0.4, 0.4, 0.0], 5), "inconclusive", "inconclusive"),
])
def test_h2b_verdict_from_made_up_scores(shift, by_rule, hypothesis):
    c, _ = adr.choices(stand_in=True)
    h = adr.h2b(FakeScores(shift), dict(c, ppo_actor="index"), 0.95)
    assert h["by_actor"] == {"index": by_rule} and h["verdict"] == hypothesis


def test_undecided_wording_blocks_the_final_results_only():
    c, undecided = adr.choices(stand_in=True)
    if undecided:                                             # until the owner has decided every point
        with pytest.raises(SystemExit):
            adr.choices(stand_in=False)
    assert set(c) >= {"resample", "statistic", "ppo_actor", "multiplicity_form"}


def test_scoring_refuses_the_evaluation_pool_without_the_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["score_final.py"])
    with pytest.raises(SystemExit, match="Refused"):
        score_final.main()
    monkeypatch.setattr(sys, "argv", ["score_final.py", "--stand-in", "--allow-evaluation-pool"])
    with pytest.raises(AssertionError):
        score_final.main()


def test_an_interrupted_scoring_run_continues(monkeypatch, tmp_path):
    todo = [("rule", tmp_path / f"{name}.npz", dict(setting="small", family="random", belief=None, params=None,
                                                    seeds=list(range(1_000_000, 1_000_005)), pool="dev (stand-in)"))
            for name in ("a", "b")]
    monkeypatch.setattr(score_final, "STANDIN", tmp_path)
    monkeypatch.setattr(score_final, "make_stand_in", lambda: None)
    monkeypatch.setattr(score_final, "jobs", lambda *a, **k: todo)
    monkeypatch.setattr(sys, "argv", ["score_final.py", "--stand-in"])
    score_final.main()
    (tmp_path / "b.npz").unlink()                              # "interrupted": one item is missing
    (tmp_path / "a.npz").write_bytes(b"kept")                  # a finished item must not be redone
    score_final.main()
    assert (tmp_path / "a.npz").read_bytes() == b"kept"
    assert len(np.load(tmp_path / "b.npz")["ret"]) == 5 and not list(tmp_path.glob("*.tmp"))


class FirstValid:
    """Stands in for a PPO model: always the first valid action."""
    def predict(self, obs, action_masks, deterministic=True):
        return np.argmax(np.atleast_2d(action_masks), axis=1), None

    def reset(self, env):
        pass

    def act(self, env):
        return int(np.argmax(env.action_masks()))


def test_batched_play_equals_one_episode_at_a_time():
    sc, seeds = load_scenario("stress"), list(range(1_000_000, 1_000_012))
    one = evaluate(sc, FirstValid(), seeds=seeds)
    many = score_final.play(sc, FirstValid(), seeds, batch=5)
    for key in one:
        assert np.array_equal(one[key], many[key]), key


def test_decision_share_is_zero_for_identical_rules_and_positive_for_different_ones():
    sc, seeds = load_scenario("stress"), list(range(1_000_000, 1_000_010))
    rule = lambda bar, belief: RULE_FAMILIES["static_threshold"]([bar], belief=belief)
    same = score_final.decision_share(sc, rule(1.2, "survival"), rule(1.2, "survival"), seeds)
    other = score_final.decision_share(sc, rule(1.2, "survival"), rule(1.2, "naive"), seeds)
    assert same["changed"].sum() == 0 and same["candidate_days"].sum() > 0
    assert 0 < other["changed"].sum() < other["candidate_days"].sum()
    assert np.array_equal(same["candidate_days"], other["candidate_days"])       # the played rule is the same
