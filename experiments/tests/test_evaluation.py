"""Evaluation harness: paired episodes, masks at evaluation, pool locks, the verdict rule."""
import numpy as np
import pytest
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from hiring.config import LockedPoolError, check_training_seed, episode_seeds, load_scenario, load_seeds
from hiring.env import HiringEnv
from hiring.evaluation import PPOPolicy, evaluate, paired_compare
from hiring.policies import IndexPolicy, index_policy_kwargs
from hiring.rules import RandomPolicy, fixed_threshold

SMALL = load_scenario("small_exact")
STRESS = load_scenario("stress")
SEEDS = list(episode_seeds("dev", 60))


@pytest.fixture(scope="module", params=["unrestricted", "index"])
def model(request):
    """Untrained agents: enough to test the evaluation code path."""
    th.manual_seed(0)
    venv = DummyVecEnv([lambda: HiringEnv(STRESS)])
    if request.param == "index":
        return MaskablePPO(IndexPolicy, venv, policy_kwargs=index_policy_kwargs(HiringEnv(STRESS)),
                           n_steps=64, batch_size=32, seed=0, device="cpu")
    return MaskablePPO("MlpPolicy", venv, n_steps=64, batch_size=32, seed=0, device="cpu")


# ---- pools ---------------------------------------------------------------------------------------

def test_evaluation_pool_is_locked_without_the_flag():
    with pytest.raises(LockedPoolError):
        episode_seeds("evaluation", 10)
    assert list(episode_seeds("evaluation", 2, allow_evaluation=True)) == [3_000_000, 3_000_001]


def test_final_training_seeds_are_locked_without_the_flag():
    final = load_seeds()["training_seeds"]["final"]
    with pytest.raises(LockedPoolError):
        check_training_seed(final[0])
    check_training_seed(final[0], allow_final=True)
    for s in load_seeds()["training_seeds"]["tuning"]:
        check_training_seed(s)


def test_offset_slices_stay_inside_the_pool_and_do_not_overlap():
    a, b = episode_seeds("dev", 100), episode_seeds("dev", 100, offset=100)
    assert a[-1] + 1 == b[0]
    with pytest.raises(AssertionError):
        episode_seeds("dev", 10, offset=99_995)
    pools = load_seeds()["episode_seeds"]
    assert pools["dev"]["start"] + pools["dev"]["count"] <= pools["tuning"]["start"]


# ---- paired episodes -------------------------------------------------------------------------------

def test_two_policies_on_the_same_seeds_face_the_same_episodes(monkeypatch):
    """The world of an episode (arrivals, qualities, noise, luck) does not depend on the policy."""
    seen = {}
    original = HiringEnv._load

    def spy(self, draws):
        seen.setdefault(self._tag, []).append((draws.arrival_day.copy(), draws.quality.copy(), draws.u_accept.copy()))
        return original(self, draws)

    monkeypatch.setattr(HiringEnv, "_load", spy)
    for tag, policy in [("a", RandomPolicy()), ("b", fixed_threshold(0.9))]:
        monkeypatch.setattr(HiringEnv, "_tag", tag, raising=False)
        evaluate(STRESS, policy, seeds=SEEDS[:20])
    assert len(seen["a"]) == len(seen["b"]) == 20
    for x, y in zip(seen["a"], seen["b"]):
        assert all(np.array_equal(u, v) for u, v in zip(x, y))


def test_evaluation_is_repeatable():
    a = evaluate(STRESS, fixed_threshold(0.9), seeds=SEEDS)["ret"]
    b = evaluate(STRESS, fixed_threshold(0.9), seeds=SEEDS)["ret"]
    assert np.array_equal(a, b)


# ---- PPO adapter -----------------------------------------------------------------------------------

def test_adapter_never_takes_an_invalid_action(model):
    """The strict environment raises on any invalid action, so finishing is the pass condition."""
    for deterministic in (True, False):
        res = evaluate(STRESS, PPOPolicy(model, deterministic=deterministic), seeds=SEEDS)
        assert len(res["ret"]) == len(SEEDS) and np.isfinite(res["ret"]).all()


def test_adapter_passes_the_mask(model, monkeypatch):
    calls = []
    original = model.predict

    def spy(obs, *args, **kwargs):
        calls.append(kwargs.get("action_masks"))
        return original(obs, *args, **kwargs)

    monkeypatch.setattr(model, "predict", spy)
    evaluate(STRESS, PPOPolicy(model), seeds=SEEDS[:3])
    assert calls and all(m is not None and m.dtype == bool and m[-1] for m in calls)


def test_adapter_sees_real_observations(model):
    """evaluate() must not hand a PPO agent the all-zero placeholder used for rules."""
    env = HiringEnv(STRESS)
    env.reset(seed=SEEDS[0])
    assert PPOPolicy.needs_obs and env._obs().any()


def test_adapter_matches_sb3_maskable_evaluate_policy(model):
    ours = evaluate(STRESS, PPOPolicy(model, deterministic=True), seeds=SEEDS)["ret"]
    venv = DummyVecEnv([lambda: Monitor(HiringEnv(STRESS, episode_seeds=SEEDS))])
    theirs, _ = evaluate_policy(model, venv, n_eval_episodes=len(SEEDS), deterministic=True,
                                return_episode_rewards=True, use_masking=True)
    assert np.allclose(ours, theirs, atol=1e-5)


def test_sampled_evaluation_is_repeatable(model):
    a = evaluate(STRESS, PPOPolicy(model, deterministic=False), seeds=SEEDS[:20])["ret"]
    b = evaluate(STRESS, PPOPolicy(model, deterministic=False), seeds=SEEDS[:20])["ret"]
    assert np.array_equal(a, b)


# ---- verdict rule ----------------------------------------------------------------------------------

def _scores(diff_mean, diff_sd, n=400, base=3.5, seed=0):
    rng = np.random.default_rng(seed)
    b = base + rng.standard_normal(n)
    d = rng.standard_normal(n)
    d = (d - d.mean()) / d.std(ddof=1) * diff_sd + diff_mean
    return b + d, b


@pytest.mark.parametrize("diff_mean, diff_sd, verdict", [
    (0.00, 0.10, "equivalent"),      # tight interval around zero, inside the 0.07 margin
    (0.03, 0.10, "equivalent"),      # above zero but inside the margin: equivalence takes precedence
    (0.20, 0.10, "better"),
    (-0.20, 0.10, "worse"),
    (0.00, 2.00, "inconclusive"),    # interval wider than the margin and straddling zero
    (0.07, 0.30, "better"),          # interval (0.04, 0.10): above zero, crosses the margin edge
    (-0.07, 0.30, "worse"),
])
def test_verdicts(diff_mean, diff_sd, verdict):
    a, b = _scores(diff_mean, diff_sd)
    r = paired_compare(a, b, margin=0.02)
    assert r["margin"] == pytest.approx(0.02 * abs(b.mean()))
    assert r["lo"] < r["diff"] < r["hi"] and r["diff"] == pytest.approx(diff_mean)
    assert r["verdict"] == verdict


def test_interval_matches_the_t_formula():
    a, b = _scores(0.05, 0.4, n=25)
    r = paired_compare(a, b)
    from scipy import stats
    d = a - b
    half = stats.t.ppf(0.975, 24) * d.std(ddof=1) / 5
    assert r["lo"] == pytest.approx(d.mean() - half) and r["hi"] == pytest.approx(d.mean() + half)


def test_unpaired_inputs_are_rejected():
    with pytest.raises(AssertionError):
        paired_compare(np.zeros(5), np.zeros(6))
