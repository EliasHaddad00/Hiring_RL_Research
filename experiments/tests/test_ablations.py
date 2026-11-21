"""The two ablations that need their own code: reward shaping and the end-day variant
(PREREGISTRATION.md, "Final run list"). No ablation run starts before these pass."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch as th
import yaml
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from helpers import random_valid_action
from hiring.config import episode_seeds, load_scenario
from hiring.env import HiringEnv
from hiring.policies import IndexEndPooledPolicy, IndexPolicy, index_policy_kwargs
from hiring.shaping import ShapedReward, potential
from hiring.training import REQUIRED_COLUMNS, load_run_config, make_training_env, override, run_config, train
from test_policies import collect_states, logits

ROOT = Path(__file__).resolve().parents[2]
STRESS = load_scenario("stress")
N, F = STRESS.n_slots, HiringEnv(STRESS).n_slot_features


# ---- reward shaping ------------------------------------------------------------------------------
def by_hand(env):
    """The pre-registered formula, written out plainly."""
    shown = env.mu_surv if env.sc.obs_belief == "survival" else env.mu_naive
    best = sorted((float(shown[c]) for c in env.order), reverse=True)[:env.roles_left]
    return sum(max(m, 0.0) for m in best)


@pytest.mark.parametrize("belief", ["survival", "naive"])
def test_potential_is_the_registered_formula_and_zero_at_the_end(belief):
    env, rng = HiringEnv(load_scenario("stress", obs_belief=belief)), np.random.default_rng(0)
    seen_big_pool = seen_fewer_than_roles = False
    for s in episode_seeds("dev", 5):
        env.reset(seed=s)
        done = False
        while not done:
            assert potential(env) == pytest.approx(by_hand(env)) and potential(env) >= 0.0
            seen_big_pool |= len(env.order) > env.N
            seen_fewer_than_roles |= len(env.order) < env.roles_left
            _, _, done, _, _ = env.step(random_valid_action(env, rng))
        assert potential(env) == 0.0
    assert seen_big_pool and seen_fewer_than_roles      # candidates outside the slots count; short pools work


def test_potential_uses_the_estimate_the_agent_is_shown():
    a, b = HiringEnv(load_scenario("stress", obs_belief="survival")), HiringEnv(load_scenario("stress", obs_belief="naive"))
    for env in (a, b):
        env.reset(seed=episode_seeds("dev", 1)[0])
        for _ in range(8):                               # eight days in: the two estimates differ
            env.step(2 * env.N)
    assert potential(a) != pytest.approx(potential(b))


@pytest.mark.parametrize("gamma", [1.0, 0.99])
def test_shaped_rewards_telescope_and_the_logged_return_is_unshaped(gamma):
    cfg = override(load_run_config("ppo_final_index"), "reward_shaping", True)
    cfg = override(cfg, "gamma", gamma)
    venv = make_training_env(cfg, STRESS, n_envs=1, seed=0)
    assert isinstance(venv.envs[0], ShapedReward)
    rng, largest = np.random.default_rng(1), 0.0
    for _ in range(5):
        venv.reset()
        env = venv.envs[0].unwrapped
        phi0, shaped, raw, discount, done = potential(env), 0.0, 0.0, 1.0, False
        while not done:
            before = env.stats["ret"]
            _, r, dones, infos = venv.step([random_valid_action(env, rng)])
            done = bool(dones[0])
            step_raw = (infos[0]["episode_stats"]["ret"] if done else env.stats["ret"]) - before
            largest = max(largest, abs(float(r[0]) - step_raw))
            shaped += discount * float(r[0])
            raw += discount * step_raw
            discount *= gamma
        # discounted shaped return = discounted unshaped return - Phi(first state): the policy-invariance identity
        assert shaped == pytest.approx(raw - phi0, abs=1e-5)
        if gamma == 1.0:
            assert infos[0]["episode"]["r"] == pytest.approx(infos[0]["episode_stats"]["ret"], abs=1e-5)   # Monitor: unshaped
    assert largest > 1e-3                                # shaping is on: some step's reward was changed
    venv.close()


def test_without_the_flag_the_training_environment_is_unshaped():
    venv = make_training_env(load_run_config("ppo_final_index"), STRESS, n_envs=1, seed=0)
    assert not isinstance(venv.envs[0], ShapedReward)
    venv.close()


# ---- end-day variant -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def variant():
    th.manual_seed(0)
    venv = DummyVecEnv([lambda: HiringEnv(STRESS)])
    return MaskablePPO(IndexEndPooledPolicy, venv, policy_kwargs=index_policy_kwargs(HiringEnv(STRESS)),
                       n_steps=64, batch_size=32, seed=0, device="cpu")


def change_one_candidate(obs, j=1):
    changed = obs.copy()
    changed[:, j * F + 1] += 0.5                         # candidate j's estimate; it stays present
    return changed


def test_variant_scores_each_candidate_from_its_own_features_only(variant):
    assert isinstance(variant.policy.action_net, nn.Identity)
    obs, _ = collect_states(32)
    a, b = logits(variant, obs), logits(variant, change_one_candidate(obs))
    others = [i for i in range(N) if i != 1]
    assert np.allclose(a[:, others], b[:, others], atol=1e-6)                           # interview logits
    assert np.allclose(a[:, [N + i for i in others]], b[:, [N + i for i in others]], atol=1e-6)   # offer logits
    assert not np.allclose(a[:, 1], b[:, 1], atol=1e-6)


def test_variant_end_day_logit_reads_the_pool_and_the_index_actor_does_not(variant):
    obs, _ = collect_states(32)
    changed = change_one_candidate(obs)
    assert np.abs(logits(variant, obs)[:, 2 * N] - logits(variant, changed)[:, 2 * N]).max() > 1e-6
    index = MaskablePPO(IndexPolicy, DummyVecEnv([lambda: HiringEnv(STRESS)]),
                        policy_kwargs=index_policy_kwargs(HiringEnv(STRESS)), n_steps=64, batch_size=32, seed=0, device="cpu")
    assert np.allclose(logits(index, obs)[:, 2 * N], logits(index, changed)[:, 2 * N], atol=1e-6)


def test_variant_end_day_logit_ignores_candidate_order(variant):
    obs, _ = collect_states(32)
    perm = np.random.default_rng(1).permutation(N)
    slots = obs[:, :N * F].reshape(-1, N, F)
    permuted = np.concatenate([slots[:, perm].reshape(len(obs), -1), obs[:, N * F:]], axis=1)
    a, b = logits(variant, obs), logits(variant, permuted)
    assert np.allclose(b[:, :N], a[:, :N][:, perm], atol=1e-5)
    assert np.allclose(b[:, 2 * N], a[:, 2 * N], atol=1e-5)


def test_variant_differs_from_the_index_actor_only_in_the_end_day_input(variant):
    index = MaskablePPO(IndexPolicy, DummyVecEnv([lambda: HiringEnv(STRESS)]),
                        policy_kwargs=index_policy_kwargs(HiringEnv(STRESS)), seed=0, device="cpu")
    shapes = lambda m, part: [tuple(p.shape) for p in getattr(m.policy.mlp_extractor, part).parameters()]
    for part in ("candidate_body", "candidate_head", "critic", "end_head"):
        assert shapes(variant, part) == shapes(index, part)
    assert shapes(variant, "end_body") != shapes(index, "end_body")


# ---- both, through the training script's code path -------------------------------------------------
def test_short_training_run_of_each_ablation_logs_every_column_and_reloads(tmp_path):
    for actor, shaping in (("index", True), ("index_end_pooled", False)):
        cfg = run_config("ppo_final_index", "h0_0.05", "survival")
        cfg["ppo"] = dict(cfg["ppo"], n_steps=64, batch_size=64, n_epochs=2)
        cfg["eval"] = dict(episodes=10, freq_steps=1024)
        if shaping:
            override(cfg, "reward_shaping", True)
        out = tmp_path / actor
        meta = train(cfg, actor, seed=101, out_dir=out, total_timesteps=2048)
        rows = list(csv.DictReader(open(out / "progress.csv", newline="")))
        for col in REQUIRED_COLUMNS:
            assert any(r.get(col) not in (None, "") for r in rows), col
        assert bool(meta["config"].get("reward_shaping")) is shaping
        assert (out / "model_80.zip").exists()
        MaskablePPO.load(out / "model.zip", device="cpu")


def test_ablation_list_is_the_registered_one():
    sys.path.insert(0, str(ROOT / "experiments" / "scripts"))
    from final_runs import ablation_commands
    spec = yaml.safe_load((ROOT / "experiments" / "configs" / "final_runs.yaml").read_text())
    ab, frozen = spec["ablations"], load_run_config("ppo_final_index")["ppo"]
    arms = {a["name"]: a for a in ab["arms"]}
    assert len(arms) == 10 and ab["seeds"] == [1001, 1002, 1003] and ab["setting"] == "h0_0.05"
    assert frozen["gamma"] == 1.0 and frozen["n_steps"] == 512
    assert [arms[n]["override"] for n in ("gamma_0.99", "gamma_0.999", "n_steps_256", "n_steps_1024")] == \
        [{"gamma": 0.99}, {"gamma": 0.999}, {"n_steps": 256}, {"n_steps": 1024}]
    for key in ("ent_coef", "learning_rate"):
        assert arms[f"{key}_third"]["override"][key] == pytest.approx(frozen[key] / 3, rel=1e-12)
        assert arms[f"{key}_triple"]["override"][key] == pytest.approx(frozen[key] * 3, rel=1e-12)
    assert arms["reward_shaping"]["override"] == {"reward_shaping": True}
    assert arms["end_day_pooled"]["actor"] == "index_end_pooled"
    assert all(a["actor"] == "index" for n, a in arms.items() if n != "end_day_pooled")
    cmds = ablation_commands(spec)
    assert len(cmds) == 10 and all("--allow-final-seeds" in c for c in cmds)
    assert "reward_shaping=true" in cmds[8]
