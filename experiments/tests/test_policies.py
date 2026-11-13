"""Index-structured actor: unit tests on the real observation layout. No training on the hiring task."""
import numpy as np
import pytest
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from helpers import random_valid_action
from hiring.config import episode_seeds, load_scenario
from hiring.env import N_GLOBAL, HiringEnv
from hiring.policies import IndexPolicy, index_policy_kwargs

STRESS = load_scenario("stress")


def collect_states(n=64, seed=0):
    env, rng = HiringEnv(STRESS), np.random.default_rng(seed)
    obs_list, masks = [], []
    for s in episode_seeds("dev", 8):
        obs, _ = env.reset(seed=s)
        done = False
        while not done and len(obs_list) < n:
            if len(env.order) >= 3:
                obs_list.append(obs); masks.append(env.action_masks())
            obs, _, done, _, _ = env.step(random_valid_action(env, rng))
    return np.array(obs_list), np.array(masks)


@pytest.fixture(scope="module")
def model():
    th.manual_seed(0)
    venv = DummyVecEnv([lambda: HiringEnv(STRESS)])
    return MaskablePPO(IndexPolicy, venv, policy_kwargs=index_policy_kwargs(HiringEnv(STRESS)),
                       n_steps=64, batch_size=32, seed=0, device="cpu")


def logits(model, obs):
    with th.no_grad():
        return model.policy.mlp_extractor.forward_actor(th.as_tensor(obs, dtype=th.float32)).numpy()


def test_final_layer_is_identity_and_logits_have_action_layout(model):
    assert isinstance(model.policy.action_net, nn.Identity)
    obs, _ = collect_states(8)
    out = logits(model, obs)
    assert out.shape == (8, 2 * STRESS.n_slots + 1) and np.isfinite(out).all()


def test_reordering_candidates_reorders_logits(model):
    obs, _ = collect_states(32)
    N, F = STRESS.n_slots, HiringEnv(STRESS).n_slot_features
    perm = np.random.default_rng(1).permutation(N)
    slots = obs[:, :N * F].reshape(-1, N, F)
    permuted = np.concatenate([slots[:, perm].reshape(len(obs), -1), obs[:, N * F:]], axis=1)
    a, b = logits(model, obs), logits(model, permuted)
    assert np.allclose(b[:, :N], a[:, :N][:, perm], atol=1e-6)            # interview logits follow the candidates
    assert np.allclose(b[:, N:2 * N], a[:, N:2 * N][:, perm], atol=1e-6)  # offer logits too
    assert np.allclose(b[:, 2 * N], a[:, 2 * N], atol=1e-6)               # "end day" is unchanged


def test_a_candidate_score_ignores_other_candidates(model):
    obs, _ = collect_states(16)
    N, F = STRESS.n_slots, HiringEnv(STRESS).n_slot_features
    changed = obs.copy()
    changed[:, F:2 * F] += 0.37                                            # alter slot 1 only
    a, b = logits(model, obs), logits(model, changed)
    others = [i for i in range(N) if i != 1]
    assert np.allclose(a[:, others], b[:, others], atol=1e-6)
    assert np.allclose(a[:, [N + i for i in others]], b[:, [N + i for i in others]], atol=1e-6)
    assert not np.allclose(a[:, 1], b[:, 1])
    assert np.allclose(a[:, 2 * N], b[:, 2 * N], atol=1e-6)               # end day sees only globals


def test_masked_actions_get_zero_probability(model):
    obs, masks = collect_states(48)
    assert (~masks).any()
    dist = model.policy.get_distribution(th.as_tensor(obs), action_masks=masks)
    probs = dist.distribution.probs.detach().numpy()
    assert np.all(probs[~masks] == 0.0)
    assert np.allclose(probs.sum(1), 1.0, atol=1e-6)
    actions, _ = model.predict(obs, action_masks=masks, deterministic=False)
    assert masks[np.arange(len(obs)), actions].all()


def test_critic_is_unrestricted_and_parameter_count(model):
    ext = model.policy.mlp_extractor
    assert ext.critic[0].in_features == STRESS.n_slots * ext.F + N_GLOBAL
    n_params = sum(p.numel() for p in model.policy.parameters())
    assert 5_000 < n_params < 20_000


def test_short_learn_runs_and_masks_are_found(model):
    """A few hundred steps only: checks the training code path, not learning."""
    model.learn(total_timesteps=128)
    assert get_action_masks(model.get_env()).shape == (1, 2 * STRESS.n_slots + 1)


def test_save_and_load_round_trip(model, tmp_path):
    obs, masks = collect_states(16)
    before, _ = model.predict(obs, action_masks=masks, deterministic=True)
    path = tmp_path / "index_policy"
    model.save(path)
    loaded = MaskablePPO.load(path, device="cpu")
    assert isinstance(loaded.policy, IndexPolicy) and isinstance(loaded.policy.action_net, nn.Identity)
    after, _ = loaded.predict(obs, action_masks=masks, deterministic=True)
    assert np.array_equal(before, after)
    assert np.allclose(logits(model, obs), logits(loaded, obs), atol=1e-6)
