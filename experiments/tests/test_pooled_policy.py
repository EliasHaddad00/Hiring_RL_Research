"""Pooled actor: weight sharing kept, pool awareness added. Unit tests on the real observation layout."""
import numpy as np
import pytest
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from hiring.config import load_scenario
from hiring.env import N_GLOBAL, HiringEnv
from hiring.policies import IndexPolicy, PooledExtractor, PooledIndexPolicy, index_policy_kwargs
from test_policies import collect_states

STRESS = load_scenario("stress")
N, F = STRESS.n_slots, HiringEnv(STRESS).n_slot_features


@pytest.fixture(scope="module")
def model():
    th.manual_seed(0)
    venv = DummyVecEnv([lambda: HiringEnv(STRESS)])
    return MaskablePPO(PooledIndexPolicy, venv, policy_kwargs=index_policy_kwargs(HiringEnv(STRESS)),
                       n_steps=64, batch_size=32, seed=0, device="cpu")


def logits(model, obs):
    with th.no_grad():
        return model.policy.mlp_extractor.forward_actor(th.as_tensor(obs, dtype=th.float32)).numpy()


def test_structure(model):
    assert isinstance(model.policy.mlp_extractor, PooledExtractor) and isinstance(model.policy.action_net, nn.Identity)
    ext = model.policy.mlp_extractor
    assert ext.critic[0].in_features == N * F + N_GLOBAL              # same unrestricted critic as the index agent
    obs, _ = collect_states(8)
    out = logits(model, obs)
    assert out.shape == (8, 2 * N + 1) and np.isfinite(out).all()


def test_reordering_candidates_reorders_logits(model):
    obs, _ = collect_states(32)
    perm = np.random.default_rng(1).permutation(N)
    slots = obs[:, :N * F].reshape(-1, N, F)
    permuted = np.concatenate([slots[:, perm].reshape(len(obs), -1), obs[:, N * F:]], axis=1)
    a, b = logits(model, obs), logits(model, permuted)
    assert np.allclose(b[:, :N], a[:, :N][:, perm], atol=1e-5)
    assert np.allclose(b[:, N:2 * N], a[:, N:2 * N][:, perm], atol=1e-5)
    assert np.allclose(b[:, 2 * N], a[:, 2 * N], atol=1e-5)           # the summary does not depend on order


def test_a_candidate_score_now_depends_on_the_pool(model):
    """The one difference from the index actor: changing candidate 1 moves the others and "end day"."""
    obs, _ = collect_states(16)
    changed = obs.copy()
    changed[:, F + 1] += 0.37                                          # estimate of the candidate in slot 1
    a, b = logits(model, obs), logits(model, changed)
    assert not np.allclose(a[:, 0], b[:, 0]) and not np.allclose(a[:, 2 * N], b[:, 2 * N])


def test_absent_slots_do_not_enter_the_summary(model):
    obs = np.zeros((1, N * F + N_GLOBAL), np.float32)
    obs[0, N * F:] = [0.5, 1.0, 1.0, 0.1, 1.0]
    for slot in range(3):                                              # three present candidates
        obs[0, slot * F:(slot + 1) * F] = [1.0, 0.2 * slot, 0.0, 0.1, 0.0]
    junk = obs.copy()
    junk[0, 5 * F + 1:6 * F] = 3.0                                     # features of an absent slot (present = 0)
    a, b = logits(model, obs), logits(model, junk)
    keep = [0, 1, 2, N, N + 1, N + 2, 2 * N]
    assert np.allclose(a[0, keep], b[0, keep], atol=1e-6)


def test_empty_pool_gives_finite_logits_and_count_is_seen(model):
    empty = np.zeros((1, N * F + N_GLOBAL), np.float32)
    empty[0, N * F:] = [0.5, 1.0, 1.0, 0.0, 1.0]
    assert np.isfinite(logits(model, empty)).all()
    # two identical candidates against four: the mean embedding is the same, only the count differs
    two, four = empty.copy(), empty.copy()
    for slot in range(2):
        two[0, slot * F:(slot + 1) * F] = [1.0, 0.3, 0.0, 0.1, 0.0]
    for slot in range(4):
        four[0, slot * F:(slot + 1) * F] = [1.0, 0.3, 0.0, 0.1, 0.0]
    assert not np.allclose(logits(model, two)[0, 2 * N], logits(model, four)[0, 2 * N])


def test_masked_actions_get_zero_probability_and_training_runs(model):
    obs, masks = collect_states(48)
    dist = model.policy.get_distribution(th.as_tensor(obs), action_masks=masks)
    probs = dist.distribution.probs.detach().numpy()
    assert np.all(probs[~masks] == 0.0) and np.allclose(probs.sum(1), 1.0, atol=1e-6)
    model.learn(total_timesteps=128)


def test_save_and_load_round_trip(model, tmp_path):
    obs, masks = collect_states(16)
    before, _ = model.predict(obs, action_masks=masks, deterministic=True)
    model.save(tmp_path / "pooled")
    loaded = MaskablePPO.load(tmp_path / "pooled", device="cpu")
    assert isinstance(loaded.policy, PooledIndexPolicy) and isinstance(loaded.policy.mlp_extractor, PooledExtractor)
    after, _ = loaded.predict(obs, action_masks=masks, deterministic=True)
    assert np.array_equal(before, after) and np.allclose(logits(model, obs), logits(loaded, obs), atol=1e-6)


def test_index_policy_is_unchanged():
    venv = DummyVecEnv([lambda: HiringEnv(STRESS)])
    m = MaskablePPO(IndexPolicy, venv, policy_kwargs=index_policy_kwargs(HiringEnv(STRESS)), n_steps=64,
                    batch_size=32, seed=0, device="cpu")
    assert type(m.policy.mlp_extractor).__name__ == "IndexExtractor"
