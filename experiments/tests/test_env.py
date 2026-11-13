"""Environment mechanics and conventions."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from helpers import end_day, hand_draws, random_valid_action, slot_features
from hiring.config import episode_seeds, load_scenario
from hiring.env import HiringEnv, N_GLOBAL
from hiring.pool import overflow_drop_index

STRESS = load_scenario("stress")
SMALL = load_scenario("small_exact")
SEEDS = list(episode_seeds("dev", 300))


@pytest.mark.parametrize("sc", [STRESS, SMALL], ids=["stress", "small"])
def test_check_env_on_tolerant_variant(sc):
    """The checkers sample actions without the mask, so they run on the variant that treats an
    invalid action as a no-op."""
    gym_check_env(HiringEnv(sc, strict=False), skip_render_check=True)
    sb3_check_env(HiringEnv(sc, strict=False))


def test_strict_env_rejects_invalid_action():
    env = HiringEnv(STRESS)
    env.reset(options={"draws": hand_draws(STRESS, [5], [0.0])})
    assert not env.action_masks()[0]            # empty pool: nobody to interview
    with pytest.raises(AssertionError):
        env.step(0)


@pytest.mark.parametrize("sc", [STRESS, SMALL, STRESS.with_(h0=0.0, accept_decay=0.0)],
                         ids=["stress", "small", "no_pressure"])
def test_random_masked_rollouts(sc):
    env = HiringEnv(sc)                         # strict: any invalid action raises
    rng = np.random.default_rng(0)
    for seed in SEEDS[:150]:
        obs, _ = env.reset(seed=seed)
        total, steps, done = 0.0, 0, False
        while not done:
            mask = env.action_masks()
            assert mask.shape == (2 * env.N + 1,) and mask[2 * env.N] and mask.any()
            assert env.observation_space.contains(obs)
            obs2, r, done, truncated, info = env.step(random_valid_action(env, rng))
            assert obs2 is not obs and not truncated
            obs, total, steps = obs2, total + r, steps + 1
            assert steps < 5000
        st = info["episode_stats"]
        assert env.t == sc.horizon or env.roles_left == 0
        assert st["ret"] == pytest.approx(total)
        assert 0 <= st["hires"] <= sc.roles
        if sc.pool_cap is not None:
            assert st["days_pool_over_slots"] == 0


def _trajectory(sc, seed, policy_seed):
    env, rng = HiringEnv(sc), np.random.default_rng(policy_seed)
    obs, _ = env.reset(seed=seed)
    out, done = [obs], False
    while not done:
        a = random_valid_action(env, rng)
        obs, r, done, _, _ = env.step(a)
        out += [np.float32(a), np.float32(r), obs]
    return out


@pytest.mark.parametrize("sc", [STRESS, SMALL], ids=["stress", "small"])
def test_same_seed_same_trajectory(sc):
    a, b = _trajectory(sc, SEEDS[0], 1), _trajectory(sc, SEEDS[0], 1)
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
    c = _trajectory(sc, SEEDS[1], 1)
    assert len(a) != len(c) or not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_same_seed_same_world_for_different_policies():
    e1, e2 = HiringEnv(STRESS), HiringEnv(STRESS)
    e1.reset(seed=SEEDS[3]); e2.reset(seed=SEEDS[3])
    rng = np.random.default_rng(5)
    for _ in range(40):
        if not e1.done:
            e1.step(random_valid_action(e1, rng))
        if not e2.done:
            end_day(e2)
    assert np.array_equal(e1.q, e2.q) and np.array_equal(e1.arr, e2.arr)
    assert np.array_equal(e1.draws.u_poach, e2.draws.u_poach)


def test_reset_without_seed_uses_seed_list_in_order():
    env = HiringEnv(STRESS, episode_seeds=SEEDS[:3])
    got = [env.reset()[1]["episode_seed"] for _ in range(4)]
    assert got == [SEEDS[0], SEEDS[1], SEEDS[2], SEEDS[0]]


# ---- conventions ------------------------------------------------------------------------------------

def two_candidates(sc=STRESS, **kw):
    """Two candidates arrive on day 0: A (q = 2) is slot 0, B (q = 1) is slot 1."""
    env = HiringEnv(sc)
    obs, _ = env.reset(options={"draws": hand_draws(sc, [0, 0], [2.0, 1.0], **kw)})
    return env, obs


def test_offers_masked_after_first_interview_of_the_day():
    env, _ = two_candidates()
    N = env.N
    assert env.action_masks()[N] and env.action_masks()[N + 1]
    env.step(0)                                              # interview A
    mask = env.action_masks()
    assert not mask[N:2 * N].any()                           # no offers at all, not even to B
    assert mask[1] and not mask[0]                           # B can still be interviewed; A cannot again today
    end_day(env)
    assert env.action_masks()[N] and env.action_masks()[N + 1]   # offers reopen next morning


def test_interview_result_not_visible_until_next_day():
    env, obs0 = two_candidates(interview_z=1.5)              # the interview will read q + 1.5
    before = slot_features(env, obs0, 0).copy()
    obs1, r, *_ = env.step(0)
    assert r == pytest.approx(-STRESS.interview_cost)
    after = slot_features(env, obs1, 0)
    assert np.array_equal(after[:4], before[:4])             # present, estimate, interviews seen, days waited
    assert before[4] == 0.0 and after[4] == 1.0              # only the interviewed-today flag changes
    assert env.mu_naive[0] == pytest.approx(before[1] * 3.0, rel=1e-6)
    obs2, *_ = end_day(env)
    nxt = slot_features(env, obs2, 0)
    expected = env.belief.naive_mean(2.0, 3.5, 1)            # resume 2.0, interview 3.5
    assert env.mu_naive[0] == pytest.approx(expected)
    assert nxt[2] == pytest.approx(1 / STRESS.max_interviews) and nxt[4] == 0.0
    assert nxt[1] != pytest.approx(before[1])


def test_pending_result_lost_if_poached_overnight():
    u = np.ones((2, STRESS.horizon)); u[0, 0] = 0.0          # A is poached on night 0
    env, _ = two_candidates(u_poach=u)
    env.step(0)
    end_day(env)
    assert not env.alive[0] and env.n_obs[0] == 0 and env.order == [1]


def test_interviewed_today_can_be_offered_tomorrow_at_the_earliest():
    env, _ = two_candidates()
    env.step(0)
    assert not env.action_masks()[env.N]
    end_day(env)
    _, r, *_ = env.step(env.N)                               # offer A the next day
    assert r == pytest.approx(2.0)


def test_interview_limits():
    sc = STRESS.with_(interviews_per_day=2)
    env = HiringEnv(sc)
    env.reset(options={"draws": hand_draws(sc, [0, 0, 0], [2.0, 1.0, 0.0])})
    env.step(0); env.step(1)
    assert not env.action_masks()[:env.N].any()              # daily limit of 2 reached
    for _ in range(2):
        end_day(env); env.step(0)
    end_day(env)
    assert env.n_done[0] == 3 and not env.action_masks()[0]  # candidate limit of 3 reached
    assert env.action_masks()[1]


def test_interviews_masked_on_last_day_and_deadline_terminates():
    env, _ = two_candidates()
    for _ in range(STRESS.horizon - 1):
        assert env.action_masks()[0]
        _, _, terminated, truncated, _ = end_day(env)
        assert not terminated and not truncated
    assert env.t == STRESS.horizon - 1
    mask = env.action_masks()
    assert not mask[:env.N].any() and mask[env.N] and mask[2 * env.N]
    obs, _, terminated, truncated, info = end_day(env)
    assert terminated and not truncated                      # deadline is terminated, never truncated
    assert obs[env.N * env.n_slot_features] == 0.0           # days left
    assert "episode_stats" in info


def test_offer_accept_decline_and_acceptance_curve():
    # A accepts only if p > 0.85 (true on day 0: p = 0.9); B needs p > 0.95 (never)
    env, _ = two_candidates(u_accept=[0.85, 0.95])
    _, r, done, _, _ = env.step(env.N + 1)                   # offer B: declined, B leaves
    assert r == 0.0 and not done and env.order == [0] and env.roles_left == 3
    _, r, _, _, _ = env.step(env.N)                          # offer A: accepted
    assert r == pytest.approx(2.0) and env.roles_left == 2 and env.order == []
    # same A two days later: p = 0.9 * exp(-0.06) = 0.8476 < 0.85, so declined
    env, _ = two_candidates(u_accept=[0.85, 0.95])
    end_day(env); end_day(env)
    _, r, *_ = env.step(env.N)
    assert r == 0.0 and env.roles_left == 3


def test_episode_ends_when_all_roles_filled():
    sc = STRESS.with_(roles=2)
    env = HiringEnv(sc)
    env.reset(options={"draws": hand_draws(sc, [0, 0, 0], [2.0, 1.0, 0.5])})
    _, _, done, _, _ = env.step(env.N)
    assert not done
    _, r, done, truncated, info = env.step(env.N)
    assert done and not truncated and info["episode_stats"]["ret"] == pytest.approx(3.0)


def test_slots_sorted_by_naive_estimate_for_both_belief_variants():
    views = []
    for variant in ("survival", "naive"):
        env = HiringEnv(STRESS.with_(obs_belief=variant, h0=0.10))
        env.reset(seed=SEEDS[7])
        for _ in range(12):
            end_day(env)
        v = env.view()
        assert np.all(np.diff(v.naive) <= 0)
        views.append((v.ids.copy(), env._obs()))
    assert np.array_equal(views[0][0], views[1][0])          # same candidates in the same slots
    assert not np.array_equal(views[0][1], views[1][1])      # but the shown estimate differs


def test_view_shows_top_n_and_logs_hidden_candidates():
    sc = STRESS.with_(n_slots=2, h0=0.10)
    env = HiringEnv(sc)
    # C has the highest resume (q = 3) but arrived long ago relative to nobody; build a case where a
    # hidden candidate has a higher survival-aware estimate than a shown one: old strong vs fresh.
    env.reset(options={"draws": hand_draws(sc, [0, 0, 30], [1.2, 1.1, 1.0])})
    for _ in range(30):
        end_day(env)
    v = env.view()
    assert env.stats["days_pool_over_slots"] >= 1 and list(v.ids) == [0, 1]
    assert env.mu_surv[2] > env.mu_surv[1]                   # the fresh hidden candidate is really better
    assert env.stats["days_hidden_better"] >= 1


def test_observation_scaling_is_identical_across_slots():
    env = HiringEnv(STRESS)
    env.reset(options={"draws": hand_draws(STRESS, [0, 0], [1.5, 1.5])})
    obs = env._obs()
    assert np.array_equal(slot_features(env, obs, 0), slot_features(env, obs, 1))
    assert obs.dtype == np.float32 and obs.shape == (env.N * env.n_slot_features + N_GLOBAL,)
    g = obs[-N_GLOBAL:]
    assert g.tolist() == pytest.approx([1.0, 1.0, 1.0, 2 / 20, 1.0])


# ---- small instance: pool cap and automatic drop ------------------------------------------------------

def test_small_instance_overflow_uses_shared_drop_rule():
    env = HiringEnv(SMALL)
    b = env.belief
    # resume bins: q = -2 -> low, q = 0 -> middle, q = 2 -> high (no resume noise)
    env.reset(options={"draws": hand_draws(SMALL, [0, 1, 2], [0.0, -2.0, 2.0])})
    end_day(env)
    assert sorted(env.order) == [0, 1]
    end_day(env)                                             # third arrival: lowest naive estimate leaves
    assert sorted(env.order) == [0, 2] and env.stats["dropped_on_overflow"] == 1
    means = [b.naive_mean(b.fresh_type(r)) for r in (1, 0, 2)]
    assert overflow_drop_index(means, [2, 1, 0]) == 1
    # tie on the estimate: the one who has waited longest leaves
    env.reset(options={"draws": hand_draws(SMALL, [0, 1, 2], [0.0, 0.0, 0.0])})
    end_day(env); end_day(env)
    assert sorted(env.order) == [1, 2]
    assert overflow_drop_index([0.3, 0.3, 0.3], [2, 1, 0]) == 0


def test_small_instance_binned_observation_and_pending():
    env = HiringEnv(SMALL)
    obs0, _ = env.reset(options={"draws": hand_draws(SMALL, [0], [2.0], interview_z=0.0)})
    f0 = slot_features(env, obs0, 0).copy()
    assert f0[5] == 1.0 and not f0[6:].any()                 # top resume bin, no interview outcomes yet
    obs1, *_ = env.step(0)
    assert np.array_equal(slot_features(env, obs1, 0)[[1, 2, 3, 5, 6, 7, 8]], f0[[1, 2, 3, 5, 6, 7, 8]])
    assert not env.action_masks()[0]                         # one interview slot per day
    obs2, *_ = end_day(env)
    f2 = slot_features(env, obs2, 0)
    assert f2[8] == pytest.approx(0.5) and f2[2] == pytest.approx(0.5)   # outcome in the top bin (q = 2 > 1)
    assert f2[1] > f0[1]
