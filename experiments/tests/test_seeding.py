import numpy as np

from hiring.config import episode_seeds, load_scenario, load_seeds
from hiring.seeding import N_INTERVIEW_DRAWS, make_draws

SC = load_scenario("stress")


def test_same_seed_same_draws():
    a, b = make_draws(1000001, SC), make_draws(1000001, SC)
    for name in ("arrival_day", "quality", "resume_z", "interview_z", "u_poach", "u_accept"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_different_seeds_differ():
    a, b = make_draws(1000001, SC), make_draws(1000002, SC)
    assert not np.array_equal(a.quality[:5], b.quality[:5])


def test_shapes_and_order():
    d = make_draws(1000003, SC)
    assert d.interview_z.shape == (d.n, N_INTERVIEW_DRAWS) and N_INTERVIEW_DRAWS == 6
    assert d.u_poach.shape == (d.n, SC.horizon)
    assert np.all(np.diff(d.arrival_day) >= 0)
    assert d.arrival_day.min() >= 0 and d.arrival_day.max() < SC.horizon


def test_candidate_draws_do_not_depend_on_arrival_count_or_pressure():
    """Candidate k keeps the same quality and luck when the arrival rate or hazard changes."""
    base = make_draws(1000004, SC)
    busy = make_draws(1000004, SC.with_(arrival_rate=3.0, h0=0.10, accept_decay=0.0))
    k = min(base.n, busy.n)
    assert k > 20
    for name in ("quality", "resume_z", "interview_z", "u_poach", "u_accept"):
        assert np.array_equal(getattr(base, name)[:k], getattr(busy, name)[:k])


def test_first_three_interview_noises_unchanged_under_cap_six():
    a = make_draws(1000005, SC)
    b = make_draws(1000005, SC.with_(max_interviews=6))
    assert np.array_equal(a.interview_z, b.interview_z)


def test_arrival_statistics():
    n = np.array([make_draws(s, SC).n for s in episode_seeds("dev", 4000)])
    expected = SC.arrival_rate * SC.horizon
    se = np.sqrt(expected / len(n))
    assert abs(n.mean() - expected) < 3 * se
    small = load_scenario("small_exact")
    m = np.array([make_draws(s, small).n for s in episode_seeds("dev", 4000)])
    p = small.arrival_rate
    assert abs(m.mean() - p * small.horizon) < 3 * np.sqrt(small.horizon * p * (1 - p) / len(m))
    assert all(np.all(np.diff(make_draws(s, small).arrival_day) >= 1) for s in episode_seeds("dev", 50))


def test_seed_pools_are_disjoint():
    pools = load_seeds()["episode_seeds"]
    spans = sorted((p["start"], p["start"] + p["count"]) for p in pools.values())
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    assert load_seeds()["training_episode_seed_start"] >= spans[-1][1]
    train = load_seeds()["training_seeds"]
    assert not set(train["tuning"]) & set(train["final"])
    assert len(train["final"]) == 15 and len(set(train["final"])) == 15
