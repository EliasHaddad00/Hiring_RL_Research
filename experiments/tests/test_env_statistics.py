"""Simulated rates match their configured values (fixed seeds, 3 standard errors)."""
import numpy as np
import pytest

from helpers import end_day
from hiring.belief import acceptance, hazard
from hiring.config import episode_seeds, load_scenario
from hiring.env import HiringEnv

STRESS = load_scenario("stress")
SEEDS = list(episode_seeds("dev", 2000))


def z_score(outcomes, probs):
    outcomes, probs = np.asarray(outcomes, float), np.asarray(probs, float)
    return (outcomes - probs).sum() / np.sqrt((probs * (1 - probs)).sum())


def test_arrival_rate_in_environment():
    env = HiringEnv(STRESS)
    counts = []
    for s in SEEDS[:600]:
        env.reset(seed=s)
        done = False
        while not done:
            *_, done, _, _ = end_day(env)
        counts.append(env.next_arrival)
    counts = np.array(counts)
    expected = STRESS.arrival_rate * STRESS.horizon
    assert abs(counts.mean() - expected) < 3 * np.sqrt(expected / len(counts))
    assert counts.var() == pytest.approx(expected, rel=0.2)          # Poisson: variance equals mean


@pytest.mark.parametrize("h0", [0.02, 0.05, 0.10])
def test_departure_hazard(h0):
    """Nobody is hired or interviewed; each night compare who left with the configured hazard."""
    sc = STRESS.with_(h0=h0)
    env = HiringEnv(sc)
    left, prob, qual = [], [], []
    for s in SEEDS[:300]:
        env.reset(seed=s)
        done = False
        while not done:
            before = np.flatnonzero(env.alive)
            *_, done, _, _ = end_day(env)
            if done:
                break
            left.append(~env.alive[before]); prob.append(hazard(env.q[before], h0, sc.quality_slope)); qual.append(env.q[before])
    left, prob, qual = map(np.concatenate, (left, prob, qual))
    assert len(left) > 20000
    assert abs(z_score(left, prob)) < 3
    for lo, hi in [(-9, -0.5), (-0.5, 0.5), (0.5, 9)]:               # hazard rises with quality as configured
        sel = (qual >= lo) & (qual < hi)
        assert abs(z_score(left[sel], prob[sel])) < 3
    assert left[qual > 0.5].mean() > 1.5 * left[qual < -0.5].mean()


def test_acceptance_curve():
    """Offer to a random shown candidate on random days; compare acceptance by days waited."""
    sc = STRESS.with_(h0=0.01, roles=10 ** 6)     # effectively unlimited roles: many offers per episode
    env = HiringEnv(sc)
    rng = np.random.default_rng(4)
    days, accepted = [], []
    for s in SEEDS:                                # about 10 offers per episode
        env.reset(seed=s)
        done = False
        while not done:
            v = env.view()
            if len(v.ids) and rng.random() < 0.15:
                slot = int(rng.integers(len(v.ids)))
                hires_before = env.stats["hires"]
                days.append(int(v.days[slot]))
                *_, done, _, _ = env.step(env.N + slot)
                accepted.append(env.stats["hires"] - hires_before)
            else:
                *_, done, _, _ = end_day(env)
    days, accepted = np.array(days), np.array(accepted)
    probs = acceptance(days, sc)
    assert len(days) > 15000
    assert abs(z_score(accepted, probs)) < 3
    for lo, hi in [(0, 1), (1, 4), (4, 11), (11, 61)]:
        sel = (days >= lo) & (days < hi)
        assert sel.sum() > 500 and abs(z_score(accepted[sel], probs[sel])) < 3
    assert accepted[days == 0].mean() == pytest.approx(0.9, abs=0.02)
    late = days >= 20
    assert accepted[late].mean() < 0.55                                # decay is real: 0.9 * exp(-0.6) = 0.49


def test_interview_noise_and_cost():
    """Interview everyone shown each day; pending signals must be q + N(0, interview_sd^2)."""
    sc = STRESS.with_(h0=0.0)
    env = HiringEnv(sc)
    err, cost = [], 0.0
    for s in SEEDS[:150]:
        env.reset(seed=s)
        done = False
        while not done:
            mask = env.action_masks()
            a = int(np.flatnonzero(mask)[0])          # lowest valid index: interviews first, then offers, then end day
            if a >= env.N:
                a = 2 * env.N
            else:
                c = env.order[a]
            _, r, done, _, info = env.step(a)
            if a < env.N:
                err.append(env.pending[c] - env.q[c]); cost += -r
    err = np.array(err)
    assert len(err) > 5000
    assert abs(err.mean()) < 3 * sc.interview_sd / np.sqrt(len(err))
    assert err.std() == pytest.approx(sc.interview_sd, rel=0.03)
    assert cost == pytest.approx(sc.interview_cost * len(err))
