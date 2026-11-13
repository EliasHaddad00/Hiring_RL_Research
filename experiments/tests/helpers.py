"""Shared test helpers: hand-built episodes and simple policies."""
import numpy as np

from hiring.seeding import N_INTERVIEW_DRAWS, EpisodeDraws


def hand_draws(sc, arrival_day, quality, resume_z=None, interview_z=None, u_poach=1.0, u_accept=0.0):
    """Episode with chosen candidates. Defaults: nobody is poached, every offer is accepted,
    signals carry no noise."""
    n = len(quality)
    full = lambda x, shape: np.broadcast_to(np.asarray(x, float), shape).copy()
    return EpisodeDraws(
        arrival_day=np.asarray(arrival_day, int),
        quality=np.asarray(quality, float),
        resume_z=full(0.0 if resume_z is None else resume_z, (n,)),
        interview_z=full(0.0 if interview_z is None else interview_z, (n, N_INTERVIEW_DRAWS)),
        u_poach=full(u_poach, (n, sc.horizon)),
        u_accept=full(u_accept, (n,)),
    )


def random_valid_action(env, rng):
    return int(rng.choice(np.flatnonzero(env.action_masks())))


def end_day(env):
    return env.step(2 * env.N)


def slot_features(env, obs, slot):
    f = env.n_slot_features
    return obs[slot * f:(slot + 1) * f]
