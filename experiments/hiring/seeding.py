"""Per-episode random draws, indexed by candidate and not by call order.

One episode seed fixes the whole "world": how many candidates arrive each day and, for candidate k
(numbered in arrival order), the hidden quality, the resume noise, up to 6 interview noises, one
departure uniform per day and one acceptance uniform. Two policies run on the same seed therefore
face the same candidates and the same luck, whatever they do. This is what makes evaluation paired.

Six interview noises are always drawn. Under the default cap of 3 interviews only the first three
are used, so a rerun with a cap of 6 replays the same episodes.

Each kind of draw has its own random stream and a fixed table size (scenario.max_candidates), so
the draws of candidate k never depend on how many candidates arrived. Arrivals beyond the table
size are dropped (with 135 rows and 90 expected arrivals this has probability about 1e-6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Scenario

N_INTERVIEW_DRAWS = 6
_STREAMS = ("arrivals", "quality", "resume", "interview", "poach", "accept")


@dataclass(frozen=True)
class EpisodeDraws:
    arrival_day: np.ndarray    # (n,) int, non-decreasing: day on which candidate k arrives
    quality: np.ndarray        # (n,) hidden q ~ N(0, 1)
    resume_z: np.ndarray       # (n,) standard normal; resume signal = q + resume_sd * z
    interview_z: np.ndarray    # (n, 6) standard normal; j-th interview = q + interview_sd * z[j]
    u_poach: np.ndarray        # (n, T) uniform; poached on day t if u < hazard(q)
    u_accept: np.ndarray       # (n,) uniform; accepts an offer if u < acceptance probability

    @property
    def n(self) -> int:
        return len(self.quality)


def make_draws(seed: int, sc: Scenario) -> EpisodeDraws:
    children = np.random.SeedSequence(int(seed)).spawn(len(_STREAMS))
    rng = {name: np.random.default_rng(child) for name, child in zip(_STREAMS, children)}
    T, M = sc.horizon, sc.max_candidates

    if sc.arrival == "poisson":
        per_day = rng["arrivals"].poisson(sc.arrival_rate, T)
    else:
        per_day = (rng["arrivals"].random(T) < sc.arrival_rate).astype(int)
    arrival_day = np.repeat(np.arange(T), per_day)[:M]
    n = len(arrival_day)

    # Full tables are always drawn, then cut to n, so candidate k's draws do not depend on n.
    quality = rng["quality"].standard_normal(M)[:n]
    resume_z = rng["resume"].standard_normal(M)[:n]
    interview_z = rng["interview"].standard_normal((M, N_INTERVIEW_DRAWS))[:n]
    u_poach = rng["poach"].random((M, T))[:n]
    u_accept = rng["accept"].random(M)[:n]
    return EpisodeDraws(arrival_day, quality, resume_z, interview_z, u_poach, u_accept)
