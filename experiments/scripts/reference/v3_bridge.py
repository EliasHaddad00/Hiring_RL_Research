"""Replay the earlier simulator's own random draws through our environment.

hiring_sim_v3.gen() draws a table per episode (arrival day, quality, resume, interview noise,
departure and acceptance uniforms per candidate). This converts one episode of that table into
our EpisodeDraws, so both simulators play the same world and can be compared episode by episode.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import hiring_sim_v3_fullprec as v3  # noqa: E402  (the full-precision copy of notes/hiring_sim_v3.py)
from hiring.seeding import N_INTERVIEW_DRAWS, EpisodeDraws  # noqa: E402

# (label, h0, acceptance decay) for the five pressure levels of the earlier check
SETTINGS = [("none", 0.0, 0.0), ("decay_only", 0.0, 0.03), ("h0_0.02", 0.02, 0.03),
            ("h0_0.05", 0.05, 0.03), ("h0_0.10", 0.10, 0.03)]
EVAL_SEEDS = [100, 101, 102, 103]       # v3 scored rules on gen(1000, 100 + j), j = 0..3


def v3_key(h0, b):
    return f"h0={h0},b={b}"


def v3_episode_draws(d: dict, e: int) -> EpisodeDraws:
    cols = np.flatnonzero(d["exists"][e])
    cols = cols[np.lexsort((cols, d["arr"][e, cols]))]          # arrival order, ties by table column
    q = d["q"][e, cols]
    iz = np.zeros((len(cols), N_INTERVIEW_DRAWS))
    iz[:, :3] = d["inoise"][e, cols] / v3.sig                   # v3 draws 3 interview noises per candidate
    return EpisodeDraws(
        arrival_day=d["arr"][e, cols].astype(int),
        quality=q,
        resume_z=(d["resume"][e, cols] - q) / v3.sig0,
        interview_z=iz,
        u_poach=d["upoach"][e, cols].astype(float),
        u_accept=d["uacc"][e, cols],
    )


def v3_rule_kwargs(family: str, par):
    """v3's own mapping from a parameter vector to run() arguments."""
    if family == "static_threshold":
        return dict(kmax=0, o=[par[0]], so=[0], i=[v3.INF], si=0)
    return v3.FAM[family][1](np.asarray(par, float))
