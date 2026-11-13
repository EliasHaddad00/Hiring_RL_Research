"""Run policies on fixed episodes. Every policy sees the same episodes, so results are paired."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .config import Scenario
from .env import HiringEnv

STAT_KEYS = ("ret", "interviews", "hires", "days", "days_pool_over_slots", "days_hidden_better",
             "hires_at_interview_cap", "dropped_on_overflow")


def run_episode(env: HiringEnv, policy, seed: Optional[int] = None, draws=None) -> dict:
    env.reset(seed=seed, options={"draws": draws} if draws is not None else None)
    policy.reset(env)
    done = False
    while not done:
        _, _, done, _, info = env.step(policy.act(env))
    return info["episode_stats"]


def evaluate(sc: Scenario, policy, seeds: Optional[Iterable[int]] = None, draws_list=None) -> dict:
    """Unshaped per-episode results as arrays keyed by STAT_KEYS (ret = payoff)."""
    env = HiringEnv(sc)
    env.build_obs = False
    if draws_list is not None:
        rows = [run_episode(env, policy, draws=d) for d in draws_list]
    else:
        rows = [run_episode(env, policy, seed=s) for s in seeds]
    return {k: np.array([r[k] for r in rows], float) for k in STAT_KEYS}


def summarize(res: dict) -> dict:
    n = len(res["ret"])
    hires = res["hires"].sum()
    return dict(
        episodes=n,
        mean=float(res["ret"].mean()),
        se=float(res["ret"].std(ddof=1) / np.sqrt(n)),
        interviews_per_episode=float(res["interviews"].mean()),
        hires_per_episode=float(res["hires"].mean()),
        share_days_pool_over_slots=float(res["days_pool_over_slots"].sum() / res["days"].sum()),
        share_days_hidden_better=float(res["days_hidden_better"].sum() / res["days"].sum()),
        share_hires_at_interview_cap=float(res["hires_at_interview_cap"].sum() / hires) if hires else 0.0,
    )
