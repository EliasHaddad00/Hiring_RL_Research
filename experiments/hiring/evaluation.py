"""Run policies on fixed episodes. Every policy sees the same episodes, so results are paired."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .config import Scenario
from .env import HiringEnv

STAT_KEYS = ("ret", "interviews", "hires", "days", "days_pool_over_slots", "days_hidden_better",
             "hires_at_interview_cap", "dropped_on_overflow", "offers")


def run_episode(env: HiringEnv, policy, seed: Optional[int] = None, draws=None) -> dict:
    env.reset(seed=seed, options={"draws": draws} if draws is not None else None)
    policy.reset(env)
    done = False
    while not done:
        _, _, done, _, info = env.step(policy.act(env))
    return info["episode_stats"]


class PPOPolicy:
    """A trained MaskablePPO model behind the same reset/act interface as the hand-written rules.

    The action mask is passed on every call; leaving it out would raise no error in SB3.
    deterministic=True takes the most probable valid action (the pre-registered scoring mode).
    With deterministic=False actions are sampled, reseeded per episode so a rerun is identical.
    """
    needs_obs = True

    def __init__(self, model, deterministic: bool = True, name: str = "ppo"):
        self.model, self.deterministic, self.name = model, deterministic, name

    @classmethod
    def load(cls, path, **kw) -> "PPOPolicy":
        from sb3_contrib import MaskablePPO
        return cls(MaskablePPO.load(path, device="cpu"), **kw)

    def reset(self, env):
        if not self.deterministic:
            import torch as th
            th.manual_seed(int(env.episode_seed or 0) % (2 ** 31))

    def act(self, env) -> int:
        action, _ = self.model.predict(env._obs(), action_masks=env.action_masks(),
                                       deterministic=self.deterministic)
        return int(action)


def evaluate(sc: Scenario, policy, seeds: Optional[Iterable[int]] = None, draws_list=None) -> dict:
    """Unshaped per-episode results as arrays keyed by STAT_KEYS (ret = payoff)."""
    env = HiringEnv(sc)
    env.build_obs = bool(getattr(policy, "needs_obs", False))
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
        offers_per_episode=float(res["offers"].mean()),
        realised_acceptance=float(hires / res["offers"].sum()) if res["offers"].sum() else float("nan"),
        share_days_pool_over_slots=float(res["days_pool_over_slots"].sum() / res["days"].sum()),
        share_days_hidden_better=float(res["days_hidden_better"].sum() / res["days"].sum()),
        share_hires_at_interview_cap=float(res["hires_at_interview_cap"].sum() / hires) if hires else 0.0,
    )


def paired_compare(a, b, margin: float = 0.02, level: float = 0.95) -> dict:
    """Paired comparison of policy A against comparator B, and the pre-registered verdict.

    a, b: scores on the same units in the same order (episodes of one pool, or training seeds).
    The interval is a t-interval on the mean of a - b. The margin is `margin` times the
    comparator's mean payoff. Interval entirely inside the margin: "equivalent" (takes
    precedence). Otherwise entirely above zero: "better"; entirely below zero: "worse";
    anything else: "inconclusive".
    """
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) >= 2, "paired scores must have the same length"
    d = a - b
    n = len(d)
    mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.5 + level / 2, n - 1)) * se
    lo, hi = mean - half, mean + half
    m = margin * abs(float(b.mean()))
    if lo > -m and hi < m:
        verdict = "equivalent"
    elif lo > 0:
        verdict = "better"
    elif hi < 0:
        verdict = "worse"
    else:
        verdict = "inconclusive"
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return dict(n=n, mean_a=float(a.mean()), mean_b=float(b.mean()), diff=mean, se=se, lo=lo, hi=hi,
                margin=m, verdict=verdict, correlation=corr)
