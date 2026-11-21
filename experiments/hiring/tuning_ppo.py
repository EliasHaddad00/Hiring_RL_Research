"""Seeded random search over PPO hyperparameters (pre-registered Nov 15).

40 configurations per architecture, the same list for every architecture. Configuration 1 is the
starting values of the run config, as for the rules. Ranges are the table in the research plan
("Hiring race RL research plan.md", starting configuration). No search library.

Each configuration is trained with the tuning seeds for SEARCH_SHARE of the architecture's final
training length; the FINALISTS best are then trained at full length.
"""
from __future__ import annotations

import copy

import numpy as np

SEARCH_SHARE = 0.30
FINALISTS = 5
FINAL_STEPS = {"index": 3_000_000, "unrestricted": 10_000_000, "pooled": 10_000_000}
TUNING_SEEDS = (101, 102, 103)
CHOSEN = {"unrestricted": 8, "index": 18, "pooled": 8}   # frozen Nov 20: configs/ppo_final_<actor>.yaml

CHOICES = {"n_steps": (256, 512, 1024, 2048),
           "batch_size": (64, 128, 256, 512, 1024),
           "n_epochs": (4, 5, 10),
           "gamma": (0.99, 0.995, 0.999, 1.0),
           "clip_range": (0.1, 0.2, 0.3),
           "hidden": (64, 128, 256)}
LOG_RANGES = {"learning_rate": (1e-4, 1e-3), "ent_coef": (1e-3, 3e-2)}
LINEAR_RANGES = {"gae_lambda": (0.90, 0.98)}
PARAMETERS = tuple(CHOICES) + tuple(LOG_RANGES) + tuple(LINEAR_RANGES)


def search_steps(actor: str) -> int:
    return int(round(SEARCH_SHARE * FINAL_STEPS[actor]))


def start_values(base_cfg: dict) -> dict:
    """The hyperparameters of the run config, in the search's own flat form."""
    out = {k: base_cfg["ppo"][k] for k in PARAMETERS if k != "hidden"}
    out["hidden"] = int(base_cfg.get("hidden", 64))
    return out


def sample_configs(base_cfg: dict, n: int = 40, search_seed: int = 0) -> list[dict]:
    """n flat hyperparameter dicts; the first is the starting values, the rest are random draws."""
    rng = np.random.default_rng(search_seed)
    configs = [start_values(base_cfg)]
    while len(configs) < n:
        c = {k: v[int(rng.integers(len(v)))] for k, v in CHOICES.items()}
        for k, (lo, hi) in LOG_RANGES.items():
            c[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        for k, (lo, hi) in LINEAR_RANGES.items():
            c[k] = float(rng.uniform(lo, hi))
        configs.append(c)
    return configs


def apply(base_cfg: dict, flat: dict, curve: bool) -> dict:
    """The run config with one flat hyperparameter dict put in. curve=False switches the learning
    curve off (search runs are scored once, at the end)."""
    cfg = copy.deepcopy(base_cfg)
    for k, v in flat.items():
        if k == "hidden":
            cfg["hidden"] = int(v)
        else:
            cfg["ppo"][k] = v
    if not curve:
        cfg["eval"] = dict(cfg["eval"], episodes=0)
    return cfg


def at_boundary(flat: dict) -> list[str]:
    """Continuous parameters within 10% (of the range, log scale where sampled so) of an end, and
    discrete ones at an end of their list. The plan widens any range whose best value sits there."""
    out = []
    for k, v in CHOICES.items():
        if flat[k] in (v[0], v[-1]):
            out.append(k)
    for k, (lo, hi) in LOG_RANGES.items():
        x = (np.log(flat[k]) - np.log(lo)) / (np.log(hi) - np.log(lo))
        if x < 0.1 or x > 0.9:
            out.append(k)
    for k, (lo, hi) in LINEAR_RANGES.items():
        x = (flat[k] - lo) / (hi - lo)
        if x < 0.1 or x > 0.9:
            out.append(k)
    return out
