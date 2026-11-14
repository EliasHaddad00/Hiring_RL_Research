"""Scenario definition and loader. One Scenario fixes every simulator parameter."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Optional, Tuple

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@dataclass(frozen=True)
class Scenario:
    name: str = "stress"
    horizon: int = 60                 # T, days; days are numbered 0 .. T-1
    roles: int = 3                    # m
    arrival: str = "poisson"          # "poisson" (mean per day) or "bernoulli" (probability per day)
    arrival_rate: float = 1.5
    max_candidates: int = 135         # size of the per-episode draw table; later arrivals are dropped
    resume_sd: float = 1.5
    interview_sd: float = 1.0
    interview_cost: float = 0.05
    interviews_per_day: int = 3       # K
    max_interviews: int = 3           # per candidate, whole episode
    h0: float = 0.05                  # daily departure hazard is min(1, h0 * exp(quality_slope * q))
    quality_slope: float = 0.6
    accept_base: float = 0.9          # acceptance is accept_base * exp(-accept_decay * days_waited)
    accept_decay: float = 0.03
    n_slots: int = 10                 # N candidates shown to the policy
    pool_cap: Optional[int] = None    # if set, overflow at arrival is resolved by pool.overflow_drop_index
    signals: str = "gaussian"         # "gaussian" or "binned" (small exact instance)
    resume_cuts: Tuple[float, ...] = ()
    interview_cuts: Tuple[float, ...] = ()
    obs_belief: str = "survival"      # which estimate the observation shows: "survival" or "naive"

    def __post_init__(self):
        assert self.arrival in ("poisson", "bernoulli")
        assert self.signals in ("gaussian", "binned")
        assert self.obs_belief in ("survival", "naive")
        assert self.max_interviews <= 6, "seeding draws 6 interview noises per candidate"
        if self.signals == "binned":
            assert self.resume_cuts and self.interview_cuts

    def with_(self, **overrides) -> "Scenario":
        return replace(self, **overrides)


def load_scenario(name_or_path: str, **overrides) -> Scenario:
    """Load configs/<name>.yaml (or a path) and apply keyword overrides."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    raw = yaml.safe_load(path.read_text())
    known = {f.name for f in fields(Scenario)}
    unknown = set(raw) - known
    assert not unknown, f"unknown scenario fields in {path}: {sorted(unknown)}"
    for key in ("resume_cuts", "interview_cuts"):
        if key in raw:
            raw[key] = tuple(float(x) for x in raw[key])
    raw.update(overrides)
    return Scenario(**raw)


def load_seeds() -> dict:
    """Seed pools from configs/seeds.yaml. Pools are disjoint; see that file."""
    return yaml.safe_load((CONFIG_DIR / "seeds.yaml").read_text())


class LockedPoolError(RuntimeError):
    """Raised when a reserved seed pool is requested without an explicit unlock."""


def episode_seeds(pool: str, count: Optional[int] = None, offset: int = 0,
                  allow_evaluation: bool = False) -> range:
    """`count` episode seeds of a pool, starting `offset` seeds into it.

    The evaluation pool is reserved for the final paired evaluation after configurations are
    frozen (gate G3). It is locked here, not in a script, so that no script can read it by accident.
    """
    if pool == "evaluation" and not allow_evaluation:
        raise LockedPoolError("The evaluation pool is reserved for final runs after gate G3. "
                              "Pass allow_evaluation=True only once configurations are frozen.")
    spec = load_seeds()["episode_seeds"][pool]
    n = spec["count"] - offset if count is None else count
    assert offset >= 0 and n >= 0 and offset + n <= spec["count"], \
        f"pool '{pool}' holds only {spec['count']} episode seeds"
    return range(spec["start"] + offset, spec["start"] + offset + n)


def check_training_seed(seed: int, allow_final: bool = False) -> None:
    """Final training seeds are reserved for the final runs (launched at gate G3)."""
    if int(seed) in load_seeds()["training_seeds"]["final"] and not allow_final:
        raise LockedPoolError(f"Training seed {seed} is a final seed, reserved until gate G3. "
                              "Use a tuning seed, or pass allow_final=True for the final runs.")
