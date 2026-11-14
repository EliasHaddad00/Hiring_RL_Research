"""One MaskablePPO training run on the hiring simulator.

Fixed for every run (CLAUDE.md, "Implementation pitfalls"):
  - DummyVecEnv and one torch thread; runs are parallelised across processes, one per core.
  - Observations are scaled inside the environment. VecNormalize is never used.
  - The evaluation callback is maskable, plays fixed development episodes and takes the most
    probable valid action, the same setting as final scoring (PREREGISTRATION.md).
  - The reward is unshaped, so the logged return is the payoff.
  - progress.csv holds: rollout/ep_rew_mean and eval/mean_reward (return), train/policy_gradient_loss,
    train/value_loss, train/entropy_loss, train/approx_kl, train/clip_fraction, and
    train/normalized_entropy (entropy divided by the log of the number of valid actions).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from .config import CONFIG_DIR, check_training_seed, episode_seeds, load_scenario
from .env import HiringEnv

REQUIRED_COLUMNS = ("rollout/ep_rew_mean", "train/policy_gradient_loss", "train/value_loss",
                    "train/entropy_loss", "train/approx_kl", "train/clip_fraction")
ACTORS = ("unrestricted", "index")
EVAL_POOL_OFFSET = 90_000       # curve episodes: dev seeds 1,090,000 onward, apart from scoring episodes
ROOT = Path(__file__).resolve().parents[2]


def load_run_config(name_or_path: str) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    return yaml.safe_load(path.read_text())


def commit_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return "unknown"


def _normalized_entropy_callback():
    import torch as th
    from stable_baselines3.common.callbacks import BaseCallback

    class NormalizedEntropy(BaseCallback):
        """Entropy / log(valid actions), averaged over the rollout's states with a real choice."""

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            buf = self.model.rollout_buffer
            obs = buf.observations.reshape(-1, buf.observations.shape[-1])
            masks = buf.action_masks.reshape(obs.shape[0], -1).astype(bool)
            n_valid = masks.sum(1)
            keep = n_valid >= 2
            if not keep.any():
                return
            with th.no_grad():
                dist = self.model.policy.get_distribution(th.as_tensor(obs[keep]), action_masks=masks[keep])
                ent = dist.entropy().numpy()
            self.logger.record("train/normalized_entropy", float(np.mean(ent / np.log(n_valid[keep]))))
            self.logger.record("rollout/valid_actions_mean", float(n_valid.mean()))

    return NormalizedEntropy()


def train(cfg: dict, actor: str, seed: int, out_dir, total_timesteps: int | None = None,
          allow_final_seed: bool = False) -> dict:
    """Train one agent; write progress.csv, evaluations.npz, model.zip and run.json to out_dir."""
    import torch as th
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from .policies import IndexPolicy, index_policy_kwargs

    assert actor in ACTORS, f"actor must be one of {ACTORS}"
    check_training_seed(seed, allow_final=allow_final_seed)
    th.set_num_threads(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = load_scenario(cfg["scenario"], **(cfg.get("scenario_overrides") or {}))
    steps = int(total_timesteps if total_timesteps is not None else cfg["total_timesteps"])
    n_envs, p, ev = int(cfg["n_envs"]), dict(cfg["ppo"]), cfg["eval"]
    hidden = int(cfg.get("hidden", 64))

    venv = make_vec_env(lambda: HiringEnv(sc), n_envs=n_envs, seed=int(seed), vec_env_cls=DummyVecEnv)
    lr0 = float(p.pop("learning_rate"))
    schedule = p.pop("lr_schedule", "linear")
    assert schedule in ("linear", "constant")
    lr = (lambda progress: lr0 * progress) if schedule == "linear" else lr0
    if actor == "index":
        policy, policy_kwargs = IndexPolicy, dict(index_policy_kwargs(HiringEnv(sc)), hidden=hidden)
    else:
        policy, policy_kwargs = "MlpPolicy", dict(net_arch=dict(pi=[hidden, hidden], vf=[hidden, hidden]))
    model = MaskablePPO(policy, venv, learning_rate=lr, policy_kwargs=policy_kwargs, seed=int(seed),
                        device="cpu", verbose=0, **p)
    model.set_logger(configure(str(out_dir), ["csv"]))

    # Fixed development episodes for the curve. One environment, so episodes are played in order;
    # the pool is cycled, so every evaluation uses the same set of episodes.
    eval_seeds = list(episode_seeds("dev", int(ev["episodes"]), offset=EVAL_POOL_OFFSET))
    eval_env = DummyVecEnv([lambda: Monitor(HiringEnv(sc, episode_seeds=eval_seeds))])
    eval_cb = MaskableEvalCallback(eval_env, n_eval_episodes=len(eval_seeds),
                                   eval_freq=max(1, int(ev["freq_steps"]) // n_envs),
                                   deterministic=True, log_path=str(out_dir), best_model_save_path=None,
                                   verbose=0, warn=False)
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps, callback=CallbackList([_normalized_entropy_callback(), eval_cb]))
    wall = time.perf_counter() - t0
    model.logger.close()
    model.save(out_dir / "model")
    venv.close()
    eval_env.close()
    meta = dict(actor=actor, seed=int(seed), scenario=sc.name, h0=sc.h0, accept_decay=sc.accept_decay,
                obs_belief=sc.obs_belief, total_timesteps=int(model.num_timesteps), wall_seconds=wall,
                steps_per_second=model.num_timesteps / wall, config=cfg, commit=commit_hash(),
                eval_episode_seeds=[eval_seeds[0], eval_seeds[-1]], eval_deterministic=True,
                last_eval_mean=float(eval_cb.last_mean_reward))
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1))
    return meta
