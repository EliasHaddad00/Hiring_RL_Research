"""Library check on a standard task: MaskablePPO on CartPole-v1 through the masked code path.

CartPole has no invalid actions, so the mask is all-true; this exercises the same classes we use
on the hiring task (MaskablePPO, ActionMasker, maskable evaluate_policy, progress.csv logging).
Known score: CartPole-v1 is capped at 500 and counts as solved at 475.
Writes results/g1/library_check.{json,txt}.
"""
import csv
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

OUT = Path(__file__).resolve().parents[2] / "results" / "g1"
STEPS, SEED, SOLVED = 150_000, 0, 475.0
LOGGED = ["rollout/ep_rew_mean", "train/policy_gradient_loss", "train/value_loss", "train/entropy_loss",
          "train/approx_kl", "train/clip_fraction"]


def make():
    return ActionMasker(gym.make("CartPole-v1"), lambda env: np.ones(2, bool))


def main():
    th.set_num_threads(1)
    venv = make_vec_env(make, n_envs=8, seed=SEED)
    model = MaskablePPO("MlpPolicy", venv, n_steps=256, batch_size=256, n_epochs=10, gamma=0.99,
                        gae_lambda=0.95, ent_coef=0.0, learning_rate=3e-4, seed=SEED, device="cpu")
    t0 = time.time()
    log_dir = OUT / "library_check"          # progress.csv is kept: return, losses, entropy, KL, clip fraction
    model.set_logger(configure(str(log_dir), ["csv"]))
    model.learn(total_timesteps=STEPS)
    model.logger.close()
    with open(log_dir / "progress.csv") as fh:
        rows = list(csv.DictReader(fh))
    eval_env = make_vec_env(make, n_envs=1, seed=10_000)
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True, use_masking=True)
    missing = [k for k in LOGGED if k not in rows[-1]]
    report = dict(task="CartPole-v1", steps=int(float(rows[-1]["time/total_timesteps"])), seed=SEED,
                  eval_mean=float(mean), eval_std=float(std), eval_episodes=50, solved_threshold=SOLVED,
                  passed=bool(mean >= SOLVED and not missing), logged_columns_missing=missing,
                  seconds=time.time() - t0, final_row={k: rows[-1].get(k) for k in LOGGED})
    text = (f"MaskablePPO on CartPole-v1, {report['steps']} steps, seed {SEED}: "
            f"evaluation return {mean:.1f} +- {std:.1f} over 50 episodes (solved at {SOLVED:.0f}, cap 500): "
            f"{'PASS' if report['passed'] else 'FAIL'}; required log columns missing: {missing or 'none'}")
    print(text)
    (OUT / "library_check.txt").write_text(text + "\n")
    (OUT / "library_check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
