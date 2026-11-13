"""G1 check: throughput on our simulator. A measurement only; the models are discarded.

  1. raw environment steps per second under the random policy (no learning)
  2. MaskablePPO time/fps over about 100,000 steps, 8 environments in DummyVecEnv, one torch thread
  3. the same with 8 environments in SubprocVecEnv
  4. the same as 2 with the index-structured actor, and on the small instance

time/fps includes gradient updates. Writes results/g1/throughput.{json,txt}.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.env import HiringEnv  # noqa: E402
from hiring.policies import IndexPolicy, index_policy_kwargs  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g1"
STEPS = 100_000
N_ENVS = 8


def make_stress():
    return HiringEnv(load_scenario("stress"))


def make_small():
    return HiringEnv(load_scenario("small_exact"))


def raw_env_speed(make, episodes=1500):
    env, rng = make(), np.random.default_rng(0)
    steps, t0 = 0, time.perf_counter()
    for s in episode_seeds("dev", episodes):
        env.reset(seed=s)
        done = False
        while not done:
            *_, done, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
            steps += 1
    dt = time.perf_counter() - t0
    return dict(steps_per_second=steps / dt, steps_per_episode=steps / episodes, episodes_per_second=episodes / dt)


def ppo_fps(name, make, vec_cls_name, policy="MlpPolicy"):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    vec_cls = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_cls_name]
    venv = make_vec_env(make, n_envs=N_ENVS, seed=0, vec_env_cls=vec_cls)
    kwargs = dict(policy_kwargs=index_policy_kwargs(make())) if policy is IndexPolicy else {}
    model = MaskablePPO(policy, venv, n_steps=512, batch_size=256, n_epochs=10, gamma=0.999, seed=0,
                        device="cpu", verbose=0, **kwargs)
    log_dir = OUT / "throughput" / name
    model.set_logger(configure(str(log_dir), ["csv"]))
    t0 = time.perf_counter()
    model.learn(total_timesteps=STEPS)
    wall = time.perf_counter() - t0
    model.logger.close()
    with open(log_dir / "progress.csv") as fh:
        rows = list(csv.DictReader(fh))
    venv.close()
    last = rows[-1]
    return dict(time_fps=float(last["time/fps"]), total_timesteps=int(float(last["time/total_timesteps"])),
                wall_seconds=wall, ep_len_mean=float(last["rollout/ep_len_mean"]))


def main():
    th.set_num_threads(1)
    report = {"torch_threads": 1, "n_envs": N_ENVS, "target_steps": STEPS}
    report["raw_env_stress"] = raw_env_speed(make_stress)
    report["raw_env_small"] = raw_env_speed(make_small, 5000)
    for name, args in [("ppo_mlp_dummy_stress", (make_stress, "dummy")),
                       ("ppo_mlp_subproc_stress", (make_stress, "subproc")),
                       ("ppo_index_dummy_stress", (make_stress, "dummy", IndexPolicy)),
                       ("ppo_mlp_dummy_small", (make_small, "dummy"))]:
        report[name] = ppo_fps(name, *args)
    lines = [
        f"Raw environment, random policy, stress:  {report['raw_env_stress']['steps_per_second']:8.0f} steps/s "
        f"({report['raw_env_stress']['steps_per_episode']:.0f} steps per episode)",
        f"Raw environment, random policy, small:   {report['raw_env_small']['steps_per_second']:8.0f} steps/s "
        f"({report['raw_env_small']['steps_per_episode']:.0f} steps per episode)",
    ]
    for key, label in [("ppo_mlp_dummy_stress", "MaskablePPO MlpPolicy, DummyVecEnv x8, stress"),
                       ("ppo_mlp_subproc_stress", "MaskablePPO MlpPolicy, SubprocVecEnv x8, stress"),
                       ("ppo_index_dummy_stress", "MaskablePPO IndexPolicy, DummyVecEnv x8, stress"),
                       ("ppo_mlp_dummy_small", "MaskablePPO MlpPolicy, DummyVecEnv x8, small")]:
        r = report[key]
        r["hours_per_3M_steps"] = 3e6 / r["time_fps"] / 3600
        lines.append(f"{label:48s} time/fps = {r['time_fps']:6.0f}  ({r['total_timesteps']} steps in {r['wall_seconds']:.0f}s; "
                     f"3M steps = {r['hours_per_3M_steps']:.2f} hours)")
    text = "\n".join(lines)
    print(text)
    (OUT / "throughput.txt").write_text(text + "\n")
    (OUT / "throughput.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
