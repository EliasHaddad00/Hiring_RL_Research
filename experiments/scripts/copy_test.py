"""Copy test: can each actor architecture represent the exact optimal policy of the small instance?

Each architecture is trained by supervised imitation of the exact optimal policy (no
reinforcement learning), then scored like a PPO agent. Reading (decision rules in the plan):
  copy passes, PPO fails   the architecture and inputs are sufficient; the shortfall is in training
  copy fails               the inputs or the architecture cannot express the optimal policy

Data: states visited on development episodes 40,000 to 59,999 when the exact policy acts but a
random valid action is taken with probability 0.2 (so off-path states are covered); the label is
always the exact policy's action. Loss: cross-entropy over valid actions.
Models go to results/g2/copy_test/<actor>_s0/; score them with
  python experiments/scripts/check_small_ppo.py --runs results/g2/copy_test --tag copy_test
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.env import HiringEnv  # noqa: E402
from hiring.exact_dp import ExactPolicy, ExactSolver  # noqa: E402
from hiring.policies import IndexPolicy, PooledIndexPolicy, index_policy_kwargs  # noqa: E402
from hiring.training import commit_hash, load_run_config  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g2" / "copy_test"
DATA_EPISODES, DATA_OFFSET, EXPLORE = 20_000, 40_000, 0.2
EPOCHS, BATCH, LR = 40, 512, 1e-3


def collect(sc):
    env, rng = HiringEnv(sc), np.random.default_rng(0)
    expert = ExactPolicy(ExactSolver(sc, "automatic"))
    obs_l, mask_l, act_l = [], [], []
    for seed in episode_seeds("dev", DATA_EPISODES, offset=DATA_OFFSET):
        obs, _ = env.reset(seed=seed)
        expert.reset(env)
        done = False
        while not done:
            mask = env.action_masks()
            a = expert.act(env)
            obs_l.append(obs); mask_l.append(mask); act_l.append(a)
            step = int(rng.choice(np.flatnonzero(mask))) if rng.random() < EXPLORE else a
            obs, _, done, _, _ = env.step(step)
    return np.array(obs_l, np.float32), np.array(mask_l), np.array(act_l)


def fit(actor, sc, obs, masks, acts):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    th.manual_seed(0)
    cfg = load_run_config("ppo_small")
    venv = DummyVecEnv([lambda: HiringEnv(sc)])
    if actor == "unrestricted":
        policy, kw = "MlpPolicy", dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    else:
        policy, kw = (IndexPolicy if actor == "index" else PooledIndexPolicy), index_policy_kwargs(HiringEnv(sc))
    model = MaskablePPO(policy, venv, policy_kwargs=kw, seed=0, device="cpu", n_steps=64, batch_size=64)
    opt = th.optim.Adam(model.policy.parameters(), lr=LR)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    o, m, a = th.as_tensor(obs), masks, th.as_tensor(acts)
    n, rng = len(o), np.random.default_rng(0)
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            dist = model.policy.get_distribution(o[idx], action_masks=m[idx])
            loss = -dist.log_prob(a[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    with th.no_grad():
        pred = th.cat([model.policy.get_distribution(o[i:i + 8192], action_masks=m[i:i + 8192]).distribution.probs.argmax(1)
                       for i in range(0, n, 8192)])
    agreement = float((pred == a).float().mean())
    folder = OUT / f"{actor}_s0"
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder / "model")
    (folder / "run.json").write_text(json.dumps(dict(
        actor=actor, seed=0, total_timesteps=0, method="supervised imitation of the exact policy",
        training_states=n, epochs=EPOCHS, agreement_on_training_states=agreement,
        final_loss=float(loss.detach()), config=cfg, commit=commit_hash()), indent=1))
    return agreement, float(loss.detach())


def main():
    th.set_num_threads(1)
    sc = load_scenario("small_exact")
    t0 = time.perf_counter()
    obs, masks, acts = collect(sc)
    print(f"{len(obs):,} states from {DATA_EPISODES:,} episodes in {time.perf_counter() - t0:.0f} s", flush=True)
    for actor in ("unrestricted", "index", "pooled"):
        t0 = time.perf_counter()
        agreement, loss = fit(actor, sc, obs, masks, acts)
        print(f"{actor:13s} agrees with the exact policy on {100 * agreement:.2f}% of training states "
              f"(loss {loss:.4f}, {time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
