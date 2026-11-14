"""Price each agent's mistakes on the small instance with the exact solver.

At every step the solver gives the value of each available action. The loss of a step is
(value of the best action) - (value of the action taken); it is zero for an optimal action.
Summed over an episode and averaged, these losses equal the exact optimum minus the agent's
expected payoff (the performance-difference identity), so the breakdown below accounts for the
whole shortfall, up to sampling error, without the luck of individual episodes.

Mistake types (action taken against the best action):
  offer too early         offered; best was to interview or to hold
  offer too late          interviewed or ended the day; best was an offer
  wrong candidate         offered or interviewed one candidate; best was the same act on another
  unneeded interview      interviewed; best was to end the day
  missed interview        ended the day; best was an interview

Writes results/g2/<tag>_mistakes.{txt,json}.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.env import HiringEnv  # noqa: E402
from hiring.evaluation import PPOPolicy  # noqa: E402
from hiring.exact_dp import ExactSolver  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g2"
TYPES = ("offer too early", "offer too late", "wrong candidate", "unneeded interview", "missed interview")
TOL = 1e-9


def action_values(solver, env):
    """{action index: exact value} for every valid action, in the environment's action numbering."""
    v, N = env.view(), env.N
    states = [(int(ty), int(d)) for ty, d in zip(v.types, v.days)]
    order = sorted(range(len(states)), key=lambda i: states[i])
    pool, t = tuple(states[i] for i in order), v.t + 1
    vals = {2 * N: solver.Pc(t, pool)}
    mask = env.action_masks()
    for j, slot in enumerate(order):
        if mask[N + slot]:
            vals[N + slot] = solver._offer_value(t, pool, j)
        if mask[slot]:
            vals[slot] = solver._interview_value(t, pool, j)
    return vals


def kind(a, N):
    return "end" if a == 2 * N else ("offer" if a >= N else "interview")


def classify(a, best, N):
    ka, kb = kind(a, N), kind(best, N)
    if ka == kb:
        return "wrong candidate"
    if ka == "offer":
        return "offer too early"
    if kb == "offer":
        return "offer too late"
    return "unneeded interview" if ka == "interview" else "missed interview"


def job(args):
    path, episodes = args
    import torch as th
    th.set_num_threads(1)
    sc = load_scenario("small_exact")
    solver, env, policy = ExactSolver(sc, "automatic"), HiringEnv(sc), PPOPolicy.load(Path(path) / "model.zip")
    N = env.N
    loss = {k: 0.0 for k in TYPES}
    count = {k: 0 for k in TYPES}
    by_day = np.zeros(sc.horizon)
    ret, steps = [], 0
    for seed in episode_seeds("dev", episodes):
        env.reset(seed=seed)
        done = False
        while not done:
            vals = action_values(solver, env)
            a = policy.act(env)
            best = max(vals, key=vals.get)
            gap = vals[best] - vals[a]
            steps += 1
            if gap > TOL:
                k = classify(a, best, N)
                loss[k] += gap
                count[k] += 1
                by_day[env.t] += gap
            _, _, done, _, info = env.step(a)
        ret.append(info["episode_stats"]["ret"])
    return dict(run=Path(path).name, episodes=episodes, steps=steps, mean_return=float(np.mean(ret)),
                loss_per_episode={k: loss[k] / episodes for k in TYPES},
                mistakes_per_100_episodes={k: 100 * count[k] / episodes for k in TYPES},
                loss_by_day=(by_day / episodes).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=OUT / "ppo_small")
    ap.add_argument("--episodes", type=int, default=40_000)
    ap.add_argument("--tag", default="small_ppo")
    a = ap.parse_args()
    optimum = ExactSolver(load_scenario("small_exact"), "automatic").value()
    runs = sorted(str(p.parent) for p in a.runs.glob("*/model.zip"))
    with Pool(min(12, len(runs))) as pool:
        rows = pool.map(job, [(r, a.episodes) for r in runs], chunksize=1)
    lines = [f"Exact optimum {optimum:.5f}. Loss per episode by mistake type, priced by the exact solver "
             f"({a.episodes:,} development episodes, greedy actions).", "",
             f"{'run':20s} " + " ".join(f"{t:>19s}" for t in TYPES) + f" {'total':>8s} {'% of optimum lost':>18s} "
             f"{'optimum - simulated':>20s}"]
    for r in rows:
        total = sum(r["loss_per_episode"].values())
        r["total_loss"], r["percent_lost"] = total, 100 * total / optimum
        lines.append(f"{r['run']:20s} " + " ".join(f"{r['loss_per_episode'][t]:19.4f}" for t in TYPES)
                     + f" {total:8.4f} {100 * total / optimum:17.2f}% {optimum - r['mean_return']:20.4f}")
    lines += ["", "Mistakes per 100 episodes:"]
    for r in rows:
        lines.append(f"{r['run']:20s} " + " ".join(f"{r['mistakes_per_100_episodes'][t]:19.1f}" for t in TYPES))
    lines += ["", "Share of the loss by day of the 12-day window (days 1-4 / 5-8 / 9-12):"]
    for r in rows:
        d = np.array(r["loss_by_day"])
        lines.append(f"{r['run']:20s} " + " / ".join(f"{100 * d[i:i + 4].sum() / max(d.sum(), 1e-12):.0f}%" for i in (0, 4, 8)))
    lines += ["", "total = exact expected shortfall of the agent (no episode luck). 'optimum - simulated' is the same",
              "quantity estimated from realised payoffs; it differs from total only by sampling error (about 0.004)."]
    text = "\n".join(lines)
    print(text)
    (OUT / f"{a.tag}_mistakes.txt").write_text(text + "\n")
    (OUT / f"{a.tag}_mistakes.json").write_text(json.dumps(dict(optimum=optimum, rows=rows), indent=1))


if __name__ == "__main__":
    main()
