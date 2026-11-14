"""G2 check: PPO against the exact optimum on the small instance.

Every saved model under --runs and the exact optimal policy play the same 40,000 development
episodes, masks on. Pass rule (decided Nov 14): for at least one agent, the lower end of the 95%
interval of (PPO minus exact policy), with greedy actions, is above -2% of 0.63333 = -0.0127.
The score with sampled actions is reported beside it and is not used for the verdict.

Writes results/g2/small_ppo.{txt,json} and per-episode returns to results/g2/small_ppo_episodes.npz.
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
from hiring.evaluation import PPOPolicy, evaluate, paired_compare  # noqa: E402
from hiring.exact_dp import ExactPolicy, ExactSolver  # noqa: E402
from hiring.rules import RandomPolicy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "g2"
EPISODES = 40_000
MARGIN = 0.02


def job(args):
    name, kind, path, episodes = args
    import torch as th
    th.set_num_threads(1)
    sc = load_scenario("small_exact")
    if kind == "exact":
        policy = ExactPolicy(ExactSolver(sc, "automatic"))
    elif kind == "random":
        policy = RandomPolicy()
    else:
        policy = PPOPolicy.load(path, deterministic=(kind == "greedy"))
    return (name, kind), evaluate(sc, policy, seeds=episode_seeds("dev", episodes))["ret"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=OUT / "ppo_small")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--tag", default="small_ppo")
    a = ap.parse_args()
    sc = load_scenario("small_exact")
    optimum = ExactSolver(sc, "automatic").value()
    limit = -MARGIN * optimum
    runs = sorted(p.parent for p in a.runs.glob("*/model.zip"))
    assert runs, f"no saved models under {a.runs}"
    jobs = [("exact", "exact", None, a.episodes), ("random", "random", None, a.episodes)]
    for r in runs:
        jobs += [(r.name, "greedy", str(r / "model.zip"), a.episodes), (r.name, "sampled", str(r / "model.zip"), a.episodes)]
    with Pool(min(a.procs, len(jobs))) as pool:
        raw = dict(pool.map(job, jobs))

    exact = raw[("exact", "exact")]
    rows, passed = [], []
    lines = [f"Exact optimum (computed, automatic drop): {optimum:.5f}. Pass limit on the paired difference: {limit:+.4f}.",
             f"Exact optimal policy on these {a.episodes} development episodes: {exact.mean():.4f} "
             f"+- {exact.std(ddof=1) / np.sqrt(len(exact)):.4f}; random policy: {raw[('random', 'random')].mean():.4f}.", "",
             f"{'run':20s} {'steps':>9s} {'greedy mean':>12s} {'minus exact':>12s} {'95% interval':>20s} "
             f"{'% of optimum':>13s} {'verdict':>8s} {'sampled mean':>13s} {'sampled minus exact':>20s}"]
    for r in runs:
        meta = json.loads((r / "run.json").read_text())
        g, s = raw[(r.name, "greedy")], raw[(r.name, "sampled")]
        cg, cs = paired_compare(g, exact, MARGIN), paired_compare(s, exact, MARGIN)
        ok = cg["lo"] > limit
        if ok:
            passed.append(r.name)
        rows.append(dict(run=r.name, actor=meta["actor"], seed=meta["seed"], steps=meta["total_timesteps"],
                         greedy=cg, sampled=cs, passes=bool(ok),
                         percent_of_optimum_paired=100 * (1 + cg["diff"] / optimum)))
        lines.append(f"{r.name:20s} {meta['total_timesteps']:9d} {cg['mean_a']:12.4f} {cg['diff']:+12.4f} "
                     f"{'[%+.4f, %+.4f]' % (cg['lo'], cg['hi']):>20s} {100 * (1 + cg['diff'] / optimum):12.1f}% "
                     f"{'PASS' if ok else 'fail':>8s} {cs['mean_a']:13.4f} {cs['diff']:+20.4f}")
    lines += ["", f"Agents passing: {len(passed)} of {len(runs)}" + (f" ({', '.join(passed)})" if passed else "") + ".",
              "% of optimum = 1 + (paired difference / computed optimum). Training seeds 101 to 103 (tuning seeds);",
              "scores are on development episodes and are a gate check, not a reported result."]
    text = "\n".join(lines)
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.txt").write_text(text + "\n")
    (OUT / f"{a.tag}.json").write_text(json.dumps(dict(optimum=optimum, limit=limit, episodes=a.episodes,
                                                         exact_mean=float(exact.mean()), rows=rows, passed=passed), indent=1))
    np.savez_compressed(OUT / f"{a.tag}_episodes.npz", **{f"{n}__{k}": v for (n, k), v in raw.items()})
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
