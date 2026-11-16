"""Doubling test at h0 = 0.05: does training for 20 million steps instead of 10 million add anything?

Pre-registered Nov 15 as the plateau evidence for the unrestricted and pooled actors: if the 95%
interval of the gain from 10 to 20 million steps lies inside +-1% of the payoff, 10 million stand.
Unlike the 80%-model test, this is not confounded by the learning rate falling to zero, since both
models are at the end of their own schedule.

10 million: results/g2/ppo_stress_10m (unrestricted), results/g2/ppo_pooled_stress_10m (pooled).
20 million: results/g2/ppo_stress_20m. Same seed in both; separate runs, not continuations.
Paired on 4,000 development episodes, greedy actions, masks on.
Writes results/g2/doubling.{txt,json}.
"""
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_plateau import BAND, BOOT, boot_interval, verdict  # noqa: E402
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.evaluation import PPOPolicy, evaluate  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g2"
N = 4000
SHORT = {"unrestricted": "ppo_stress_10m", "pooled": "ppo_pooled_stress_10m"}
LONG = "ppo_stress_20m"
SEEDS = (101, 102, 103)


def job(path):
    import torch as th
    th.set_num_threads(1)
    sc = load_scenario("stress", h0=0.05, accept_decay=0.03)
    return path, evaluate(sc, PPOPolicy.load(Path(path) / "model.zip"), seeds=episode_seeds("dev", N))["ret"]


def main():
    pairs = {(actor, seed): (str(OUT / SHORT[actor] / f"{actor}_s{seed}"), str(OUT / LONG / f"{actor}_s{seed}"))
             for actor in SHORT for seed in SEEDS}
    pairs = {k: v for k, v in pairs.items() if all((Path(p) / "run.json").exists() for p in v)}
    paths = sorted({p for v in pairs.values() for p in v})
    with Pool(min(12, len(paths))) as pool:
        ret = dict(pool.map(job, paths))
    rng = np.random.default_rng(0)
    lines = [f"{'run':20s} {'10M':>8s} {'20M':>8s} {'gain':>8s} {'95% interval':>20s}"]
    rows, actors = [], []
    for actor in SHORT:
        diffs, finals = [], []
        for seed in SEEDS:
            if (actor, seed) not in pairs:
                continue
            a, b = (ret[p] for p in pairs[(actor, seed)])
            for p, want in zip(pairs[(actor, seed)], (10e6, 20e6)):
                steps = json.loads((Path(p) / "run.json").read_text())["total_timesteps"]
                assert abs(steps - want) < 1e5, (p, steps)
            d = b - a
            se = d.std(ddof=1) / np.sqrt(N)
            rows.append(dict(actor=actor, seed=seed, at_10m=float(a.mean()), at_20m=float(b.mean()), gain=float(d.mean()),
                             interval=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]))
            lines.append(f"{actor + '_s' + str(seed):20s} {a.mean():8.3f} {b.mean():8.3f} {d.mean():+8.3f} "
                         f"{'[%+.3f, %+.3f]' % tuple(rows[-1]['interval']):>20s}")
            diffs.append(d)
            finals.append(b.mean())
        if not diffs:
            continue
        d = np.stack(diffs)
        lo, hi = boot_interval(d, rng)
        final = float(np.mean(finals))
        v = verdict(lo, hi, final)
        actors.append(dict(actor=actor, runs=len(diffs), gain=float(d.mean()), interval=[float(lo), float(hi)],
                           band=BAND * final, verdict=v))
        lines += [f"{actor}: mean gain {d.mean():+.3f}, 95% interval [{lo:+.3f}, {hi:+.3f}], band +-{BAND * final:.3f}: {v}"
                  + (" (10 million steps stand)" if v == "flat" else " (10 million steps are NOT shown to be enough)"), ""]
    lines += [f"Paired on {N:,} development episodes; interval: bootstrap over runs and episodes, {BOOT:,} draws.",
              "flat: the interval lies inside +-1% of the 20-million-step payoff.",
              "Tuning seeds 101 to 103: a check, not a reported result."]
    text = "\n".join(lines)
    print(text)
    (OUT / "doubling.txt").write_text(text + "\n")
    (OUT / "doubling.json").write_text(json.dumps(dict(episodes=N, runs=rows, actors=actors), indent=1))


if __name__ == "__main__":
    main()
