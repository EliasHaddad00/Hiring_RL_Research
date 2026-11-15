"""Plateau test from saved models: the model at 80% of training against the final model.

Same pre-registered rule as check_plateau.py (flat if the 95% interval of the change over the last
20% of training lies inside +-1% of the final payoff; otherwise inconclusive or changing), but the
change is measured by scoring model_80.zip and model.zip on 4,000 development episodes (greedy
actions, masks on, paired by episode) instead of the 300 curve episodes. Needs runs made after
Nov 15, which save model_80.zip.

Usage: check_plateau_checkpoints.py results/g2/ppo_pooled_stress_10m [--tag ...] [--episodes 4000]
Writes results/g2/<tag>.{txt,json}.
"""
import argparse
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_plateau import BAND, BOOT, boot_interval, iqm, verdict  # noqa: E402
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.evaluation import PPOPolicy, evaluate  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g2"


def job(spec):
    import torch as th
    th.set_num_threads(1)
    folder, which, episodes = spec
    cfg = json.loads((Path(folder) / "run.json").read_text())["config"]
    sc = load_scenario(cfg["scenario"], **(cfg.get("scenario_overrides") or {}))
    return (folder, which), evaluate(sc, PPOPolicy.load(Path(folder) / which), seeds=episode_seeds("dev", episodes))["ret"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    root = Path(args.runs)
    tag = args.tag or f"plateau_checkpoints_{root.name}"
    folders = sorted(str(p.parent) for p in root.glob("*/model_80.zip") if (p.parent / "run.json").exists())
    if not folders:
        sys.exit(f"no finished run with model_80.zip under {root}")
    specs = [(f, which, args.episodes) for f in folders for which in ("model_80.zip", "model.zip")]
    with Pool(min(args.procs, len(specs))) as pool:
        ret = dict(pool.map(job, specs))

    rng = np.random.default_rng(0)
    rows, by_actor, problems = [], {}, []
    lines = [f"{'run':22s} {'steps':>9s} {'80% at':>9s} {'at 80%':>8s} {'final':>8s} {'change':>8s} {'95% interval':>20s} {'+-1%':>6s} {'verdict':>13s}"]
    for f in folders:
        meta = json.loads((Path(f) / "run.json").read_text())
        a, b = ret[(f, "model_80.zip")], ret[(f, "model.zip")]
        diff = b - a
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        lo, hi = diff.mean() - 1.96 * se, diff.mean() + 1.96 * se
        v = verdict(lo, hi, b.mean())
        rows.append(dict(run=Path(f).name, actor=meta["actor"], seed=meta["seed"], steps=meta["total_timesteps"],
                         step_80=meta["checkpoint_80_step"], at_80=float(a.mean()), final=float(b.mean()),
                         change=float(diff.mean()), interval=[float(lo), float(hi)], band=float(BAND * abs(b.mean())), verdict=v))
        by_actor.setdefault(meta["actor"], []).append((diff, float(b.mean())))
        lines.append(f"{Path(f).name:22s} {meta['total_timesteps']:9d} {meta['checkpoint_80_step']:9d} {a.mean():8.3f} {b.mean():8.3f} "
                     f"{diff.mean():+8.3f} {'[%+.3f, %+.3f]' % (lo, hi):>20s} {BAND * abs(b.mean()):6.3f} {v:>13s}")

    lines += ["", f"{'actor':14s} {'runs':>4s} {'change in IQM':>14s} {'95% interval':>20s} {'+-1%':>6s} {'verdict':>13s}"]
    actors = []
    for actor, pairs in by_actor.items():
        d = np.stack([p[0] for p in pairs])
        change, interval = float(iqm(d.mean(axis=1))), boot_interval(d, rng)
        final = float(iqm(np.array([p[1] for p in pairs])))
        v = verdict(interval[0], interval[1], final)
        actors.append(dict(actor=actor, runs=len(pairs), final=final, band=BAND * abs(final), change=change,
                           interval=[float(x) for x in interval], verdict=v))
        lines.append(f"{actor:14s} {len(pairs):4d} {change:+14.3f} {'[%+.3f, %+.3f]' % tuple(interval):>20s} {BAND * abs(final):6.3f} {v:>13s}")
        if v != "flat":
            problems.append(f"{actor}: plateau test is {v}, not flat")

    lines += ["", f"change = final model minus the model saved at 80% of training, on the same {args.episodes:,} development episodes (paired).",
              "flat: the 95% interval lies inside +-1% of the final payoff. inconclusive: it reaches outside and contains zero. "
              "changing: it excludes zero.",
              f"Per actor: interquartile mean over runs (the plain mean with three runs); bootstrap over runs and episodes, {BOOT:,} draws.",
              "Tuning seeds, development episodes: a check, not a reported result.",
              "", "Problems: " + ("none" if not problems else "")] + [f"  {p}" for p in problems]
    text = "\n".join(lines)
    print(text)
    (OUT / f"{tag}.txt").write_text(text + "\n")
    (OUT / f"{tag}.json").write_text(json.dumps(dict(episodes=args.episodes, runs=rows, actors=actors, problems=problems), indent=1))


if __name__ == "__main__":
    main()
