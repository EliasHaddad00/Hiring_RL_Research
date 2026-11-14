"""Plateau test on saved learning curves, plus the list of logged columns. Reads only saved files.

Pre-registered rule: "the change in interquartile mean over the last 20% of training lies within
its interval". Applied here as follows (the wording leaves these details open; they are printed
with the result):

  - The curve is the greedy evaluation on the fixed development episodes (evaluations.npz). Every
    evaluation of a run plays the same set of episodes, each time starting one episode later; the
    script rotates them back, checks the alignment by correlation, and pairs changes by episode.
  - change = score at the last evaluation minus score at the evaluation at 80% of training.
  - Per actor: the interquartile mean over runs of each run's mean score (with three runs this is
    the plain mean), and a 95% bootstrap interval that resamples runs and episodes.
  - Flat = the interval contains zero. Per-run paired changes are listed beside it, and so is the
    same test on wider windows (last three evaluations against the three ending at 80%), which
    is less noisy.

Usage: check_plateau.py results/g2/ppo_stress [--tag plateau_stress]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.training import REQUIRED_COLUMNS  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g2"
BOOT = 10_000


def iqm(x):
    x = np.sort(np.asarray(x, float), axis=0)
    n = len(x)
    cut = int(np.floor(0.25 * n))
    return x[cut:n - cut].mean(axis=0)


def boot_interval(diffs, rng):
    """diffs: runs x episodes array of paired per-episode changes. Resample runs, then episodes."""
    runs, eps = diffs.shape
    out = np.empty(BOOT)
    for b in range(BOOT):
        r = rng.integers(0, runs, runs)
        e = rng.integers(0, eps, eps)
        out[b] = iqm(diffs[r][:, e].mean(axis=1))
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    root = Path(args.runs)
    tag = args.tag or f"plateau_{root.name}"
    rng = np.random.default_rng(0)
    by_actor, lines, rows, problems = {}, [], [], []

    lines.append(f"{'run':22s} {'steps':>9s} {'at 80%':>8s} {'final':>8s} {'change':>8s} {'95% interval':>20s} {'flat':>5s}  columns")
    for d in sorted(p.parent for p in root.glob("*/evaluations.npz")):
        meta_path = d / "run.json"
        if not meta_path.exists():
            lines.append(f"{d.name:22s} still running, skipped")
            continue
        meta = json.loads(meta_path.read_text())
        z = np.load(d / "evaluations.npz")
        steps, res = z["timesteps"], z["results"]
        # Each evaluation starts one episode later in the cycled list (the vectorised environment
        # resets once more after the last episode), so evaluation k is rotated back by k places.
        res = np.stack([np.roll(r, k) for k, r in enumerate(res)])
        i80 = int(np.argmin(np.abs(steps - 0.8 * steps[-1])))
        pairing = float(np.corrcoef(res[i80], res[-1])[0, 1])
        if pairing < 0.5:
            problems.append(f"{d.name}: evaluations do not line up by episode (correlation {pairing:.2f})")
        diff = res[-1] - res[i80]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        lo, hi = diff.mean() - 1.96 * se, diff.mean() + 1.96 * se
        wide = res[-3:].mean(axis=0) - res[i80 - 2:i80 + 1].mean(axis=0)
        with open(d / "progress.csv", newline="") as f:
            header = next(csv.reader(f))
            filled = {c: 0 for c in REQUIRED_COLUMNS}
            for row in csv.DictReader(f, fieldnames=header):
                for c in REQUIRED_COLUMNS:
                    filled[c] += bool(row.get(c))
        missing = [c for c in REQUIRED_COLUMNS if c not in header or filled[c] == 0]
        if missing:
            problems.append(f"{d.name}: columns missing or empty: {missing}")
        flat = bool(lo <= 0 <= hi)
        rows.append(dict(run=d.name, actor=meta["actor"], seed=meta["seed"], steps=int(steps[-1]),
                         step_80=int(steps[i80]), at_80=float(res[i80].mean()), final=float(res[-1].mean()),
                         change=float(diff.mean()), interval=[float(lo), float(hi)], flat=flat,
                         pairing_correlation=pairing,
                         missing_columns=missing, curve=[float(v) for v in res.mean(axis=1)],
                         curve_steps=[int(s) for s in steps]))
        by_actor.setdefault(meta["actor"], []).append((diff, wide))
        lines.append(f"{d.name:22s} {int(steps[-1]):9d} {res[i80].mean():8.3f} {res[-1].mean():8.3f} {diff.mean():+8.3f} "
                     f"{'[%+.3f, %+.3f]' % (lo, hi):>20s} {'yes' if flat else 'no':>5s}  "
                     f"{'all present' if not missing else 'MISSING ' + ', '.join(missing)}")

    lines += ["", f"{'actor':14s} {'runs':>4s} {'change in IQM':>14s} {'95% interval':>20s} {'flat':>5s}   "
                  f"{'three-evaluation windows':>25s} {'95% interval':>20s} {'flat':>5s}"]
    actors = []
    for actor, pairs in by_actor.items():
        d1 = np.stack([p[0] for p in pairs])
        d3 = np.stack([p[1] for p in pairs])
        c1, c3 = float(iqm(d1.mean(axis=1))), float(iqm(d3.mean(axis=1)))
        i1, i3 = boot_interval(d1, rng), boot_interval(d3, rng)
        f1, f3 = bool(i1[0] <= 0 <= i1[1]), bool(i3[0] <= 0 <= i3[1])
        actors.append(dict(actor=actor, runs=len(pairs), change=c1, interval=[float(v) for v in i1], flat=f1,
                           change_windows=c3, interval_windows=[float(v) for v in i3], flat_windows=f3))
        lines.append(f"{actor:14s} {len(pairs):4d} {c1:+14.3f} {'[%+.3f, %+.3f]' % tuple(i1):>20s} {'yes' if f1 else 'no':>5s}   "
                     f"{c3:+25.3f} {'[%+.3f, %+.3f]' % tuple(i3):>20s} {'yes' if f3 else 'no':>5s}")
        if not f1:
            problems.append(f"{actor}: curve not flat over the last 20% of training")

    lines += ["", "change = last evaluation minus the evaluation at 80% of training, same development episodes (paired).",
              "Per actor: interquartile mean over runs (the plain mean with three runs); bootstrap over runs and episodes, "
              f"{BOOT:,} draws.",
              "Columns checked: " + ", ".join(REQUIRED_COLUMNS) + ".",
              "Training seeds 101 to 103, development episodes: a gate check, not a reported result.",
              "", "Problems: " + ("none" if not problems else "")] + [f"  {p}" for p in problems]
    text = "\n".join(lines)
    print(text)
    (OUT / f"{tag}.txt").write_text(text + "\n")
    (OUT / f"{tag}.json").write_text(json.dumps(dict(runs=rows, actors=actors, problems=problems), indent=1))


if __name__ == "__main__":
    main()
