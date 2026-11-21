"""Figure 1 of the paper: learning curves of the final runs at the primary setting.

Reads only progress.csv of FINISHED runs (folders with run.json) under
results/final/h0_0.05_survival/ and the bound from results/g2/bounds.json. Four panels: payoff of
the greedy policy on the 300 curve episodes, policy loss, value loss, entropy. Line = median over
seeds, band = 25th to 75th percentile. Writes paper/figures/curves.pdf and curves_count.tex (how
many seeds each line has, used in the caption).

The curve episodes are development episodes: the figure is evidence of convergence, not a result.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "final" / "h0_0.05_survival"
ACTORS = [("index", "Index", "#1b6ca8"), ("unrestricted", "Unrestricted", "#c8553d"), ("pooled", "Pooled", "#3a7d44")]
PANELS = [("eval/mean_reward", "Payoff (greedy, 300 episodes)", 1.0), ("train/policy_gradient_loss", "Policy loss", 1.0),
          ("train/value_loss", "Value loss", 1.0), ("train/entropy_loss", "Entropy", -1.0)]


def column(run: Path, name: str):
    steps, values = [], []
    with open(run / "progress.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get(name) not in (None, ""):
                steps.append(float(row["time/total_timesteps"]))
                values.append(float(row[name]))
    return np.array(steps), np.array(values)


def main():
    bound = next(s["bound"] for s in json.loads((ROOT / "results/g2/bounds.json").read_text())["settings"] if s["h0"] == 0.05)
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.75))
    counts = []
    for actor, label, colour in ACTORS:
        runs = sorted(p.parent for p in RUNS.glob(f"{actor}_s*/run.json"))
        counts.append(f"{label.lower()} {len(runs)}")
        if not runs:
            continue
        for ax, (name, _, sign) in zip(axes, PANELS):
            series = [column(r, name) for r in runs]
            n = min(len(v) for _, v in series)
            x = series[0][0][:n] / 1e6
            y = sign * np.array([v[:n] for _, v in series])
            lo, mid, hi = np.percentile(y, [25, 50, 75], axis=0)
            ax.plot(x, mid, color=colour, lw=0.9, label=label)
            ax.fill_between(x, lo, hi, color=colour, alpha=0.2, lw=0)
    axes[0].axhline(bound, color="0.3", lw=0.7, ls="--")
    axes[0].text(0.98, bound, "upper bound", ha="right", va="bottom", fontsize=5.5, transform=axes[0].get_yaxis_transform())
    axes[0].set_ylim(2.6, bound + 0.25)
    axes[2].set_yscale("log")
    for ax, (_, title, _) in zip(axes, PANELS):
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("Training steps (millions)", fontsize=6.5)
        ax.tick_params(labelsize=6, length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=5.5, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    out = ROOT / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "curves.pdf")
    (out / "curves_count.tex").write_text("% Written by experiments/analysis/fig_curves.py. Do not edit.\n"
                                          r"\newcommand{\curveseeds}{" + ", ".join(counts) + "}\n")
    print("seeds per line:", ", ".join(counts))


if __name__ == "__main__":
    main()
