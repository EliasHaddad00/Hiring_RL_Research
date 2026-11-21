"""Learning curves of the final runs.

Reads only progress.csv of FINISHED runs (folders with run.json), the bound
(results/g2/bounds.json) and, once score_final.py has run, the best rule's payoff on the same
300 curve episodes (rules/<setting>/curve_refs.json). Line = median over training seeds,
band = 25th to 75th percentile. Writes

  paper/figures/curves.pdf         primary setting: payoff of the greedy policy on the 300 curve
                                   episodes, policy loss, value loss, entropy
  paper/figures/curves_all.pdf     supplement: every setting, all six logged quantities
  paper/figures/curves_count.tex   seeds per line and the reference payoffs, used in the caption

The curve episodes are development episodes: the figures are evidence of convergence, not results.
"""
import csv
import json

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

from common import setup
from apply_decision_rule import bound  # noqa: E402
from score_final import ROOT  # noqa: E402

ACTORS = [("index", "Index", "#1b6ca8"), ("unrestricted", "Unrestricted", "#c8553d"), ("pooled", "Pooled", "#3a7d44")]
PANELS = [("eval/mean_reward", "Payoff (greedy, 300 episodes)", 1.0), ("train/policy_gradient_loss", "Policy loss", 1.0),
          ("train/value_loss", "Value loss", 1.0), ("train/entropy_loss", "Entropy", -1.0),
          ("train/approx_kl", "Approximate KL", 1.0), ("train/clip_fraction", "Clip fraction", 1.0)]
GROUPS = [("small_survival", "Small"), ("none_survival", "No pressure"), ("h0_0_survival", "$h_0 = 0$"),
          ("h0_0.02_survival", "$h_0 = 0.02$"), ("h0_0.05_survival", "$h_0 = 0.05$"), ("h0_0.05_naive", "$h_0 = 0.05$, naive"),
          ("h0_0.10_survival", "$h_0 = 0.10$"), ("calibrated_survival", "Calibrated")]


def column(run, name: str):
    steps, values = [], []
    with open(run / "progress.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get(name) not in (None, ""):
                steps.append(float(row["time/total_timesteps"]))
                values.append(float(row[name]))
    return np.array(steps), np.array(values)


def references(root, group: str) -> dict:
    """Yardstick payoffs for the payoff panel: the bound (the exact optimum on the small instance) and the best rule."""
    setting = group.rsplit("_", 1)[0]
    if setting == "small":
        top = json.loads((ROOT / "results" / "g3" / "rules_small.json").read_text())["optimum"]
    else:
        top = bound(setting)
    refs = root / "rules" / setting / "curve_refs.json"
    return dict(top=top, **(json.loads(refs.read_text()) if refs.exists() else {}))


def draw(axes, root, group: str, panels) -> list:
    """One row of panels for one group of runs; returns the seed count per actor."""
    counts = []
    for actor, label, colour in ACTORS:
        runs = sorted(p.parent for p in (root / group).glob(f"{actor}_s*/run.json"))
        counts.append(f"{label.lower()} {len(runs)}")
        for ax, (name, _, sign) in zip(axes, panels):
            series = [column(r, name) for r in runs]
            if not runs or min(len(v) for _, v in series) == 0:
                continue
            n = min(len(v) for _, v in series)
            lo, mid, hi = np.percentile(sign * np.array([v[:n] for _, v in series]), [25, 50, 75], axis=0)
            ax.plot(series[0][0][:n] / 1e6, mid, color=colour, lw=0.9, label=label)
            ax.fill_between(series[0][0][:n] / 1e6, lo, hi, color=colour, alpha=0.2, lw=0)
    ref = references(root, group)
    axes[0].axhline(ref["top"], color="0.3", lw=0.7, ls="--")
    if "best_rule" in ref:
        axes[0].axhline(ref["best_rule"], color="0.3", lw=0.7, ls=":")
        axes[0].set_ylim(0.75 * ref["best_rule"], 1.06 * ref["top"])
    for ax in axes:
        ax.tick_params(which="both", labelsize=6, length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return counts


def main():
    root, out, stand_in = setup(__doc__)
    mark = "STAND-IN, NOT RESULTS. " if stand_in else ""

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.75))
    counts = draw(axes, root, "h0_0.05_survival", PANELS[:4])
    ref = references(root, "h0_0.05_survival")
    axes[0].text(0.98, ref["top"], "upper bound", ha="right", va="bottom", fontsize=5.5, transform=axes[0].get_yaxis_transform())
    if "best_rule" in ref:
        axes[0].text(0.98, ref["best_rule"], "best rule", ha="right", va="top", fontsize=5.5, transform=axes[0].get_yaxis_transform())
    else:
        axes[0].set_ylim(2.6, ref["top"] + 0.25)
    axes[2].set_yscale("log")
    for ax, (_, title, _) in zip(axes, PANELS):
        ax.set_title(mark + title if ax is axes[0] else title, fontsize=7)
        ax.set_xlabel("Training steps (millions)", fontsize=6.5)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=5.5, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(out / "figures" / "curves.pdf")

    groups = [(g, label) for g, label in GROUPS if any((root / g).glob("*/run.json"))] or [GROUPS[4]]
    fig, grid = plt.subplots(len(groups), 6, figsize=(7.0, 1.15 * len(groups) + 0.3), squeeze=False)
    for row, (group, label) in zip(grid, groups):
        draw(row, root, group, PANELS)
        row[2].set_yscale("log")
        row[0].set_ylabel(label, fontsize=6.5)
    for ax, (_, title, _) in zip(grid[0], PANELS):
        ax.set_title(mark + "Payoff" if ax is grid[0][0] else title, fontsize=7)
    for ax in grid[-1]:
        ax.set_xlabel("Steps (millions)", fontsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.5)
    fig.savefig(out / "figures" / "curves_all.pdf")

    number = lambda key: f"{ref[key]:.2f}" if key in ref else "--"
    (out / "figures" / "curves_count.tex").write_text(
        "% Written by experiments/analysis/fig_curves.py. Do not edit.\n"
        r"\newcommand{\curveseeds}{" + ", ".join(counts) + "}\n"
        r"\newcommand{\curverule}{" + number("best_rule") + "}\n"
        r"\newcommand{\curverandom}{" + number("random") + "}\n")
    print("seeds per line:", ", ".join(counts), "| settings in curves_all.pdf:", len(groups))


if __name__ == "__main__":
    main()
