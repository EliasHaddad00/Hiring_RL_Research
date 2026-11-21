"""Median search score by hyperparameter value, per actor (descriptive check for the widening decision).

Reads results/g3/ppo_tuning_search.json (40 configurations per actor, 30% length, mean over seeds
101 to 103 on the tuning episodes) and results/g3/ppo_tuning_finals.json.
Writes results/g3/search_medians.txt. Not a controlled test: each configuration changes every
hyperparameter at once, so a value's median also reflects what it happened to be drawn with.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.tuning_ppo import CHOICES  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g3"
ACTORS = ("unrestricted", "index", "pooled")


def main():
    rows = json.loads((OUT / "ppo_tuning_search.json").read_text())
    lines = ["Median search score by hyperparameter value (40 configurations per actor, 30% length, tuning episodes).",
             "Descriptive only: configurations differ in every hyperparameter at once. n = configurations with that value.", ""]
    for k, values in CHOICES.items():
        lines.append(f"{k}")
        lines.append(f"  {'value':>7s} {'n':>3s} " + " ".join(f"{a:>13s}" for a in ACTORS))
        for v in values:
            cells, n = [], 0
            for a in ACTORS:
                s = [r["mean"] for r in rows if r["actor"] == a and r["params"][k] == v]
                n = len(s)
                cells.append(f"{np.median(s):13.3f}" if s else f"{'-':>13s}")
            lines.append(f"  {v:>7g} {n:3d} " + " ".join(cells))
        lines.append("")
    lines.append("Full-length finalists (mean over seeds 101 to 103): gap of each to the best of its actor")
    finals = json.loads((OUT / "ppo_tuning_finals.json").read_text())
    for a in ACTORS:
        f = sorted((r for r in finals if r["actor"] == a), key=lambda r: r["rank"])
        lines.append(f"  {a:13s} " + "  ".join(f"c{r['configuration']:02d} {r['mean']:.4f} ({r['mean'] - f[0]['mean']:+.4f})" for r in f))
        sd = np.mean([np.std(list(r["by_seed"].values()), ddof=1) for r in f])
        lines.append(f"  {'':13s} mean standard deviation over the three seeds within a configuration: {sd:.4f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "search_medians.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
