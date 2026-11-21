"""Failure-mode table of the paper: where the agents' actions differ from the reference, by type.

Reads only mistakes.json files written by score_final.py. Left block, primary setting: how often
per 100 episodes an agent's action differs from what the best tuned rule would do in the same
state. Right block, small instance: the exact cost of each type of mistake (from the exact
solver's action values), as a percentage of the optimum. Each cell is the mean over training
seeds. A cell without scores shows "--". Writes paper/tables/mistakes.tex.
"""
import json

import numpy as np

from common import ACTORS, PENDING, setup, write

TYPES = [("offer too early", "Offer too early"), ("offer too late", "Offer too late"), ("wrong candidate", "Other candidate"),
         ("unneeded interview", "Extra interview"), ("missed interview", "Missed interview")]


def load(folder, actor):
    return [json.loads(p.read_text()) for p in sorted(folder.glob(f"{actor}_s*/mistakes.json"))]


def main():
    root, out, stand_in = setup(__doc__)
    exact = sorted(root.glob("small_survival/*/exact.json"))
    optimum = json.loads(exact[0].read_text())["optimum"] if exact else None
    lines = [r"\begin{tabular}{@{}l@{\hskip 5pt}c@{\hskip 4pt}c@{\hskip 4pt}c@{\hskip 8pt}c@{\hskip 4pt}c@{\hskip 4pt}c@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{vs.\ rule, per 100 ep.} & \multicolumn{3}{c}{Small: \% of opt.} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             "Type & " + " & ".join(["Ind.", "Unr.", "Pool."] * 2) + r" \\", r"\midrule"]
    for key, label in TYPES:
        cells = []
        for actor, _ in ACTORS:
            d = load(root / "h0_0.05_survival", actor)
            cells.append(f"{np.mean([x['per_100_episodes'][key] for x in d]):.0f}" if d else PENDING)
        for actor, _ in ACTORS:
            d = load(root / "small_survival", actor)
            cells.append(f"{100 * np.mean([x['loss_per_episode'][key] for x in d]) / optimum:.2f}" if d and optimum else PENDING)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tables" / "mistakes.tex", "table_mistakes.py", lines, stand_in)


if __name__ == "__main__":
    main()
