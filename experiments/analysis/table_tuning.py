"""Tuning table of the paper: the hyperparameter search of each actor.

Reads only results/g3/ppo_tuning_search.json (40 configurations at 30% of the final length),
results/g3/ppo_tuning_finals.json (the five best at full length) and results/g2/tuning_40.json
(the best rule). All scores are means over tuning seeds 101 to 103 on the 4,000 tuning episodes:
sensitivity evidence, not results. Writes paper/tables/tuning.tex.
"""
import json

import numpy as np

from common import ACTORS, ROOT, setup, write


def main():
    _, out, stand_in = setup(__doc__)
    load = lambda name: json.loads((ROOT / "results" / name).read_text())
    search, finals = load("g3/ppo_tuning_search.json"), load("g3/ppo_tuning_finals.json")
    rule = max(r["best"] for r in load("g2/tuning_40.json") if r["setting"] == "h0_0.05")
    lines = [r"\begin{tabular}{@{}l@{\hskip 5pt}c@{\hskip 5pt}c@{\hskip 5pt}c@{\hskip 5pt}c@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{40 config., 30\% length} & \multicolumn{2}{c}{5 best, full length} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"Actor & Worst/median/best & Start & Range & Chosen \\", r"\midrule"]
    for actor, label in ACTORS:
        s = np.array([r["mean"] for r in search if r["actor"] == actor])
        start = next(r["mean"] for r in search if r["actor"] == actor and r["configuration"] == 1)
        f = [r for r in finals if r["actor"] == actor]
        top = min(f, key=lambda r: r["rank"])
        lines.append(f"{label} & {s.min():.2f}/{np.median(s):.2f}/{s.max():.2f} & {start:.2f} & "
                     f"{min(r['mean'] for r in f):.2f}--{max(r['mean'] for r in f):.2f} & {top['mean']:.3f} \\\\")
    lines += [r"\midrule", rf"Best rule (40 vectors) & \multicolumn{{4}}{{c}}{{{rule:.3f}}} \\", r"\bottomrule", r"\end{tabular}"]
    write(out / "tables" / "tuning.tex", "table_tuning.py", lines, stand_in)


if __name__ == "__main__":
    main()
