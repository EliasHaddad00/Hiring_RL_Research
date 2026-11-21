"""Main table of the paper: best tuned rule, the three agents and the yardstick at every setting.

Reads only files written by score_final.py (eval.npz, exact.json, rules/) and earlier gates.
An agent's cell is the interquartile mean over training seeds of each seed's mean payoff, with
its 95% bootstrap interval over seeds as sub- and superscript. On the small instance every number
is exact (no sampling) and the last column is the exact optimum. A cell without scores shows "--".
Writes paper/tables/main.tex.
"""
import json

import numpy as np

from common import ACTORS, PENDING, ROOT, setup, write
from apply_decision_rule import Scores, bound  # noqa: E402
from hiring.decision import boot_stat  # noqa: E402
from score_final import SETTINGS, best_rule  # noqa: E402

LABELS = {"small": "Small (exact)", "none": "No pressure", "h0_0": "$h_0 = 0$", "h0_0.02": "$h_0 = 0.02$",
          "h0_0.05": "$h_0 = 0.05$", "h0_0.10": "$h_0 = 0.10$", "calibrated": "Calibrated"}


def cell(values, digits):
    s = boot_stat(values, "iqm")
    return f"${s['value']:.{digits}f}_{{{s['lo']:.{digits}f}}}^{{{s['hi']:.{digits}f}}}$"


def main():
    root, out, stand_in = setup(__doc__)
    S = Scores(root)
    lines = [r"\begin{tabular}{@{}l@{\hskip 5pt}c@{\hskip 5pt}c@{\hskip 5pt}c@{\hskip 5pt}c@{\hskip 5pt}c@{}}", r"\toprule",
             "Setting & Rule & " + " & ".join(label for _, label in ACTORS) + r" & Bound \\", r"\midrule"]
    for s in SETTINGS:
        small = s == "small"
        digits = 3 if small else 2
        family, belief, _ = best_rule(s)
        runs = S.runs(f"{s}_survival")
        if small:
            d = json.loads((ROOT / "results" / "g3" / "rules_small.json").read_text())
            rule = next(f"{r['exact_value']:.3f}" for r in d["rows"] if (r["family"], r["belief"]) == (family, belief))
            last = f"{d['optimum']:.3f}"
            scores = {a: [json.loads((f / "exact.json").read_text())["exact_value"] for _, f in v if (f / "exact.json").exists()]
                      for a, v in runs.items()}
        else:
            path = root / "rules" / s / f"{family}__{belief}.npz"
            rule = f"{S.rule(s, path.stem).mean():.2f}" if path.exists() else PENDING
            last = f"{bound(s):.2f}"
            scores = {a: [np.load(f / "eval.npz")["ret"].mean() for _, f in v if (f / "eval.npz").exists()] for a, v in runs.items()}
        cells = [cell(scores[a], digits) if len(scores.get(a, [])) >= 2 else PENDING for a, _ in ACTORS]
        lines.append(f"{LABELS[s]} & {rule} & " + " & ".join(cells) + f" & {last} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tables" / "main.tex", "table_main.py", lines, stand_in)


if __name__ == "__main__":
    main()
