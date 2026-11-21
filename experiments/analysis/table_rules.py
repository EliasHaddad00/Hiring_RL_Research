"""Rule table of the paper (research question 1): upper bound and best tuned rule of each family.

Reads only saved results: results/g2/bounds.json (the bound; no episodes involved), the tuned
rules (results/g2/tuning/40) and, once score_final.py has run, the rules' payoffs on the
evaluation episodes (results/final/rules/). Until then the rule scores are those on the 4,000
TUNING episodes the rules were fitted on: in-sample, marked provisional in the paper. The
estimate shown for a family and the rule in bold are always chosen on the tuning episodes.
Writes paper/tables/rules.tex and paper/tables/prov.tex (the provisional mark, switched off
once every rule has its evaluation score).
"""
import numpy as np

from common import setup, write
from apply_decision_rule import bound  # noqa: E402
from score_final import FULL_SETTINGS, best_rule, tuned_rules  # noqa: E402

LABELS = {"none": "No press.", "h0_0": "$h_0{=}0$", "h0_0.02": "$h_0{=}0.02$", "h0_0.05": "$h_0{=}0.05$",
          "h0_0.10": "$h_0{=}0.10$", "calibrated": "Calibrated"}
FAMILIES = ["static_threshold", "resume_falling_bar", "one_interview", "up_to_two_interviews", "price_directed"]


def evaluated(root) -> bool:
    return all((root / "rules" / s / f"{f}__{b}.npz").exists() for s in FULL_SETTINGS for f, b in tuned_rules(s))


def main():
    root, out, stand_in = setup(__doc__)
    final = evaluated(root)
    lines = [r"\begin{tabular}{@{}l@{\hskip 4pt}c@{\hskip 4pt}c@{\hskip 4pt}c@{\hskip 4pt}c@{\hskip 4pt}c@{\hskip 4pt}c@{\hskip 4pt}c@{}}", r"\toprule",
             r"Setting & Bound & Static & Falling & 1 int. & 2 int. & Price & \% \\", r"\midrule"]
    for s in FULL_SETTINGS:
        rules, b, top = tuned_rules(s), bound(s), best_rule(s)[0]
        scores = {}
        for f in FAMILIES:
            belief = max(("naive", "survival"), key=lambda e: rules[(f, e)][0])        # the better estimate, on the tuning episodes
            scores[f] = float(np.load(root / "rules" / s / f"{f}__{belief}.npz")["ret"].mean()) if final else rules[(f, belief)][0]
        cells = [(r"\textbf{%.2f}" if f == top else "%.2f") % scores[f] for f in FAMILIES]
        lines.append(f"{LABELS[s]} & {b:.2f} & " + " & ".join(cells) + f" & {100 * scores[top] / b:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tables" / "rules.tex", "table_rules.py", lines, stand_in)
    write(out / "tables" / "prov.tex", "table_rules.py",
          [r"\renewcommand{\prov}{}% rule scores are on the evaluation episodes" if final
           else r"% rule scores are still on the tuning episodes: the provisional mark stays"], stand_in)


if __name__ == "__main__":
    main()
