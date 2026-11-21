"""Estimate table of the paper (hypothesis H4a): payoff of each rule family with the survival-aware
estimate minus the same family with the naive estimate, each tuned separately.

Reads only saved results. Once score_final.py has run, the difference is the paired mean on the
evaluation episodes (results/final/rules/). Until then it is the difference of the two scores on
the 4,000 TUNING episodes (results/g2/tuning/40): in-sample and unpaired, marked provisional in
the paper. Writes paper/tables/belief.tex.
"""
import numpy as np

from common import setup, write
from score_final import tuned_rules  # noqa: E402
from table_rules import evaluated  # noqa: E402

SETTINGS = [("h0_0", "$0$"), ("h0_0.02", "$0.02$"), ("h0_0.05", "$0.05$"), ("h0_0.10", "$0.10$"), ("calibrated", "Calib.")]
FAMILIES = [("static_threshold", "Static bar"), ("resume_falling_bar", "Falling bar"), ("one_interview", "One interview"),
            ("up_to_two_interviews", "Two interviews"), ("price_directed", "Price-directed")]


def main():
    root, out, stand_in = setup(__doc__)
    final = evaluated(root)

    def gap(setting, family):
        if final:
            ret = lambda e: np.load(root / "rules" / setting / f"{family}__{e}.npz")["ret"]
            return float((ret("survival") - ret("naive")).mean())
        rules = tuned_rules(setting)
        return rules[(family, "survival")][0] - rules[(family, "naive")][0]

    lines = [r"\begin{tabular}{@{}l@{\hskip 5pt}c@{\hskip 5pt}c@{\hskip 5pt}c@{\hskip 5pt}c@{\hskip 5pt}c@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{Stress, $h_0$} & \\", r"\cmidrule(lr){2-5}",
             "Rule family & " + " & ".join(label for _, label in SETTINGS) + r" \\", r"\midrule"]
    for fam, label in FAMILIES:
        lines.append(label + " & " + " & ".join(f"${gap(s, fam):+.3f}$" for s, _ in SETTINGS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tables" / "belief.tex", "table_belief.py", lines, stand_in)


if __name__ == "__main__":
    main()
