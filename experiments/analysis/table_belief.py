"""Table 3 of the paper (hypothesis H4a): payoff of each rule family with the survival-aware
estimate minus the same family with the naive estimate, each tuned separately.

Reads only results/g2/tuning_40.json (4,000 TUNING episodes, in-sample; no paired interval is
saved there). Marked provisional in the paper; the paired evaluation-pool values replace it.
Writes paper/tables/belief.tex.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = [("h0_0", "$0$"), ("h0_0.02", "$0.02$"), ("h0_0.05", "$0.05$"), ("h0_0.10", "$0.10$"), ("calibrated", "Calib.")]
FAMILIES = [("static_threshold", "Static bar"), ("resume_falling_bar", "Falling bar"), ("one_interview", "One interview"),
            ("up_to_two_interviews", "Two interviews"), ("price_directed", "Price-directed")]


def main():
    score = {(r["setting"], r["family"], r["belief"]): r["best"] for r in json.loads((ROOT / "results/g2/tuning_40.json").read_text())}
    lines = [r"\begin{tabular}{@{}l@{\hskip 5pt}c@{\hskip 5pt}c@{\hskip 5pt}c@{\hskip 5pt}c@{\hskip 5pt}c@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{Stress, $h_0$} & \\", r"\cmidrule(lr){2-5}",
             "Rule family & " + " & ".join(label for _, label in SETTINGS) + r" \\", r"\midrule"]
    for fam, label in FAMILIES:
        gaps = [score[(s, fam, "survival")] - score[(s, fam, "naive")] for s, _ in SETTINGS]
        lines.append(label + " & " + " & ".join(f"${g:+.3f}$" for g in gaps) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = ROOT / "paper" / "tables" / "belief.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("% Written by experiments/analysis/table_belief.py. Do not edit.\n" + "\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
