"""Table 2 of the paper (research question 1): upper bound and best tuned rule of each family.

Reads only saved results: results/g2/bounds.json (the bound; no episodes involved) and
results/g2/tuning_40.json (rule scores on the 4,000 TUNING episodes the rules were fitted on).
The rule scores are therefore in-sample and marked provisional in the paper; the evaluation-pool
values replace them when that pool is opened. Writes paper/tables/rules.tex.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = [("none", "stress, no pressure", "No press."), ("h0_0", "stress, h0 = 0 (decay only)", "$h_0{=}0$"),
            ("h0_0.02", "stress, h0 = 0.02", "$h_0{=}0.02$"), ("h0_0.05", "stress, h0 = 0.05 (primary)", "$h_0{=}0.05$"),
            ("h0_0.10", "stress, h0 = 0.10", "$h_0{=}0.10$"), ("calibrated", "calibrated", "Calibrated")]
FAMILIES = ["static_threshold", "resume_falling_bar", "one_interview", "up_to_two_interviews", "price_directed"]


def load():
    bounds = {s["setting"]: s["bound"] for s in json.loads((ROOT / "results/g2/bounds.json").read_text())["settings"]}
    best = {}
    for r in json.loads((ROOT / "results/g2/tuning_40.json").read_text()):
        key = (r["setting"], r["family"])
        best[key] = max(best.get(key, -1e9), r["best"])        # the better of the two estimates
    return bounds, best


def main():
    bounds, best = load()
    lines = [r"\begin{tabular}{@{}l@{\hskip 4pt}c@{\hskip 4pt}c@{\hskip 4pt}c@{\hskip 4pt}c@{\hskip 4pt}c@{\hskip 4pt}c@{\hskip 4pt}c@{}}", r"\toprule",
             r"Setting & Bound & Static & Falling & 1 int. & 2 int. & Price & \% \\", r"\midrule"]
    for key, bound_name, label in SETTINGS:
        b = bounds[bound_name]
        scores = [best[(key, f)] for f in FAMILIES]
        top = max(scores)
        cells = [(r"\textbf{%.2f}" if s == top else "%.2f") % s for s in scores]
        lines.append(f"{label} & {b:.2f} & " + " & ".join(cells) + f" & {100 * top / b:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = ROOT / "paper" / "tables" / "rules.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("% Written by experiments/analysis/table_rules.py. Do not edit.\n" + "\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
