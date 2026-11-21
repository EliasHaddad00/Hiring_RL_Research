"""Ablation table of the paper: each arm against the frozen index-structured configuration.

Reads only eval.npz files written by score_final.py. An arm's run is paired with the main run of
the same training seed at the primary setting, on the same episodes. With three seeds per arm no
interval is formed: the cell is the mean difference in payoff and the smallest and largest
per-seed difference. An arm without scores shows "--". Writes paper/tables/ablations.tex.
"""
import numpy as np
import yaml

from common import PENDING, ROOT, setup, write

LABELS = {"gamma_0.99": "Discount 0.99", "gamma_0.999": "Discount 0.999", "ent_coef_third": r"Entropy coef.\ $\div 3$",
          "ent_coef_triple": r"Entropy coef.\ $\times 3$", "learning_rate_third": r"Learning rate $\div 3$",
          "learning_rate_triple": r"Learning rate $\times 3$", "n_steps_256": "Rollout 256", "n_steps_1024": "Rollout 1024",
          "reward_shaping": "Reward shaping on", "end_day_pooled": "End-day score reads pool"}


def main():
    root, out, stand_in = setup(__doc__)
    arms = yaml.safe_load((ROOT / "experiments" / "configs" / "final_runs.yaml").read_text())["ablations"]["arms"]
    lines = [r"\begin{tabular}{@{}l@{\hskip 6pt}c@{\hskip 6pt}c@{}}", r"\toprule", r"Change & Payoff change & Per-seed range \\", r"\midrule"]
    for arm in arms:
        diffs = []
        for run in sorted((root / "ablations" / arm["name"]).glob("*/eval.npz")):
            base = root / "h0_0.05_survival" / f"index_s{run.parent.name.rsplit('_s', 1)[1]}" / "eval.npz"
            if base.exists():
                a, b = np.load(run), np.load(base)
                assert np.array_equal(a["episode_seed"], b["episode_seed"]), "an arm and its main run must share episodes"
                diffs.append(float((a["ret"] - b["ret"]).mean()))
        cells = f"${np.mean(diffs):+.3f}$ & ${min(diffs):+.3f}$ to ${max(diffs):+.3f}$" if diffs else f"{PENDING} & {PENDING}"
        lines.append(f"{LABELS[arm['name']]} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tables" / "ablations.tex", "table_ablations.py", lines, stand_in)


if __name__ == "__main__":
    main()
