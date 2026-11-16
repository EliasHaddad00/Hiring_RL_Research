"""Robustness of the rule tuning budget at the primary setting (h0 = 0.05).

Pre-registered rule: every rule family also gets a 360-evaluation search. If the best rule gains
more than the 2% margin over its 40-configuration score, this is reported and the stronger rule
becomes the comparator.

With --setting calibrated the same comparison is made for the calibrated scenario (added Nov 15:
its naive rules were poorly tuned by 40 configurations), and the gap between the two estimates is
listed for both budgets.

Reads results/g2/tuning/40 and /360. Tuning scores are biased upward (each configuration was picked
on those episodes), so both chosen rules are also scored on 4,000 development episodes, paired.
Writes results/g2/robustness.{txt,json} (robustness_calibrated.{txt,json} for the calibrated scenario).
"""
import argparse
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.evaluation import evaluate, paired_compare  # noqa: E402
from hiring.rules import RULE_FAMILIES  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g2"
SETTING, EPISODES, MARGIN = "h0_0.05", 4000, 0.02
SCENARIOS = {"h0_0.05": ("stress", dict(h0=0.05, accept_decay=0.03)), "calibrated": ("calibrated", {})}
FAMILIES = ["static_threshold", "resume_falling_bar", "one_interview", "up_to_two_interviews", "price_directed"]
BELIEFS = ["naive", "survival"]


def load(budget, family, belief):
    d = json.loads((OUT / "tuning" / str(budget) / f"{SETTING}__{family}__{belief}.json").read_text())
    best = d["best"]["mean"] if isinstance(d["best"], dict) else d["best"]
    return d["best_params"], float(best)


def job(args):
    global SETTING
    budget, family, belief, SETTING = args
    params, _ = load(budget, family, belief)
    name, overrides = SCENARIOS[SETTING]
    sc = load_scenario(name, **overrides)
    return args[:3], evaluate(sc, RULE_FAMILIES[family](params, belief=belief), seeds=episode_seeds("dev", EPISODES))["ret"]


def main():
    global SETTING
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default=SETTING, choices=list(SCENARIOS))
    SETTING = ap.parse_args().setting
    tag = "robustness" if SETTING == "h0_0.05" else f"robustness_{SETTING}"
    keys = [(b, f, e, SETTING) for f in FAMILIES for e in BELIEFS for b in (40, 360)]
    with Pool(12) as pool:
        dev = dict(pool.map(job, keys))
    rows = []
    lines = [f"Rule tuning budget at {SETTING}: 40 against 360 configurations.", "",
             f"{'family':22s} {'estimate':9s} {'tuning 40':>10s} {'tuning 360':>11s} {'gain':>7s} {'dev 40':>8s} {'dev 360':>8s} "
             f"{'dev gain':>9s} {'95% interval':>19s}"]
    for f in FAMILIES:
        for e in BELIEFS:
            (_, t40), (_, t360) = load(40, f, e), load(360, f, e)
            c = paired_compare(dev[(360, f, e)], dev[(40, f, e)])
            rows.append(dict(family=f, belief=e, tuning_40=t40, tuning_360=t360, tuning_gain_percent=100 * (t360 - t40) / t40,
                             dev_40=c["mean_b"], dev_360=c["mean_a"], dev_gain=c["diff"], dev_interval=[c["lo"], c["hi"]]))
            lines.append(f"{f:22s} {e:9s} {t40:10.3f} {t360:11.3f} {100 * (t360 - t40) / t40:+6.1f}% {c['mean_b']:8.3f} {c['mean_a']:8.3f} "
                         f"{c['diff']:+9.3f} {'[%+.3f, %+.3f]' % (c['lo'], c['hi']):>19s}")
    b40 = max(rows, key=lambda r: r["tuning_40"])
    b360 = max(rows, key=lambda r: r["tuning_360"])
    d40 = max(rows, key=lambda r: r["dev_40"])
    d360 = max(rows, key=lambda r: r["dev_360"])
    gain = (b360["tuning_360"] - b40["tuning_40"]) / b40["tuning_40"]
    fired = gain > MARGIN
    lines += ["",
              f"Best rule with 40 configurations (tuning episodes):  {b40['family']} ({b40['belief']}), {b40['tuning_40']:.3f}",
              f"Best rule with 360 configurations (tuning episodes): {b360['family']} ({b360['belief']}), {b360['tuning_360']:.3f}",
              f"Gain of the best rule: {100 * gain:+.2f}% (margin {100 * MARGIN:.0f}%). "
              + ("RULE FIRES: the 360-configuration rule becomes the comparator." if fired
                 else "The rule does not fire: the comparator stays the 40-configuration rule."),
              f"On development episodes: best of the 40-configuration rules {d40['family']} ({d40['belief']}) {d40['dev_40']:.3f}; "
              f"best of the 360-configuration rules {d360['family']} ({d360['belief']}) {d360['dev_360']:.3f}.",
              "", f"dev = {EPISODES:,} development episodes, not used for tuning; dev gain is paired (360 minus 40).",
              "", "Survival-aware minus naive estimate on development episodes (paired), by tuning budget:",
              f"{'family':22s} {'40 configs':>11s} {'95% interval':>19s} {'360 configs':>12s} {'95% interval':>19s}"]
    gaps = []
    for f in FAMILIES:
        g = {b: paired_compare(dev[(b, f, "survival")], dev[(b, f, "naive")]) for b in (40, 360)}
        gaps.append(dict(family=f, **{f"gap_{b}": g[b]["diff"] for b in g}, **{f"interval_{b}": [g[b]["lo"], g[b]["hi"]] for b in g}))
        lines.append(f"{f:22s} " + " ".join(f"{g[b]['diff']:+{w}.3f} {'[%+.3f, %+.3f]' % (g[b]['lo'], g[b]['hi']):>19s}"
                                           for b, w in ((40, 11), (360, 12))))
    text = "\n".join(lines)
    print(text)
    (OUT / f"{tag}.txt").write_text(text + "\n")
    (OUT / f"{tag}.json").write_text(json.dumps(dict(setting=SETTING, gaps=gaps, rows=rows, best_gain=gain, rule_fires=bool(fired)), indent=1))


if __name__ == "__main__":
    main()
