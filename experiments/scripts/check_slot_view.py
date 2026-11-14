"""Does the 10-slot view (sorted by naive estimate) cost a rule that uses the survival-aware estimate?

For each poaching level and rule family, the same rule plays the same 2,000 development episodes
through a 10-slot view, a 20-slot view and the full pool, once with the naive estimate and once
with the survival-aware estimate. Reported: mean payoff, the paired difference to the full pool
with its standard error, and the share of days on which a hidden candidate has a higher
survival-aware estimate than a shown one.

Parameters are the earlier simulator's (tuned for the naive estimate, results/g1/v3_fullprec.json),
so the survival-aware rules here are untuned: read the view gaps, not the payoff levels.
Writes results/g1/slot_view.{txt,json}.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.evaluation import evaluate  # noqa: E402
from hiring.rules import FAMILIES  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g1"
EPISODES = 2000
DECAY = 0.03
H0S = [0.02, 0.05, 0.10]
VIEWS = {"10": 10, "20": 20, "full": 135}
BELIEFS = ["naive", "survival"]
LIMIT = 0.02            # the gap agreed on Nov 12 as the point to reconsider 10 slots


def job(args):
    h0, family, par, belief, view = args
    sc = load_scenario("stress", h0=h0, accept_decay=DECAY, n_slots=VIEWS[view])
    res = evaluate(sc, FAMILIES[family](par, belief=belief), seeds=episode_seeds("dev", EPISODES))
    return args[:2] + args[3:], dict(ret=np.asarray(res["ret"], float),
                                     hidden=float(np.sum(res["days_hidden_better"]) / np.sum(res["days"])))


def main():
    params = json.loads((OUT / "v3_fullprec.json").read_text())
    jobs = [(h0, fam, params[f"h0={h0},b={DECAY}"][fam]["par_full"], bel, view)
            for h0 in H0S for fam in FAMILIES for bel in BELIEFS for view in VIEWS]
    with Pool(12) as pool:
        raw = dict(pool.map(job, jobs))

    report, flagged = [], []
    lines = [f"{'h0':>5s} {'rule':22s} {'belief':8s} {'10 slots':>8s} {'20 slots':>8s} {'full':>7s} "
             f"{'10-full':>8s} {'se':>6s} {'20-full':>8s} {'se':>6s} {'hidden10':>8s} {'hidden20':>8s}"]
    for h0 in H0S:
        for fam in FAMILIES:
            for bel in BELIEFS:
                r = {v: raw[(h0, fam, bel, v)] for v in VIEWS}
                row = dict(h0=h0, family=fam, belief=bel)
                for v in VIEWS:
                    row[f"mean_{v}"] = float(r[v]["ret"].mean())
                    row[f"hidden_share_{v}"] = r[v]["hidden"]
                for v in ("10", "20"):
                    d = r[v]["ret"] - r["full"]["ret"]
                    row[f"gap_{v}"] = float(d.mean())
                    row[f"gap_{v}_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
                if abs(row["gap_10"]) > LIMIT:
                    flagged.append(f"h0={h0} {fam} {bel}: 10 slots vs full pool = {row['gap_10']:+.4f}")
                report.append(row)
                lines.append(f"{h0:5.2f} {fam:22s} {bel:8s} {row['mean_10']:8.3f} {row['mean_20']:8.3f} {row['mean_full']:7.3f} "
                             f"{row['gap_10']:+8.4f} {row['gap_10_se']:6.4f} {row['gap_20']:+8.4f} {row['gap_20_se']:6.4f} "
                             f"{row['hidden_share_10']:8.1%} {row['hidden_share_20']:8.1%}")
    lines.append("")
    lines.append(f"{EPISODES} development episodes, paired. Gaps are view minus full pool. hidden = share of days on which a "
                 "hidden candidate has a higher survival-aware estimate than a shown one.")
    lines.append("Parameters were tuned for the naive estimate; survival-aware rows are untuned.")
    lines.append(f"Rows with |10 slots - full| above {LIMIT}: " + ("; ".join(flagged) if flagged else "none"))
    text = "\n".join(lines)
    print(text)
    (OUT / "slot_view.txt").write_text(text + "\n")
    (OUT / "slot_view.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
