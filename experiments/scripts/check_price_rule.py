"""G2 check: the price-directed rule, untuned (price_scale 1, interview_price 0).

  1. Timing over 100 episodes, and the projected cost of tuning (12 settings x 40 configurations
     x 4,000 episodes on 12 processes). Limit agreed on Nov 14: 2 hours.
  2. Score on 4,000 development episodes at h0 = 0, 0.02, 0.05 (decay 0.03), with both estimates,
     beside the prototype figures 3.91 / 3.72 / 3.52 (targets from notes/prototypes/sim.py, 3,000
     episodes, +-0.03; not results) and beside the earlier tuned one-interview rule on the same
     episodes (paired).
  3. The same rule through the full pool instead of 10 slots.
  4. The share of hires that reached the cap of 3 interviews per candidate, and the same rule
     with a cap of 6 on the same episodes (CLAUDE.md: above 5%, rerun with 6 and tell the user).

Writes results/g2/price_rule.{txt,json}.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.bounds import price_tables  # noqa: E402
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.evaluation import evaluate, paired_compare, summarize  # noqa: E402
from hiring.rules import FAMILIES, PriceDirectedRule  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "g2"
EPISODES = 4000
H0S = [0.0, 0.02, 0.05]
PROTOTYPE = {0.0: 3.91, 0.02: 3.72, 0.05: 3.52}
TUNING_JOBS = 12 * 40          # settings (6 scenarios x 2 estimates) x configurations
PROCESSES = 12
LIMIT_HOURS = 2.0


def job(args):
    h0, kind, n_slots, cap = args
    sc = load_scenario("stress", h0=h0, accept_decay=0.03, n_slots=n_slots, max_interviews=cap)
    if kind == "one_interview":
        par = json.loads((ROOT / "results" / "g1" / "v3_fullprec.json").read_text())[f"h0={h0},b=0.03"]
        policy = FAMILIES["one_interview"](par["one_interview"]["par_full"])
    else:
        policy = PriceDirectedRule(belief=kind)
    res = evaluate(sc, policy, seeds=episode_seeds("dev", EPISODES))
    return args, res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 1. timing, one process
    sc = load_scenario("stress")
    t0 = time.perf_counter()
    tb = price_tables(sc, "survival", 0.0)
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    evaluate(sc, PriceDirectedRule(), seeds=episode_seeds("dev", 100))
    per_episode = (time.perf_counter() - t0) / 100
    hours = TUNING_JOBS * (EPISODES * per_episode + build) / PROCESSES / 3600
    timing = dict(table_build_seconds=build, table_megabytes=tb.megabytes, seconds_per_episode=per_episode,
                  projected_tuning_hours=hours, limit_hours=LIMIT_HOURS)

    # 2 and 3. scores
    jobs = [(h0, kind, 10, 3) for h0 in H0S for kind in ("survival", "naive", "one_interview")]
    jobs += [(h0, "survival", 135, 3) for h0 in H0S]
    jobs += [(h0, kind, 10, 6) for h0 in H0S for kind in ("survival", "naive")]
    with Pool(PROCESSES) as pool:
        raw = dict(pool.map(job, jobs))

    rows = []
    lines = [f"Timing: tables {build:.1f} s and {tb.megabytes:.0f} MB per setting; {1000 * per_episode:.1f} ms per episode; "
             f"projected tuning {hours:.2f} h on {PROCESSES} processes (limit {LIMIT_HOURS:.0f} h): "
             f"{'within the limit' if hours <= LIMIT_HOURS else 'OVER THE LIMIT'}.", "",
             f"{'h0':>5s} {'price, survival':>16s} {'se':>6s} {'price, naive':>13s} {'earlier one-interview':>22s} "
             f"{'price minus earlier':>20s} {'95% interval':>18s} {'full pool minus 10 slots':>25s} {'prototype':>10s}"]
    for h0 in H0S:
        s, n, o = raw[(h0, "survival", 10, 3)], raw[(h0, "naive", 10, 3)], raw[(h0, "one_interview", 10, 3)]
        full = raw[(h0, "survival", 135, 3)]
        cmp_ = paired_compare(s["ret"], o["ret"])
        gap = paired_compare(full["ret"], s["ret"])
        ss = summarize(s)
        rows.append(dict(h0=h0, price_survival=ss, price_naive=summarize(n), one_interview=summarize(o),
                         price_minus_one_interview=cmp_, full_pool_minus_10_slots=gap, prototype=PROTOTYPE[h0]))
        lines.append(f"{h0:5.2f} {ss['mean']:16.3f} {ss['se']:6.3f} {n['ret'].mean():13.3f} {o['ret'].mean():22.3f} "
                     f"{cmp_['diff']:+20.3f} {'[%+.3f, %+.3f]' % (cmp_['lo'], cmp_['hi']):>18s} "
                     f"{gap['diff']:+25.4f} {PROTOTYPE[h0]:10.2f}")
    lines += ["",
              f"{EPISODES} development episodes, paired, decay 0.03. Rule untuned: price_scale 1, interview_price 0.",
              "prototype = notes/prototypes/sim.py (3,000 episodes, +-0.03): a target, not a result.",
              "earlier one-interview = parameters tuned in the earlier simulator, naive estimate (results/g1/v3_fullprec.json).",
              "Survival-aware rule: interviews per episode "
              + ", ".join(f"{r['price_survival']['interviews_per_episode']:.1f}" for r in rows)
              + "; hires per episode " + ", ".join(f"{r['price_survival']['hires_per_episode']:.2f}" for r in rows)
              + "; share of hires at the 3-interview cap "
              + ", ".join(f"{100 * r['price_survival']['share_hires_at_interview_cap']:.1f}%" for r in rows) + ".",
              "",
              "Interview cap of 6 instead of 3, same episodes (cap 6 minus cap 3, paired):",
              f"{'h0':>5s} {'estimate':>9s} {'cap 3':>7s} {'at cap':>7s} {'cap 6':>7s} {'at cap':>7s} {'difference':>11s} "
              f"{'95% interval':>18s} {'interviews 3 / 6':>17s}"]
    caps = []
    for h0 in H0S:
        for kind in ("survival", "naive"):
            a3, a6 = raw[(h0, kind, 10, 3)], raw[(h0, kind, 10, 6)]
            s3, s6, c = summarize(a3), summarize(a6), paired_compare(a6["ret"], a3["ret"])
            caps.append(dict(h0=h0, belief=kind, cap3=s3, cap6=s6, cap6_minus_cap3=c))
            lines.append(f"{h0:5.2f} {kind:>9s} {s3['mean']:7.3f} {100 * s3['share_hires_at_interview_cap']:6.1f}% "
                         f"{s6['mean']:7.3f} {100 * s6['share_hires_at_interview_cap']:6.1f}% {c['diff']:+11.4f} "
                         f"{'[%+.4f, %+.4f]' % (c['lo'], c['hi']):>18s} "
                         f"{s3['interviews_per_episode']:8.1f} / {s6['interviews_per_episode']:.1f}")
    lines += ["at cap = share of hired candidates who had reached the per-candidate interview cap."]
    text = "\n".join(lines)
    print(text)
    (OUT / "price_rule.txt").write_text(text + "\n")
    (OUT / "price_rule.json").write_text(json.dumps(dict(timing=timing, rows=rows, interview_cap=caps), indent=1))


if __name__ == "__main__":
    main()
