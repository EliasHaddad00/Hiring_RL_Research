"""Tune every rule family, with both estimates, at every setting. Tuning episodes only.

  python experiments/scripts/tune_rules.py                 40 configurations each (the budget PPO gets)
  python experiments/scripts/tune_rules.py --robustness    360 configurations, primary setting only

One configuration is one parameter vector scored on all 4,000 tuning episodes (hiring/tuning.py).
Writes results/g2/tuning/<tag>/<setting>__<family>__<estimate>.json (every configuration) and
results/g2/tuning_<tag>.{txt,json} (the table of best configurations). Finished searches are
reused on a rerun; delete the file to redo one.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import load_scenario  # noqa: E402
from hiring.tuning import BELIEFS, SCALES, tune  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g2"
SETTINGS = {"none": ("stress", dict(h0=0.0, accept_decay=0.0)),
            "h0_0": ("stress", dict(h0=0.0, accept_decay=0.03)),
            "h0_0.02": ("stress", dict(h0=0.02, accept_decay=0.03)),
            "h0_0.05": ("stress", dict(h0=0.05, accept_decay=0.03)),
            "h0_0.10": ("stress", dict(h0=0.10, accept_decay=0.03)),
            "calibrated": ("calibrated", {})}
PRIMARY = "h0_0.05"


def job(args):
    setting, family, belief, budget, folder = args
    path = Path(folder) / f"{setting}__{family}__{belief}.json"
    if path.exists():
        return json.loads(path.read_text())
    name, overrides = SETTINGS[setting]
    t0 = time.perf_counter()
    res = tune(load_scenario(name, **overrides), family, belief, budget=budget)
    res.update(setting=setting, seconds=time.perf_counter() - t0)
    path.write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robustness", action="store_true", help="360 configurations at the primary setting")
    ap.add_argument("--budget", type=int)
    ap.add_argument("--settings", nargs="+", choices=list(SETTINGS))
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    budget = a.budget or (360 if a.robustness else 40)
    settings = a.settings or ([PRIMARY] if a.robustness else list(SETTINGS))
    tag = f"{budget}"
    folder = OUT / "tuning" / tag
    folder.mkdir(parents=True, exist_ok=True)
    # longest searches first, so the pool stays busy
    order = ["price_directed", "up_to_two_interviews", "one_interview", "resume_falling_bar", "static_threshold"]
    jobs = [(s, f, b, budget, str(folder)) for f in order for s in settings for b in BELIEFS]
    assert set(order) == set(SCALES)
    with Pool(a.procs) as pool:
        results = pool.map(job, jobs, chunksize=1)

    by = {(r["setting"], r["family"], r["belief"]): r for r in results}
    lines = [f"Best of {budget} configurations per rule family, estimate and setting, on the 4,000 tuning episodes.",
             "Scores are biased upward for the chosen configuration (it was picked on these episodes).", ""]
    table = []
    for s in settings:
        lines.append(f"{s}")
        lines.append(f"  {'family':22s} {'estimate':9s} {'start':>7s} {'best':>7s} {'se':>6s} {'gain':>7s} {'found at':>9s} "
                     f"{'interviews':>10s} {'hires':>6s} {'accept':>7s} {'at cap':>7s}  parameters")
        for f in order[::-1]:
            for b in BELIEFS:
                r = by[(s, f, b)]
                bs = r["best"]
                table.append(dict(setting=s, family=f, belief=b, start=r["start"]["mean"], best=bs["mean"], se=bs["se"],
                                  best_configuration=r["best_configuration"], params=r["best_params"], summary=bs))
                lines.append(f"  {f:22s} {b:9s} {r['start']['mean']:7.3f} {bs['mean']:7.3f} {bs['se']:6.3f} "
                             f"{bs['mean'] - r['start']['mean']:+7.3f} {r['best_configuration']:9d} "
                             f"{bs['interviews_per_episode']:10.1f} {bs['hires_per_episode']:6.2f} "
                             f"{100 * bs['realised_acceptance']:6.1f}% {100 * bs['share_hires_at_interview_cap']:6.1f}%  "
                             + " ".join(f"{x:.3f}" for x in r["best_params"]))
        top = max((t for t in table if t["setting"] == s), key=lambda t: t["best"])
        lines.append(f"  best rule: {top['family']} with the {top['belief']} estimate, {top['best']:.3f}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (OUT / f"tuning_{tag}.txt").write_text(text + "\n")
    (OUT / f"tuning_{tag}.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
