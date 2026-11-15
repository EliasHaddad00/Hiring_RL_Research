"""Full-problem PPO runs at h0 = 0.05, 3 million against 10 million steps, and each against the tuned
price-directed rule. Paired on 4,000 development episodes (greedy actions, masks on).
Tuning seeds: a diagnostic, no verdict. Writes results/g2/stress_10m_scores.{txt,json}."""
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.evaluation import PPOPolicy, evaluate, paired_compare  # noqa: E402
from hiring.rules import RULE_FAMILIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "g2"
N = 4000
SEEDS = (101, 102, 103)
ACTORS = ("index", "unrestricted")


def job(spec):
    import torch as th
    th.set_num_threads(1)
    sc = load_scenario("stress", h0=0.05, accept_decay=0.03)
    kind, arg = spec
    if kind == "rule":
        p = json.loads((OUT / "tuning" / "40" / "h0_0.05__price_directed__survival.json").read_text())["best_params"]
        pol = RULE_FAMILIES["price_directed"](p, belief="survival")
    else:
        pol = PPOPolicy.load(OUT / arg / "model.zip")
    return spec, evaluate(sc, pol, seeds=episode_seeds("dev", N))["ret"]


def main():
    specs = [("rule", "")] + [("ppo", f"{d}/{a}_s{s}") for a in ACTORS for d in ("ppo_stress", "ppo_stress_10m") for s in SEEDS]
    with Pool(13) as pool:
        out = dict(pool.map(job, specs))
    rule = out[("rule", "")]
    rows, gains = [], []
    lines = [f"tuned price-directed rule (survival-aware): {rule.mean():.3f}", ""]
    for (k, a), r in out.items():
        if k == "rule":
            continue
        c = paired_compare(r, rule)
        rows.append(dict(run=a, mean=float(r.mean()), minus_rule=c["diff"], lo=c["lo"], hi=c["hi"], verdict=c["verdict"]))
        lines.append(f"{a:34s} {r.mean():.3f}  minus rule {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] {c['verdict']}")
    lines.append("")
    for a in ACTORS:
        diffs = []
        for s in SEEDS:
            long, short = out[("ppo", f"ppo_stress_10m/{a}_s{s}")], out[("ppo", f"ppo_stress/{a}_s{s}")]
            c = paired_compare(long, short)
            diffs.append(long - short)
            gains.append(dict(actor=a, seed=s, gain=c["diff"], lo=c["lo"], hi=c["hi"]))
            lines.append(f"{a} seed {s}: 10M minus 3M {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")
        m = np.mean(diffs, axis=0)                     # per-episode gain, averaged over the three seeds
        se = m.std(ddof=1) / np.sqrt(len(m))
        lo, hi = m.mean() - 1.96 * se, m.mean() + 1.96 * se
        gains.append(dict(actor=a, seed="mean of three", gain=float(m.mean()), lo=float(lo), hi=float(hi)))
        lines.append(f"{a} mean of three seeds: 10M minus 3M {m.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] (episode noise only, not seed noise)")
    text = "\n".join(lines)
    print(text)
    (OUT / "stress_10m_scores.txt").write_text(text + "\n")
    (OUT / "stress_10m_scores.json").write_text(json.dumps(dict(episodes=N, rule=float(rule.mean()), rows=rows, gains=gains), indent=1))


if __name__ == "__main__":
    main()
