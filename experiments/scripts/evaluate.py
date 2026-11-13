"""Run one policy on a fixed pool of episodes and save unshaped per-episode results.

Examples
  python experiments/scripts/evaluate.py --policy random --episodes 2000
  python experiments/scripts/evaluate.py --policy one_interview --h0 0.02 --belief survival \
      --params 1.65 1.31 0.99 0.44 0.47 -0.06 0.02 --out results/dev/one_interview_h002.csv

Every policy run on the same pool and count sees the same episodes, so results are paired.
The evaluation pool is locked until configurations are frozen (gate G3).
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.evaluation import STAT_KEYS, evaluate, summarize  # noqa: E402
from hiring.rules import FAMILIES, RandomPolicy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def commit_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="stress")
    ap.add_argument("--h0", type=float)
    ap.add_argument("--accept-decay", type=float)
    ap.add_argument("--n-slots", type=int)
    ap.add_argument("--policy", default="random", choices=["random", *FAMILIES])
    ap.add_argument("--params", type=float, nargs="*", help="rule parameter vector (see hiring/rules.py)")
    ap.add_argument("--belief", default="naive", choices=["naive", "survival"])
    ap.add_argument("--pool", default="dev", choices=["dev", "tuning", "evaluation"])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--allow-evaluation-pool", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.pool == "evaluation" and not a.allow_evaluation_pool:
        sys.exit("The evaluation pool is reserved for final runs. Pass --allow-evaluation-pool once configurations are frozen.")

    overrides = {k: v for k, v in dict(h0=a.h0, accept_decay=a.accept_decay, n_slots=a.n_slots).items() if v is not None}
    sc = load_scenario(a.scenario, **overrides)
    if a.policy == "random":
        policy = RandomPolicy()
    else:
        assert a.params, "--params is required for rule policies"
        policy = FAMILIES[a.policy](a.params, belief=a.belief)
    seeds = episode_seeds(a.pool, a.episodes)
    res = evaluate(sc, policy, seeds=seeds)
    summary = dict(summarize(res), scenario=sc.name, h0=sc.h0, accept_decay=sc.accept_decay, n_slots=sc.n_slots,
                   policy=a.policy, params=a.params, belief=a.belief, pool=a.pool, commit=commit_hash())
    print(json.dumps(summary, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["episode_seed", *STAT_KEYS, "commit"])
            for i, s in enumerate(seeds):
                w.writerow([s, *(res[k][i] for k in STAT_KEYS), summary["commit"]])
        a.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
