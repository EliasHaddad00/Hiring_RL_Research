"""Run one policy on a fixed pool of episodes and save unshaped per-episode results.

Examples
  python experiments/scripts/evaluate.py --policy random --episodes 2000
  python experiments/scripts/evaluate.py --policy one_interview --h0 0.02 --belief survival \
      --params 1.65 1.31 0.99 0.44 0.47 -0.06 0.02 --out results/dev/one_interview_h002.csv

  python experiments/scripts/evaluate.py --scenario small_exact --policy exact --episodes 40000 \
      --out results/g2/small/exact.csv
  python experiments/scripts/evaluate.py --scenario small_exact --policy ppo \
      --model results/g2/ppo_small/index_s101/model.zip --episodes 40000 \
      --out results/g2/small/index_s101.csv --paired-with results/g2/small/exact.csv

A PPO model is scored with the action mask on and the most probable valid action (--sampled
samples instead). --paired-with compares with an earlier output on the same episodes and prints
the mean paired difference, its 95% interval and the verdict at the 2% margin.
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
from hiring.evaluation import STAT_KEYS, PPOPolicy, evaluate, paired_compare, summarize  # noqa: E402
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
    ap.add_argument("--policy", default="random", choices=["random", "ppo", "exact", *FAMILIES])
    ap.add_argument("--model", type=Path, help="saved MaskablePPO model (for --policy ppo)")
    ap.add_argument("--sampled", action="store_true", help="sample PPO actions instead of the most probable one")
    ap.add_argument("--paired-with", type=Path, help="CSV written earlier by this script on the same episodes")
    ap.add_argument("--params", type=float, nargs="*", help="rule parameter vector (see hiring/rules.py)")
    ap.add_argument("--belief", default="naive", choices=["naive", "survival"])
    ap.add_argument("--pool", default="dev", choices=["dev", "tuning", "evaluation"])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--allow-evaluation-pool", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    overrides = {k: v for k, v in dict(h0=a.h0, accept_decay=a.accept_decay, n_slots=a.n_slots).items() if v is not None}
    sc = load_scenario(a.scenario, **overrides)
    if a.policy == "random":
        policy = RandomPolicy()
    elif a.policy == "ppo":
        assert a.model, "--model is required for --policy ppo"
        policy = PPOPolicy.load(a.model, deterministic=not a.sampled)
    elif a.policy == "exact":
        from hiring.exact_dp import ExactPolicy, ExactSolver
        policy = ExactPolicy(ExactSolver(sc, overflow="automatic"))
    else:
        assert a.params, "--params is required for rule policies"
        policy = FAMILIES[a.policy](a.params, belief=a.belief)
    seeds = episode_seeds(a.pool, a.episodes, allow_evaluation=a.allow_evaluation_pool)   # lock is in config.py
    res = evaluate(sc, policy, seeds=seeds)
    summary = dict(summarize(res), scenario=sc.name, h0=sc.h0, accept_decay=sc.accept_decay, n_slots=sc.n_slots,
                   policy=a.policy, params=a.params, belief=a.belief, pool=a.pool, commit=commit_hash(),
                   model=str(a.model) if a.model else None,
                   ppo_actions=("sampled" if a.sampled else "most probable") if a.policy == "ppo" else None)
    if a.paired_with:
        with open(a.paired_with, newline="") as fh:
            other = {int(r["episode_seed"]): float(r["ret"]) for r in csv.DictReader(fh)}
        assert all(s in other for s in seeds), "--paired-with does not cover the same episodes"
        summary["paired"] = dict(paired_compare(res["ret"], [other[s] for s in seeds]),
                                 comparator=str(a.paired_with))
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
