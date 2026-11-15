"""Full-problem PPO runs at h0 = 0.05 against the tuned price-directed rule, paired on 4,000 development
episodes (greedy actions, masks on). Tuning seeds: a diagnostic, no verdict. Writes results/g2/stress_10m_scores.json."""
import json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import episode_seeds, load_scenario
from hiring.evaluation import PPOPolicy, evaluate, paired_compare
from hiring.rules import RULE_FAMILIES
N = 4000
def job(spec):
    import torch as th; th.set_num_threads(1)
    sc = load_scenario("stress", h0=0.05, accept_decay=0.03)
    kind, arg = spec
    if kind == "rule":
        p = json.loads(Path("results/g2/tuning/40/h0_0.05__price_directed__survival.json").read_text())["best_params"]
        pol = RULE_FAMILIES["price_directed"](p, belief="survival")
    else:
        pol = PPOPolicy.load(Path(arg) / "model.zip")
    return spec, evaluate(sc, pol, seeds=episode_seeds("dev", N))["ret"]
if __name__ == "__main__":
    specs = [("rule", "")] + [("ppo", f"results/g2/{d}/unrestricted_s{s}") for d in ("ppo_stress_10m", "ppo_stress") for s in (101, 102, 103)] \
          + [("ppo", f"results/g2/ppo_stress/index_s{s}") for s in (101, 102, 103)]
    with Pool(10) as pool: out = dict(pool.map(job, specs))
    rule = out[("rule", "")]
    rows = []
    print(f"tuned price-directed rule (survival-aware): {rule.mean():.3f}")
    for (k, a), r in out.items():
        if k == "rule": continue
        c = paired_compare(r, rule)
        rows.append(dict(run=a, mean=float(r.mean()), minus_rule=c["diff"], lo=c["lo"], hi=c["hi"], verdict=c["verdict"]))
        print(f"{a:45s} {r.mean():.3f}  minus rule {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] {c['verdict']}")
    for s in (101, 102, 103):
        c = paired_compare(out[("ppo", f"results/g2/ppo_stress_10m/unrestricted_s{s}")], out[("ppo", f"results/g2/ppo_stress/unrestricted_s{s}")])
        print(f"unrestricted seed {s}: 10M minus 3M {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]")
    Path("results/g2/stress_10m_scores.json").write_text(json.dumps(dict(episodes=N, rule=float(rule.mean()), rows=rows), indent=1))
