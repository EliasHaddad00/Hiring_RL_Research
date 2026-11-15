"""Known-answer and pooled-actor runs on the full problem, against the best tuned rule of their setting.

  no pressure (h0 = 0, no decay): results/g2/ppo_no_pressure (3 million steps, one run per
                                  architecture) and ppo_no_pressure_10m (10 million steps)
  h0 = 0.05: the pooled actor, results/g2/ppo_pooled_stress (3 million) and ppo_pooled_stress_10m

Paired on 4,000 development episodes (greedy actions, masks on). Tuning seeds: a check, no verdict.
Writes results/g2/extra_runs.{txt,json}.
"""
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.evaluation import PPOPolicy, evaluate, paired_compare, summarize  # noqa: E402
from hiring.rules import RULE_FAMILIES  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g2"
N = 4000
SETTINGS = {"none": (dict(h0=0.0, accept_decay=0.0), ["ppo_no_pressure", "ppo_no_pressure_10m"], "stress, no pressure"),
            "h0_0.05": (dict(h0=0.05, accept_decay=0.03), ["ppo_pooled_stress", "ppo_pooled_stress_10m"],
                        "stress, h0 = 0.05 (primary)")}


def best_rule(setting):
    """Best tuned rule of the setting by its score on the tuning episodes."""
    best = None
    for path in (OUT / "tuning" / "40").glob(f"{setting}__*.json"):
        d = json.loads(path.read_text())
        score = d["best"]["mean"] if isinstance(d["best"], dict) else d["best"]
        if best is None or score > best[0]:
            best = (score, d["family"], d["belief"], d["best_params"])
    return best


def job(spec):
    import torch as th
    th.set_num_threads(1)
    setting, kind, arg = spec
    sc = load_scenario("stress", **SETTINGS[setting][0])
    if kind == "rule":
        _, family, belief, params = best_rule(setting)
        pol = RULE_FAMILIES[family](params, belief=belief)
    else:
        pol = PPOPolicy.load(Path(arg) / "model.zip")
    return spec, evaluate(sc, pol, seeds=episode_seeds("dev", N))


def main():
    specs = []
    for setting, (_, folders, _) in SETTINGS.items():
        specs.append((setting, "rule", ""))
        specs += [(setting, "ppo", str(p.parent)) for folder in folders for p in sorted((OUT / folder).glob("*/model.zip"))
                  if (p.parent / "run.json").exists()]
    with Pool(min(12, len(specs))) as pool:
        out = dict(pool.map(job, specs))
    bounds = {r["setting"]: r["bound"] for r in json.loads((OUT / "bounds.json").read_text())["settings"]}
    lines, rows = [], []
    for setting, (_, _, label) in SETTINGS.items():
        rule = out[(setting, "rule", "")]
        _, family, belief, _ = best_rule(setting)
        bound = bounds[label]
        lines += [f"{label}: bound {bound:.3f}; best tuned rule {family} ({belief}) {rule['ret'].mean():.3f}", ""]
        for (s, kind, arg), res in out.items():
            if s != setting or kind == "rule":
                continue
            c, sm = paired_compare(res["ret"], rule["ret"]), summarize(res)
            steps = json.loads((Path(arg) / "run.json").read_text())["total_timesteps"]
            name = f"{Path(arg).name}, {steps / 1e6:.0f}M"
            rows.append(dict(setting=setting, run=Path(arg).name, folder=Path(arg).parent.name, steps=steps, mean=sm["mean"], se=sm["se"], minus_rule=c["diff"], lo=c["lo"], hi=c["hi"],
                             verdict=c["verdict"], bound_minus_run=bound - sm["mean"], interviews=sm["interviews_per_episode"],
                             hires=sm["hires_per_episode"], at_cap=sm["share_hires_at_interview_cap"]))
            lines.append(f"  {name:20s} {sm['mean']:.3f} +- {sm['se']:.3f}  minus rule {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] "
                         f"{c['verdict']:12s} bound - run {bound - sm['mean']:+.3f}  interviews {sm['interviews_per_episode']:.1f}, "
                         f"hires {sm['hires_per_episode']:.2f}, at cap {100 * sm['share_hires_at_interview_cap']:.1f}%")
        lines.append("")
    lines.append("Tuning seeds 101 to 103, development episodes: a check, not a reported result.")
    text = "\n".join(lines)
    print(text)
    (OUT / "extra_runs.txt").write_text(text + "\n")
    (OUT / "extra_runs.json").write_text(json.dumps(dict(episodes=N, rows=rows), indent=1))


if __name__ == "__main__":
    main()
