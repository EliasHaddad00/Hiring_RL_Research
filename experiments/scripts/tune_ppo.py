"""PPO hyperparameter search at the primary setting (h0 = 0.05), same budget as the rules.

  python experiments/scripts/tune_ppo.py --stage search    40 configurations x 3 seeds per actor, 30% length
  python experiments/scripts/tune_ppo.py --stage finals    the 5 best per actor at full length
  python experiments/scripts/tune_ppo.py --stage report    tables only, from what is saved

Seeded random search (hiring/tuning_ppo.py); configuration 1 is the starting values. Every run is
scored once, with greedy actions and masks on, on the 4,000 tuning episodes the rules were tuned on.
A configuration's score is the mean over tuning seeds 101 to 103.

Each run writes to results/g3/ppo_tuning/<stage>/<actor>/c<NN>_s<seed>/ and is skipped if its
score.json exists, so an interrupted search continues where it stopped.
Writes results/g3/ppo_tuning_<stage>.{txt,json}.
"""
import argparse
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
from hiring.training import ACTORS, load_run_config, train  # noqa: E402
from hiring.tuning_ppo import (FINAL_STEPS, FINALISTS, PARAMETERS, TUNING_SEEDS, apply, at_boundary,  # noqa: E402
                               sample_configs, search_steps)

OUT = Path(__file__).resolve().parents[2] / "results" / "g3"
BASE, N_CONFIGS, EPISODES = "ppo_stress", 40, 4000


def run_dir(stage, actor, idx, seed):
    return OUT / "ppo_tuning" / stage / actor / f"c{idx + 1:02d}_s{seed}"


def job(spec):
    import torch as th
    from hiring.evaluation import PPOPolicy, evaluate, summarize
    th.set_num_threads(1)
    stage, actor, idx, seed, steps, episodes = spec
    out = run_dir(stage, actor, idx, seed)
    score_path = out / "score.json"
    if score_path.exists():
        return spec, json.loads(score_path.read_text())
    base = load_run_config(BASE)
    flat = sample_configs(base, N_CONFIGS)[idx]
    cfg = apply(base, flat, curve=(stage == "finals"))
    meta = train(cfg, actor, seed, out, total_timesteps=steps)
    sc = load_scenario(cfg["scenario"], **(cfg.get("scenario_overrides") or {}))
    res = evaluate(sc, PPOPolicy.load(out / "model.zip"), seeds=episode_seeds("tuning", episodes))
    np.save(out / "tuning_episode_payoffs.npy", res["ret"].astype(np.float32))
    sm = summarize(res)
    score = dict(actor=actor, configuration=idx + 1, seed=seed, steps=meta["total_timesteps"], mean=sm["mean"], se=sm["se"],
                 interviews=sm["interviews_per_episode"], hires=sm["hires_per_episode"],
                 at_cap=sm["share_hires_at_interview_cap"], wall_seconds=meta["wall_seconds"], params=flat)
    score_path.write_text(json.dumps(score, indent=1))
    return spec, score


def table(stage, actors, scores, configs):
    """Per actor: configurations ranked by the mean over seeds."""
    lines, rows = [], []
    for actor in actors:
        per = {}
        for (st, a, idx, seed, _, _), s in scores.items():
            if a == actor:
                per.setdefault(idx, {})[seed] = s["mean"]
        ranked = sorted(per, key=lambda i: -np.mean(list(per[i].values())))
        lines += [f"{actor}: {len(per)} configurations, ranked by mean payoff over seeds {', '.join(map(str, TUNING_SEEDS))}",
                  f"  {'rank':>4s} {'config':>6s} {'mean':>7s} {'seeds':>23s}  " + " ".join(f"{k:>13s}" for k in PARAMETERS)]
        for rank, idx in enumerate(ranked, 1):
            vals = [per[idx].get(s, float("nan")) for s in TUNING_SEEDS]
            flat = configs[idx]
            rows.append(dict(actor=actor, rank=rank, configuration=idx + 1, mean=float(np.mean(list(per[idx].values()))),
                             by_seed=dict(zip(map(str, TUNING_SEEDS), vals)), seeds_done=len(per[idx]), params=flat))
            lines.append(f"  {rank:4d} {idx + 1:6d} {rows[-1]['mean']:7.3f} {' '.join(f'{v:7.3f}' for v in vals):>23s}  "
                         + " ".join(f"{flat[k]:13.5g}" for k in PARAMETERS))
        if ranked:
            edge = at_boundary(configs[ranked[0]])
            lines.append(f"  best configuration {ranked[0] + 1}; parameters at an end of their range: {', '.join(edge) if edge else 'none'}")
        lines.append("")
    return lines, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["search", "finals", "report"])
    ap.add_argument("--actor", nargs="+", default=list(ACTORS), choices=ACTORS)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--configs", type=int, nargs="+", help="1-based configuration numbers (default: all 40)")
    a = ap.parse_args()
    configs = sample_configs(load_run_config(BASE), N_CONFIGS)

    def saved(stage):
        out = {}
        for p in (OUT / "ppo_tuning" / stage).glob("*/*/score.json"):
            s = json.loads(p.read_text())
            if s["actor"] in a.actor:
                out[(stage, s["actor"], s["configuration"] - 1, s["seed"], s["steps"], a.episodes)] = s
        return out

    stages = ["search", "finals"] if a.stage == "report" else [a.stage]
    if a.stage == "search":
        idxs = [i - 1 for i in a.configs] if a.configs else range(N_CONFIGS)
        # longest runs first, so the pool stays busy
        specs = [("search", actor, i, seed, search_steps(actor), a.episodes)
                 for actor in sorted(a.actor, key=lambda x: -search_steps(x)) for i in idxs for seed in TUNING_SEEDS]
    elif a.stage == "finals":
        _, rows = table("search", a.actor, saved("search"), configs)
        specs = []
        for actor in a.actor:
            done = [r for r in rows if r["actor"] == actor]
            if len(done) < N_CONFIGS or any(r["seeds_done"] < len(TUNING_SEEDS) for r in done):
                sys.exit(f"{actor}: the search is not complete ({len(done)} configurations saved); finals need all {N_CONFIGS}")
            specs += [("finals", actor, r["configuration"] - 1, seed, FINAL_STEPS[actor], a.episodes)
                      for r in done[:FINALISTS] for seed in TUNING_SEEDS]
    if a.stage != "report":
        with Pool(a.procs) as pool:
            for spec, s in pool.imap_unordered(job, specs, chunksize=1):
                print(f"{spec[0]} {spec[1]} c{spec[2] + 1:02d} seed {spec[3]}: {s['mean']:.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        scores = saved(stage)
        if not scores:
            continue
        steps = {actor: (search_steps(actor) if stage == "search" else FINAL_STEPS[actor]) for actor in a.actor}
        lines, rows = table(stage, a.actor, scores, configs)
        head = [f"PPO tuning, stage '{stage}', primary setting (h0 = 0.05). Steps per run: "
                + ", ".join(f"{k} {v:,}" for k, v in steps.items()) + ".",
                f"Scored on the {a.episodes:,} tuning episodes, greedy actions, masks on. Configuration 1 = starting values.", ""]
        text = "\n".join(head + lines)
        print(text)
        (OUT / f"ppo_tuning_{stage}.txt").write_text(text + "\n")
        (OUT / f"ppo_tuning_{stage}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
