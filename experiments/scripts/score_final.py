"""Gate G4, step 1: score every final run and every tuned rule, paired by episode.

  python experiments/scripts/score_final.py --allow-evaluation-pool      the real thing (results/final)
  python experiments/scripts/score_final.py --stand-in                   pipeline test, NOT results

Refuses to run on the evaluation episodes without --allow-evaluation-pool (owner's word, gate
G4). Every item is one output file; an item whose file exists is skipped, so an interrupted run
continues where it was. One process unless --procs says otherwise.

Written into each finished run folder (results/final/<setting>_<estimate>/<actor>_s<seed>/ and
results/final/ablations/<arm>/<actor>_s<seed>/):
  eval.npz           greedy actions (the pre-registered score): per-episode payoff and statistics
  eval_sampled.npz   sampled actions, reported beside it, used for no verdict
  plateau.npz        model at 80% of training and final model on 4,000 development episodes
  exact.json         small instance only: exact value of the greedy policy (no sampling)
  mistakes.json      agent's action against the reference in the same state, by type, on the
                     first --mistake-episodes episodes: the best tuned rule on the full problem,
                     the exact solver (with the exact cost of each mistake) on the small instance
Written to results/final/rules/<setting>/:
  <family>__<estimate>.npz        each tuned rule (best of 40 on the tuning episodes)
  <family>__naive_swapped.npz     the survival-aware rule's parameters with the naive estimate
  <family>__share_{swapped,tuned}.npz   candidate-days on which the two versions decide differently (H4a)
  random.npz, curve_refs.json     random policy; random and best rule on the 300 curve episodes

--stand-in plays the same code on tuning-seed models (results/g3) and development episodes and
writes only to results/standin_not_results/. Models trained at h0 = 0.05 stand in for every other
setting there, so its numbers mean nothing; it exists to test the pipeline.
"""
import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import SETTINGS, episode_seeds, load_scenario  # noqa: E402
from hiring.env import HiringEnv  # noqa: E402
from hiring.evaluation import STAT_KEYS, evaluate  # noqa: E402
from hiring.rules import RULE_FAMILIES, RandomPolicy  # noqa: E402
from price_mistakes import TOL, TYPES, action_values, classify  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "results" / "final"
STANDIN = ROOT / "results" / "standin_not_results"
FULL_SETTINGS = [s for s in SETTINGS if s != "small"]
PLATEAU_EPISODES = 4000           # development episodes, as in check_plateau_checkpoints.py
CURVE = dict(count=300, offset=90_000)   # the curve episodes of every training run (hiring/training.py)
BATCH = 64


# ---- what is scored -------------------------------------------------------------------------------

def tuned_rules(setting: str) -> dict:
    """{(family, estimate): (tuning-episode mean, parameters)} of the registered 40-configuration search."""
    if setting == "small":
        files = (ROOT / "results" / "g3" / "rules_small").glob("*.json")
    else:
        files = (ROOT / "results" / "g2" / "tuning" / "40").glob(f"{setting}__*.json")
    out = {}
    for f in files:
        r = json.loads(f.read_text())
        out[(r["family"], r["belief"])] = (r["best"]["mean"], r["best_params"])
    assert out, f"no tuned rules found for {setting}"
    return out


def best_rule(setting: str):
    """(family, estimate, parameters) of the best rule, chosen on the tuning episodes only."""
    rules = tuned_rules(setting)
    family, belief = max(rules, key=lambda k: rules[k][0])
    return family, belief, rules[(family, belief)][1]


def scenario(setting: str, **kw):
    name, overrides = SETTINGS[setting]
    return load_scenario(name, **{**overrides, **kw})


def run_setting(meta: dict) -> str:
    """The setting name of a run, from the scenario recorded in its run.json."""
    cfg = meta["config"]
    over = {k: v for k, v in (cfg.get("scenario_overrides") or {}).items() if k != "obs_belief"}
    for name, (scen, o) in SETTINGS.items():
        if scen == cfg["scenario"] and all(over.get(k) == v for k, v in o.items()):
            return name
    raise ValueError(f"run matches no registered setting: {cfg['scenario']} {over}")


def finished_runs(root: Path) -> list:
    found = list(root.glob("*/*/run.json")) + list(root.glob("ablations/*/*/run.json"))
    return sorted(p.parent for p in found)


# ---- playing ----------------------------------------------------------------------------------------

def load_model(folder: Path, which: str = "model.zip"):
    import torch as th
    from sb3_contrib import MaskablePPO
    th.set_num_threads(1)
    meta = json.loads((folder / "run.json").read_text())
    source = ROOT / meta["model_dir"] if "model_dir" in meta else folder       # stand-in runs point at their source
    sc = load_scenario(meta["config"]["scenario"], **(meta["config"].get("scenario_overrides") or {}))
    return MaskablePPO.load(source / which, device="cpu"), sc, meta


def play(sc, model, seeds, deterministic: bool = True, batch: int = BATCH) -> dict:
    """hiring.evaluation.evaluate for a PPO model, with one forward pass per step for `batch`
    episodes at a time. Masks are always passed. Same episodes, same order of results."""
    seeds = list(seeds)
    envs = [HiringEnv(sc) for _ in range(min(batch, len(seeds)))]
    rows, obs, active, nxt = [None] * len(seeds), {}, {}, 0
    for k, env in enumerate(envs):
        obs[k], active[k], nxt = env.reset(seed=seeds[nxt])[0], nxt, nxt + 1
    while active:
        ks = list(active)
        actions, _ = model.predict(np.stack([obs[k] for k in ks]), deterministic=deterministic,
                                   action_masks=np.stack([envs[k].action_masks() for k in ks]))
        for k, a in zip(ks, actions):
            obs[k], _, done, _, info = envs[k].step(int(a))
            if done:
                rows[active[k]] = info["episode_stats"]
                if nxt < len(seeds):
                    obs[k], active[k], nxt = envs[k].reset(seed=seeds[nxt])[0], nxt, nxt + 1
                else:
                    del active[k]
    return {key: np.array([r[key] for r in rows], float) for key in STAT_KEYS}


def mistakes(sc, model, seeds, rule=None, solver=None) -> dict:
    """The agent's greedy action against a reference action in the same state, by type.

    Reference: `solver` (small instance: the best action by exact value; the gap in value is the
    exact cost of the mistake) or `rule` (full problem: what the rule would do; no cost)."""
    env = HiringEnv(sc)
    count, loss, steps, n = {k: 0 for k in TYPES}, {k: 0.0 for k in TYPES}, 0, 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        if rule is not None:
            rule.reset(env)
        done, n = False, n + 1
        while not done:
            a = int(model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0])
            if solver is not None:
                vals = action_values(solver, env)
                ref = max(vals, key=vals.get)
                gap = vals[ref] - vals[a]
                differs = gap > TOL
            else:
                ref, gap = rule.act(env), 0.0
                differs = a != ref
            steps += 1
            if differs:
                kind = classify(a, ref, env.N)
                count[kind] += 1
                loss[kind] += gap
            obs, _, done, _, _ = env.step(a)
    return dict(episodes=n, steps=steps, per_100_episodes={k: 100 * count[k] / n for k in TYPES},
                loss_per_episode={k: loss[k] / n for k in TYPES} if solver is not None else None,
                reference="exact solver" if solver is not None else "best tuned rule")


def day_decisions(env, rule):
    """Play `rule` on `env` until the day ends; ({candidate id: "offer" or "interview"}, episode over)."""
    day, N, out, done = env.t, env.N, {}, False
    while not done and env.t == day:
        a = rule.act(env)
        if a < 2 * N:
            out[int(env.view().ids[a % N])] = "offer" if a >= N else "interview"
        _, _, done, _, _ = env.step(a)
    return out, done


def decision_share(sc, rule_a, rule_b, seeds) -> dict:
    """H4a. Play rule A. Every morning, copy the simulator and let rule B play that one day in
    the copy. A candidate-day counts as changed if the two rules treat a candidate who is in the
    pool that morning differently on that day (offer, interview, or neither)."""
    env = HiringEnv(sc)
    env.build_obs = False
    changed, total = [], []
    for seed in seeds:
        env.reset(seed=seed)
        rule_a.reset(env)
        rule_b.reset(env)
        c, n, done = 0, 0, False
        while not done:
            pool = [int(i) for i in env.order]
            b, _ = day_decisions(copy.deepcopy(env), rule_b)
            a, done = day_decisions(env, rule_a)
            c += sum(a.get(i) != b.get(i) for i in pool)
            n += len(pool)
        changed.append(c)
        total.append(n)
    return dict(changed=np.array(changed, float), candidate_days=np.array(total, float))


# ---- jobs: one output file each ---------------------------------------------------------------------

def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".json":
        tmp.write_text(json.dumps(data, indent=1))
    else:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **data)
    os.replace(tmp, path)                          # a half-written file never counts as done


def do(job) -> str:
    kind, path, a = job
    path, t0 = Path(path), time.time()
    seeds = a.get("seeds")
    if kind == "eval":
        model, sc, meta = load_model(Path(a["run"]))
        if a["sampled"]:
            import torch as th
            th.manual_seed(int(meta["seed"]))
            data = dict(ret=play(sc, model, seeds, deterministic=False)["ret"])
        else:
            data = play(sc, model, seeds)
        save(path, dict(data, episode_seed=np.array(seeds), pool=a["pool"]))
    elif kind == "plateau":
        dev = episode_seeds("dev", a["episodes"])
        model, sc, _ = load_model(Path(a["run"]))
        early, _, _ = load_model(Path(a["run"]), "model_80.zip")
        save(path, dict(final=play(sc, model, dev)["ret"], at_80=play(sc, early, dev)["ret"], episode_seed=np.array(dev)))
    elif kind == "exact":
        from hiring.exact_dp import ExactSolver, PolicyEvaluator, greedy_choice
        model, sc, _ = load_model(Path(a["run"]))
        solver = ExactSolver(sc, "automatic")
        save(path, dict(optimum=solver.value(), exact_value=PolicyEvaluator(solver, greedy_choice(solver, model)).value()))
    elif kind == "mistakes":
        model, sc, _ = load_model(Path(a["run"]))
        if a["setting"] == "small":
            from hiring.exact_dp import ExactSolver
            out = mistakes(sc, model, seeds, solver=ExactSolver(sc, "automatic"))
        else:
            family, belief, params = best_rule(a["setting"])
            out = mistakes(sc, model, seeds, rule=RULE_FAMILIES[family](params, belief=belief))
        save(path, dict(out, pool=a["pool"]))
    elif kind == "rule":
        policy = RandomPolicy() if a["family"] == "random" else RULE_FAMILIES[a["family"]](a["params"], belief=a["belief"])
        save(path, dict(evaluate(scenario(a["setting"]), policy, seeds=seeds), episode_seed=np.array(seeds), pool=a["pool"]))
    elif kind == "share":
        fam = RULE_FAMILIES[a["family"]]
        save(path, dict(decision_share(scenario(a["setting"]), fam(a["params"], belief="survival"),
                                       fam(a["other_params"], belief="naive"), seeds), pool=a["pool"]))
    elif kind == "refs":
        curve = episode_seeds("dev", **CURVE)
        family, belief, params = best_rule(a["setting"])
        sc = scenario(a["setting"])
        save(path, dict(random=float(evaluate(sc, RandomPolicy(), seeds=curve)["ret"].mean()), best_rule_family=family,
                        best_rule=float(evaluate(sc, RULE_FAMILIES[family](params, belief=belief), seeds=curve)["ret"].mean())))
    else:
        raise ValueError(kind)
    return f"{kind:9s} {path}  {time.time() - t0:.0f} s"


def jobs(root: Path, seeds, pool: str, mistake_episodes: int, plateau_episodes: int) -> list:
    seeds = list(seeds)
    few = seeds[:mistake_episodes]
    out = []
    for setting in SETTINGS:
        folder, rules = root / "rules" / setting, tuned_rules(setting)
        rule = lambda family, belief, params, label: out.append(
            ("rule", folder / f"{label}.npz", dict(setting=setting, family=family, belief=belief, params=params, seeds=seeds, pool=pool)))
        rule("random", None, None, "random")
        for (family, belief), (_, params) in sorted(rules.items()):
            rule(family, belief, params, f"{family}__{belief}")
        out.append(("refs", folder / "curve_refs.json", dict(setting=setting)))
        if setting == "small":                      # small-instance scores are not used to compare the estimates
            continue
        for family in sorted({f for f, _ in rules}):
            own, other = rules[(family, "survival")][1], rules[(family, "naive")][1]
            rule(family, "naive", own, f"{family}__naive_swapped")
            for label, params in (("swapped", own), ("tuned", other)):
                out.append(("share", folder / f"{family}__share_{label}.npz",
                            dict(setting=setting, family=family, params=own, other_params=params, seeds=few, pool=pool)))
    for run in finished_runs(root):
        setting = run_setting(json.loads((run / "run.json").read_text()))
        a = dict(run=str(run), seeds=seeds, pool=pool, setting=setting)
        out.append(("eval", run / "eval.npz", dict(a, sampled=False)))
        out.append(("eval", run / "eval_sampled.npz", dict(a, sampled=True)))
        out.append(("plateau", run / "plateau.npz", dict(run=str(run), episodes=plateau_episodes)))
        out.append(("mistakes", run / "mistakes.json", dict(a, seeds=few)))
        if setting == "small":
            out.append(("exact", run / "exact.json", dict(run=str(run))))
    return out


# ---- stand-in data ----------------------------------------------------------------------------------

def make_stand_in() -> None:
    """A run tree shaped like results/final, made of tuning-seed models (seeds 101 to 103)."""
    g3 = ROOT / "results" / "g3"
    finals = lambda cfg: (lambda actor, seed: g3 / "ppo_tuning" / "finals" / actor / f"c{cfg[actor]:02d}_s{seed}")
    chosen, second = finals(dict(index=18, unrestricted=8, pooled=8)), finals(dict(index=29, unrestricted=20, pooled=10))
    groups = {f"{s}_survival": (chosen, s, "survival") for s in FULL_SETTINGS}
    groups["h0_0.05_naive"] = (second, "h0_0.05", "naive")
    groups["small_survival"] = (lambda actor, seed: g3 / "small_check" / f"{actor}_s{seed}", "small", "survival")
    plan = [(g, actor, src, setting, belief) for g, (src, setting, belief) in groups.items() for actor in ("index", "unrestricted", "pooled")]
    for arm, cfg in (("gamma_0.999", 1), ("n_steps_256", 16), ("end_day_pooled", 10)):
        plan.append((f"ablations/{arm}", "index", finals(dict(index=cfg)), "h0_0.05", "survival"))
    for group, actor, src, setting, belief in plan:
        for seed in (101, 102, 103):
            source, dest = src(actor, seed), STANDIN / group / f"{actor}_s{seed}"
            if (dest / "run.json").exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            meta = json.loads((source / "run.json").read_text())
            name, overrides = SETTINGS[setting]
            meta["config"].update(scenario=name, scenario_overrides=dict(overrides, obs_belief=belief))
            meta.update(model_dir=source.relative_to(ROOT).as_posix(),
                        stand_in=f"tuning-seed model standing in for a final run at {setting}, {belief}; NOT a result")
            for f in ("progress.csv", "evaluations.npz"):
                shutil.copy(source / f, dest / f)
            (dest / "run.json").write_text(json.dumps(meta, indent=1))
    (STANDIN / "README.txt").write_text(
        "PIPELINE TEST. NOT RESULTS.\nTuning-seed models (seeds 101 to 103) and development episodes, pushed through the G4\n"
        "pipeline to test it. Models trained at h0 = 0.05 stand in for every other setting and for the\n"
        "naive-estimate arm, and other finalist configurations stand in for ablation arms, so no number\n"
        "in this folder means anything. Made by experiments/scripts/score_final.py --stand-in.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-evaluation-pool", action="store_true", help="the owner's word at gate G4")
    ap.add_argument("--stand-in", action="store_true", help="pipeline test on tuning-seed models and development episodes")
    ap.add_argument("--episodes", type=int, default=300, help="stand-in only: development episodes per item")
    ap.add_argument("--mistake-episodes", type=int, default=1000, help="episodes for mistakes.json and the H4a decision share")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--list", action="store_true", help="show what is left to do and stop")
    a = ap.parse_args()
    if a.stand_in:
        assert not a.allow_evaluation_pool, "the stand-in never touches the evaluation pool"
        make_stand_in()
        root, pool, seeds = STANDIN, "dev (stand-in)", episode_seeds("dev", a.episodes, offset=20_000)
        todo = jobs(root, seeds, pool, min(a.mistake_episodes, a.episodes), a.episodes)
    else:
        if not a.allow_evaluation_pool:
            sys.exit("Refused: this scores on the evaluation episodes. They are opened once, at gate G4, on the "
                     "owner's word: pass --allow-evaluation-pool then. For a pipeline test use --stand-in.")
        root, pool = FINAL, "evaluation"
        todo = jobs(root, episode_seeds("evaluation", allow_evaluation=True), pool, a.mistake_episodes, PLATEAU_EPISODES)
    left = [j for j in todo if not Path(j[1]).exists()]
    print(f"{len(todo)} items, {len(todo) - len(left)} already done, {len(left)} to do ({pool} episodes, {root})", flush=True)
    if a.list:
        return
    if a.procs > 1:
        from multiprocessing import Pool
        with Pool(a.procs) as p:
            for line in p.imap_unordered(do, left, chunksize=1):
                print(line, flush=True)
    else:
        for j in left:
            print(do(j), flush=True)


if __name__ == "__main__":
    main()
