"""Exact expected payoff of each trained agent on the small instance, beside two estimates of it.

  exact      backward pass over every state with the agent's greedy action in place of the maximum
             (hiring/exact_dp.py, PolicyEvaluator). No sampling at all.
  priced     sum of solver-priced mistakes along simulated episodes (price_mistakes.py)
  realised   paired difference of realised payoffs to the exact policy on 40,000 development
             episodes (check_small_ppo.py)

Writes results/g2/exact_values.{txt,json}.

  python experiments/scripts/exact_values.py --runs results/g3/small_check
      exact values only, for every <runs>/<actor>_s<seed>/model.zip, with the mean per actor against
      the pass mark (98% of the optimum). Writes <runs>_exact_values.{txt,json}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import load_scenario  # noqa: E402
from hiring.exact_dp import ExactSolver, PolicyEvaluator, greedy_choice, optimal_choice, uniform_choice  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g2"
SOURCES = [("ppo_small", "small_ppo"), ("copy_test", "copy_test"), ("ppo_small_10m", "small_ppo_10m"),
           ("ppo_pooled_small", "small_ppo_pooled")]
SOURCES = [(folder, tag) for folder, tag in SOURCES                      # later diagnostics are optional
           if (OUT / f"{tag}.json").exists() and (OUT / f"{tag}_mistakes.json").exists()]


def runs_only(folder: Path):
    """Exact value of every agent in a runs folder; pass rule on the mean per actor."""
    from sb3_contrib import MaskablePPO
    th.set_num_threads(1)
    solver = ExactSolver(load_scenario("small_exact"), "automatic")
    optimum = solver.value()
    mark = 0.98 * optimum
    rows, lines = [], [f"Exact optimum {optimum:.5f}; pass mark {mark:.5f} (98%). Exact value of the greedy policy, no sampling.", ""]
    for path in sorted(folder.glob("*/model.zip")):
        actor, seed = path.parent.name.rsplit("_s", 1)
        value = PolicyEvaluator(solver, greedy_choice(solver, MaskablePPO.load(path, device="cpu"))).value()
        rows.append(dict(actor=actor, seed=int(seed), exact_value=value, percent_of_optimum=100 * value / optimum,
                         passes=bool(value >= mark)))
        lines.append(f"{path.parent.name:22s} {value:.5f}  {100 * value / optimum:6.2f}% of the optimum  "
                     f"{'pass' if value >= mark else 'BELOW THE MARK'}")
        print(lines[-1], flush=True)
    lines.append("")
    summary = {}
    for actor in sorted({r["actor"] for r in rows}):
        v = [r["exact_value"] for r in rows if r["actor"] == actor]
        summary[actor] = dict(mean=float(np.mean(v)), seeds=len(v), passes=bool(np.mean(v) >= mark))
        lines.append(f"{actor:13s} mean of {len(v)} seeds {np.mean(v):.5f}  {100 * np.mean(v) / optimum:6.2f}% of the optimum  "
                     f"{'PASS' if np.mean(v) >= mark else 'FAIL'}")
    lines += ["", "Tuning seeds; a gate check, not a reported result."]
    text = "\n".join(lines)
    print("\n" + text)
    out = folder.parent / f"{folder.name}_exact_values"
    out.with_suffix(".txt").write_text(text + "\n")
    out.with_suffix(".json").write_text(json.dumps(dict(optimum=optimum, pass_mark=mark, rows=rows, by_actor=summary), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, help="a folder of runs: exact values only")
    a = ap.parse_args()
    if a.runs:
        return runs_only(a.runs)
    from sb3_contrib import MaskablePPO
    th.set_num_threads(1)
    sc = load_scenario("small_exact")
    solver = ExactSolver(sc, "automatic")
    optimum = solver.value()
    limit = -0.02 * optimum
    check = PolicyEvaluator(solver, optimal_choice(solver)).value()
    random_value = PolicyEvaluator(solver, uniform_choice(solver)).value()
    lines = [f"Exact optimum {optimum:.5f}; the optimal policy evaluated by the same pass {check:.5f}; random policy {random_value:.5f}.",
             f"Offer-phase states evaluated per agent: {len(solver.memo['U']):,}.", "",
             f"{'agent':28s} {'exact value':>12s} {'exact shortfall':>16s} {'% of optimum':>13s} {'within 2%':>10s} "
             f"{'priced shortfall':>17s} {'realised shortfall':>19s} {'95% interval':>18s}"]
    rows = []
    for folder, tag in SOURCES:
        priced = {r["run"]: r["total_loss"] for r in json.loads((OUT / f"{tag}_mistakes.json").read_text())["rows"]}
        realised = {r["run"]: r["greedy"] for r in json.loads((OUT / f"{tag}.json").read_text())["rows"]}
        for path in sorted((OUT / folder).glob("*/model.zip")):
            name = path.parent.name
            model = MaskablePPO.load(path, device="cpu")
            value = PolicyEvaluator(solver, greedy_choice(solver, model)).value()
            g = realised[name]
            rows.append(dict(agent=f"{folder}/{name}", exact_value=value, exact_shortfall=optimum - value,
                             percent_of_optimum=100 * value / optimum, within_2_percent=bool(value - optimum > limit),
                             priced_shortfall=priced[name], realised_shortfall=-g["diff"],
                             realised_interval=[-g["hi"], -g["lo"]]))
            r = rows[-1]
            lines.append(f"{r['agent']:28s} {value:12.5f} {r['exact_shortfall']:16.5f} {r['percent_of_optimum']:12.2f}% "
                         f"{'yes' if r['within_2_percent'] else 'no':>10s} {r['priced_shortfall']:17.5f} "
                         f"{r['realised_shortfall']:19.5f} {'[%.4f, %.4f]' % tuple(r['realised_interval']):>18s}")
            print(lines[-1], flush=True)
    d_priced = np.array([r["exact_shortfall"] - r["priced_shortfall"] for r in rows])
    d_real = np.array([r["exact_shortfall"] - r["realised_shortfall"] for r in rows])
    lines += ["", f"exact minus priced shortfall:   largest |difference| {np.abs(d_priced).max():.5f}, mean {d_priced.mean():+.5f}",
              f"exact minus realised shortfall: largest |difference| {np.abs(d_real).max():.5f}, mean {d_real.mean():+.5f}",
              "", "Agents use training seeds 101 to 103 or supervised imitation (copy_test); a gate check, not a reported result."]
    text = "\n".join(lines)
    print("\n" + text)
    (OUT / "exact_values.txt").write_text(text + "\n")
    (OUT / "exact_values.json").write_text(json.dumps(dict(optimum=optimum, random=random_value, rows=rows), indent=1))


if __name__ == "__main__":
    main()
