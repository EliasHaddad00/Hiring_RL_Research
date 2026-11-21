"""Hypothesis H1a: tune the threshold rules on the exact small instance and score the best exactly.

  python experiments/scripts/tune_rules_small.py        (one process)

The registered budget: 40 parameter vectors per rule family and estimate, each scored on the
4,000 tuning episodes, plus the best vector of each family a family contains (hiring/tuning.py).
The price-directed rule is built for Gaussian signals and does not exist on the small instance
(binned signals), so four families are tuned. This registered search was run after the tag
(PREREGISTRATION.md, "Deviations").

Each family's best rule is then scored EXACTLY: the backward pass of hiring/exact_dp.py with the
rule's action in place of the maximum, no sampling. `rule_choice` restates ThresholdRule.act on
solver states; the table shows the exact value beside the simulated mean as a check on it.

Writes results/g3/rules_small/<family>__<estimate>.json and results/g3/rules_small.{txt,json}.
"""
import json
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import load_scenario  # noqa: E402
from hiring.exact_dp import ExactSolver, PolicyEvaluator, slot_order  # noqa: E402
from hiring.rules import RULE_FAMILIES  # noqa: E402
from hiring.tuning import BELIEFS, CHAIN, seed_contained, tune  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g3"
BUDGET = 40


def rule_choice(solver: ExactSolver, rule):
    """ThresholdRule.act as a function of the solver's offer-phase state (day t = 1..T, pool)."""
    b, T = solver.b, solver.T

    def choose(t, pool):
        order = slot_order(solver, pool)                      # the environment's slot order
        if not order:
            return [(1.0, "end", None)]
        f = (t - 1) / T                                       # the environment counts days from 0
        est = {i: (solver.M[pool[i][0]][pool[i][1]] if rule.belief == "survival" else solver.M[pool[i][0]][0]) for i in order}
        n = {i: sum(b.types[pool[i][0]][1]) for i in order}   # interview results seen
        if t - 1 >= rule.start_day:
            if rule.last_day_zero and t == T:
                eligible = [i for i in order if est[i] > 0.0]
            else:
                eligible = [i for i in order if est[i] >= rule.offer_bar(min(n[i], len(rule.o) - 1), f, 0)]
            if eligible:
                return [(1.0, "offer", max(eligible, key=lambda i: est[i]))]     # ties: first in slot order
        if rule.kmax > 0 and t < T:
            want = [i for i in order if solver.nx[pool[i][0]][0] >= 0 and n[i] < rule.kmax
                    and est[i] >= float(rule.interview_bar(n[i], f))]
            if want:
                return [(1.0, "interview", max(want, key=lambda i: est[i]))]
        return [(1.0, "end", None)]
    return choose


def main():
    sc = load_scenario("small_exact")
    folder = OUT / "rules_small"
    folder.mkdir(parents=True, exist_ok=True)
    results = {}
    for family in CHAIN:
        for belief in BELIEFS:
            path = folder / f"{family}__{belief}.json"
            if not path.exists():
                path.write_text(json.dumps(tune(sc, family, belief, budget=BUDGET), indent=1))
            results[(family, belief)] = json.loads(path.read_text())
            print(f"searched {family} {belief}", flush=True)
    own = {k: r.get("search_best_params", r["best_params"]) for k, r in results.items()}
    for family in CHAIN[1:]:
        for belief in BELIEFS:
            r = results[(family, belief)]
            if "search_best" not in r:
                r = seed_contained(sc, r, {n: own[(n, belief)] for n in CHAIN[:CHAIN.index(family)]})
                (folder / f"{family}__{belief}.json").write_text(json.dumps(r, indent=1))
                results[(family, belief)] = r

    solver = ExactSolver(sc, "automatic")
    optimum = solver.value()
    lines = [f"Threshold rules on the exact small instance: best of {BUDGET} configurations per family and estimate",
             "(4,000 tuning episodes; plus the best vector of each contained family), then scored exactly.",
             f"Exact optimum {optimum:.5f}. H1a asks for at least 95% of it: {0.95 * optimum:.5f}.",
             "The price-directed rule does not exist on this instance (binned signals).", "",
             f"{'family':22s} {'estimate':9s} {'tuning mean':>11s} {'se':>6s} {'exact value':>11s} {'% of optimum':>12s} "
             f"{'exact - mean, in se':>19s}  parameters"]
    rows = []
    for family in CHAIN:
        for belief in BELIEFS:
            r = results[(family, belief)]
            exact = PolicyEvaluator(solver, rule_choice(solver, RULE_FAMILIES[family](r["best_params"], belief=belief))).value()
            mean, se = r["best"]["mean"], r["best"]["se"]
            rows.append(dict(family=family, belief=belief, tuning_mean=mean, se=se, exact_value=exact,
                             percent_of_optimum=100 * exact / optimum, params=r["best_params"]))
            lines.append(f"{family:22s} {belief:9s} {mean:11.4f} {se:6.4f} {exact:11.5f} {100 * exact / optimum:11.2f}% "
                         f"{(exact - mean) / se:+19.2f}  " + " ".join(f"{x:.3f}" for x in r["best_params"]))
    top = max(rows, key=lambda x: x["exact_value"])
    lines += ["", f"Best rule by exact value: {top['family']} with the {top['belief']} estimate, {top['exact_value']:.5f}, "
                  f"{top['percent_of_optimum']:.2f}% of the optimum: H1a {'holds' if top['percent_of_optimum'] >= 95 else 'does not hold'}.",
              "The rule was chosen on the tuning episodes; its exact value involves no sampling."]
    text = "\n".join(lines)
    print("\n" + text)
    (OUT / "rules_small.txt").write_text(text + "\n")
    (OUT / "rules_small.json").write_text(json.dumps(dict(optimum=optimum, rows=rows), indent=1))


if __name__ == "__main__":
    main()
