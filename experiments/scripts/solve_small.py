"""G1 check: exact optimum of the small instance, both overflow modes, and a Monte Carlo
run of the optimal policy inside our environment. Writes results/g1/exact_small.{json,txt}.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.evaluation import evaluate  # noqa: E402
from hiring.exact_dp import ExactPolicy, ExactSolver  # noqa: E402
from hiring.rules import RandomPolicy  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g1"
PROTOTYPE = {10: 0.5838, 12: 0.6335, 15: 0.6947}
MC_EPISODES = 40000


def main():
    sc = load_scenario("small_exact")
    report, lines = {"horizons": {}}, []
    lines.append(f"{'T':>3s} {'prototype':>9s} {'choose':>9s} {'automatic':>9s} {'diff':>9s} {'diff %':>7s} {'states':>8s} {'sec':>5s}")
    for T in (10, 12, 15):
        t0 = time.time()
        s = sc.with_(horizon=T, max_candidates=T)
        choose, auto = ExactSolver(s, "choose"), ExactSolver(s, "automatic")
        vc, va = choose.value(), auto.value()
        report["horizons"][T] = dict(prototype=PROTOTYPE[T], choose=vc, automatic=va, difference=vc - va,
                                     difference_pct=100 * (vc - va) / vc, start_of_day_states=len(choose.memo["W"]),
                                     reproduces_prototype=abs(vc - PROTOTYPE[T]) <= 0.0005)
        lines.append(f"{T:3d} {PROTOTYPE[T]:9.4f} {vc:9.5f} {va:9.5f} {vc - va:9.6f} {100 * (vc - va) / vc:6.3f}% "
                     f"{len(choose.memo['W']):8d} {time.time() - t0:5.1f}")
        if T == sc.horizon:
            solver = auto
    stop = report["horizons"][12]["difference_pct"] > 1.0
    lines.append("")
    if stop:
        lines.append("STOP: the two optima differ by more than 1%. Monte Carlo check not run.")
    else:
        seeds = episode_seeds("dev", MC_EPISODES)
        opt = evaluate(sc, ExactPolicy(solver), seeds=seeds)["ret"]
        rnd = evaluate(sc, RandomPolicy(), seeds=seeds)["ret"]
        se = opt.std(ddof=1) / np.sqrt(len(opt))
        z = (opt.mean() - solver.value()) / se
        report["monte_carlo"] = dict(episodes=MC_EPISODES, mean=float(opt.mean()), se=float(se),
                                     computed=solver.value(), z=float(z), within_3_se=bool(abs(z) < 3),
                                     random_policy_mean=float(rnd.mean()),
                                     random_policy_se=float(rnd.std(ddof=1) / np.sqrt(len(rnd))))
        lines.append(f"Optimal policy (automatic drop) simulated in our environment, {MC_EPISODES} episodes: "
                     f"{opt.mean():.4f} +- {se:.4f}; computed {solver.value():.5f}; z = {z:+.2f} "
                     f"({'within' if abs(z) < 3 else 'OUTSIDE'} 3 standard errors)")
        lines.append(f"Random policy on the same episodes: {rnd.mean():.4f} +- {report['monte_carlo']['random_policy_se']:.4f}")
    report["stop"] = stop
    text = "\n".join(lines)
    print(text)
    (OUT / "exact_small.txt").write_text(text + "\n")
    (OUT / "exact_small.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
