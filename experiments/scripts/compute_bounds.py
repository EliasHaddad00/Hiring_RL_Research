"""G2 check: the Lagrangian upper bound at every setting. Writes results/g2/bounds.{txt,json}.

For each setting: the bound on the default grid, the same on a finer grid (the difference is the
numerical error; it must stay under 0.01), the best prices, the bound with an interview cap of 6,
and a simulation of the relaxed policy on development episodes (it must match the computed value).
Prototype figures are printed beside ours as targets only; they come from notes/prototypes/lag.py.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# One numerical thread per worker: several multi-threaded workers at once slow each other down badly.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.bounds import GaussianTables, lagrangian_bound, relaxed_value, simulate_relaxed  # noqa: E402
from hiring.config import CONFIG_DIR, episode_seeds, load_scenario  # noqa: E402
from hiring.exact_dp import ExactSolver  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g2"
SETTINGS = [("stress, no pressure", "stress", dict(h0=0.0, accept_decay=0.0)),
            ("stress, h0 = 0 (decay only)", "stress", dict(h0=0.0, accept_decay=0.03)),
            ("stress, h0 = 0.02", "stress", dict(h0=0.02, accept_decay=0.03)),
            ("stress, h0 = 0.05 (primary)", "stress", dict(h0=0.05, accept_decay=0.03)),
            ("stress, h0 = 0.10", "stress", dict(h0=0.10, accept_decay=0.03))]
if (CONFIG_DIR / "calibrated.yaml").exists():
    SETTINGS.append(("calibrated", "calibrated", {}))
PROTOTYPE = {"stress, h0 = 0 (decay only)": 4.31, "stress, h0 = 0.02": 4.17, "stress, h0 = 0.05 (primary)": 4.00}
LAMS = (0.0, 0.005, 0.02)
SIM_EPISODES = 4000
GRID_LIMIT = 0.01


def job(setting):
    label, name, overrides = setting
    sc = load_scenario(name, **overrides)
    t0 = time.perf_counter()
    tb = GaussianTables(sc)
    base = lagrangian_bound(sc, lam_grid=LAMS, tables=tb)
    fine = lagrangian_bound(sc, n_grid=361, n_nodes=160, n_noise=30)
    finest = lagrangian_bound(sc, n_grid=721)             # one more doubling of the grid of naive means
    cap6 = lagrangian_bound(sc.with_(max_interviews=6))
    z = base["z"]
    computed = float(relaxed_value(sc, tb.solve([z]), [z])[0])
    sim = simulate_relaxed(sc, tb, z, episode_seeds("dev", SIM_EPISODES))
    return dict(setting=label, scenario=name, h0=sc.h0, accept_decay=sc.accept_decay,
                bound=base["bound"], z=z, lam=base["lam"], per_lam=base["per_lam"],
                bound_fine_grid=fine["bound"], grid_change=base["bound"] - fine["bound"],
                bound_finest_grid=finest["bound"], second_change=fine["bound"] - finest["bound"],
                bound_interview_cap_6=cap6["bound"], state_count=base["state_count"],
                build_seconds=base["build_seconds"], solve_seconds=base["solve_seconds"],
                relaxed_computed=computed, relaxed_simulated=float(sim.mean()),
                relaxed_se=float(sim.std(ddof=1) / np.sqrt(len(sim))),
                prototype=PROTOTYPE.get(label), seconds=time.perf_counter() - t0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(min(6, len(SETTINGS))) as pool:
        rows = pool.map(job, SETTINGS)

    small = load_scenario("small_exact")
    sb = lagrangian_bound(small, lam_grid=(0.0, 0.01, 0.02, 0.05))
    small_row = dict(bound=sb["bound"], z=sb["z"], lam=sb["lam"], state_count=sb["state_count"],
                     exact_choose=ExactSolver(small, "choose").value(),
                     exact_automatic=ExactSolver(small, "automatic").value())

    lines = [f"{'setting':30s} {'bound':>7s} {'fine':>7s} {'change':>7s} {'finest':>7s} {'change':>7s} {'cap 6':>7s} {'z':>6s} {'lam':>6s} "
             f"{'relaxed: computed':>18s} {'simulated':>10s} {'se':>6s} {'prototype':>10s}"]
    problems = []
    for r in rows:
        proto = f"{r['prototype']:.2f}" if r["prototype"] is not None else "none"
        lines.append(f"{r['setting']:30s} {r['bound']:7.4f} {r['bound_fine_grid']:7.4f} {r['grid_change']:+7.4f} "
                     f"{r['bound_finest_grid']:7.4f} {r['second_change']:+7.4f} "
                     f"{r['bound_interview_cap_6']:7.4f} {r['z']:6.3f} {r['lam']:6.3f} {r['relaxed_computed']:18.4f} "
                     f"{r['relaxed_simulated']:10.4f} {r['relaxed_se']:6.4f} {proto:>10s}")
        if abs(r["grid_change"]) >= GRID_LIMIT:
            problems.append(f"{r['setting']}: finer grid changes the bound by {r['grid_change']:+.4f}")
        if abs(r["relaxed_simulated"] - r["relaxed_computed"]) > 3 * r["relaxed_se"]:
            problems.append(f"{r['setting']}: simulated relaxed value is more than 3 standard errors from the computed one")
    lines += ["",
              "bound = default grid (181 points, 80 quality nodes, 15 noise nodes); fine = 361 / 160 / 30; finest = 721 / 80 / 15.",
              "change = bound minus fine; the coarser grid errs upward, which is the safe side for an upper bound.",
              f"relaxed: the priced problem without the role and slot limits, at the best z; simulated on {SIM_EPISODES} development episodes.",
              "prototype = notes/prototypes/lag.py as reported in context/ (a target, not a result). Rerun here, the prototype",
              "gives 4.172 at h0 = 0.02; our code gives the same 4.17237 when set to its grid (91 points on [-3, 4], 15 / 9",
              "nodes) and its arrival integration (41 Gauss-Hermite nodes), so there the added wait action changes nothing.",
              f"one-candidate table: {rows[0]['state_count']:,} states per price; build {rows[0]['build_seconds']:.2f} s, "
              f"bound search {rows[0]['solve_seconds']:.1f} s.",
              "",
              f"small instance (T = 12): bound {small_row['bound']:.5f} (z = {small_row['z']:.3f}, lam = {small_row['lam']}), "
              f"exact optimum {small_row['exact_choose']:.5f} (choose) / {small_row['exact_automatic']:.5f} (automatic drop); "
              f"{small_row['state_count']:,} states.",
              "",
              "Problems: " + ("none" if not problems else "")] + [f"  {p}" for p in problems]
    text = "\n".join(lines)
    print(text)
    (OUT / "bounds.txt").write_text(text + "\n")
    (OUT / "bounds.json").write_text(json.dumps(dict(settings=rows, small_instance=small_row, problems=problems), indent=1))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
