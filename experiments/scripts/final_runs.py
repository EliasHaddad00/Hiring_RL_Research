"""The final training runs: list them, or launch them in the pre-registered order.

  python experiments/scripts/final_runs.py --list       the run table, run count and hours estimate
  python experiments/scripts/final_runs.py --launch --allow-final-seeds
                                                         train every run not yet finished, block by block

The list is experiments/configs/final_runs.yaml. Each run uses the frozen configuration of its
actor (configs/ppo_final_<actor>.yaml) at the block's setting and writes to
results/final/<setting>_<estimate>/<actor>_s<seed>/. A run with run.json is finished and skipped,
so a stopped launch continues where it was. Final seeds are refused without --allow-final-seeds
(gate G3: only after PREREGISTRATION.md is tagged and the owner says so).
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import CONFIG_DIR, SETTINGS  # noqa: E402
from hiring.training import load_run_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def expand(spec):
    """One row per block and actor: (group, setting, belief, actor, seeds, steps, hours per run, folder)."""
    rows = []
    for b in spec["blocks"]:
        assert b["setting"] in SETTINGS and b["belief"] in ("naive", "survival")
        hours = spec["hours_per_run"]["small" if b["setting"] == "small" else "full"]
        for actor in spec["actors"]:
            steps = load_run_config(f"ppo_final_{actor}")["total_timesteps"]
            rows.append(dict(group=b["group"], setting=b["setting"], belief=b["belief"], actor=actor, seeds=spec["seeds"],
                             steps=steps, hours=hours[actor], out=f"{spec['out']}/{b['setting']}_{b['belief']}"))
    return rows


def table(spec):
    rows = expand(spec)
    lines = ["| Group | Setting | Estimate | Actor | Steps | Seeds | Runs | Run-hours |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        n = len(r["seeds"])
        lines.append(f"| {r['group']} | {r['setting']} | {r['belief']} | {r['actor']} | {r['steps'] // 10**6} million | "
                     f"{r['seeds'][0]} to {r['seeds'][-1]} | {n} | {n * r['hours']:.0f} |")
    runs = sum(len(r["seeds"]) for r in rows)
    hours = sum(len(r["seeds"]) * r["hours"] for r in rows)
    wall = hours / spec["procs"]
    lines += ["", f"Total: {runs} runs, {hours:.0f} run-hours; with {spec['procs']} runs at a time about {wall:.0f} hours "
                  f"({wall / 24:.1f} days) of machine time. An estimate from the tuning runs' times."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--allow-final-seeds", action="store_true")
    a = ap.parse_args()
    import yaml
    spec = yaml.safe_load((CONFIG_DIR / "final_runs.yaml").read_text())
    if a.list or not a.launch:
        print(table(spec))
        return
    for b in spec["blocks"]:   # one pool per block, so the six slots stay full across actors
        cmd = [sys.executable, str(Path(__file__).with_name("train.py")), "--config", "ppo_final_{actor}",
               "--setting", b["setting"], "--belief", b["belief"], "--actor", *spec["actors"],
               "--out", str(ROOT / spec["out"] / f"{b['setting']}_{b['belief']}"),
               "--procs", str(spec["procs"]), "--skip-finished", "--seed", *map(str, spec["seeds"])]
        if a.allow_final_seeds:
            cmd.append("--allow-final-seeds")
        print(f"{b['group']}: {b['setting']} / {b['belief']}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            sys.exit("stopped: a run failed or final seeds are locked; see the message above")


if __name__ == "__main__":
    main()
