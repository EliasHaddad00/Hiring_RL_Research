"""Train one MaskablePPO agent, or several in parallel (one single-threaded process per run).

Examples
  python experiments/scripts/train.py --config ppo_small --actor index --seed 101 --out results/g2/ppo_small
  python experiments/scripts/train.py --config ppo_small --actor unrestricted index --seed 101 102 103 \
      --out results/g2/ppo_small --procs 6
  python experiments/scripts/train.py --config ppo_final_index --setting small --actor index --seed 101 \
      --out results/g3/small_check          (a frozen configuration at another setting)

Each run writes to <out>/<actor>_s<seed>/: progress.csv (return, policy loss, value loss, entropy,
approximate KL, clip fraction), evaluations.npz, model.zip, run.json and log.txt.
Final training seeds (1001 to 1015) are refused until gate G3 (--allow-final-seeds).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import SETTINGS, check_training_seed  # noqa: E402
from hiring.training import ACTORS, run_config, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="name in experiments/configs or a path")
    ap.add_argument("--actor", nargs="+", required=True, choices=ACTORS)
    ap.add_argument("--seed", nargs="+", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, help="override total_timesteps")
    ap.add_argument("--setting", choices=list(SETTINGS), help="run the config's hyperparameters at another setting")
    ap.add_argument("--belief", choices=["naive", "survival"], help="estimate shown in the observation (default: the scenario's)")
    ap.add_argument("--procs", type=int, default=6, help="runs in parallel")
    ap.add_argument("--allow-final-seeds", action="store_true")
    a = ap.parse_args()
    for s in a.seed:
        check_training_seed(s, allow_final=a.allow_final_seeds)
    runs = [(actor, seed) for actor in a.actor for seed in a.seed]

    if len(runs) == 1:
        actor, seed = runs[0]
        meta = train(run_config(a.config, a.setting, a.belief), actor, seed, a.out / f"{actor}_s{seed}", a.steps,
                     allow_final_seed=a.allow_final_seeds)
        print(f"{actor} seed {seed}: {meta['total_timesteps']} steps in {meta['wall_seconds']:.0f} s, "
              f"last evaluation {meta['last_eval_mean']:.4f}")
        return

    pending, running, failed = list(runs), [], []
    while pending or running:
        while pending and len(running) < a.procs:
            actor, seed = pending.pop(0)
            out = a.out / f"{actor}_s{seed}"
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, __file__, "--config", a.config, "--actor", actor, "--seed", str(seed),
                   "--out", str(a.out)]
            if a.steps:
                cmd += ["--steps", str(a.steps)]
            if a.setting:
                cmd += ["--setting", a.setting]
            if a.belief:
                cmd += ["--belief", a.belief]
            if a.allow_final_seeds:
                cmd += ["--allow-final-seeds"]
            log = open(out / "log.txt", "w")
            running.append((actor, seed, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(2)
        for item in list(running):
            actor, seed, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                if proc.returncode != 0:
                    failed.append((actor, seed))
                print(f"{actor} seed {seed}: {status}", flush=True)
    if failed:
        sys.exit(f"{len(failed)} run(s) failed: {failed}. See log.txt in each run folder.")


if __name__ == "__main__":
    main()
