"""Why is training slower than the G1 throughput measurement?

Trains the unrestricted actor on the small instance for 200,000 steps in four set-ups:
  A  one run alone, evaluation curve off
  B  one run alone, evaluation curve on (the configured episodes and frequency)
  C  six runs in parallel, evaluation off
  D  six runs in parallel, evaluation on
and reports steps per second, the share of wall time spent in evaluation, and torch's thread
count. Run it on an otherwise idle machine. Writes results/g2/throughput_diagnosis.{txt,json}.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.training import load_run_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "g2"
WORK = OUT / "throughput_diagnosis"
STEPS = 200_000
CONFIG = "ppo_small"


def launch(tag, n_runs, evaluate):
    cfg = load_run_config(CONFIG)
    if not evaluate:
        cfg["eval"] = dict(episodes=0, freq_steps=100000)
    folder = WORK / tag
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text(json.dumps(cfg))          # JSON is valid YAML
    code = ("import sys; sys.path.insert(0, r'%s'); from hiring.training import load_run_config, train; "
            "train(load_run_config(r'%s'), 'unrestricted', int(sys.argv[1]), sys.argv[2], %d)"
            % (ROOT / "experiments", folder / "config.yaml", STEPS))
    t0 = time.perf_counter()
    procs = [subprocess.Popen([sys.executable, "-c", code, str(101 + i % 3), str(folder / f"run{i}")])
             for i in range(n_runs)]
    codes = [p.wait() for p in procs]
    wall = time.perf_counter() - t0
    assert not any(codes), f"{tag}: a run failed"
    metas = [json.loads((folder / f"run{i}" / "run.json").read_text()) for i in range(n_runs)]
    sps = [m["steps_per_second"] for m in metas]
    return dict(tag=tag, runs=n_runs, evaluation=evaluate, wall_seconds=wall,
                steps_per_second_mean=sum(sps) / len(sps), steps_per_second_min=min(sps), steps_per_second_max=max(sps),
                eval_share=sum(m["eval_seconds"] / m["wall_seconds"] for m in metas) / len(metas),
                train_only_steps_per_second=sum(m["total_timesteps"] / (m["wall_seconds"] - m["eval_seconds"]) for m in metas) / len(metas),
                torch_threads=metas[0]["torch_threads"])


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    rows = [launch("A_alone_no_eval", 1, False), launch("B_alone_eval", 1, True),
            launch("C_six_no_eval", 6, False), launch("D_six_eval", 6, True)]
    cfg = load_run_config(CONFIG)
    lines = [f"Unrestricted actor, small instance, {STEPS:,} steps per run, 8 environments. Evaluation when on: "
             f"{cfg['eval']['episodes']} episodes every {cfg['eval']['freq_steps']:,} steps.", "",
             f"{'set-up':20s} {'runs':>4s} {'steps/s mean':>13s} {'min':>7s} {'max':>7s} {'time in evaluation':>19s} "
             f"{'steps/s without evaluation time':>32s} {'torch threads':>14s}"]
    for r in rows:
        lines.append(f"{r['tag']:20s} {r['runs']:4d} {r['steps_per_second_mean']:13.0f} {r['steps_per_second_min']:7.0f} "
                     f"{r['steps_per_second_max']:7.0f} {100 * r['eval_share']:18.1f}% {r['train_only_steps_per_second']:32.0f} "
                     f"{r['torch_threads']:14d}")
    text = "\n".join(lines)
    print(text)
    (OUT / "throughput_diagnosis.txt").write_text(text + "\n")
    (OUT / "throughput_diagnosis.json").write_text(json.dumps(rows, indent=1))
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
