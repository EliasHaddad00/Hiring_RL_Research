"""Is the default training path the same at the tagged commit and now?

The training code gained the ablation options (commit d2f45d6) while the final runs were in
progress, so final runs record different commits (PREREGISTRATION.md, "Deviations"). This check
trains each actor's frozen configuration for a short run with tuning seed 101, once with the code
of the tagged commit (a temporary git worktree) and once with the current code, and compares every
trained weight bit for bit (model.zip and model_80.zip).

  python experiments/scripts/check_default_path.py

Writes results/final/default_path_check.txt. Exit code 1 if any weight differs.
"""
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import torch as th

ROOT = Path(__file__).resolve().parents[2]
TAGGED = "0f64099"
ACTORS = ("index", "unrestricted", "pooled")
STEPS, SEED, SETTING = 20480, 101, "h0_0.05"   # five updates; the 80% model is saved before the fourth


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def weights(model_zip: Path) -> dict:
    with zipfile.ZipFile(model_zip) as z:
        return th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")


def train(tree: Path, actor: str, out: Path):
    subprocess.run([sys.executable, str(tree / "experiments" / "scripts" / "train.py"), "--config", f"ppo_final_{actor}",
                    "--setting", SETTING, "--actor", actor, "--seed", str(SEED), "--steps", str(STEPS), "--out", str(out)],
                   check=True, capture_output=True)


def main():
    head = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "--", "experiments/hiring", "experiments/scripts/train.py", "experiments/configs")
    lines = [f"Default training path: tagged commit {TAGGED} against current commit {head}"
             + (" plus uncommitted changes" if dirty else " (training code, configs: no uncommitted changes)") + ".",
             f"Each actor's frozen configuration, setting {SETTING}, tuning seed {SEED}, {STEPS} steps, one run with each code.",
             "Compared: every tensor of the saved policy (model.zip and model_80.zip), bit for bit.", ""]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        old_tree = tmp / "tagged"
        git("worktree", "add", "--detach", str(old_tree), TAGGED)
        try:
            for actor in ACTORS:
                train(old_tree, actor, tmp / "old")
                train(ROOT, actor, tmp / "new")
                for name in ("model.zip", "model_80.zip"):
                    a, b = (weights(tmp / side / f"{actor}_s{SEED}" / name) for side in ("old", "new"))
                    same = a.keys() == b.keys() and all(th.equal(a[k], b[k]) for k in a)
                    start = weights(tmp / "old" / f"{actor}_s{SEED}" / "model_80.zip")
                    moved = name == "model_80.zip" or any(not th.equal(a[k], start[k]) for k in a)   # training did change the weights
                    ok &= same and moved
                    n = sum(v.numel() for v in a.values())
                    lines.append(f"{actor:13s} {name:13s} {len(a):3d} tensors, {n:,} values: "
                                 f"{'IDENTICAL' if same else 'DIFFERENT'}" + ("" if moved else " (but training changed nothing: check invalid)"))
                    print(lines[-1], flush=True)
        finally:
            git("worktree", "remove", "--force", str(old_tree))
    lines += ["", "Result: " + ("the default path gives identical weights at both commits." if ok else "WEIGHTS DIFFER.")]
    text = "\n".join(lines)
    print("\n" + text)
    (ROOT / "results" / "final" / "default_path_check.txt").write_text(text + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
