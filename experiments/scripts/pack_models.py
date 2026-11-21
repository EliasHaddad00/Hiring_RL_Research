"""Zip the models of every finished setting for the GitHub Release, and record their checksums.

  python experiments/scripts/pack_models.py

Final-run models are not in git. For each setting whose runs have all finished this writes
results/final/release/models_<setting>.zip (model.zip and model_80.zip of every run) and adds to
results/final/model_checksums.sha256 (in git) the SHA-256 of every model and of the zip itself.
The ten ablation arms go into one zip, models_ablations.zip, once all of them have finished.
Upload the zips to the Release; fetch_models.py downloads and verifies them.
A setting already in the checksum file is checked, never repacked: a changed model is an error.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHECKSUMS = "model_checksums.sha256"
MODELS = ("model.zip", "model_80.zip")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_checksums(final: Path) -> dict:
    """{path relative to results/final: sha256}"""
    path = final / CHECKSUMS
    if not path.exists():
        return {}
    return {name: digest for digest, name in (line.split("  ", 1) for line in path.read_text().splitlines() if line)}


def write_checksums(final: Path, sums: dict):
    (final / CHECKSUMS).write_text("".join(f"{sums[k]}  {k}\n" for k in sorted(sums)), newline="\n")


def expected_groups(spec: dict, final: Path) -> dict:
    """{zip name: [run folders relative to results/final]} for the settings that have started."""
    groups = {}
    for b in spec["blocks"]:
        folder = f"{b['setting']}_{b['belief']}"
        if (final / folder / "seeds.json").exists():
            seeds = json.loads((final / folder / "seeds.json").read_text())
            groups[f"models_{folder}.zip"] = [f"{folder}/{a}_s{s}" for a in spec["actors"] for s in seeds]
    ab = spec["ablations"]
    groups["models_ablations.zip"] = [f"ablations/{arm['name']}/{arm['actor']}_s{s}" for arm in ab["arms"] for s in ab["seeds"]]
    return groups


def finished(final: Path, runs: list) -> bool:
    return all((final / r / "run.json").exists() and all((final / r / m).exists() for m in MODELS) for r in runs)


def pack(final: Path, spec: dict) -> list:
    """Pack every finished setting not yet packed. Returns the zip names written."""
    sums, written = read_checksums(final), []
    (final / "release").mkdir(parents=True, exist_ok=True)
    for name, runs in expected_groups(spec, final).items():
        files = [f"{r}/{m}" for r in runs for m in MODELS]
        if f"release/{name}" in sums:
            changed = [f for f in files if (final / f).exists() and sha256(final / f) != sums[f]]
            if changed:
                sys.exit(f"STOP: models changed after their checksum was recorded: {changed[:3]}")
            continue
        if not finished(final, runs):
            continue
        with zipfile.ZipFile(final / "release" / name, "w", zipfile.ZIP_STORED) as z:
            for f in files:                              # fixed timestamp: the same models give the same zip
                z.writestr(zipfile.ZipInfo(f, date_time=(1980, 1, 1, 0, 0, 0)), (final / f).read_bytes())
                sums[f] = sha256(final / f)
        sums[f"release/{name}"] = sha256(final / "release" / name)
        written.append(name)
    write_checksums(final, sums)
    return written


def main():
    spec = yaml.safe_load((ROOT / "experiments" / "configs" / "final_runs.yaml").read_text())
    final = ROOT / spec["out"]
    written = pack(final, spec)
    sums = read_checksums(final)
    packed = sorted(k for k in sums if k.startswith("release/"))
    print(f"Packed now: {written or 'nothing new'}")
    print(f"Settings packed so far: {len(packed)}; models with a checksum: {len(sums) - len(packed)}")
    for name in written:
        size = (final / "release" / name).stat().st_size / 1e6
        print(f"Upload to the Release: {final / 'release' / name}  ({size:.0f} MB)")
    if written:
        print(f"Then commit {final / CHECKSUMS}")


if __name__ == "__main__":
    main()
