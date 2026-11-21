"""Download the final-run models from the GitHub Release and verify every checksum.

  python experiments/scripts/fetch_models.py                 download, verify, unpack into results/final/
  python experiments/scripts/fetch_models.py --verify-only   check the models already on disk
  python experiments/scripts/fetch_models.py --source <folder or URL>   take the zips from elsewhere
  python experiments/scripts/fetch_models.py --tuning-search  also the 720 tuning-search models (about
                                                              1 GB, into results/g3/ppo_tuning/search);
                                                              optional, no result depends on them

The list of zips and the SHA-256 of every zip and every model is results/final/model_checksums.sha256
(in git, written by pack_models.py). A zip is unpacked only if its checksum matches, and only the
files named in the checksum file are taken from it. Exit code 1 if anything is missing or differs.
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from pack_models import OPTIONAL, ROOT, read_checksums, sha256

RELEASE = "https://github.com/EliasHaddad00/Hiring_RL_Research/releases/download/final-models"


def fetch(final: Path, source: str, verify_only: bool = False, optional: bool = False) -> list:
    """Returns the list of problems (empty: every model is present and matches).

    optional=False: the final-run models, unpacked into `final`. optional=True: only the entries
    marked optional in the checksum file (the tuning-search models), unpacked into results/."""
    sums = read_checksums(final)
    if optional:
        sums = {k[len(OPTIONAL):]: v for k, v in sums.items() if k.startswith(OPTIONAL)}
    else:
        sums = {k: v for k, v in sums.items() if not k.startswith(OPTIONAL)}
    base = final.parent if optional else final        # where the models go
    zips = sorted(k for k in sums if k.startswith("release/"))
    models = sorted(k for k in sums if not k.startswith("release/"))
    problems = []
    if not verify_only:
        for z in zips:
            target = final / z
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (target.exists() and sha256(target) == sums[z]):
                name = z.split("/", 1)[1]
                print(f"getting {name}", flush=True)
                if source.startswith("http"):
                    urllib.request.urlretrieve(f"{source}/{name}", target)
                else:
                    shutil.copyfile(Path(source) / name, target)
            if sha256(target) != sums[z]:
                problems.append(f"{z}: checksum differs, not unpacked")
                continue
            with zipfile.ZipFile(target) as archive:
                for member in archive.namelist():
                    if member in sums:                   # nothing outside the recorded list is written
                        (base / member).parent.mkdir(parents=True, exist_ok=True)
                        (base / member).write_bytes(archive.read(member))
    for m in models:
        if not (base / m).exists():
            problems.append(f"{m}: missing")
        elif sha256(base / m) != sums[m]:
            problems.append(f"{m}: checksum differs")
    print(f"{len(models) - sum(': ' in p and not p.startswith('release/') for p in problems)} of {len(models)} models verified, "
          f"{len(zips)} zip(s) listed")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--source", default=RELEASE, help="URL or folder holding the zips")
    ap.add_argument("--tuning-search", action="store_true", help="also fetch the optional tuning-search models")
    ap.add_argument("--final", type=Path, default=ROOT / "results" / "final", help="folder holding model_checksums.sha256")
    a = ap.parse_args()
    problems = fetch(a.final, a.source, a.verify_only)
    if a.tuning_search:
        problems += fetch(a.final, a.source, a.verify_only, optional=True)
    for p in problems:
        print("PROBLEM", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
