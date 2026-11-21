"""Shared by the analysis scripts: where results are read from and where paper files are written.

Without arguments a script reads results/final and writes into paper/. With --stand-in it reads
results/standin_not_results (the pipeline test) and writes into that folder's paper/, never into
the real paper.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

ACTORS = [("index", "Index"), ("unrestricted", "Unrestr."), ("pooled", "Pooled")]
PENDING = "--"


def setup(doc: str):
    """(results folder, paper folder, stand-in?) from the command line."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stand-in", action="store_true", help="read and write the pipeline-test folder")
    stand_in = ap.parse_args().stand_in
    root = ROOT / "results" / ("standin_not_results" if stand_in else "final")
    out = root / "paper" if stand_in else ROOT / "paper"
    for sub in ("tables", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    return root, out, stand_in


def write(path: Path, script: str, lines, stand_in: bool) -> None:
    head = f"% Written by experiments/analysis/{script}. Do not edit.\n"
    if stand_in:
        head += "% STAND-IN: tuning-seed models on development episodes. A pipeline test, NOT results.\n"
    path.write_text(head + "\n".join(lines) + "\n")
    print(path.read_text())
