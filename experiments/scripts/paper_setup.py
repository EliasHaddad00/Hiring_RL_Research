"""Download the official ICML 2026 style files into paper/ (`make paper-setup`).

The style files are not kept in the repository (icml2026.sty carries no redistribution licence).
Uses only the standard library.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip"
FILES = ("icml2026.sty", "icml2026.bst", "fancyhdr.sty", "algorithm.sty", "algorithmic.sty")
PAPER = Path(__file__).resolve().parents[2] / "paper"


def main():
    with urllib.request.urlopen(URL, timeout=60) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    names = {Path(n).name: n for n in archive.namelist()}
    missing = [f for f in FILES if f not in names]
    if missing:
        sys.exit(f"the archive at {URL} does not contain: {missing}")
    for f in FILES:
        (PAPER / f).write_bytes(archive.read(names[f]))       # written by base name only
        print("wrote", PAPER / f)


if __name__ == "__main__":
    main()
