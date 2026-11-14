# Hiring as a race against poaching

Final project for Columbia ELEN E6885 (reinforcement learning), due Dec 16, 2026. A simulated
hiring problem (fill 3 roles in 60 days while stronger candidates are poached faster), solved with
MaskablePPO and compared against tuned rules, an exact optimum on a small instance, and an upper bound.

Status: gate G1 (simulator trusted) passed; see `results/g1/G1_STATUS.md`. No agent results yet.

## Install

Python 3.13 (all tests pass). **Not 3.12.0**: its `zipfile` cannot reload saved models, and two
tests fail there.

The supported install is the CPU build of PyTorch from the PyTorch index:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
```

On Linux or macOS use `.venv/bin/python`. The lockfile pins `torch==2.15.0+cpu`, which exists only
on that index, so the `--extra-index-url` part is required. If the PyTorch index is not reachable,
this fallback takes torch 2.15.0 from PyPI instead. It pins only the RL libraries and torch, so
other packages may differ from the lockfile; it is not the build the results were produced with:

```
python -m pip install -r requirements.txt torch==2.15.0
```

## Run

| Command | What it does |
| --- | --- |
| `python -m pytest experiments/tests` | Fast tests, about 2 minutes |
| `python -m pytest experiments/tests -m slow` | Replay of the earlier simulator and a 40,000-episode check, about 3 minutes |
| `make g1` | All gate G1 checks; each writes to `results/g1/` |
| `python experiments/scripts/evaluate.py --help` | Run one policy on a fixed pool of episodes |

Without `make`, run the scripts listed under `g1` in the `Makefile` one by one.

## Layout

`experiments/` holds the code (`hiring/` simulator, belief, rules, exact solver; `scripts/`; `tests/`;
`configs/`). `results/` holds outputs. `context/` holds the plan and proposal, `notes/` the reading
notes, `PREREGISTRATION.md` the draft pre-registration. `CLAUDE.md` records the working conventions.
