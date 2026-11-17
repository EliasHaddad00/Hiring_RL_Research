# Hiring as a race against poaching

Final project for Columbia ELEN E6885 (reinforcement learning), due Dec 16, 2026. A simulated
hiring problem (fill 3 roles in 60 days while stronger candidates are poached faster), solved with
MaskablePPO and compared against tuned rules, an exact optimum on a small instance, and an upper bound.

Status: gates G1 (simulator trusted) and G2 (yardsticks built, first agents trained) passed; see
`results/g1/G1_STATUS.md` and `results/g2/G2_STATUS.md`. Everything so far uses tuning seeds and
development or tuning episodes: there are no reported results yet.

## Where the project stands (Nov 16, 2026)

Working towards gate G3 (Dec 2): configurations frozen, pre-registration tagged, final seeds launched.
Read `CLAUDE.md` first (binding rules), then `PREREGISTRATION.md` and `results/g2/README.md` (index of every result file).

**PPO hyperparameter tuning is half done and paused** (stopped by hand on Nov 16 to rest the machine).

| Stage | State | Where |
| --- | --- | --- |
| Search: 40 configurations x 3 seeds x 3 actors, 30% length | Done, 360 of 360 | `results/g3/ppo_tuning_search.txt` |
| Finals: 5 best per actor x 3 seeds, full length | 15 of 45 done: unrestricted 12 of 15, index 3 of 15, pooled 0 of 15 | `results/g3/ppo_tuning/finals/` |

A run is finished only if its folder has `score.json`. Folders without it were interrupted; their
files are partial and are overwritten when the run restarts. To continue (finished runs are skipped):

```
.venv\Scripts\python.exe experiments\scripts\tune_ppo.py --stage finals --actor unrestricted index
.venv\Scripts\python.exe experiments\scripts\tune_ppo.py --stage finals --actor pooled
```

About 3 hours for the first line and about 8 for the second on 12 cores (estimates). Each writes
`results/g3/ppo_tuning_finals.txt` when it ends. Run nothing else heavy at the same time.

**Then, in order:**

1. Pick one configuration per actor (best mean over seeds 101 to 103 in the finals table).
2. Rerun the small-instance check on the winners (`check_small_ppo.py`, `exact_values.py`); tuning was done at h0 = 0.05 only.
3. Get the owner's decisions (below), record them in `PREREGISTRATION.md`.
4. Open the calibrated-scenario benchmark sources before citing them.
5. Freeze the configs, tag `PREREGISTRATION.md`, and only then launch final training seeds 1001 to 1015.

**Decisions waiting for the project owner (do not choose for them):**

| Decision | Options | Recommendation given to the owner (Nov 16) |
| --- | --- | --- |
| Interview cap: the cap-6 agent without pressure scored 0.055 below the cap-3 agent, outside the 0.02 closing rule (`results/g2/cap6_check.txt`) | A: record it and keep 3. B: two more seeds for each cap. C: move to 6 | A |
| Tuning ranges: the plan widens a range whose best value sits on its edge; it fires mechanically for every actor (`results/g3/ppo_tuning_search.txt`) | A: no widening, record why. B: add network width 512. C: add width 512 and learning rates above 0.001 | A (the only edge that clearly wins is a discount of 1.0, which cannot be widened) |
| Calibrated scenario: the naive one-interview rule scores below the naive static rule after 360 configurations (`results/g2/robustness_calibrated.txt`) | A: check whether the search misses the "never interview" region, and seed it if so. B: leave it and state the estimate gap as an upper estimate | A |
| Calibrated benchmark sources: none has been opened, so none is cited | Owner opens them, or asks the agent to | none |

None of these blocks finishing the tuning runs. All four must be settled and recorded in `PREREGISTRATION.md` before the configs are frozen.

**Not allowed before G3:** the evaluation episode pool and final training seeds 1001 to 1015 (both locked in `hiring/config.py`).
Search models (`results/g3/ppo_tuning/search/**/model*.zip`) are kept locally only, by decision.

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
| `python experiments/scripts/train.py --config ppo_stress --actor index --seed 101 --out results/g2/ppo_stress` | Train one PPO agent (several actors and seeds run in parallel) |
| `python experiments/scripts/compute_bounds.py` | Upper bound at every setting |
| `python experiments/scripts/tune_rules.py` | Tune every rule family (40 configurations; `--robustness` for 360 at h0 = 0.05) |
| `python experiments/scripts/check_g2.py` | Gate G2 check: bound above every rule and PPO run, slot view, interview cap |
| `python experiments/scripts/exact_values.py` | Exact value of the trained small-instance agents |
| `python experiments/scripts/tune_ppo.py --stage search` (then `finals`, `report`) | PPO hyperparameter search, results in `results/g3/` |
| `python experiments/scripts/check_plateau.py results/g2/ppo_stress` | Logged columns and the plateau test on the curve, for a folder of runs |
| `python experiments/scripts/check_plateau_checkpoints.py results/g2/ppo_pooled_stress_10m` | Plateau test from the saved 80% and final models, 4,000 development episodes |

Without `make`, run the scripts listed under `g1` in the `Makefile` one by one.

## Layout

`experiments/` holds the code (`hiring/` simulator, belief, rules, exact solver; `scripts/`; `tests/`;
`configs/`). `results/` holds outputs. `context/` holds the plan and proposal, `notes/` the reading
notes, `PREREGISTRATION.md` the draft pre-registration. `CLAUDE.md` records the working conventions.
