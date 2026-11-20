# Hiring as a race against poaching

Final project for Columbia ELEN E6885 (reinforcement learning), due Dec 16, 2026. A simulated
hiring problem (fill 3 roles in 60 days while stronger candidates are poached faster), solved with
MaskablePPO and compared against tuned rules, an exact optimum on a small instance, and an upper bound.

Status: gates G1 (simulator trusted) and G2 (yardsticks built, first agents trained) passed; see
`results/g1/G1_STATUS.md` and `results/g2/G2_STATUS.md`. Everything so far uses tuning seeds and
development or tuning episodes: there are no reported results yet.

## Where the project stands (Nov 20, 2026)

Working towards gate G3 (Dec 2): configurations frozen, pre-registration tagged, final seeds launched.
Read `CLAUDE.md` first (binding rules), then `PREREGISTRATION.md` and `results/g2/README.md` (index of every result file).

**PPO hyperparameter tuning is finished (Nov 20): 45 of 45 full-length runs.**

| Stage | State | Where |
| --- | --- | --- |
| Search: 40 configurations x 3 seeds x 3 actors, 30% length | Done, 360 of 360 | `results/g3/ppo_tuning_search.txt` |
| Finals: 5 best per actor x 3 seeds, full length | Done, 45 of 45 | `results/g3/ppo_tuning_finals.txt` |

Best configuration per actor at full length (mean over seeds 101 to 103, tuning episodes; the tuned
rule scores 3.594 on the same episodes): unrestricted 8 (3.535), index 18 (3.573), pooled 8 (3.586).
Not yet chosen: see the open decision below. One sitting was stopped when the machine ran low on
memory; the last 6 runs were made with `--procs 6`.

Rule tuning was corrected on Nov 20: each threshold family is also scored at the best vector of every
family it contains (`PREREGISTRATION.md`, "Checks triggered before the tag"). The best rule is
unchanged at every setting. Corrected calibrated gaps between the two estimates: falling bar +0.097
(was +0.243), one interview +0.192 (was +0.319), two interviews +0.120 (was +0.105).

**Then, in order:**

1. Pick one configuration per actor (best mean over seeds 101 to 103 in the finals table).
2. Rerun the small-instance check on the winners (`check_small_ppo.py`, `exact_values.py`); tuning was done at h0 = 0.05 only.
3. Get the owner's decision (below), record it in `PREREGISTRATION.md`.
4. Freeze the configs, tag `PREREGISTRATION.md`, and only then launch final training seeds 1001 to 1015.

**Decisions waiting for the project owner (do not choose for them):**

| Decision | Options | Recommendation given to the owner (Nov 16) |
| --- | --- | --- |
| Tuning ranges, reopened by the owner's own condition: each actor's best full-length configuration has values on the edge of a range that can be widened (unrestricted and pooled 8: 10 epochs, clip 0.1, width 256; index 18: batch 64, width 256; `results/g3/ppo_tuning_finals.txt`) | A: no widening, record why. B: add network width 512 (new search runs). C: wider still | none yet |

Decided Nov 20 and recorded in `PREREGISTRATION.md`: the interview cap stays at 3; no widening of the PPO tuning ranges unless a chosen configuration sits on the edge of a range that can be widened (it does: see the table); contained rules are scored in every threshold search. The calibrated benchmark sources were opened and confirmed (`experiments/configs/README.md`).

It must be settled and recorded in `PREREGISTRATION.md` before the configs are frozen.

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
