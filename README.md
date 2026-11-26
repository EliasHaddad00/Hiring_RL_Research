# Hiring as a race against poaching

Final project for Columbia ELEN E6885 (reinforcement learning), due Dec 16, 2026. A simulated
hiring problem (fill 3 roles in 60 days while stronger candidates are poached faster), solved with
MaskablePPO and compared against tuned rules, an exact optimum on a small instance, and an upper bound.

Status: gates G1 (simulator trusted) and G2 (yardsticks built, first agents trained) passed; see
`results/g1/G1_STATUS.md` and `results/g2/G2_STATUS.md`. Everything so far uses tuning seeds and
development or tuning episodes: there are no reported results yet.

## Where the project stands (Nov 21, 2026)

Working towards gate G3 (Dec 2): configurations frozen, pre-registration tagged, final seeds launched.
Read `CLAUDE.md` first (binding rules), then `PREREGISTRATION.md` and `results/g2/README.md` (index of every result file).

| Step | State | Where |
| --- | --- | --- |
| PPO tuning: search (360 runs) and finals (45 runs) | Done Nov 20 | `results/g3/ppo_tuning_search.txt`, `ppo_tuning_finals.txt` |
| Tuning ranges | Not widened (owner, Nov 20); edge values do not win in the search | `results/g3/search_medians.txt` |
| Configurations chosen and frozen | Unrestricted 8, index 18, pooled 8 | `experiments/configs/ppo_final_<actor>.yaml` |
| Small-instance check on the chosen configurations | Passed Nov 21: 99.0%, 99.3%, 99.4% of the optimum (index, unrestricted, pooled; mean of seeds 101 to 103) | `results/g3/small_check_exact_values.txt` |
| Final run list | Written: 360 runs, about 870 run-hours | `PREREGISTRATION.md`, "Final run list"; `experiments/configs/final_runs.yaml` |
| `PREREGISTRATION.md` tagged | Done Nov 21, tag `preregistration` | |
| Final seeds launched | Launched Nov 21 with the owner's word; log in `results/final/launch_log.txt`. A stopped launch is continued with the same command | `python experiments/scripts/final_runs.py --launch --allow-final-seeds` |

Rule tuning was corrected on Nov 20: each threshold family is also scored at the best vector of every
family it contains (`PREREGISTRATION.md`, "Checks triggered before the tag"). The best rule is
unchanged at every setting. Corrected calibrated gaps between the two estimates: falling bar +0.097
(was +0.243), one interview +0.192 (was +0.319), two interviews +0.120 (was +0.105).

**No decision is waiting for the project owner.** The 18 points where the wording of the pre-registered
decision rule was open were decided on Nov 25, before any final run was scored
(`experiments/configs/decision_rule.yaml`; recorded in `PREREGISTRATION.md`, Deviations).

**Gate G4 pipeline (written Nov 21, tested on stand-in data only).** After the last final run, and on the owner's word:

```
python experiments/scripts/score_final.py --allow-evaluation-pool    # every run and rule on the evaluation episodes; resumes if interrupted
python experiments/scripts/apply_decision_rule.py                    # one verdict per hypothesis: results/final/verdicts.txt
python experiments/analysis/make_figures.py                          # every table and figure of the paper ("make figures")
```

With `--stand-in` the same three commands run on tuning-seed models and development episodes and write
only to `results/standin_not_results/` (git-ignored): a pipeline test whose numbers mean nothing.

Decided and recorded in `PREREGISTRATION.md`: interview cap stays at 3; no widening of the PPO tuning
ranges; chosen configurations; small-instance check (pass rule on the mean of three seeds) and its
fallback; model storage (Nov 20). Full list of 360 runs, with a date cutoff of Dec 4 after which a
setting not yet started gets 10 seeds; launch order primary, H4b arm, other settings, ablations;
the 30 ablation runs (Nov 21).

**Final-run models are not in git.** They stay in `results/final/` on the training machine
(`.gitignore`) and will be attached to a GitHub Release; every other file of a run is committed.
The tuning models already in the repository stay.

**Still locked:** the evaluation episode pool (`hiring/config.py`); it is opened only for the final paired evaluation, on the owner's word. Final training seeds 1001 to 1015 are used only by `final_runs.py`.
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

## Trained models (the step after cloning and installing)

The final-run models are not in git. They are attached to the GitHub Release `final-models`,
one zip per setting. Download them and check every file against the checksums kept in git
(`results/final/model_checksums.sha256`):

```
.venv\Scripts\python.exe experiments/scripts/fetch_models.py
```

It unpacks the models into `results/final/<setting>/<actor>_s<seed>/` and ends with
"N of N models verified"; any missing or differing file is listed and the exit code is 1.
`--verify-only` re-checks the files on disk without downloading. Everything else from the final
runs (logs, curves, `run.json`) is already in the clone. The clean-clone reproduction of Dec 14
includes this step: clone, install, fetch and verify the models, then regenerate the results.

For the owner, after a setting finishes: `python experiments/scripts/pack_models.py` writes
`results/final/release/models_<setting>.zip`, prints the files to upload to the Release, and
updates the checksum file, which is then committed.

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
| `python experiments/scripts/exact_values.py --runs results/g3/small_check` | Exact value of every agent in a folder of small-instance runs, against the pass mark |
| `python experiments/scripts/search_medians.py` | Median search score by hyperparameter value (`results/g3/search_medians.txt`) |
| `python experiments/scripts/pack_models.py` / `fetch_models.py` | Zip the models of finished settings and record checksums / download from the Release and verify |
| `python experiments/scripts/final_runs.py --launch-ablations --allow-final-seeds` | The 30 must-run ablation runs (after the main runs) |
| `python experiments/scripts/final_runs.py --list` | The final run table, run count and hours (`--launch --allow-final-seeds` only after the owner says so) |
| `python experiments/scripts/check_plateau.py results/g2/ppo_stress` | Logged columns and the plateau test on the curve, for a folder of runs |
| `python experiments/scripts/check_plateau_checkpoints.py results/g2/ppo_pooled_stress_10m` | Plateau test from the saved 80% and final models, 4,000 development episodes |

Without `make`, run the scripts listed under `g1` in the `Makefile` one by one.

## Layout

`experiments/` holds the code (`hiring/` simulator, belief, rules, exact solver; `scripts/`; `tests/`;
`configs/`). `results/` holds outputs. `context/` holds the plan and proposal, `notes/` the reading
notes, `PREREGISTRATION.md` the draft pre-registration. `CLAUDE.md` records the working conventions.
