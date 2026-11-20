# What is in `results/g2/`

Everything here was made for gate G2 (Nov 14 to 15, 2026) with tuning seeds 101 to 103 and
development or tuning episodes. The evaluation episodes and the final training seeds 1001 to 1015
were not used. **Nothing here is a reported result.** Start with `G2_STATUS.md`; decisions and
their reasons are in `PREREGISTRATION.md` at the repository root.

Scripts are in `experiments/scripts/`. Each `.txt` table has a `.json` twin with the same numbers.

## Yardsticks

| File or folder | Question it answers | Made by | Redone? |
| --- | --- | --- | --- |
| `bounds.txt` | What is the upper bound at each setting, and is it stable on a finer grid? | `compute_bounds.py` | Calibrated line recomputed after base acceptance moved from 0.85 to 0.80 |
| `price_rule.txt` | Does the untuned price-directed rule match the prototype targets? | `check_price_rule.py` | No. Uses the earlier last-day rule form on purpose (`last_day_zero=False`) |
| `tuning/40/` | Every rule configuration tried and its score: 5 families x 2 estimates x 6 settings, 40 configurations each, 4,000 tuning episodes | `tune_rules.py` | Yes, see "What was redone" below |
| `tuning_40.txt` | The best configuration of each of those 60 searches | `tune_rules.py` | Rewritten with the folder above |
| `tuning/360/` | The same searches with 360 configurations: h0 = 0.05 (pre-registered robustness search) and calibrated (added Nov 15) | `tune_rules.py --budget 360 --settings h0_0.05 calibrated` | Four naive searches at h0 = 0.05 redone |
| `tuning_360.txt` | The best configuration of each 360-configuration search | same | Rewritten with the folder above |
| `robustness.txt` | Does the larger search improve the best rule at h0 = 0.05 by more than 2%? | `check_robustness.py` | No |
| `calibrated_check.txt` | Realised acceptance, roles filled and interviews per hire in the calibrated scenario | `check_g2.py` | Yes, after the move to 0.80 |
| `g2_check.txt` | Is the bound above every tuned rule and every PPO run? Slot view, interview cap, belief gaps | `check_g2.py` | Rerun after each tuning change |

## Trained agents

Each run folder holds `model.zip`, `run.json` (actor, seed, full configuration, steps, commit),
`progress.csv` (the six logged columns), `evaluations.npz` (the learning curve) and `log.txt`.
Folders ending in `_s101` to `_s103` are the three tuning seeds.

| Folder | Problem | Actors | Steps | Why it exists |
| --- | --- | --- | --- | --- |
| `ppo_small/` | Small exact instance | index, unrestricted | 3 million | G2 condition 3: within 2% of the optimum |
| `ppo_small_10m/` | Small exact instance | unrestricted | 10 million | The unrestricted actor missed 2% at 3 million steps; is that training length? (yes) |
| `ppo_pooled_small/` | Small exact instance | pooled, seed 101 | 3 million | Same check for the pooled actor |
| `copy_test/` | Small exact instance | all three, supervised copies of the exact policy | n/a | Can each architecture represent a near-optimal policy at all? (yes) |
| `ppo_stress/` | Full problem, h0 = 0.05 | index, unrestricted | 3 million | G2 condition 4: first full-problem curves |
| `ppo_stress_10m/` | Full problem, h0 = 0.05 | index, unrestricted | 10 million | Sets the training length: does 3 to 10 million steps add anything? |
| `ppo_pooled_stress/` | Full problem, h0 = 0.05 | pooled, seed 101 | 3 million | First pooled run on the full problem |
| `ppo_pooled_stress_10m/` | Full problem, h0 = 0.05 | pooled | 10 million | Three seeds and longer training, to set the pooled actor's length |
| `ppo_stress_20m/` | Full problem, h0 = 0.05 | pooled, unrestricted | 20 million | Doubling test: does 10 to 20 million steps add anything? (no) |
| `ppo_no_pressure_cap6/` | Full problem, no pressure, interview cap 6 | unrestricted, seed 101 | 10 million | Interview-cap check for a trained agent |
| `ppo_no_pressure/` | Full problem, no pressure | all three, seed 101 | 3 million | Known-answer check: can each agent match the best rule when waiting is free? |
| `ppo_no_pressure_10m/` | Full problem, no pressure | pooled (three seeds), unrestricted (seed 101) | 10 million | The two actors that failed that check at 3 million steps, trained longer |

No run was repeated with the same settings. A folder with `_10m` is a longer training of the same
kind of agent, started from scratch, not a continuation.

**Two generations of runs.** Runs made after the fix of Nov 15 (`ppo_pooled_stress_10m/`,
`ppo_no_pressure_10m/`, `ppo_stress_20m/`, `ppo_no_pressure_cap6/`) have `"eval_episode_order": "fixed"` in `run.json` and also hold
`model_80.zip`, the model at 80% of training. In earlier runs each curve evaluation starts one
episode later than the one before (same episodes, rotated); `check_plateau.py` rotates them back.
The earlier models were re-saved once, with identical weights, to remove a local file path that the
library had stored inside them.

## Scores of the trained agents

| File | Question it answers | Made by |
| --- | --- | --- |
| `exact_values.txt` | Exact expected payoff of every small-instance agent (no sampling); the pass rule for condition 3 | `exact_values.py` |
| `small_ppo*.txt`, `small_ppo*_episodes.npz` | The same agents against the exact policy on 40,000 development episodes (cross-check) | `check_small_ppo.py` |
| `small_ppo*_mistakes.txt`, `copy_test_mistakes.txt` | Each agent's mistakes by type, priced with the exact solver | `price_mistakes.py` |
| `copy_test.txt` | Scores of the supervised copies | `copy_test.py` |
| `stress_10m_scores.txt` | h0 = 0.05: index and unrestricted at 3 and 10 million steps, against the tuned rule | `score_stress_runs.py` |
| `extra_runs.txt` | No-pressure runs and pooled runs at h0 = 0.05, against the best tuned rule | `score_extra_runs.py` |
| `plateau_*.txt` | Plateau test on the learning curve (300 episodes) and the check of the six logged columns | `check_plateau.py <folder>` |
| `plateau_checkpoints_*.txt` | Plateau test from the 80% model and the final model on 4,000 development episodes (runs after Nov 15) | `check_plateau_checkpoints.py <folder>` |
| `doubling.txt` | Gain from 10 to 20 million steps at h0 = 0.05: the plateau evidence for the unrestricted and pooled actors | `check_doubling.py` |
| `cap6_check.txt` | Unrestricted no-pressure agent trained with an interview cap of 6 against the cap-3 agent | `check_cap6.py` |
| `robustness_calibrated.txt` | Calibrated scenario: 40 against 360 configurations, and the gap between the two estimates at both budgets | `check_robustness.py --setting calibrated` |
| `throughput_diagnosis.txt` | Training speed alone and with several runs in parallel | `diagnose_throughput.py` |

## What was redone, and why

| What | Why | What happened to the old files |
| --- | --- | --- |
| The four naive threshold families in `tuning/40/` (every setting) and `tuning/360/` (h0 = 0.05) | The last-day rule changed on Nov 14: the zero bar on the last day now applies with either estimate. The first version applied it to the survival-aware estimate only, which biased the comparison of the two estimates | Replaced in `tuning/40/`; the earlier files are in the git history at commit `e45737f`. The earlier 360-configuration naive searches were never committed |
| All ten calibrated searches in `tuning/40/` | Base acceptance moved from 0.85 to 0.80 so that realised acceptance falls inside the benchmark range | Replaced; earlier files at commit `e45737f` |
| `bounds.txt` (calibrated line), `g2_check.txt`, `calibrated_check.txt` | They depend on the two changes above | Replaced |
| `tuning/40/` and `tuning/360/` threshold searches, `tuning_40.txt`, `tuning_360.txt`, `robustness.txt`, `robustness_calibrated.txt`, `g2_check.txt`, `calibrated_check.txt` (Nov 20) | A wider threshold family scored below a family it contains in 8 searches; each is now also scored at the best vector of every contained family (`PREREGISTRATION.md`) | Replaced; earlier files at commit `632b2e9`. Each search file keeps its own best under `search_best` |
| Plateau tables | The plateau rule was fixed on Nov 15 (+-1% band), and the curve rotation was found | Replaced |

Searches with the survival-aware estimate and the price-directed rule were not affected by the
last-day change and were not redone.
