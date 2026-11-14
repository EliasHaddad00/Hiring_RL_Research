# Gate G1 status (Nov 12, 2026)

Every number below was produced by this repository's code. "Prototype" and "published" values are the targets they were compared against.

| G1 condition | Result | Evidence |
| --- | --- | --- |
| Environment tests pass | Yes | `pytest experiments/tests`: both environment checkers, masked random rollouts, determinism, arrival / acceptance / departure rates, and every convention test |
| Earlier rule scores reproduce within 0.05 | Yes, all ten gate rows; largest gap 0.006 | `rule_reproduction.txt` |
| Exact instance reproduces 0.6335 | Yes: 0.63348 (prototype convention) | `exact_small.txt` |
| Throughput measured | Yes | `throughput.txt` |
| Pre-registration drafted | Yes, untagged | `PREREGISTRATION.md` |

Open problem, not a G1 condition: saved SB3 models cannot be loaded under Python 3.12.0 (standard-library `zipfile` bug). Two tests fail for this reason: `test_setup.py::test_zipfile_streams_seek_correctly` and `test_policies.py::test_save_and_load_round_trip`. Needs a later Python patch release before week 2.

**Resolved Nov 14.** The environment was rebuilt on Python 3.13.17 from the same lockfile. All 97 fast tests and all 17 slow tests pass, including the two above.

## Rule reproduction

Method: a copy of `notes/hiring_sim_v3.py` was rerun with fixed seeds and reproduced every score and every rounded parameter in the original results file. Its 4,000 evaluation episodes were then replayed draw for draw through our environment with its full-precision parameters.

| Pressure | Rule | Published | Ours, 10 slots | Ours, full pool |
| --- | --- | --- | --- | --- |
| all five | fixed threshold | 3.10 | 3.099 | 3.099 |
| none | up to two interviews | 3.93 | 3.928 | 3.928 |
| decay only | one interview | 3.71 | 3.716 | 3.716 |
| h0 = 0.02 | one interview | 3.54 | 3.541 | 3.541 |
| h0 = 0.05 | up to two interviews | 3.42 | 3.423 | 3.423 |
| h0 = 0.10 | one interview | 3.14 | 3.144 | 3.144 |

- With the full pool, every one of the 4,000 episodes matches the earlier simulator exactly (largest gap 4e-15) once its last-day interviews are added back at 0.05 each.
- Last-day interviews in the earlier simulator: 0 to 0.024 per episode, worth at most 0.0012 of score.
- 10 slots versus full pool: at most 0.0006 apart at every level (limit agreed: 0.02).
- Default rule behaviour (bar re-checked after each hire, no 12-offer cap) versus the earlier simulator's: at most +0.011.
- No hired candidate reached the 3-interview cap under any rule (limit agreed: 5%).
- Shape facts for the earlier tuned parameters: the bar is lower with more open roles in every tuned rule. The bar is not zero on the last day (falling-bar rule: 0.07 to 0.54); the rule's linear form does not force it.

**The 10-slot view hides candidates often.** The pool exceeds 10 on 53 to 87% of days, because a passed-over candidate is never removed. With poaching on, a hidden candidate has a higher survival-aware estimate than a shown one on 34 to 51% of days. It cost these naive-estimate rules nothing measurable, but it is untested for policies that use the survival-aware estimate.

**Follow-up, Nov 14 (`slot_view.txt`).** The same rules with the survival-aware estimate, on 2,000 development episodes at h0 = 0.02, 0.05, 0.10: the 10-slot view, a 20-slot view and the full pool give identical payoff in every episode, for both estimates and all four rule families. A hidden candidate outranks a shown one on 33 to 55% of days, but these rules never act that far down the list (a separate 500-episode check: payoffs start to differ only at 6 slots or fewer). The parameters were tuned for the naive estimate, so this says nothing yet about survival-aware rules tuned in week 2, or about PPO.

## Exact small instance

| T | Prototype | Choose whom to drop | Automatic drop | Difference |
| --- | --- | --- | --- | --- |
| 10 | 0.5838 | 0.58376 | 0.58367 | 0.016% |
| 12 | 0.6335 | 0.63348 | **0.63333** | 0.024% |
| 15 | 0.6947 | 0.69473 | 0.69453 | 0.029% |

State counts match the prototype exactly (86,194 / 165,558 / 357,279). The automatic-drop optimal policy simulated in our environment over 40,000 episodes gives 0.6308 +- 0.0041 against a computed 0.63333 (0.6 standard errors below). The random policy scores -0.010 on the same episodes.

## Belief

Quadrature gives 0.883 / 0.721 / 0.482 for a naive estimate of 1.0 after 7 days at h0 = 0.02 / 0.05 / 0.10, and reproduces the prototype check values 1.312 and 0.806. The corrected mean is calibrated against 1.5 million simulated candidates.

## Throughput (i7-13700F, one torch thread, 8 environments, about 100,000 steps)

| Run | Steps per second | 3 million steps |
| --- | --- | --- |
| Environment alone, random policy | 22,400 | |
| MaskablePPO, unrestricted actor, DummyVecEnv | 4,760 | 10.5 minutes |
| MaskablePPO, unrestricted actor, SubprocVecEnv | 4,850 | 10.3 minutes on 8 cores |
| MaskablePPO, index actor, DummyVecEnv | 3,720 | 13.4 minutes |
| MaskablePPO, unrestricted actor, small instance | 4,990 | 10.0 minutes |

About 370 final runs at 3 million steps come to roughly 65 to 85 core-hours, inside the plan's 45 to 80 estimate at the upper end. Random-policy episodes last only 16 steps; trained policies will have longer episodes, so steps per second may differ somewhat during real training.

## Also done

- Library check: MaskablePPO on CartPole-v1 reaches 500.0 over 50 evaluation episodes after 150,000 steps; all six required log columns are written.
- Index-structured actor: reordering candidates reorders the logits, one candidate's score ignores the others, masked actions get zero probability, logits are finite. Save and load is blocked by the Python problem above.
- Random policy at the primary setting (stress, h0 = 0.05), 2,000 development episodes: -0.48 +- 0.04.
