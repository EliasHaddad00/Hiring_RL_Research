# Pre-registration: hiring as a race against poaching

**Status: DRAFT, not tagged.** Written Nov 12, 2026 for gate G1. Nothing here is binding until this
file is committed and tagged at gate G3 (Dec 2), before the final seeds run. Every value that was
a planning default is now marked *(final, Nov 20)* with what fixed it. After the tag,
anything not in this file is labelled exploratory, and changes are recorded under Deviations.

## Hypotheses

| | Hypothesis |
| --- | --- |
| H1a | On the exact small instance, the best tuned rule reaches at least 95% *(final, Nov 20: kept as written on Nov 12, before any rule was scored on the small instance)* of the exact optimum. |
| H1b | On the full problem, the best tuned rule reaches 85 to 91% of the Lagrangian upper bound, and the shortfall grows with the poaching level h0. |
| H2a | PPO's mean payoff is within 2% of the exact optimum on the small instance. |
| H2b | On the full problem, PPO's mean payoff is at least that of the best tuned rule. |
| H3 | The payoff of the unrestricted agent minus the index-structured agent is within the 2% margin at h0 = 0 and increases with h0. |
| H4a | Inside a fixed rule, replacing the naive estimate by the survival-aware estimate changes decisions on a measurable share of candidate-days and raises payoff at h0 = 0.05 and above. |
| H4b | For PPO with days waited in the observation, the two estimates give final payoffs within 2%, and the survival-aware agent reaches a given payoff in fewer steps. |

## Primary metric and comparison

- **Metric.** Mean unshaped payoff (sum of hired quality minus interview costs) over the 10,000
  evaluation episodes. Reported for every policy, also when training uses reward shaping.
- **Unit of analysis.** The training seed; each seed's score is its 10,000-episode mean.
- **Primary comparison.** PPO against the best tuned rule at the primary setting, stress scenario
  with h0 = 0.05 *(final, Nov 20: rules and PPO were tuned at this setting)*, paired on the same evaluation episodes.
- **How a PPO agent is scored** *(decided Nov 14)*. The action mask is always passed and the agent
  takes its most probable valid action (greedy). The evaluation callback that draws the training
  curves uses the same setting, so curves and final scores measure the same thing. The score with
  sampled actions is reported beside it and is not used for any verdict.
- **Reference points.** Random valid policy, best tuned rule, exact optimum under the
  automatic-drop rule (small instance), Lagrangian upper bound (full problem).

## Decision rule

Take the 95% interval of the mean paired difference and a margin of 2% *(final, Nov 20: unchanged since Nov 12 and used in every gate check)* of the
comparator's payoff. Interval entirely inside the margin: **equivalent**. Otherwise, entirely above
zero: **better**; entirely below zero: **worse**. Anything else: **inconclusive**, reported as such.
Equivalence takes precedence. Statistics: interquartile mean with bootstrap 95% intervals over
seeds, the interval on the paired difference as the headline, Welch's t-test alongside, and a
multiplicity correction over the primary comparisons.

## Seeds (experiments/configs/seeds.yaml)

| Pool | Seeds | Use |
| --- | --- | --- |
| Tuning training seeds | 101, 102, 103 (3 per configuration; *final, Nov 20: the tuning was run with them*) | Scoring PPO configurations |
| Final training seeds | 1001 to 1015 (15 per condition; 10 is the floor) | Final runs only |
| Tuning episodes | 2,000,000 to 2,003,999 (4,000) | Fitting rules; scoring configurations |
| Evaluation episodes | 3,000,000 to 3,009,999 (10,000) | Final paired evaluation only |
| Development episodes | 1,000,000 to 1,099,999 | Tests and gate checks; never reported as results |

No choice is made on the final training seeds or the evaluation episodes. **No seed is dropped.**

## Budgets

- **Tuning.** 40 configurations *(final, Nov 20: the searches were run with this budget)* for each rule and each PPO architecture; PPO never
  gets less than the rules. *(Nov 20)* A threshold rule family is in addition scored at the best
  vector of each family it contains, up to 3 evaluations; the price-directed rule gets none (see
  "Checks triggered before the tag").
- **One PPO configuration** *(decided Nov 15)*. Seeded random search over the ranges of the research
  plan, 40 configurations per architecture; configuration 1 is the starting values, as for the rules.
  Each configuration is trained with tuning seeds 101 to 103 for 30% of that architecture's final
  training length (the same fraction for every architecture) and scored as the mean of the three
  seeds. The five best are then trained at full length with the same seeds, and the winner is picked
  among those five. No search library is used. The same 40 sampled configurations are used for every
  architecture, so a difference between architectures cannot come from one drawing luckier
  settings. Every run is scored once, with greedy actions, on the 4,000 tuning episodes the rules
  were tuned on. *Departure from the research plan:* the plan tunes at the primary setting and on
  the small instance; we tune at the primary setting (h0 = 0.05) only, because every architecture
  already reaches 98.9 to 99.3% of the small-instance optimum with the starting values. The
  small-instance check is rerun on the tuned winners instead. Search-stage models are kept locally
  and not committed; the 15 full-length finalist models are committed
  (`experiments/scripts/tune_ppo.py`, results in `results/g3/`).
- **One rule configuration** *(decided Nov 14)* is one full parameter vector of a rule family, scored
  on all 4,000 tuning episodes. Each rule family gets 40 per estimate (naive, survival-aware) and
  per setting. Configuration 1 is the earlier tuned vector (`results/g1/v3_fullprec.json`) where one
  exists; the rest come from a seeded local search.
- **Robustness search for the rules** *(decided Nov 14)*. At the primary setting every rule family
  also gets a 360-evaluation search, with both estimates, on the tuning episodes. If the best rule
  gains more than the 2% margin over its 40-configuration score, this is reported and the stronger
  rule becomes the comparator.
- **Training length** *(decided Nov 15)*. 3 million steps for the index actor and 10 million for
  the unrestricted actor. Rule set in advance: if the index actor gained under 0.01 from 3 to 10
  million steps at the primary setting, use 3 and 10 million; otherwise 10 million for every actor.
  Measured on tuning seeds 101 to 103 and 4,000 development episodes
  (`results/g2/stress_10m_scores.txt`): index +0.006 (interval -0.010 to +0.021, episode noise
  only), unrestricted +0.022 (+0.008 to +0.037). The index actor reaches its 3-million-step score
  by about 1.5 million steps. The unrestricted actor is not shown to be flat at 10 million steps
  (plateau test: inconclusive), so "twice past the plateau" is established for neither actor by the
  plateau test; the final runs repeat that test with 15 seeds. The pooled variant *(decided Nov 15)*:
  10 million steps. Without pressure it was 0.14 below the best rule at 3 million steps (seed 101)
  and within 0.04 at 10 million (seeds 101 to 103); at the primary setting it scores 3.575 to 3.585
  at 10 million (`results/g2/extra_runs.txt`). **Doubling test** *(decided Nov 15)*: the saved-model
  plateau test cannot separate "still learning" from "settling as the learning rate falls to zero",
  so for the unrestricted and pooled actors the plateau evidence is the gain from 10 to 20 million
  steps at the primary setting, seeds 101 to 103, on 4,000 development episodes. If its 95% interval
  lies inside +-1% of the payoff, 10 million steps stand; if not, the gain and the cost of 20 million
  are reported before the length is fixed. This is the test the index actor passed between 3 and 10
  million steps. The 80% model test is reported beside it as a description only.
  *Result (Nov 16, `results/g2/doubling.txt`):* unrestricted +0.011 (interval -0.008 to +0.030),
  pooled +0.001 (-0.022 to +0.023); both inside the band of about +-0.036, so both are flat and 10
  million steps stand. For description: the 80% model test on the 20-million-step runs gives
  unrestricted +0.015 (flat) and pooled +0.023 (inconclusive); the models at 16 of 20 million steps
  score at or below the finished 10-million-step models, which is what settling under a falling
  learning rate would look like. Earlier text, superseded by the lengths above: 3 million steps per run, then reset to at least twice the point where the curves flatten. Plateau test *(wording fixed Nov 15)*: a curve is flat if the 95%
  interval of the change in interquartile mean over the last 20% of training lies inside +-1% of the
  final payoff. An interval that reaches outside that band is reported as "inconclusive" (if it
  contains zero) or "changing" (if it does not), never as flat. The change is the last evaluation
  minus the evaluation at 80% of training, paired by episode; the interval is a bootstrap over runs
  and episodes (`experiments/scripts/check_plateau.py`). *Measurement changed Nov 15, rule
  unchanged:* every run saves its model at 80% of training (`model_80.zip`); the change is the
  final model minus that model, both scored on the same 4,000 development episodes
  (`check_plateau_checkpoints.py`), instead of on the 300 curve episodes. Reason: with 300 episodes
  the interval was wider than the band (about +-0.06 against +-0.035 with three seeds), so the test
  could not return "flat". The curve keeps 300 episodes per evaluation. Runs made before this
  change have no 80% model and keep the curve-based test.
- **Logged for every run.** Evaluation payoff, policy loss, value loss, entropy, approximate KL,
  clip fraction.

## Settings

Exact small instance; stress with no pressure (h0 = 0, no acceptance decay); stress at h0 = 0, 0.02,
0.05 (primary), 0.10; calibrated scenario; optional: stress at h0 = 0.05 with one interview per day.
Calibrated scenario *(decided Nov 14; frozen Nov 15)*: as stress,
with acceptance 0.80 x exp(-0.003 x days), h0 = 0.02, interview noise 2.2 and resume noise 2.4
(base acceptance was 0.85 until Nov 14; see "Changes before the tag").
Realised acceptance under the best tuned rule on the tuning episodes is reported against the
76 to 84% benchmark before the file is frozen. Result: 79.3% at base acceptance 0.80
(`results/g2/calibrated_check.txt`); the scenario is frozen with these values.
Rule form *(decided Nov 14, revised the same day; see "Changes before the tag")*: every threshold
rule, with either estimate, offers to anyone with a positive estimate on the last day. The
price-directed rule's naive version uses its own tables, solved
under the naive belief (tonight's loss risk averaged over that belief), and is tuned separately.
Simulator conventions are those in `experiments/hiring/env.py` (strict offer-then-interview phases,
interview results visible next morning, 3 interviews per candidate, 10 slots *(final: decided Nov 15, confirmed Nov 20)*,
last-day interviews masked, automatic drop on the small instance).

## Planned ablations

Must run: one-at-a-time sweeps of discount, entropy coefficient, learning rate and rollout size
(three seeds each); naive versus survival-aware estimate; pooled variant; reward shaping on or off;
end-day logit from global features versus a pooled summary.
Optional, only if gate G3 is met on time: masking versus an invalid-action penalty; network width;
20 slots; posterior-mean reward; wrong assumed h0; one interview per day.
The runs for the must-run items, with their seeds, are under "Final run list".

## Final run list (written Nov 21, from the owner's decisions of Nov 20)

Every run uses the frozen configuration of its actor (`experiments/configs/ppo_final_<actor>.yaml`:
unrestricted 8, index 18, pooled 8) and final training seeds 1001 to 1015. The list is
`experiments/configs/final_runs.yaml`; `experiments/scripts/final_runs.py --list` prints this
table and `--launch` runs it. Setting names: `small` = exact small instance, `none` = no pressure,
`h0_x` = stress at that h0, `calibrated` = calibrated scenario.

**Main runs.** Three actors at every setting, with the survival-aware estimate in the observation
(the estimate used in tuning). **H4b arm.** The same three actors at h0 = 0.05 only, with the naive
estimate; no other setting gets a naive-estimate PPO run.

| Group | Setting | Estimate | Actor | Steps | Seeds | Runs | Run-hours |
| --- | --- | --- | --- | --- | --- | --- | --- |
| primary | h0_0.05 | survival | index | 3 million | 1001 to 1015 | 15 | 18 |
| primary | h0_0.05 | survival | unrestricted | 10 million | 1001 to 1015 | 15 | 30 |
| primary | h0_0.05 | survival | pooled | 10 million | 1001 to 1015 | 15 | 68 |
| other settings | small | survival | index | 3 million | 1001 to 1015 | 15 | 6 |
| other settings | small | survival | unrestricted | 10 million | 1001 to 1015 | 15 | 21 |
| other settings | small | survival | pooled | 10 million | 1001 to 1015 | 15 | 34 |
| other settings | none | survival | index | 3 million | 1001 to 1015 | 15 | 18 |
| other settings | none | survival | unrestricted | 10 million | 1001 to 1015 | 15 | 30 |
| other settings | none | survival | pooled | 10 million | 1001 to 1015 | 15 | 68 |
| other settings | h0_0 | survival | index | 3 million | 1001 to 1015 | 15 | 18 |
| other settings | h0_0 | survival | unrestricted | 10 million | 1001 to 1015 | 15 | 30 |
| other settings | h0_0 | survival | pooled | 10 million | 1001 to 1015 | 15 | 68 |
| other settings | h0_0.02 | survival | index | 3 million | 1001 to 1015 | 15 | 18 |
| other settings | h0_0.02 | survival | unrestricted | 10 million | 1001 to 1015 | 15 | 30 |
| other settings | h0_0.02 | survival | pooled | 10 million | 1001 to 1015 | 15 | 68 |
| other settings | h0_0.10 | survival | index | 3 million | 1001 to 1015 | 15 | 18 |
| other settings | h0_0.10 | survival | unrestricted | 10 million | 1001 to 1015 | 15 | 30 |
| other settings | h0_0.10 | survival | pooled | 10 million | 1001 to 1015 | 15 | 68 |
| other settings | calibrated | survival | index | 3 million | 1001 to 1015 | 15 | 18 |
| other settings | calibrated | survival | unrestricted | 10 million | 1001 to 1015 | 15 | 30 |
| other settings | calibrated | survival | pooled | 10 million | 1001 to 1015 | 15 | 68 |
| belief arm (H4b) | h0_0.05 | naive | index | 3 million | 1001 to 1015 | 15 | 18 |
| belief arm (H4b) | h0_0.05 | naive | unrestricted | 10 million | 1001 to 1015 | 15 | 30 |
| belief arm (H4b) | h0_0.05 | naive | pooled | 10 million | 1001 to 1015 | 15 | 68 |

Total: 360 runs, 870 run-hours; with 6 runs at a time about 145 hours (6.0 days) of machine time. An estimate from the tuning runs' times.
The hours per run are the measured times of the same configurations on tuning seeds: index 1.20,
unrestricted 1.97, pooled 4.54 on the full problem (`results/g3/ppo_tuning/finals`); 0.40, 1.39,
2.25 on the small instance (`results/g3/small_check`). The first two full-problem times were
measured with about 12 runs on the machine and the others with 6, so the total leans high.

**Machine time is above the owner's line of about 4 days.** Nothing has been cut. The list above
is the list as specified; the owner chooses whether to run it as it is or change it, and the choice
is recorded here before the tag.

**Must-run ablations: PROPOSAL, NOT YET DECIDED BY THE OWNER.** Two items of the list under
"Planned ablations" are already in the table: naive versus survival-aware estimate (the H4b arm)
and the pooled variant (the pooled actor). The other three need choices this file had not fixed
(values, actor, seeds), so they are written here as a proposal and are not binding until the
owner confirms them:

| Ablation | Proposed runs | Seeds | Runs | Run-hours |
| --- | --- | --- | --- | --- |
| Sweeps, one value changed at a time from the frozen index configuration, h0 = 0.05: discount 0.99 and 0.999 (frozen 1.0); entropy coefficient one third and three times the frozen value; learning rate one third and three times; rollout size 256 and 1024 (frozen 512) | index actor, 8 arms | 1001 to 1003 | 24 | 29 |
| Reward shaping on (potential-based; off is the main run) | index actor, h0 = 0.05 | 1001 to 1003 | 3 | 4 |
| End-day logit from the other source (global features or pooled summary; the main run has one) | index actor, h0 = 0.05 | 1001 to 1003 | 3 | 4 |

The unchanged arm of each ablation is the main run with the same seed. As proposed: 30 runs, about
36 run-hours, about 6 hours of machine time. Reward shaping and the second end-day variant are not
implemented yet; unshaped return is reported in every case.

**Model storage (decided Nov 20).** Final-run models (`model.zip`, `model_80.zip`) are not put in
git: `results/final/**/model*.zip` is in `.gitignore`, the files stay in `results/final/` on the
training machine and are attached to a GitHub Release afterwards. Everything else in a run folder
is committed: `run.json`, `progress.csv`, `evaluations.npz`, `log.txt`, and the per-episode
payoffs once the run is evaluated. The finalist models already in git stay.

**Launch order (decided Nov 20).** Primary setting (h0 = 0.05) first, then the other settings in
the order of the table, then the H4b arm, then the ablations. Six runs at a time: every chosen
configuration has width 256, and the machine ran out of memory with 12 during tuning. A finished
run (one with `run.json`) is never rerun; a stopped launch continues where it was.

**Not before the owner says so.** The tag does not unlock anything: the evaluation episodes and
seeds 1001 to 1015 stay locked until the owner gives the word after the tag.

## Known-answer tests

Exact small instance (PPO within 2% of the optimum, measured by the exact value of the greedy
policy: a backward pass over every state, with no sampling, must give at least 0.62067, which is
98% of 0.63333 (`experiments/scripts/exact_values.py`; decided Nov 15). The paired interval on
40,000 development episodes, PPO minus the exact optimal policy with its lower end above -0.0127,
is reported beside it as a cross-check; it was the pass rule from Nov 14 to Nov 15 and failed two
agents on episode noise alone); degenerate regimes (zero interview cost, zero
interview noise, no pressure, very high hazard); imitation of the exact optimal policy by each
architecture; a standard reference task (CartPole); simulator statistics (arrival, acceptance,
departure rates); upper bound at least as large as every evaluated policy at every setting.

## Checks triggered before the tag

- **Interview cap (Nov 14).** The rule "if more than 5% of hired candidates reach the cap of 3
  interviews, rerun with a cap of 6" fired for the untuned price-directed rule: 18.8%, 12.8% and
  5.7% of hires at h0 = 0, 0.02 and 0.05. Rerun with a cap of 6 on the same 4,000 development
  episodes: payoff changes by at most 0.008 (the 2% margin is about 0.07) and the upper bound by
  at most 0.001 (`results/g2/price_rule.txt`, `results/g2/bounds.txt`). Decision: the cap stays
  at 3. The check is repeated once on a trained full-problem PPO agent, which may use interviews
  differently from the rule.
- **Interview cap, trained agent (decided Nov 20).** The repeat on a trained agent: unrestricted
  actor, no pressure, 10 million steps, seed 101, 12.5% of hires at the cap of 3. An agent trained
  with a cap of 6 scores 3.916 against 3.971 with a cap of 3: -0.055, interval -0.078 to -0.032,
  4,000 development episodes, no hire at its cap (`results/g2/cap6_check.txt`). This is outside the
  0.02 closing rule, in the direction where a higher cap does not help. One seed each, so the
  difference includes training luck. Decision (owner): the cap stays at 3; the result is recorded
  as a finding.
- **PPO tuning ranges (decided Nov 20).** The plan widens a range whose best value sits on its
  edge. In the 30%-length search this fires for every actor (`results/g3/ppo_tuning_search.txt`),
  and the only edge that clearly wins is a discount of 1.0, which cannot be widened. Decision
  (owner): no widening on the search, with one condition: if the configuration chosen for an actor
  after the full-length runs has a value on the edge of a range that can be widened, the question
  is reopened for that actor before the configurations are frozen.
- **PPO tuning ranges, reopened and closed (decided Nov 20).** The condition fired: the best
  full-length configuration of every actor has values on an edge that can be widened
  (unrestricted and pooled 8: 10 epochs, clip 0.1, width 256; index 18: batch 64, width 256).
  Decision (owner): no widening. Reason: across the 40 search configurations the edge values do
  not win. Median search score by value (`results/g3/search_medians.txt`, from
  `experiments/scripts/search_medians.py`; unrestricted / index / pooled):

  | Value on an edge | Its median | Compared with | Its median |
  | --- | --- | --- | --- |
  | Width 256 | 3.456 / 3.361 / 3.526 | Width 128 | 3.485 / 3.417 / 3.548 |
  | Batch 64 | 3.461 / 3.437 / 3.523 | Batch 256 | 3.484 / 3.498 / 3.551 |
  | Clip 0.1 | 3.460 / 3.283 / 3.516 | Clip 0.2 | 3.459 / 3.464 / 3.539 |
  | 10 epochs | 3.426 / 3.435 / 3.508 | 4 epochs | 3.443 / 3.347 / 3.523 |

  Width 128 beats 256 and batch 256 beats 64 for every actor. Clip 0.1 is below 0.2 for the index
  and pooled actors and level with it for the unrestricted actor (+0.001). Ten epochs help only
  the index actor. The top finalists are tied within noise: pooled 8, 10 and 30 within 0.0011,
  index 18, 29 and 1 within 0.0045, against a seed-to-seed standard deviation of about 0.016
  within a configuration. This is a descriptive check, not a controlled test: every configuration
  changes all hyperparameters at once, so a value's median also reflects what it was drawn with.
- **Chosen configurations (decided Nov 20).** By the rule under "Budgets" (best mean over seeds
  101 to 103 among the five full-length finalists, `results/g3/ppo_tuning_finals.txt`):
  unrestricted 8 (3.535), index 18 (3.573), pooled 8 (3.586); the tuned rule scores 3.594 on the
  same tuning episodes. Frozen in `experiments/configs/ppo_final_<actor>.yaml`.
- **Small-instance check on the chosen configurations (rule written Nov 20, before the runs).**
  Each chosen configuration is trained on the small instance with seeds 101 to 103 at its own
  length (index 3 million steps, unrestricted and pooled 10 million) and scored by the exact
  value of its greedy policy (`exact_values.py`). Pass mark as before: at least 0.62067 (98% of
  0.63333). A configuration passes if the mean of its three exact values reaches the mark (a
  configuration's score is its mean over seeds everywhere else in this file); all three values
  are reported, and a single seed below the mark is reported to the owner. Fallback, fixed before running: if a
  chosen configuration fails, the next-ranked finalist of that actor that passes is used instead,
  and the owner is told before anything else is done.
  **Result (Nov 21): all three pass; the fallback did not fire.** Every one of the nine runs is
  above the mark (`results/g3/small_check_exact_values.txt`, runs in `results/g3/small_check/`):

  | Actor (configuration) | Seed 101 | Seed 102 | Seed 103 | Mean | Share of the optimum |
  | --- | --- | --- | --- | --- | --- |
  | Index (18) | 0.62548 | 0.62733 | 0.62879 | 0.62720 | 99.03% |
  | Unrestricted (8) | 0.62758 | 0.62929 | 0.62927 | 0.62871 | 99.27% |
  | Pooled (8) | 0.63013 | 0.63041 | 0.62864 | 0.62972 | 99.43% |

  Tuning seeds: a gate check, not a reported result.
- **Rule search misses contained rules (found and fixed Nov 20).** Each threshold
  family contains the narrower ones (one interview with interviews off is the falling bar; a
  falling bar with no fall is the static rule), so its best score cannot be lower. The local
  search starts from the earlier tuned vector and does not always reach those regions. With 40
  configurations a wider family scored below a family it contains in 8 of 48 threshold searches
  (calibrated: 3; h0 = 0.02: 1; h0 = 0.10: 4); with 360 configurations in 3 of the 8 calibrated
  threshold searches, the largest being the naive one-interview rule at 2.018 against the naive
  static rule at 2.129. Checked directly: the naive one-interview and falling-bar rules with the
  extra feature switched off score exactly the static rule's 2.129 on the 4,000 tuning episodes.
  - *Fix (owner's decision).* After its search, every threshold family is also scored at the best
    vector of each family it contains, at every setting and for both estimates, and keeps the best
    (`hiring/tuning.py`, `seed_contained`; a test requires identical per-episode payoffs from a
    rule and the same rule written as a wider family). This adds up to 3 evaluations beyond the 40
    (or 360) of a threshold search. The price-directed rule, the rule PPO is compared with, gets
    none, so PPO's 40 configurations are not fewer than its comparator's.
  - *Effect.* 8 of the 60 searches at 40 configurations and 3 of the 20 at 360 rose; none fell;
    no wider family is below a contained one any more. The best rule and its score are unchanged
    at every setting (`results/g2/tuning_40.txt`, `tuning_360.txt`, `g2_check.txt`: no problems).
  - *The earlier calibrated gaps were overstated.* Survival-aware minus naive estimate, paired,
    4,000 development episodes, 360 configurations (`results/g2/robustness_calibrated.txt`):

    | Family | Before the fix | After the fix (95% interval) |
    | --- | --- | --- |
    | Static | +0.038 | +0.038 (+0.009 to +0.067) |
    | Falling bar | +0.243 | +0.097 (+0.057 to +0.136) |
    | One interview | +0.319 | +0.192 (+0.143 to +0.241) |
    | Up to two interviews | +0.105 | +0.120 (+0.085 to +0.155) |
    | Price-directed | +0.052 | +0.052 (+0.021 to +0.084) |

    The +0.243 and +0.319 were overstated because the naive searches for those two families had
    not reached the static rule they contain: their naive scores were too low, not their
    survival-aware scores too high. The two-interview gap rose slightly because its
    survival-aware search had not reached the one-interview rule. After the fix the naive
    falling-bar and one-interview rules are the static rule (no fall, no interviews).
  - *Slot view.* The one firing of the 0.02 stop rule (naive one-interview rule, calibrated,
    "Changes before the tag") concerned a rule that is now replaced; every rule and setting is
    within 0.004. The decision (10 slots) stands.

## Changes before the tag

These were made while this file was still a draft, so they are not deviations; they are recorded
because each reverses something written here earlier.

- **Last-day rule applies to both estimates (Nov 14, evening).** The first decision of Nov 14
  forced the zero bar on the last day only for rules using the survival-aware estimate. That
  biased the comparison of the two estimates (research question 4): at h0 = 0 nobody is poached and
  the two estimates are equal, yet the tuned "survival-aware" one-interview rule scored 3.815
  against 3.706 for the naive one on 4,000 development episodes (`results/g2/g2_check.txt` at commit e233f7a). With equal estimates, the difference can only come from the last-day rule and from separate tuning; the
  retuned scores will show the split. Now the rule applies to both, the naive threshold rules were retuned at every
  setting, and a test requires identical per-episode payoffs from the two estimates at h0 = 0
  (`experiments/tests/test_tuning.py`). The price-directed rule is not affected: its tables handle
  the last day for both versions.
- **Calibrated base acceptance 0.85 to 0.80 (Nov 14).** At 0.85 the realised acceptance under the
  best tuned rule was 84.6% on the tuning episodes, outside the 76 to 84% benchmark. 0.80 is below
  the 0.85 to 0.90 range in our notes; it is chosen to put realised acceptance near the middle of
  the benchmark. All calibrated rules were retuned. The realised figure at 0.80 is in
  `results/g2/calibrated_check.txt`.

- **Slot view (Nov 15).** The stop rule "full pool against 10 slots differs by more than 0.02"
  fired once: the naive one-interview rule on the calibrated scenario scores 0.0201 lower with the
  full pool than with 10 slots (interval -0.030 to -0.010, 2,000 development episodes, after the
  retune under the new last-day rule). The 10-slot view costs that rule nothing; every other rule
  and setting is within 0.004. Decision: 10 slots stay.

## Deviations

*(empty)*
