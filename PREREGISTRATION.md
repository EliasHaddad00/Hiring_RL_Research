# Pre-registration: hiring as a race against poaching

**Status: DRAFT, not tagged.** Written Nov 12, 2026 for gate G1. Nothing here is binding until this
file is committed and tagged at gate G3 (Dec 2), before the final seeds run. Items marked
*(planning default)* are this plan's choices and may still change before the tag. After the tag,
anything not in this file is labelled exploratory, and changes are recorded under Deviations.

## Hypotheses

| | Hypothesis |
| --- | --- |
| H1a | On the exact small instance, the best tuned rule reaches at least 95% *(planning default)* of the exact optimum. |
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
  with h0 = 0.05 *(planning default)*, paired on the same evaluation episodes.
- **How a PPO agent is scored** *(decided Nov 14)*. The action mask is always passed and the agent
  takes its most probable valid action (greedy). The evaluation callback that draws the training
  curves uses the same setting, so curves and final scores measure the same thing. The score with
  sampled actions is reported beside it and is not used for any verdict.
- **Reference points.** Random valid policy, best tuned rule, exact optimum under the
  automatic-drop rule (small instance), Lagrangian upper bound (full problem).

## Decision rule

Take the 95% interval of the mean paired difference and a margin of 2% *(planning default)* of the
comparator's payoff. Interval entirely inside the margin: **equivalent**. Otherwise, entirely above
zero: **better**; entirely below zero: **worse**. Anything else: **inconclusive**, reported as such.
Equivalence takes precedence. Statistics: interquartile mean with bootstrap 95% intervals over
seeds, the interval on the paired difference as the headline, Welch's t-test alongside, and a
multiplicity correction over the primary comparisons.

## Seeds (experiments/configs/seeds.yaml)

| Pool | Seeds | Use |
| --- | --- | --- |
| Tuning training seeds | 101, 102, 103 (3 per configuration, *planning default*) | Scoring PPO configurations |
| Final training seeds | 1001 to 1015 (15 per condition; 10 is the floor) | Final runs only |
| Tuning episodes | 2,000,000 to 2,003,999 (4,000) | Fitting rules; scoring configurations |
| Evaluation episodes | 3,000,000 to 3,009,999 (10,000) | Final paired evaluation only |
| Development episodes | 1,000,000 to 1,099,999 | Tests and gate checks; never reported as results |

No choice is made on the final training seeds or the evaluation episodes. **No seed is dropped.**

## Budgets

- **Tuning.** 40 configurations *(planning default)* for each rule and each PPO architecture; PPO never
  gets less than the rules.
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
  plateau test; the final runs repeat that test with 15 seeds. The pooled variant uses the length
  of the index actor unless its own check says otherwise. Earlier text: 3 million steps per run
  *(planning default)*, then reset to at least twice the point where the curves flatten. Plateau test *(wording fixed Nov 15)*: a curve is flat if the 95%
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
interview results visible next morning, 3 interviews per candidate, 10 slots *(planning default)*,
last-day interviews masked, automatic drop on the small instance).

## Planned ablations

Must run: one-at-a-time sweeps of discount, entropy coefficient, learning rate and rollout size
(three seeds each); naive versus survival-aware estimate; pooled variant; reward shaping on or off;
end-day logit from global features versus a pooled summary.
Optional, only if gate G3 is met on time: masking versus an invalid-action penalty; network width;
20 slots; posterior-mean reward; wrong assumed h0; one interview per day.

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
