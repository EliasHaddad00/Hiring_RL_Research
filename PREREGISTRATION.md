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
  gets less than the rules. Open: what counts as one "configuration" for a rule (the earlier check
  used 360 evaluations per rule family); to be settled in week 2.
- **Training length.** 3 million steps per run *(planning default)*, then reset to at least twice the
  point where the curves flatten. Plateau test: the change in interquartile mean over the last 20%
  of training lies within its interval.
- **Logged for every run.** Evaluation payoff, policy loss, value loss, entropy, approximate KL,
  clip fraction.

## Settings

Exact small instance; stress with no pressure (h0 = 0, no acceptance decay); stress at h0 = 0, 0.02,
0.05 (primary), 0.10; calibrated scenario; optional: stress at h0 = 0.05 with one interview per day.
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

Exact small instance (PPO within 2% of the optimum); degenerate regimes (zero interview cost, zero
interview noise, no pressure, very high hazard); imitation of the exact optimal policy by each
architecture; a standard reference task (CartPole); simulator statistics (arrival, acceptance,
departure rates); upper bound at least as large as every evaluated policy at every setting.

## Deviations

*(empty)*
