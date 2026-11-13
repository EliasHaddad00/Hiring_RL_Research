# Research plan: hiring as a race against poaching

Nov 11, 2026 · @EH

## Purpose and what done looks like

The plan's main move is to build the yardsticks before the agents: an exact optimum on a small instance, an upper bound on the full problem and a stronger theory-based rule, all in place by Nov 25. Every later PPO number is then read against a known ceiling.

The proposal says what the project is. This plan says what gets built, why, what we expect to see and the test that ends each piece of work. It is the working version; the full plan with every source and derivation is `plans/Hiring race RL research plan.md` in the project files.

**What the research changed in our expectations**

- **Headroom is narrow.** A prototype bound puts the ceiling near 4.17 at h0 = 0.02, against 3.54 for our best rule and 3.72 for a prototype price-directed rule. The optimum is at most about 12 to 18% above today's rules.
- **A tie is the likely result on the full problem.** We plan for it: a 2% equivalence margin is fixed in advance, and three of the four research questions give reportable findings however PPO training goes.
- **Our time-pressure parameters are harsher than real hiring.** We keep them as a stress scenario and add a calibrated scenario beside it.

"Prototype" below means a number from the researchers' own short scripts, not yet reproduced in our simulator. Reproducing them is week 1 and 2 work.

**Definition of done**

| # | Done when | Serves |
| --- | --- | --- |
| 1 | Each research question has a verdict from a decision rule written before the final runs, with a 95% confidence interval | Completion, Contribution |
| 2 | Every final PPO condition has 15 training seeds (10 minimum), none dropped, with curves for return, policy loss, value loss and entropy that meet a stated plateau test | Completion |
| 3 | A tuning table and the planned ablations are reported for PPO and for every rule | Completion |
| 4 | All known-answer tests pass and sit in the test suite | Completion |
| 5 | Hurdle experiments and failure-mode analysis are complete, whatever the result | Completion, Contribution |
| 6 | One command regenerates every figure and table; a clean clone reproduces them; a tagged release matches the PDF | Completion |
| 7 | The paper is at most five pages in ICML format and uses the hedged novelty wording | Writing, Novelty |

## Research questions, hypotheses and expectations

Each question has a hypothesis the data can refute and an outcome we expect in advance.

| Question | Hypothesis | What we expect, and why | Main evidence |
| --- | --- | --- | --- |
| RQ1. How close do simple rules get to the optimum? | The best tuned rule reaches at least 95% of the exact optimum on the small instance, and 85 to 91% of the upper bound on the full problem, with the shortfall growing as poaching rises | Likely true. A prototype price-directed rule sits 9, 11 and 12% below the bound at h0 = 0, 0.02 and 0.05 | Percent of exact optimum; percent of bound |
| RQ2. Does RL reach the optimum and beat the best rule? | PPO is within 2% of the exact optimum on the small instance, and at least as good as the best rule on the full problem | Small instance: expected to pass, as a positive control. Full problem: expected to end as "equivalent within 2%"; a clear win would be the surprise | Gap to optimum; interval on PPO minus best rule |
| RQ3. Does index structure survive departures? | The unrestricted agent's advantage over the index agent is within 2% at h0 = 0 and grows with h0 | Small gaps at most settings; most likely to open at h0 = 0.10 or when interview slots are scarce. The index agent may even lead, because structured learners often learn faster | Gap against h0; how well each network can copy the exact optimal policy |
| RQ4. Does the belief correction change decisions? | Inside a fixed rule, swapping in the corrected estimate changes decisions and raises payoff at h0 of 0.05 and above. For PPO that sees days waited, final payoff is within 2% either way, with faster learning for the corrected version | Likely for rules: the bias (1.0 is really 0.71 at h0 = 0.05) is as large as the width of the "interview" zone. For PPO, a network can learn the correction from days waited | Share of decisions that change; payoff difference; training steps to a target |

**Two confounds the plan removes**

- **Architecture versus optimization.** A gap between the two agents could come from training noise. Both get the same critic, tuning budget and seeds, and both are tested on the exactly solved instance.
- **Pool awareness versus symmetry.** The unrestricted network differs from the index network in two ways: it sees the whole pool, and it does not share weights across candidates. A third "pooled" variant keeps the weight sharing and adds only a summary of the pool, which separates the two. It has not been tested yet.

## What gets built

Eight components. Each has a reason to exist and a test that says it is finished.

| Component | What it is | Why we need it | Accepted when |
| --- | --- | --- | --- |
| 1. Environment | One Gymnasium class with an action mask; 2N+1 actions (interview or offer for each of N = 10 shown candidates, plus "end day"); the same class runs the small instance | Everything else reads it | Environment checks pass; same seed gives same episode; arrival, acceptance and departure rates match their settings; our earlier rule scores reproduce within 0.05 |
| 2. Two parameter sets | A stress scenario (the current values) and a calibrated scenario matched to hiring benchmarks | Answers "is this realistic?" and tests whether weaker time pressure makes simple rules enough | Both frozen in config files; fill rate, acceptance rate and interviews per hire reported for each |
| 3. Belief module | Exact quality estimate that accounts for strong candidates leaving first; also returns the naive estimate | Used by the corrected agent and rules, the exact solver and the bound | Reproduces our 0.87, 0.71, 0.47 figures; passes a check against simulated survivors |
| 4. Rule baselines | Fixed threshold, falling-bar threshold, one-interview rule, price-directed index rule, random policy; each with both estimates | Answer RQ1, set the bar for RQ2, carry the clean test of RQ4 | Tuned on a separate episode pool with the same budget as PPO; known shape facts hold (bar lower with more open roles, zero on the last day) |
| 5. Exact small instance | One role, 12 days, one interview slot, at most two candidates in the pool, coarsened signals, solved by dynamic programming | Gives a true optimum, a positive control for PPO, and the exact cost of each mistake | Our port reproduces the prototype optimum of 0.6335 (solved in about 3 seconds); simulated payoff of the optimal policy matches it |
| 6. Upper bound | A Lagrangian bound: put a price on each hire, solve one candidate at a time | A ceiling for the full problem, where no exact optimum is possible | Regenerated for every setting; never below any evaluated policy; stable on a finer grid |
| 7. Two agents | MaskablePPO with an unrestricted actor, and with an index-structured actor (one shared network scores each candidate); identical critic | RQ2 and RQ3 | Unit tests pass (candidate order does not matter, masked actions get zero probability); solves a tiny instance in minutes; at least one agent within 2% of the exact optimum |
| 8. Ablations | Discount, entropy bonus, learning rate, rollout size; naive versus corrected estimate; pooled variant; reward shaping on or off | Each rules out one explanation for the result | List fixed before the final runs |

**Prototype yardsticks, to be reproduced**

| Poaching level | Upper bound | Price-directed rule | Our best one-interview rule |
| --- | --- | --- | --- |
| h0 = 0 | 4.31 | 3.91 | 3.71 |
| h0 = 0.02 | 4.17 | 3.72 | 3.54 |
| h0 = 0.05 | 4.00 | 3.52 | 3.42 |

The first two columns come from a researcher's simulator and the third from ours, so the 0.1 to 0.2 lead of the price-directed rule is a lead to check, not a result. If it holds, the rule PPO must beat moves up.

**Stress and calibrated scenarios**

| Parameter | Stress (current) | Calibrated | Basis for the calibrated value |
| --- | --- | --- | --- |
| Horizon, roles, arrivals, interview limit | 60 days, 3, 1.5 per day, 3 per day | Same | Supported: time to fill is 60 to 61 days at large employers ([SHRM 2025](https://www.shrm.org/content/dam/en/shrm/research/2025-recruiting-benchmarking-report.pdf)) |
| Offer acceptance | 0.9 x exp(-0.03 x days) | 0.9 x exp(-0.003 x days) | Inference: real acceptance is 76 to 84% ([Gem 2025](https://lp.gem.com/rs/972-IVV-330/images/2025%20Recruiting%20Benchmarks%20-%20Gem.pdf?version=0)); the current curve gives 37% at day 30 |
| Poaching h0 | 0 to 0.10 | 0.02 | Weak evidence from stated candidate intentions ([Cronofy 2024](https://www.cronofy.com/reports/candidate-expectations-report-2024)) |
| Interview noise | 1.0 | 2.2 | Secondary source on interview validity |
| Resume noise | 1.5 | 2.4 | Secondary source; our choice within a range |

The quality slope of 0.6, the exponential form and the interview cost have no empirical basis and will be stated as limitations. The claim that top candidates are gone in ten days traces to an unsupported assertion and will not be cited.

**Implementation rules that prevent silent failures**

- The deadline ends the episode as "terminated", not "truncated".
- Features are scaled by hand inside the environment, identically for every candidate slot. Automatic observation normalization would break weight sharing.
- "End day" is always a valid action, and the environment checks that at least one action is valid.
- Masks are always passed at evaluation; leaving them out raises no error.
- The index actor must replace the library's final dense layer, which would otherwise mix candidates.
- Start with a discount of 0.999. At 0.99 over 200 steps only 13% of a final reward survives.

Library versions verified to work together: Gymnasium 1.4.0, Stable-Baselines3 2.9.2, SB3-Contrib 2.9.2. The starting hyperparameters and search ranges are in the full plan.

## Experiment protocol

Seeds, budgets and statistics are fixed before the final runs, following the standard guidance on RL evaluation ([Agarwal et al. 2021](https://arxiv.org/pdf/2108.13264), [Patterson et al. 2024](https://arxiv.org/pdf/2304.01315), [Henderson et al. 2018](https://ar5iv.labs.arxiv.org/html/1709.06560)). Numbers marked "default" are this plan's choices and can change until they are frozen at the Dec 2 gate.

| Item | Rule | Why |
| --- | --- | --- |
| Seed pools | Three separate pools written to a config file: tuning seeds (3 per configuration), final training seeds (15 per condition), and episode seeds split into 4,000 for tuning and 10,000 for evaluation | Picking the best of many noisy configurations and scoring it on the same seeds inflates results |
| Evaluation | Every policy runs on the same 10,000 episodes, so comparisons are paired | 10,000 episodes pin a policy's mean to about 0.7%; pairing sharpens differences further |
| Unit of analysis | The training seed. Each seed's score is its 10,000-episode mean | The claim is about PPO as a procedure, not one lucky run |
| Tuning budget | 40 configurations for each rule and each PPO architecture (default); PPO never gets less than the rules | Removes "PPO was under-tuned" as an objection |
| Training length | 3 million steps per run (default), then reset to at least twice the point where curves flatten | No source says how many steps this task needs |
| Statistics | Interquartile mean with bootstrap 95% intervals over seeds; interval on the paired difference as the headline; Welch's t-test alongside, since we have under 20 seeds | Recommended for experiments with few runs |
| Pre-registration | A one-page `PREREGISTRATION.md` with hypotheses, the decision rule, seed lists and budgets, committed and tagged before the final seeds run | Turns whichever verdict arrives into a planned result |

**Settings and runs**

| Setting | Role | PPO conditions (15 seeds each) |
| --- | --- | --- |
| Exact small instance | Positive control; true optimum for RQ1 | Index, unrestricted |
| Stress, no pressure at all | Known-answer check | 5 seeds per architecture |
| Stress, h0 = 0 | RQ3 baseline | Index, unrestricted, pooled |
| Stress, h0 = 0.02 | RQ2 to RQ4 | Index and unrestricted, each with naive and corrected estimate; pooled |
| Stress, h0 = 0.05 (primary, default) | RQ2 to RQ4; tuning; learning-curve figure | Same |
| Stress, h0 = 0.10 | Extreme for RQ3 and RQ4 | Same |
| Calibrated | Realism check | Index and unrestricted, each with both estimates |
| Stress, one interview per day | Optional slot-scarcity test | Optional |

Rules (with both estimates) and the bound are computed at every setting. That is about 370 final training runs. On a toy environment this costs roughly 45 to 80 core-hours plus 10 to 20 for tuning, about a day on eight cores. The estimate must be re-measured on our simulator in week 1.

**What goes in the five pages**

1. One full-width learning-curve figure at the primary setting: payoff with reference lines for random, best rule and bound; policy and value loss; entropy.
2. One main results table: methods by settings with intervals, percent of bound and percent of exact optimum.
3. One figure of the unrestricted-minus-index gap against poaching level.
4. One figure for the belief effect, or the failure-mode breakdown if the result is negative.
5. One compact tuning and ablation table.

Per-seed curves and all other diagnostics live in the repository.

## Schedule and gates

Work is grouped into five workstreams so it fits any team size: A simulator and calibration, B yardsticks (belief, rules, exact instance, bound), C agents and training, D evaluation and diagnosis, E paper and repository. With two people, merge A with B and C with D.

| Week | A: simulator | B: yardsticks | C: agents | D and E: evaluation, paper | Gate |
| --- | --- | --- | --- | --- | --- |
| 1, Nov 12 to 18 | Gymnasium wrapper, masks, seeding, unit tests; fix the day-order conventions; reproduce earlier rule scores; measure speed | Belief module; port the exact small instance; port tuned rules | Library check on a standard task; index-policy unit tests | Repository skeleton and lockfile; draft pre-registration; forward-citation search; paper skeleton | G1: simulator trusted |
| 2, Nov 19 to 25 | Freeze the calibrated scenario | Upper bound and price-directed rule; tune rules with both estimates | Train both agents on the small instance; known-answer regimes; first full-problem runs; build pooled variant | Evaluation harness on shared episodes; write problem and methods sections | G2: yardsticks in place, positive control passed |
| 3, Nov 26 to Dec 2 | Optional slot-scarcity setting | Bounds for every setting; rule scores on held-out episodes | 40-configuration search; sensitivity sweeps; freeze settings; launch final seeds | Tag pre-registration; test statistics scripts; write protocol section | G3: frozen and launched |
| 4, Dec 3 to 9 | Support | Belief test inside rules; exact cost of PPO's mistakes on the small instance | Final runs done; ablations | Full evaluation; apply decision rules; hurdle experiments; failure analysis; all figures | G4: results locked |
| 5, Dec 10 to 16 |  |  |  | Full draft Dec 12; clean-clone reproduction Dec 14; PDF frozen Dec 15; submit Dec 16 | Submission |

Thanksgiving is Nov 26, the first day of week 3, so week 3's compute should be queued before it.

**Gates**

| Gate | Pass when | If not |
| --- | --- | --- |
| G1, Nov 18 | Environment tests pass; earlier rule scores reproduce within 0.05; exact instance reproduces 0.6335; speed measured | No agent training until it passes. If more than three days late, cut PPO runs in the calibrated scenario |
| G2, Nov 25 | Bound is above every rule at every setting; price-directed rule evaluated; at least one PPO agent within 2% of the exact optimum; full-problem curves have flattened | Week 3 becomes diagnosis on the small instance; the full-problem grid shrinks to the primary setting; the paper commits to the negative-result path |
| G3, Dec 2 | Settings frozen; pre-registration tagged; final seeds launched; statistics scripts run end to end | Skip the 40-configuration search and keep the sweep winners; drop to 10 seeds; drop optional ablations |
| G4, Dec 9 | Every run evaluated; every decision rule applied; figures regenerate with one command | Apply the cut list; no new experiments |

**Cut list, in order:** optional theory-constant rules, the slot-scarcity setting, optional ablations, the 40-configuration search, calibrated-scenario PPO runs, the pooled variant.

**Never cut:** the exact small instance, the upper bound, complete learning curves including entropy, the tuning table and the failure-mode analysis. The grading clarification rests on these.

## Decision rules for every outcome

One rule decides every comparison. Take the 95% interval of the mean paired difference and a margin of 2% of the comparator's payoff (about 0.07 at a payoff of 3.5). Entirely inside the margin means **equivalent**. Otherwise, entirely above zero means **better** and entirely below zero means **worse**. Anything else is **inconclusive** and is reported that way. The 2% margin is our choice; no source prescribes one.

| Question and outcome | What the paper claims | What it triggers |
| --- | --- | --- |
| RQ1: rules reach 95% or more of the exact optimum | The task's structure makes simple rules near-optimal; state how much headroom any learner has | Headroom becomes the first "fundamental hurdle" |
| RQ1: rules fall below 95% | There is structure the rules miss; name the states where the optimal policy differs | Those states become the test set for PPO |
| RQ2 small instance: PPO within 2% of optimum | The code and observation design are sound | Full-problem results are read as scaling behaviour |
| RQ2 small instance: PPO outside 2% | PPO does not solve even the small problem at this budget | Copy test: can a network trained to imitate the optimal policy match it? If yes, the failure is in training; if no, in the inputs or architecture |
| RQ2 full problem: PPO better | PPO finds value the rules miss; report the share of the rule-to-bound gap it closes | Show where, by comparing actions with the best rule on shared episodes |
| RQ2 full problem: equivalent | PPO matches the best rule within 2%; with the bound, state the most either could still gain | Negative-result package |
| RQ2 full problem: worse or inconclusive | PPO does not reach the best rule under a stated budget | Negative-result package |
| RQ3: gap inside the margin everywhere | Index structure survives departures in this regime | Point to slot scarcity as the next place to look |
| RQ3: gap grows with poaching | The size and location of the loss from index structure | Confirm on the small instance and with the pooled variant |
| RQ3: index agent ahead | The restriction helps learning at this budget | Check whether the pooled variant or longer training closes it; do not claim index optimality |
| RQ4: rules change decisions and gain payoff | The correction matters at the stated poaching levels | Test with a wrong assumed poaching rate to bound the claim |
| RQ4: PPO versions equivalent | A network given days waited learns the correction; the benefit is learning speed | Report steps to target; claim no final gain |

**The negative-result path**

Every item below is produced in the positive case too, so choosing the branch at the Dec 9 gate changes the paper's framing and one figure, not the work.

| Instructor's condition | Evidence we produce | Where |
| --- | --- | --- |
| Complete learning curves (reward, loss, entropy) | Payoff, policy loss, value loss and entropy against steps, with 95% intervals over all final seeds; training runs at least twice past the plateau | Figure 1; all settings in the repository |
| Tuning and ablations to rule out bugs | Search table with budgets and chosen values; four sensitivity sweeps; known-answer tests; ablation list | Table 2; test suite |
| Fundamental task hurdles | Experiments that move one hurdle at a time and show the PPO-to-rule gap moving with it: headroom, payoff variance, length of credit assignment, time pressure, slot scarcity | Results table, one paragraph each |
| Failure-mode analysis | On shared episodes, PPO's actions against the best rule's and the exact optimal policy's; mistakes counted by type (offer too late, too early, too many interviews, too few, ending the day early, roles unfilled) and priced exactly on the small instance | Figure in the negative branch |

The mistake categories and their exact pricing are this plan's design, not taken from a source.

## Risks, report outline and repository

**Risks**

| Risk | Why it is real | What blunts it |
| --- | --- | --- |
| Prototype numbers do not reproduce | The exact optimum, bound and price-directed scores come from researchers' scripts with their own timing conventions | Treat each as a target to regenerate; fix conventions in week 1; "bound above every policy" blocks the gate |
| "PPO was under-tuned" | The standard objection to a negative RL result | Equal or larger tuning budget, stated; fresh seeds; positive control on the small instance |
| Headroom too small to resolve | The optimum is within roughly a tenth of good rules | 10,000 paired episodes, 15 seeds, a margin fixed in advance; "equivalent" is a reportable verdict |
| Agent gap confounded | The unrestricted network differs in two ways at once | Same critic and budget; pooled variant; copy test on the small instance |
| Belief effect vanishes for PPO | Days waited is an input, so the correction is learnable | The rule-level test carries RQ4 |
| Silent masking or scaling errors | Several library failure modes raise no error | Assertions, unit tests, hand scaling inside the environment |
| Showing only 10 candidates hides others | Which candidates are shown is part of the policy | Log how often the pool exceeds 10; raise to 20 if it is not rare |
| Calibration evidence is thin | Vendor data lean toward technology employers; some figures are secondary | Label every value; keep the stress scenario primary |
| Novelty undercut by unseen work | Google Scholar and cited-by lists were not searched | Forward-citation pass in week 1; "to our knowledge" wording with the search scope stated |

**Report outline (five pages)**

| Section | Length | Content |
| --- | --- | --- |
| Abstract and introduction | 0.6 page | The race, the four questions, three contributions, headline verdicts |
| Related work | 0.3 page | Analytic hiring models; learning in Pandora's box; learned index policies; selection by departure in economics |
| Problem and simulator | 0.6 page | Model, day order, stress and calibrated scenarios |
| Methods | 0.9 page | Corrected belief; rules and where they come from; exact instance and bound; the two agents |
| Protocol | 0.4 page | Seeds, budgets, episodes, statistics, pre-registration |
| Results | 1.4 pages | Learning curves, main table, gap figure, belief result; one verdict per question |
| Hurdles and failure modes | 0.5 page | Mechanisms tested versus conjectured; mistake breakdown |
| Limitations and conclusion | 0.3 page | Instant interviews, uncalibrated values, small-instance scope, loose bound |

The ICML 2026 style files are on the [ICML author instructions page](https://icml.cc/Conferences/2026/AuthorInstructions).

**What the paper may and may not claim**

- May claim: to our knowledge, the first learned policy for this combined hiring setting; a comparison of index-structured and unrestricted policies used to map where index structure costs value; a standard statistical correction applied to an RL hiring agent's belief, with its effect measured.
- May not claim: first to learn index policies, first to show index policies fail, or first to notice that those still available are a weaker group.
- Nearest analytic work to cite: rolling recruitment with departures ([Gong et al. 2025](https://pubsonline.informs.org/doi/fpi/10.1287/msom.2024.1456)) and approximation schemes for sequential hiring ([Segev and Stein 2026](https://arxiv.org/abs/2601.12750)).

**Repository layout**

```
hiring-race/
  README.md             quick start, runtime per experiment, map from paper items to scripts
  PREREGISTRATION.md    tagged before final runs; deviations filled in afterwards
  lockfile              pinned Python 3.11+, gymnasium, stable-baselines3, sb3-contrib, torch, numpy
  configs/              one file per experiment: hyperparameters, scenario, seed lists
  hiring/               env.py, belief.py, rules.py, exact_dp.py, bounds.py, policies.py, seeding.py
  experiments/          tune_rules.py, tune_ppo.py, train.py, evaluate.py, ablations.py, diagnose.py
  analysis/             one script per figure and table, reading only saved results
  results/              per-seed training logs and per-episode evaluation files
  tests/                simulator statistics, determinism, belief, exact values, bound checks, policy tests
  paper/                ICML source and figures
  Makefile              one target regenerates all figures and tables
```

A clean-clone run by someone who did not write the code is the final test, set for Dec 14.

## Open items

| Open item | Why it matters | When it closes |
| --- | --- | --- |
| Team size | Sets how many workstreams run in parallel | Day 1 |
| Whether references or an appendix count toward five pages | Decides where tuning tables and per-seed curves go | Ask the instructor in week 1; the plan assumes no appendix |
| Simulator timing conventions | Change the exact solver and the bound | Week 1, before the first gate |
| Plan defaults: primary setting, 2% margin, 95% threshold, 40 configurations, 10 shown candidates, 3 million steps | All are this plan's choices | Frozen in the pre-registration by Dec 2 |
| Price-directed rule versus our one-interview rule | A real 0.1 to 0.2 lead raises PPO's target | Week 2, on our simulator |
| Bound not yet computed for h0 = 0.10 or the calibrated scenario | Needed at every setting | Weeks 2 and 3 |
| Training steps and real simulator speed | Only a toy measurement exists | Week 1 measurement; Nov 25 gate |
| Pooled variant untested | Schedule risk | Built by Nov 25, or cut |
| Prior-work coverage | Supports the novelty wording | Forward-citation pass in week 1 |
| Citations read only in summary | Several figures came through summaries or secondary sources | Citation pass in week 5; cite only what has been opened |
| Calibrated resume noise and base acceptance | Sources give ranges, not values | Team decision in week 2 |

## Sources

The full plan, `plans/Hiring race RL research plan.md`, carries every citation in place. The four sets of research notes behind it are in `research-notes/`, and the researchers' prototype scripts are in `topic-checks/prototypes/`.

- RL evaluation: [Agarwal et al. 2021](https://arxiv.org/pdf/2108.13264), [Patterson et al. 2024](https://arxiv.org/pdf/2304.01315), [Henderson et al. 2018](https://ar5iv.labs.arxiv.org/html/1709.06560), [Jordan et al. 2024](https://arxiv.org/pdf/2406.16241)
- PPO practice: [Huang et al. 2022, implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details), [Andrychowicz et al. 2020](https://arxiv.org/pdf/2006.05990v1), [Huang and Ontañón on action masking](https://arxiv.org/pdf/2006.14171v3), [MaskablePPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- Bounds and rules: [Brown and Smith 2020](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Dynamic-Selection-Problems.pdf), [Qin et al. 2026](https://arxiv.org/html/2610.13775v2), [Beyhaghi and Kleinberg](https://arxiv.org/pdf/1905.01428)
- Prior work: [Gong et al. 2025](https://pubsonline.informs.org/doi/fpi/10.1287/msom.2024.1456), [Segev and Stein 2026](https://arxiv.org/abs/2601.12750), [Aouad, Ji and Shaposhnik](https://arxiv.org/pdf/2507.07508), [Scully and Terenin](https://arxiv.org/pdf/2506.10872)
- Calibration: [SHRM 2025](https://www.shrm.org/content/dam/en/shrm/research/2025-recruiting-benchmarking-report.pdf), [Gem 2025](https://lp.gem.com/rs/972-IVV-330/images/2025%20Recruiting%20Benchmarks%20-%20Gem.pdf?version=0), [Cronofy 2024](https://www.cronofy.com/reports/candidate-expectations-report-2024)
- Companion document: [Research proposal: hiring as a race against poaching](https://claude.ai/artifact/NrVa2C4x5Se4JvnzkYUZ3h)
