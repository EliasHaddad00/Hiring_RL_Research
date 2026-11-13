# Research proposal: hiring as a race against poaching

Nov 11, 2026 · @EH

## Summary

*Revised after an external review. Each point in the review was checked against the cited papers and a rewritten simulation before anything was changed; the table near the end lists what changed and what did not.*

We propose to study hiring as a race: a firm must fill a few roles by a deadline, can pay for interviews to learn about candidates, and loses its best candidates to competitors while it learns. Theory has near-optimal simple rules for each piece of this problem separately. We build our own simulator of the combined problem and ask whether simple rules stay near-optimal in it, and whether a reinforcement learning agent does better.

The method contribution is two testable ideas: whether a policy that scores each candidate independently (index structure) is enough once candidates leave, and whether correcting the quality estimate for the fact that strong candidates leave first changes decisions. Both are measured against an exactly solved small instance.

A first check shows the problem is not trivial but the bar is high. Tuned hand-written rules that use one interview beat a fast threshold by 10 to 27% at low and moderate pressure, and by about 2% at the highest poaching level. RL has to beat those rules, and a tie is plausible. The plan is built so that a tie, scored against the exact optimum, is still a defensible result under the grading clarification.

## Motivation and research questions

Every interview tells an employer more about a candidate, and gives competitors more time to hire them first. Theory has studied the pieces separately: stopping rules when candidates leave, inspection rules when options expire, bandits when arms die. Each piece has a simple near-optimal rule. We ask whether simple rules survive when the pieces are combined, and whether a learned policy finds something better.

1. **RQ1. How good are simple rules?** How close do tuned threshold and index rules get to the true optimum, measured exactly on small instances?
2. **RQ2. Does learning help?** Can an RL agent reach the optimum where it is computable, and beat the best tuned rule on the full problem?
3. **RQ3. Does index structure survive departures?** A policy that scores each candidate independently is optimal for classical costly inspection. Where does it break once candidates leave and interview slots are limited?
4. **RQ4. Does the belief need correcting?** Strong candidates leave faster, so those still waiting are worse than their scores suggest. Does accounting for that change decisions?

That waiting policies lose value under poaching is true by construction, so it is motivation here, not a finding.

## Related work and the gap

Each ingredient of our problem has been studied; the combination, and a learned policy for it, has not turned up in our search. Every paper below was opened and checked for this revision.

| Line of work | What it covers | What it leaves out for us |
| --- | --- | --- |
| [Hiring Under Uncertainty, Purohit, Gollapudi and Raghavan, ICML 2019](https://icml.cc/virtual/2019/poster/4304) | Ordering offers when candidates may decline and there is a deadline; approximation algorithms | Acceptance odds are given; no interviews to learn quality, no departures |
| [Selection and Ordering Policies for Hiring Pipelines, Epstein and Ma](https://arxiv.org/pdf/2210.04059v3) | Interviewing and offering for k positions via linear programming, with 1 - 1/e guarantees | Interviews are a budget with no cost; no explicit departure process |
| [How to Hire Secretaries with Stochastic Departures, Kesselheim, Psomas and Vardi](https://arxiv.org/abs/1909.08660) | Candidates stay a random time, then leave; a single threshold is near-optimal for large pools | One hire, quality seen on arrival, no interviews |
| [Secretary Problems with Competing Employers, Immorlica, Kleinberg and Mahdian, 2006](https://link.springer.com/chapter/10.1007/11944874_35) and [On a Competitive Secretary Problem, Karlin and Lei](https://ojs.aaai.org/index.php/AAAI/article/download/9312/9171) | Employers competing for the same candidates, solved as a game; competition pulls offers earlier | No noisy signals or costly learning |
| [Pandora's Problem with Deadlines, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/download/30015/31784), its [1/2-approximation](https://ar5iv.labs.arxiv.org/html/2610.13775), and [Pandora's Box Problem Over Time](https://arxiv.org/html/2407.15261v1) | Paying to inspect options that expire or lose value: the classical form of learn versus lose | Known deadlines and distributions, one prize, one inspection reveals everything |
| [Mortal Multi-Armed Bandits, NeurIPS 2008](https://mlanthology.org/neurips/2008/chakrabarti2008neurips-mortal) and [Reducing Exploration of Dying Arms](https://ar5iv.arxiv.org/html/1907.02571) | Bandit arms that expire | Repeated pulls of an arm for reward; no one-shot hire or deadline |
| [Efficient candidate screening under multiple tests, Cohen, Lipton and Mansour](https://arxiv.org/html/1905.11361) | After each noisy test: hire, reject or test again | Infinite pool, no departures, deadline or competing roles |
| [Tiered interviewing, Schumann et al., NeurIPS 2019](https://mlanthology.org/neurips/2019/schumann2019neurips-making) and [DR-UCB, 2026](https://arxiv.org/pdf/2606.18438) | Bandits for cohort selection and for keeping a team of m workers | No departures; DR-UCB's own future work names contextual features and workers who accept or reject based on outside options |
| [Hiring as Exploration, Li, Raymond and Bergman](https://www.nber.org/papers/w27736) | Contextual bandit for which resumes to interview, on real firm data | One screening decision |
| [A New Dog Learns Old Tricks, Kong et al., ICLR 2019](https://wwkong.github.io/files/publications/rl_opt.pdf) | RL recovers the classic secretary rule, with weaker results on harder variants | Shows RL for hiring-style stopping is not new by itself |

**The claim we can defend.** To our knowledge, no published work gives a learned policy for the combined setting: noisy quality that can be refined by costly interviews, candidates who leave at a quality-dependent rate, declining offers, and several roles under a deadline. Our contribution is that combined model, plus a test of whether the simple rules each neighbouring literature prescribes (a single threshold, an inspection index) stay near-optimal in it. This search was moderate, not exhaustive, and the wording must be re-checked before the report.

**Not a tweak of an existing RL method.** None of the work above trains a deep RL agent on this problem. It uses proven rules, approximation algorithms and bandits; the closest RL paper, Kong et al., applies a simple policy-gradient method to the classic secretary problem with no interviews or departures. So we are not modifying a published RL algorithm for this task. We bring RL to a problem handled so far by other methods, in a model we built, with two ideas of our own on top of a standard training algorithm (PPO). A dedicated search for deep RL work on this exact setting is scheduled for week 1 to confirm this.

## Our model: the recruiting pipeline

The model is the simulator: a partially observable Markov decision process (POMDP) that we write ourselves as a Gymnasium environment. One episode is one hiring window.

| Element | Definition | Value in the current check |
| --- | --- | --- |
| Horizon and goal | Fill m roles within T days | m = 3, T = 60 |
| Arrivals | Poisson number of new candidates per day | 1.5 per day |
| Hidden state | Each candidate's true quality q, standard normal | never observed |
| Observations | Noisy resume score on arrival; each interview returns q plus noise | resume sd 1.5, interview sd 1.0 |
| Actions (daily) | For each candidate: interview, make offer, or wait; at most K interviews per day | K = 3 |
| Poaching | Each waiting candidate leaves with a daily probability that rises with quality | h0 x exp(0.6 q), h0 from 0 to 0.10 |
| Offer response | Acceptance probability falls with days since arrival; a candidate who declines leaves | 0.9 x exp(-0.03 x days) |
| Reward | Quality of each accepted hire minus interview cost | 0.05 per interview |

**Timing rules.** Each day runs in this order: arrivals, offers, interviews, poaching. An interview result can be acted on from the next day. Offers resolve the same day. Results depend on these rules, so they are fixed here and in the code.

**Known weaknesses to fix in week 1.**

- The interview decision is too easy with these values: the slot limit never binds and a second interview never pays. We will make it harder (noisier resumes, costlier or slower interviews, a tighter limit) and re-run the rule baselines before any training.
- The closed-form quality estimate ignores that strong candidates leave first. The approach section covers the correction.

**Calibration anchors.** There is no public hiring dataset, so parameters will be set against published benchmarks and every result reported across a sweep. Anchors checked so far, all secondary sources to be traced to the originals:

- Global median time to hire of 38 days and US offer acceptance of 79% ([SmartRecruiters benchmarks](https://ta.smartrecruiters.com/rs/664-NIC-529/images/Recruiting-Benchmarks%202025-2026.pdf), 89 million applications)
- Accepted offers spent about 2 days in the offer stage, declined ones about 6 ([Ashby data via candidate.fyi](https://candidate.fyi/post/recruiting-funnel-benchmarks))
- 32% of North American candidates who withdrew had accepted another offer ([2023 CandE report via JobScore](https://www.jobscore.com/articles/candidate-experience-statistics/))

**Deferred extensions.** Multi-stage interviews, a salary lever and fairness across candidate groups are out of scope for the five weeks.

## Our approach

The approach is the agent that learns inside the simulator. PPO is the training algorithm, and it is standard. What we propose as ours are the two ideas below, each a hypothesis with a yes or no answer, plus an exact reference to measure against.

**Idea 1: index-structured versus unrestricted policy (RQ3).** In classical costly inspection, where options never expire, Weitzman's rule is optimal: give each option a score computed from that option alone, and act on the highest. With deadlines that ordering can fail badly ([1/2-approximation paper](https://ar5iv.labs.arxiv.org/html/2610.13775)). We train two agents. The index agent uses one shared network to score each candidate from that candidate's own features plus the clock and roles left. The unrestricted agent lets a candidate's score depend on the rest of the pool. If the two tie everywhere, hiring under poaching has index structure, which is a strong and useful statement. If a gap opens as poaching or slot scarcity rises, we have located where the classical structure breaks.

**Idea 2: survival-aware belief (RQ4).** The standard Gaussian update is biased here. In our check, a candidate whose estimate is 1.0 on arrival and who is still available a week later has true average quality 0.87, 0.71 and 0.47 at poaching levels 0.02, 0.05 and 0.10. The exact belief multiplies the Gaussian by the probability of having survived, a one-dimensional integral we can compute numerically. We compare agents fed the naive and the corrected belief.

**Reference optimum.** A small instance (one role, short horizon, capped pool, discretized estimates) is solved exactly by dynamic programming. Agents and rules are scored as a percentage of that optimum before anything is scaled up.

**Action scheme.** Each step the agent picks one candidate and one action (interview or offer), or ends the day. Invalid choices are masked. This is a single categorical choice per step, so PPO has a well-defined action probability, and it fits [masked PPO in SB3-Contrib](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html). The agent sees the top N candidates by estimate in a fixed-size input; attention over the full pool is a later ablation, not the starting point.

**Ablations.** Index versus unrestricted; naive versus corrected belief; fixed top-N versus attention; one agent across the poaching dial versus one per level.

Neither idea has had a dedicated novelty search yet. A light search for learned index policies in inspection problems with expiring options found theory papers and bandit-style learning of Weitzman thresholds, but no deep RL study of this question.

## Baselines and experiments

The opponents are the strongest simple rules we can build, not adapted bandits. Every method runs in the same simulator on the same seeds, with the same stated tuning budget.

| Baseline | Where it comes from |
| --- | --- |
| Fixed threshold | Single-threshold result for secretaries with departures |
| Threshold with a bar that falls toward the deadline | The natural deadline-aware rule; a baseline, not something RL gets credit for finding |
| Interview rule | Interview promising candidates, offer above a falling bar, fast-track exceptional resumes; the best rule in our check |
| Inspection-index rule | Weitzman-style reservation value per candidate, from the Pandora's box literature |
| Exact optimum on a small instance | Dynamic programming; the reference for "how close to optimal" |

A waiting policy and a rule that sees true quality appear only as illustrations. Tiered interviewing and DR-UCB were built for other goals, so they stay in related work.

**Experiments**

1. **Known-answer test (RQ1, RQ2).** On the solvable instance, score every rule and agent as a share of the exact optimum.
2. **Regime map (RQ2).** On the full problem, payoff of rules and agents across poaching pressure and offer decay, 10 seeds each, 95% confidence intervals. The headline figure.
3. **Index-structure test (RQ3).** Index agent against unrestricted agent across poaching pressure and slot scarcity.
4. **Belief test (RQ4).** Naive against survival-aware estimate.
5. **Robustness.** Train at one pressure level, test at another; vary arrival rate, interview noise and number of roles.

**Evidence for a negative result.** Learning curves for reward, value loss and policy entropy on every run; a documented hyperparameter search; and failure cases grouped by poaching level, day and candidate type. With the exact optimum as a reference, a tie between RL and the rules can be attributed to the task and not to the implementation.

## Preliminary check

Simple hand-written rules already beat the fast threshold by 10 to 27% at low and moderate pressure, so the number RL has to beat is 3.4 to 3.9, not 3.1. No RL agent has been trained yet. This replaces the first draft's check, which an external review showed was too weak; the simulator was rewritten and every rule family re-tuned (tuned on 600 episodes with shared random numbers, scored on 4,000 fresh ones, standard error about 0.02). Code and output: `topic-checks/hiring_sim_v3.py` and `hiring_sim_v3_results.txt` in the project files.

| Time pressure | Fast threshold, fixed bar | Threshold, bar falls toward deadline | Best interview rule | Best rule over fixed threshold | Best waiting policy | Rule that sees true quality |
| --- | --- | --- | --- | --- | --- | --- |
| None | 3.10 | 3.42 | 3.93 | +27% | 3.87 | 6.07 |
| Offer decay only | 3.10 | 3.31 | 3.71 | +20% | 2.94 | 5.94 |
| Poaching h0 = 0.02 | 3.10 | 3.17 | 3.54 | +14% | 1.90 | 5.87 |
| Poaching h0 = 0.05 | 3.10 | 3.15 | 3.42 | +10% | 1.18 | 5.84 |
| Poaching h0 = 0.10 | 3.10 | 3.17 | 3.14 | +2% | 0.66 | 5.83 |

Payoff is total quality of hires minus interview costs; higher is better. The interview rule interviews promising candidates, offers when the updated estimate clears a falling bar, and fast-tracks exceptional resumes.

- **Interviewing pays except at the highest poaching level.** The first draft said the best simple rule uses no interviews. That was a search failure on our side.
- **At h0 = 0.10 we found no interview rule that beats a no-interview threshold.** The review reports +5% there from its own simulator; ours shows +2%, from the falling bar alone. Unresolved, and small either way.
- **A second interview was never worth it.** Given the option, the tuned rules never used one. The best rules use 8 to 12 interviews per episode out of 180 possible, and the daily limit of 3 was never binding. The interview decision is currently too easy.
- **The first draft's "patient" baseline was under-built.** It interviewed everyone and scored 2.31 with no time pressure. A sensible waiting policy scores 3.87 there, above the fast threshold, then collapses to 0.66 or lower under heavy poaching. The first draft's 46% figure depended on that construction and is withdrawn.
- **The fixed threshold is flat by construction.** It offers on arrival, so poaching and decay cannot touch it. The 3.04 to 3.13 spread in the first draft was noise.
- **The last column is not an upper bound.** It is a tuned rule with perfect information. Better tuning moved it from 5.82 to 6.07, which is why the plan now includes an exactly solved instance.

These numbers come from tuned heuristics with a modest search budget, so better rules probably exist.

## Timeline, risks and rubric

The report is due Dec 16, which leaves five weeks. Scope is cut from the first draft: the adapted bandit baselines and the DQN variant are dropped, the fairness, salary and multi-stage extensions are deferred, and attention is optional.

| Week | Dates | Work |
| --- | --- | --- |
| 1 | Nov 12 to 18 | Dedicated search for deep RL work on this setting; Gymnasium environment with the timing rules written down; survival-aware belief; tuned rule baselines; exact solver for the small instance |
| 2 | Nov 19 to 25 | Masked PPO with fixed top-N input; known-answer test against the exact optimum; logging for reward, loss and entropy |
| 3 | Nov 26 to Dec 2 | Full problem: index and unrestricted agents; hyperparameter search with equal, stated budgets for agents and rules; first regime map |
| 4 | Dec 3 to 9 | Belief ablation, robustness runs, failure cases; attention only if time allows |
| 5 | Dec 10 to 16 | Write the 5-page ICML report, clean the GitHub repo, reproduce all figures from scripts |

**Risks and fallbacks**

| Risk | Fallback |
| --- | --- |
| RL only ties the tuned rules | Report both as a share of the exact optimum on small instances; a tie near 100% is a result about the task, not the execution |
| Index and unrestricted agents are indistinguishable | That is the finding: the problem has index structure in the tested range |
| The exact solver is too slow | Shrink the instance (shorter horizon, coarser estimates); it only needs to be a known-answer test |
| Parameters look arbitrary | Anchor to the published benchmarks in the model section and report sweeps |
| Compute | The simulator runs 1,000 episodes in about 0.2 seconds; PPO fits on CPU |

**Coverage of the grading clarification**

| Requirement | Plan |
| --- | --- |
| Complete learning curves | Reward, value loss and entropy for every run, shown per poaching level |
| Rule out bugs and poor tuning | Known-answer test on the solvable instance; same tuning budget for rules and agents |
| Fundamental hurdles | Named and tested: reward arrives only at hire, quality is hidden, the daily action is a set of candidates, and the naive estimate is biased by survival |
| Failure cases | Episodes where RL loses, grouped by poaching level, day in the window and candidate type |

**How it maps to the rubric**

| Criterion | Weight | What delivers it |
| --- | --- | --- |
| Contribution | 20% | Open simulator with a solvable test case and strong rule baselines; a regime map scored against the optimum |
| Novelty | 20% | The combined model, the index-structure test and the survival-aware belief |
| Writing | 20% | Clear model and approach split; one headline figure; every citation read in full |
| Completion | 40% | Environment, exact solver, two agents, four rule baselines, ablations and reproducible scripts |

**Open questions for EH**

- [ ] Team size, which decides whether attention and any extension fit
- [ ] Whether the two method ideas (index-structure test, survival-aware belief) are the contribution you want to lead with

## What changed after the review

| Review point | Checked how | Outcome |
| --- | --- | --- |
| Gap claim too broad; related work missing | Opened every paper the review named | Confirmed. Related work rebuilt, claim narrowed |
| DR-UCB row misstates the paper | Re-read its concluding section | Partly confirmed. The paper does not list multi-stage interviews; it does name workers who accept or reject offers. Row corrected |
| "Hiring as Exploration" cited from memory | Opened the NBER page | Confirmed real; now linked |
| Patient baseline under-built; 46% headline is by construction | Rebuilt waiting policies in a new simulator | Confirmed. Headline withdrawn, RQ1 replaced |
| A one-interview rule beats the fast threshold everywhere | Re-tuned rule families | Confirmed at four of five pressure levels (+10 to +27%). At h0 = 0.10 we see +2%, not the review's +5% |
| Second interviews do not help; slot limit rarely binds | Measured directly | Confirmed. Limit never bound |
| "Oracle" is not an upper bound | Re-tuned it | Confirmed (5.82 became 6.07). Exact small instance added |
| Quality estimate is biased under poaching | Direct simulation | Confirmed (1.0 becomes 0.87, 0.71, 0.47 after a week). Now a method idea |
| PPO action undefined; reject equals hold | Reasoning, SB3-Contrib docs | Accepted. One masked choice per step; reject removed |
| Method is off-the-shelf | Light novelty search | Accepted. Two method ideas added, not yet fully novelty-checked |
| Scope too large | Judgment | Accepted. Bandit baselines and DQN dropped, extensions deferred |
| Hiring benchmarks | Opened the three cited pages | Numbers match; added as calibration anchors |
| Preliminary script missing from project files | Listed the folder | Not confirmed. `topic-checks/hiring_headroom2.py` is there. The reviewer's own scripts are not, so its exact numbers could not be rerun |

## Sources

All linked in the related work and model sections, and each was opened for this revision except where noted.

- [Mortal Multi-Armed Bandits](https://mlanthology.org/neurips/2008/chakrabarti2008neurips-mortal): existence confirmed by search, paper not opened
- [Tiered interviewing](https://mlanthology.org/neurips/2019/schumann2019neurips-making): abstract-level reading only
- One possibly relevant paper could not be opened (rate limited) and still needs checking: [The Pandora's Box Problem with Sequential Inspections](https://arxiv.org/html/2507.07508)
- Course handout and grading clarification, in the project files
- Earlier shortlist: [New RL project topics: round 2](https://claude.ai/artifact/1J1XAKFNeQ1hBjRTxMFKZh)
