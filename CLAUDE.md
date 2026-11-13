# ELEN E6885 final project: hiring as a race against poaching

Reinforcement learning course project. Final report due **Dec 16, 2026**: at most 5 pages, double-column ICML format, code on GitHub, every result reproducible. Grading: Contribution 20%, Novelty 20%, Writing 20%, Completion 40%.

## Read these first

Everything about the what, why and how is in `context/`. Read the relevant file before making a design decision; do not re-derive or re-argue what they already settle.

| File | Use it for |
| --- | --- |
| `context/Research plan hiring as a race against poaching.md` | The working blueprint: components, acceptance tests, protocol, schedule, gates, decision rules. Start here. |
| `context/Hiring race RL research plan.md` | Full plan with sources, formulas, starting hyperparameters and derivations. Look things up here. |
| `context/Research_proposal_hiring_as_a_race_against_poaching.md` | The model, the research questions, related work and the novelty claim. |
| `context/Grading_criteria_clarification` | What a negative result must show to count. Plain text, no extension. |
| `context/Final Project - 2026.pdf` | The assignment and report structure. |

If the plan and the proposal disagree, the plan wins. If a file and the user disagree, the user wins; say so and offer to update the file.

## The project in brief

- **Problem.** Fill 3 roles in 60 days. Candidates arrive with hidden quality and a noisy resume. Interviews cost money and refine the estimate. Stronger candidates are poached faster, and offers are declined more often the longer a candidate has waited.
- **Our model.** A simulator we write ourselves (Gymnasium POMDP). Day order: arrivals, offers, interviews, poaching.
- **Our approach.** MaskablePPO (SB3-Contrib), one masked choice per step: pick a candidate and interview or offer, or end the day.
- **Our two ideas.** (1) Index-structured actor (one shared network scores each candidate) versus an unrestricted actor. (2) Survival-aware belief (corrects for strong candidates leaving first) versus the naive Gaussian estimate.
- **Yardsticks, built before the agents.** Tuned rules, an exact dynamic-programming optimum on a small instance, and a Lagrangian upper bound on the full problem.
- **Expected outcome.** PPO most likely ties the best rule on the full problem. That is a planned result, judged with a 2% equivalence margin fixed in advance.

Research questions: RQ1 how close simple rules get to the optimum; RQ2 whether PPO reaches the optimum on the small instance and beats the best rule on the full problem; RQ3 whether index structure survives departures; RQ4 whether the belief correction changes decisions.

## Folder layout

```
context/       the five reference files above; treat as read-only unless asked
experiments/   all code and configs
  hiring/      env.py, belief.py, rules.py, exact_dp.py, bounds.py, policies.py, seeding.py
  configs/     one file per experiment: hyperparameters, scenario, seed lists
  scripts/     tune_rules.py, tune_ppo.py, train.py, evaluate.py, ablations.py, diagnose.py
  analysis/    one script per figure or table, reading only saved results
  tests/       simulator statistics, determinism, belief, exact values, bound checks, policy tests
notes/         research papers and reading notes
paper/         ICML 2026 source and vector figures
results/       per-seed training logs, per-episode evaluation files, plots, summaries
```

Create subfolders as needed. Keep `PREREGISTRATION.md`, the lockfile and the `Makefile` at the project root.

## Stack

Python 3.11 or later. Gymnasium 1.4.0, Stable-Baselines3 2.9.2, SB3-Contrib 2.9.2, PyTorch, NumPy. Pin versions in a lockfile. CPU only.

Commands (fill in as they come to exist, and keep this list current):

- Tests: `pytest experiments/tests`
- Regenerate all figures and tables: `make figures`

## Rules for working in this project

### Evidence and honesty

- **Verify before you change.** Reviews, notes and earlier documents are claims, not facts. Check a claim against the paper or by running code before editing anything because of it, and say what you checked.
- **Label what is not ours.** Numbers marked "prototype" in the plan (exact optimum 0.6335, bound 4.31 / 4.17 / 4.00, price-directed rule 3.91 / 3.72 / 3.52) came from researchers' scripts, not this simulator. Treat them as targets to reproduce. Never present one as a result.
- **Report outcomes as they are.** If a test fails or a number does not reproduce, say so with the output. A mismatch is first investigated as a simulator-convention difference.
- **Cite only what has been opened.** Never cite a paper from memory or from a summary. Mark anything read only as an abstract.
- **Keep the novelty wording hedged.** "To our knowledge", with the search scope stated. Do not claim to be first to learn index policies, first to show index policies fail, or first to notice survivorship in beliefs.

### Experiments

- **Yardsticks before agents.** No full-problem PPO conclusions until the exact small instance, the bound and the tuned rules are in place and pass their checks.
- **Seed pools are separate and fixed in a config file:** tuning seeds, final training seeds, and episode seeds split into tuning and evaluation pools. Never tune on the final or evaluation pools. Never drop a seed.
- **Same tuning budget for rules and PPO.** PPO never gets less than the rules.
- **Paired evaluation.** Every policy runs on the same evaluation episodes. Each episode seed fixes arrivals, qualities, interview noise and the departure and acceptance draws, indexed by candidate, not by call order.
- **Log every run:** return, policy loss, value loss, entropy, approximate KL, clip fraction. The grading clarification requires complete curves.
- **Always report unshaped return**, even when training uses reward shaping.
- **Pre-registration.** Once `PREREGISTRATION.md` is tagged, anything not in it is labelled exploratory. Do not change hypotheses, margins or seed lists after that; record deviations in the file.
- **Never skip, weaken or delete a test to make it pass.**

### Implementation pitfalls (each fails silently)

- The day-60 deadline returns `terminated=True`, not `truncated=True`.
- Scale observations by hand inside the environment, identically for every candidate slot. Do not use `VecNormalize` on observations; it breaks weight sharing.
- "End day" is always valid, and the environment asserts that the mask has at least one valid action.
- Always pass action masks at evaluation (maskable `evaluate_policy`, `MaskableEvalCallback`).
- The index-structured actor must replace SB3's final `action_net` with `nn.Identity()`; the default dense layer mixes candidates.
- Start with `gamma = 0.999`. At 0.99 over 200 steps only 13% of a terminal reward survives.
- Use `DummyVecEnv` and parallelise across runs, one process per core.
- Fix these three conventions once and keep the simulator, the exact solver and the bound consistent with them: at most one interview per candidate per day; offers resolved one at a time; a candidate interviewed today can be offered tomorrow at the earliest.

### Scope

Out of scope unless the user says otherwise: adapted bandit baselines, DQN, attention policies, and the fairness, salary and multi-stage interview extensions. If time runs short, follow the cut list in the plan. Never cut the exact small instance, the upper bound, complete learning curves, the tuning table or the failure-mode analysis.

## Schedule and gates

| Gate | Date | Pass when |
| --- | --- | --- |
| G1 | Nov 18 | Environment tests pass; earlier rule scores reproduce within 0.05; exact instance reproduces; throughput measured |
| G2 | Nov 25 | Bound is above every rule at every setting; price-directed rule evaluated; at least one PPO agent within 2% of the exact optimum |
| G3 | Dec 2 | Configurations frozen; pre-registration tagged; final seeds launched |
| G4 | Dec 9 | Every run evaluated; every decision rule applied; figures regenerate with one command |
| Submit | Dec 16 | Draft Dec 12, clean-clone reproduction Dec 14, PDF frozen Dec 15 |

When asked what to do next, check which gate is current and what its pass test still needs.

## How to write for the user

- Lead with the answer or the result. Keep explanations short; use a simple table when comparing things.
- Plain words over jargon. Define a term the first time it appears.
- State what was verified and what was not.
- In the paper, keep tested mechanisms separate from conjecture.

## Keeping this file current

Update this file when a command, a convention or a decision changes. Keep it short: detail belongs in `context/`.
