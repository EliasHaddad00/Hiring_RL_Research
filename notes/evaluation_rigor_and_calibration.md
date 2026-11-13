# Evaluation rigor, negative-result reporting, and hiring calibration statistics (as of 12 Nov 2026)

Scope: (A) an experimental protocol the ELEN E6885 team can adopt and cite; (B) real hiring statistics mapped to simulator parameters. Conventions used below: "Cited Findings" were read from the linked page in this session; anything computed or recommended by me is under "Inferences"; anything not traced to an original is flagged "SECONDARY" or listed under "Gaps".

---

## A1. What do Agarwal 2021 (rliable), Henderson 2018, Colas 2018, Patterson 2024 and Jordan 2020/2024 recommend (seeds, tuning vs evaluation seeds, IQM, bootstrap CIs, performance profiles, probability of improvement, search budgets)?

### Takeaway
The literature converges on: report interval estimates rather than point estimates, never tune and evaluate on the same seeds, give every method the same tuning budget, and publish all hyperparameters and per-run results. For a CPU-bound student project a defensible numeric target is at least 3 seeds per configuration while tuning and 15-20 fresh seeds per final condition, with bootstrap (or Welch) intervals on the difference between methods.

### Cited Findings
**Agarwal et al., NeurIPS 2021, "Deep RL at the Edge of the Statistical Precipice"**
- "3-10 runs are prevalent in deep RL"; Atari 100k results were mostly from 3 or 5 runs — [arXiv 2108.13264](https://arxiv.org/pdf/2108.13264)
- Recommendations (Table 1 / Sec. 4): report stratified-bootstrap confidence intervals for aggregate performance; show performance profiles (score distributions) over all runs and tasks; use the interquartile mean (IQM) as the main aggregate; add probability of improvement and optimality gap; prefer effect sizes with CIs over dichotomous p<0.05 tests; publish all individual run scores — [arXiv 2108.13264](https://arxiv.org/pdf/2108.13264)
- IQM = "25% trimmed mean": drop the top and bottom 25% of runs (pooled over tasks) and average the middle 50%; it is "robust to outlier scores but more statistically efficient than median" — [arXiv 2108.13264](https://arxiv.org/pdf/2108.13264)
- Stratified bootstrap: resample runs with replacement independently within each task; 95% CI = 2.5th/97.5th percentiles; the paper uses 50,000 resamples for aggregates and 2,000 for pointwise bands and probability of improvement; percentile CIs "provide good interval estimates for as few as N = 10 runs" for median and IQM — [arXiv 2108.13264](https://arxiv.org/pdf/2108.13264)
- Performance profile = empirical tail distribution: fraction of runs (over all tasks) with normalized score above tau, with pointwise bootstrap bands — [arXiv 2108.13264](https://arxiv.org/pdf/2108.13264)
- Probability of improvement P(X>Y) is the Mann-Whitney U statistic normalized by N*K, averaged over tasks; significant if the CI excludes 0.5; it "does not account for the size of improvement" — [arXiv 2108.13264](https://arxiv.org/pdf/2108.13264)
- Optimality gap = amount by which runs fall short of a target normalized score gamma (default 1.0) — [arXiv 2108.13264](https://arxiv.org/pdf/2108.13264)

**Patterson, Neumann, White, White, JMLR 2024, "Empirical Design in Reinforcement Learning"**
- "5 runs is insufficient to make strong claims, but even 30 runs can be insufficient if the distribution is heavily skewed" (Sec. 2.7) — [arXiv 2304.01315](https://arxiv.org/pdf/2304.01315); journal version [JMLR v25 23-0183](https://jmlr.org/papers/v25/23-0183.html)
- Workflow figure (paraphrased by the reader tool, verify wording before quoting): run each hyperparameter setting at least 3 times in the sweep, then at least 15 runs per algorithm for the final comparison — [arXiv 2304.01315](https://arxiv.org/pdf/2304.01315)
- Maximization bias: picking the best of many noisy configurations inflates reported performance. Two-stage fix: evaluate the chosen configuration "with a new set of random seeds" (Sec. 3.2) — [arXiv 2304.01315](https://arxiv.org/pdf/2304.01315)
- Fair sweeps: "At a minimum, you should ensure all algorithms have the same number of hyperparameter settings that are tested"; "if you find optimal parameters lie on the boundary, expand the range" — [arXiv 2304.01315](https://arxiv.org/pdf/2304.01315)
- "In general we advocate that you do not report standard errors"; default to bootstrap intervals; use tolerance intervals when describing the spread of agent behaviour (they "do not shrink to zero with more runs") — [arXiv 2304.01315](https://arxiv.org/pdf/2304.01315)
- "beware of untuned baselines"; "It is better to risk giving the baselines a small advantage to mitigate over-claiming"; include random/oracle calibration baselines (Sec. 4.1-4.2) — [arXiv 2304.01315](https://arxiv.org/pdf/2304.01315)
- Paired comparisons are more powerful; non-overlapping CIs are "a low powered test"; put the CI on the difference D = A - B (Sec. 4.4) — [arXiv 2304.01315](https://arxiv.org/pdf/2304.01315)

**Colas, Sigaud, Oudeyer 2018, "How Many Random Seeds?"**
- Final recommendations: use Welch's t-test over the bootstrap CI test; set alpha below 0.05; correct for multiple comparisons; use at least 20 samples in the pilot study; use more seeds than the power analysis prescribes — [arXiv 1806.08295 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1806.08295)
- The bootstrap test "should not be used with less than N=20 samples"; with N<10 its empirical type-I error was about 10% against a nominal 5% — [arXiv 1806.08295 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1806.08295)
- Procedure: pilot runs to estimate SDs, compute power as a function of N and effect size, then run N seeds. Code: [flowersteam/rl-difference-testing](https://github.com/flowersteam/rl-difference-testing)

**Henderson et al., AAAI 2018, "Deep RL that Matters"**
- Ten runs of one configuration split into two groups of five produced "statistically different distributions just from varying random seeds" — [arXiv 1709.06560 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1709.06560)
- "implementation differences which are often not reflected in publications can have dramatic impacts on performance"; reward rescaling "can have a large effect" — [arXiv 1709.06560 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1709.06560)
- Recommendations: run many trials with different seeds, do proper significance testing, tune baselines so they at least match reported performance, and "report all hyperparameters, implementation details, experimental setup, and evaluation methods" — [arXiv 1709.06560 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1709.06560)

**Jordan et al., ICML 2020 and ICML 2024**
- 2020: tune-then-report-best "shows what an algorithm can achieve not what it is likely to achieve"; tuning "hides the lack of knowledge required to use the algorithm"; proposes "complete" algorithm definitions (tuning procedure is part of the algorithm) and performance-percentile normalization — [arXiv 2006.16958 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2006.16958)
- 2020: finds that "the bootstrap technique has the tightest intervals, but has a high failure rate" for their aggregate, so they prefer their PBP-t method — [arXiv 2006.16958 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2006.16958)
- 2024 position paper: trustworthy benchmark rankings can need 100+ seeds per algorithm-environment pair; recommends softening ranking claims, making hypothesis-driven "scientific testing" the main paradigm (it "sheds light on how an algorithm works"), and reporting where an algorithm fails — [arXiv 2406.16241](https://arxiv.org/pdf/2406.16241)

### Inferences
Protocol the team can adopt (my synthesis; cite the papers above for each element):
1. **Three disjoint seed pools, fixed in a config file before any run**: tuning seeds, final training seeds, evaluation-episode seeds. No configuration choice is ever made on the final or evaluation pools (Patterson Sec. 3.2).
2. **Tuning**: same number of configurations for every tunable method (PPO, each heuristic). At least 3 seeds per PPO configuration; widen any range whose best value is on the boundary.
3. **Final runs**: 15-20 fresh training seeds per PPO condition (minimum 10, which is the smallest N for which Agarwal reports good percentile-bootstrap coverage). With fewer than 20 seeds, report Welch's t-test alongside the bootstrap CI (Colas).
4. **Statistics**: treat each simulator setting (for example each h0 level) as a "task". Normalize return as (J - J_random) / (J_reference - J_random), where J_reference is the exact optimum on the small instance and the best tuned heuristic elsewhere. Report IQM with 95% stratified bootstrap CI, a performance profile, probability of improvement of PPO over the best heuristic, and the optimality gap on the small instance.
5. **Uncertainty display**: 95% CIs (not standard errors) on learning curves; a CI on the PPO-minus-baseline difference as the headline number.
6. **Reporting**: one table with the search space, number of configurations tried per method, seeds per configuration, total environment steps and CPU-hours, and the selected values.
- The 2024 Jordan paper is useful framing for this project: with a CPU budget, a hypothesis-driven study ("why does PPO fail to beat a threshold rule here?") is more defensible than a ranking claim.
- In a negative-result paper the conservative direction flips relative to Patterson's advice: to avoid the objection "PPO was under-tuned", PPO should get at least as much tuning budget as the heuristics, preferably more, and this should be stated.

### Gaps
- I did not re-fetch the rliable GitHub README this session; the library exists as `rliable` on PyPI ([package page](https://socket.dev/pypi/package/rliable/overview/1.0.8)). Function names should be checked against the repository before being cited.
- Colas et al. 2019 ("A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms") was not retrieved; only the 2018 paper was read.
- Patterson's Section 7 (common errors list) was beyond the portion of the PDF read; the "3 runs / 15 runs" figure is a tool paraphrase of Figure 1, not a verified quote.

---

## A2. How should a learned agent be compared fairly with tuned heuristics, and how many evaluation episodes are needed to detect a 2-5% difference (SD about 1.3, mean about 3.5)?

### Takeaway
Use equal (or PPO-favouring) tuning budgets, tune heuristics on episodes disjoint from evaluation, and evaluate every policy on the same held-out episode seeds so differences are paired. Unpaired, detecting a 2% difference needs about 5,400 episodes per policy and 5% needs about 870 (80% power); common random numbers cut this in proportion to (1 - correlation).

### Cited Findings
- Equal number of hyperparameter settings per algorithm is the minimum standard for fairness — [Patterson et al. 2024](https://arxiv.org/pdf/2304.01315)
- Selecting the best configuration and reporting its tuning-time score is biased upward; re-evaluate on new seeds — [Patterson et al. 2024](https://arxiv.org/pdf/2304.01315)
- Paired comparisons are more powerful than unpaired; report a CI on the difference rather than checking CI overlap — [Patterson et al. 2024](https://arxiv.org/pdf/2304.01315)
- Baselines must be tuned until they at least match known performance — [Henderson et al. 2018](https://ar5iv.labs.arxiv.org/html/1709.06560)
- Keep evaluation protocols identical within a comparison — [Agarwal et al. 2021](https://arxiv.org/pdf/2108.13264)

### Inferences
Sample-size arithmetic (my calculation; two-sided alpha = 0.05, normal approximation, n = 2 (z_alpha/2 + z_power)^2 sd^2 / delta^2, sd = 1.3, mean = 3.5):

| Difference to detect | Unpaired, 80% power | Unpaired, 90% power | Paired, rho = 0.5 (80%) | Paired, rho = 0.8 (80%) |
|---|---|---|---|---|
| 2% (0.070) | 5,414 per policy | 7,248 | 2,707 pairs | 1,083 pairs |
| 3% (0.105) | 2,406 | 3,221 | 1,203 | 481 |
| 5% (0.175) | 866 | 1,160 | 433 | 173 |

- Precision of one policy's mean: 1,000 episodes gives a 95% half-width of 0.081 (2.3% of the mean); 10,000 gives 0.025 (0.73%); 50,000 gives 0.011 (0.33%).
- Recommendation: 10,000 evaluation episodes per policy on a fixed list of episode seeds shared by all policies. This makes evaluation noise small relative to a 2% effect and is cheap on CPU for a 60-step episode.
- Common random numbers: the same episode seed should fix the arrival times, hidden qualities, interview noise draws and the uniform draws used for departures and offer acceptance, indexed by candidate rather than by call order, so two policies face the same "world". The achieved correlation rho should be measured and reported; the paired-n column above scales as (1 - rho). This is standard simulation practice; I did not fetch a specific source for it beyond Patterson's paired-comparison advice.
- Two levels of variance must not be mixed. Episode noise answers "is this one trained policy better than the heuristic?"; training-seed variance answers "does PPO as a procedure produce policies better than the heuristic?". The second is the paper's claim, so the unit of analysis is the training seed (15-20 values, each a 10,000-episode mean), and the bootstrap is over seeds.
- Heuristic thresholds are fitted parameters too: fit them on a tuning episode pool, then report on the held-out pool, otherwise the baseline enjoys the same maximization bias.
- Report three reference lines on every plot: random policy, best tuned heuristic, and (small instance) exact optimum. The optimum turns "PPO loses by x" into "PPO recovers y% of the achievable gap".

### Gaps
- The SD of 1.3 was given, not measured by me; if the return distribution is strongly skewed or discrete (0-3 hires), the normal approximation is still adequate at these n, but the team should confirm with a bootstrap.
- No fetched source gives a rule for an equivalence margin; the 2% margin is the team's choice and should be pre-specified (see A4).

---

## A3. What makes a negative RL result credible (ICBINB, Troubling Trends, Pineau checklist, NeurIPS/ICML checklists, known-answer tests)?

### Takeaway
A negative result is credible when the reader can rule out the three boring explanations (a bug, under-tuning, under-training) and is given a mechanism for the failure. The ICBINB four-part structure (problem, expected solution, observed outcome with error bars, diagnosed reason) maps almost one-to-one onto the instructor's four conditions.

### Cited Findings
- ICBINB (ICLR 2026 workshop) requires four elements: a clearly specified problem with target metrics; a proposed solution with "expected working hypotheses"; the negative outcome with "metrics, error bars, compute, data, and seed details"; and an investigation of why it failed "supported by diagnostics or ablations", ending in boundary conditions and takeaways. Review criteria: clarity, rigor and transparency of methodology, novelty and significance of insight, discussion of limitations, reproducibility — [ICBINB 2026 call](https://sites.google.com/view/icbinb-2026/submit)
- Lipton and Steinhardt name four trends to avoid, including "Failure to distinguish between explanation and speculation" and "Failure to identify the sources of empirical gains"; they recommend "error analysis, ablation studies, and robustness checks" and asking "what worked?" and "why?" rather than only "how well?" — [Troubling Trends, arXiv 1807.03341](https://ar5iv.arxiv.org/html/1807.03341)
- Lipton and Steinhardt cite Melis et al. 2018: published gains attributed to new architectures were "actually due to better hyper-parameter tuning" (the same risk in reverse applies to a claimed RL failure) — [Troubling Trends](https://ar5iv.arxiv.org/html/1807.03341)
- NeurIPS checklist questions: do claims "accurately reflect the paper's contributions and scope"; "Did you describe the limitations"; code, data and instructions to reproduce; "all the training details (e.g., data splits, hyperparameters, how they were chosen)"; "error bars (e.g., with respect to the random seed...)"; amount and type of compute — [NeurIPS 2022 checklist](https://neurips.cc/Conferences/2022/PaperInformation/PaperChecklist)
- The ML Reproducibility Checklist was one of three parts of the NeurIPS 2019 reproducibility program (with a code submission policy and a reproducibility challenge) — [Pineau et al., JMLR 2021](https://jmlr.org/papers/v22/20-303.html)
- PPO debug variables to log: policy_loss, value_loss, entropy, clipfrac, approx_kl; the ratio should equal 1 in the first minibatch of the first epoch; approx_kl should stay below about 0.02 — [Huang et al., "The 37 Implementation Details of PPO", ICLR Blog Track 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details)
- The same post lists invalid-action masking (set invalid logits to minus infinity) and the 13 core details (orthogonal init, Adam eps 1e-5, LR annealing, GAE, advantage normalization, value clipping, entropy bonus, gradient clipping, etc.) — [Huang et al. 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details)
- Debugging guidance: validate on toy problems with known optimal return; check entropy does not collapse too fast or stay flat; early actions should look random; keep rewards roughly in [-1, 1] or normalize; normalize observations; watch the agent, since it "will reveal things the learning curves won't tell you" — [TorchRL debugging guide](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/DEBUGGING_RL.html)
- Benchmarks should be accompanied by reports of where an algorithm fails — [Jordan et al. 2024](https://arxiv.org/pdf/2406.16241)

### Inferences
Mapping to the instructor's four conditions, with concrete evidence to produce:

| Instructor condition | Evidence to include | Supporting source |
|---|---|---|
| Complete learning curves proving convergence | Return, policy loss, value loss, entropy, approx_kl and clipfrac versus steps, mean with 95% CI over all final seeds; train at least 2x past the plateau; state a plateau criterion (for example, change in IQM over the last 20% of training within the CI) | Huang 2022; Patterson 2024 |
| Tuning and ablations ruling out bugs | Search table with budgets; sensitivity curves; known-answer tests (below); ablations on observation features, reward shaping, action masking, network size, entropy coefficient | Patterson 2024; Henderson 2018; Lipton and Steinhardt |
| Fundamental task hurdles | Controlled experiments that vary one hurdle at a time (noise level, departure hazard, horizon, with or without belief features/recurrence) and show the PPO-to-baseline gap moving with it | Jordan 2024 "scientific testing"; ICBINB element 4 |
| Failure-mode analysis | State-level comparison of PPO and baseline/optimal actions on shared episodes; episodes ranked by paired return difference; categorized error types with counts | Lipton and Steinhardt "error analysis"; ICBINB |

Known-answer tests suited to this simulator (positive controls):
1. **Exact optimum on the small instance**: PPO should reach within a stated tolerance of the dynamic-programming value. If it does on the small instance but not the large one, the code is exonerated and the failure is a scaling or credit-assignment issue.
2. **Degenerate regimes with obvious optimal behaviour**: zero interview noise; zero interview cost (interview everyone); h0 = 0 with no acceptance decay (waiting is free, so pure selectivity wins); very high h0 (hire immediately).
3. **Imitation test**: train the same network by supervised learning to copy the best heuristic. If it can, the observation and architecture are sufficient, so a PPO shortfall is an optimization or exploration problem.
4. **Warm start**: initialize PPO from the cloned heuristic. If return degrades, the gradient signal is the problem; if it stays flat, the heuristic is near a local optimum of the policy class.
5. **Simulator unit tests**: empirical arrival rate, acceptance curve and departure hazard match the configured values; same seed gives identical trajectories.
6. **Reference check**: the PPO implementation reproduces a known score on a standard task (for example CartPole) with the same code path.
- Explained variance of the value function is widely logged by PPO libraries and is a useful extra curve, but none of the sources read this session recommends it explicitly.
- Wording discipline: label mechanism claims that were tested ("removing belief features widens the gap by x") separately from conjecture, per Lipton and Steinhardt's first trend.

### Gaps
- The itemized ML Reproducibility Checklist was not on the JMLR abstract page; the full list is in the [PDF](https://jmlr.org/papers/volume22/20-303/20-303.pdf), which I did not read.
- Andy Jones's "Debugging RL, Without the Pain" (probe environments) was not retrieved; a package implementing probe environments exists ([probing-environments on PyPI](https://pypi.org/project/probing-environments)) but I did not verify its contents.
- The current ICML reviewer checklist was not located; the ICML author page only states that reproducibility and code availability "will be taken into account" ([ICML 2026 author instructions](https://icml.cc/Conferences/2026/AuthorInstructions)).

---

## A4. Pre-specifying hypotheses and decision rules (pre-registration in ML): what is recommended and practical?

### Takeaway
ML pre-registration has been piloted as a workshop format in which the protocol is reviewed before any results exist; there is no binding standard. For a course project the practical equivalent is a timestamped protocol file committed to the repository before the final seeds are run.

### Cited Findings
- NeurIPS 2020 workshop "The pre-registration experiment: an alternative publication model for machine learning research": papers are submitted "without any experimental results", describing a protocol to be executed after acceptance; reviewers judge the argument and experimental design; the aim is to counter an incentive system that rewards mainly positive results and leaves negative results unreported. Organizers: Bertinetto, Henriques, Albanie, Paganini, Varol; project site preregister.science — [NeurIPS 2020 workshop page](https://neurips.cc/virtual/2020/workshop/16158)
- The workshop ran again at NeurIPS 2021 — [NeurIPS 2021 workshop page](https://neurips.cc/virtual/2021/workshop/21885)
- Colas et al. advise fixing the test, the significance level (below 0.05) and the multiple-comparison correction, and sizing N from a pilot before the main runs — [arXiv 1806.08295](https://ar5iv.labs.arxiv.org/html/1806.08295)

### Inferences
A one-page `PREREGISTRATION.md`, committed and tagged before final training (cite the commit hash in the report), containing:
- Hypotheses, e.g. H1: PPO's mean return is at least that of the best tuned heuristic; H2: the PPO-to-heuristic gap grows with interview noise and departure hazard.
- Primary metric (mean return over 10,000 shared evaluation episodes per trained seed) and primary comparison (PPO versus best heuristic, paired).
- Decision rule on the 95% CI of the mean paired difference with a pre-set margin (for example 2% of baseline return): CI entirely above zero means PPO better; entirely inside the margin means equivalent; entirely below zero means PPO worse; otherwise inconclusive.
- Fixed seed lists, tuning budget per method, training length, the list of planned ablations and known-answer tests.
- Exclusion rule: no seed is dropped; diverged runs are reported.
- A "deviations" section filled in afterwards. Anything not in the file is labelled exploratory in the report.
- This costs about one hour and directly supports the Completion and Writing criteria, because it shows a negative outcome was a possibility planned for rather than a rescue narrative.

### Gaps
- "Preregistering NLP research" appeared in search results ([talk slides](https://myrthereuver.github.io/talks/PreregisteringNLP.pdf)) but was not read; it may contain a reusable question template.
- I found no evidence of pre-registration being required or formally recommended by ICML or NeurIPS main tracks.

---

## A5. ICML format, page budget for 5 pages, and repository reproducibility practices

### Takeaway
The ICML 2026 style files are current and downloadable from icml.cc; ICML's own limit is 8 pages, so the course's 5-page limit is the binding one. Five double-column pages realistically hold about four display items plus roughly 3,000 words, so most diagnostics must live in the repository.

### Cited Findings
- Style files: https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip ; example paper: https://media.icml.cc/Conferences/ICML2026/Styles/example_paper.pdf — [ICML 2026 author instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
- ICML 2026 submissions have "a main body, which can be up to eight pages long, followed by any number of pages for references and appendices"; camera-ready body limit is 9 pages — [ICML 2026 author instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
- Non-anonymous (author names shown) version uses `\usepackage[accepted]{icml2026}`; LaTeX only; US letter; "Abstracts should be a single paragraph and ideally 4-6 sentences"; "please use vector graphics (eps or pdf figures) for experimental results" — [ICML 2026 author instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
- "Authors are encouraged to submit code to foster reproducibility"; final code should go to a repository such as GitHub with a link in the paper — [ICML 2026 author instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
- Report all hyperparameters, implementation details and evaluation methods — [Henderson et al. 2018](https://ar5iv.labs.arxiv.org/html/1709.06560); publish individual run scores — [Agarwal et al. 2021](https://arxiv.org/pdf/2108.13264)

### Inferences
- Page budget (my estimate, not from a source; a text-only ICML page holds very roughly 900 words): 5 pages is about 4,500 words with no displays. A workable allocation is one full-width three-panel learning-curve figure (return, losses, entropy), one main results table (methods by simulator settings, with CIs), one single-column ablation/sensitivity figure, one single-column failure-mode figure, and one compact calibration or hyperparameter table, leaving about 3,000 words.
- Whether an appendix is allowed beyond 5 pages is a course rule, not an ICML one; if not allowed, put the full hyperparameter tables, per-seed curves and known-answer test outputs in the repository and reference them by path.
- Repository practices: pinned environment lockfile and Python version; one config file per experiment containing every hyperparameter and seed list; a single seeding function for Python, NumPy, PyTorch and the simulator; raw per-seed, per-episode results saved as CSV with the git commit hash; one script per figure and table reading only saved results; one top-level command to regenerate everything; a README table of runtime per experiment; unit tests for the simulator and the exact solver; tagged release matching the submitted PDF.

### Gaps
- The contents of the 2026 style archive (whether the example paper changed from 2025) were not inspected.
- ICML's generative-AI policy is in the Call for Papers, which I did not read; irrelevant to formatting but relevant if the course adopts ICML policies.

---

## B1. Time to hire and time to fill

### Takeaway
Original sources put time to fill for a single role at roughly 30-75 calendar days depending on definition, role type and employer size, with a daily vacancy-filling rate of about 3.5-5%. A 60-day horizon is therefore realistic and moderately tight for three roles.

### Cited Findings
- SHRM 2025 Recruiting Benchmarking (2,371 HR-member respondents, fielded 9 Jan - 3 Mar 2025, unweighted, medians): executive time to fill fell "from 60 days to 45 days"; executive and nonexecutive are each "about a month and a half"; organizations with 5,000+ employees: 61 days nonexecutive, 60 days executive; median requisitions per recruiter 20 — [SHRM 2025 report PDF](https://www.shrm.org/content/dam/en/shrm/research/2025-recruiting-benchmarking-report.pdf)
- Ashby Talent Trends (54M applications, 93K jobs, Ashby customers globally, Jan 2021 - Mar 2026): median time to first fill 60 days for business roles and 75 days for technical roles; junior 52, mid-level 63, senior 71 days — [Ashby recruiting operations benchmarks](https://www.ashbyhq.com/talent-trends-report/reports/recruiting-operations-benchmarks-talent-trends)
- Gem 2025 Recruiting Benchmarks (140M+ applicants, 1.3M hires, Jan 2021 - Dec 2024, Gem customers, skewed to technology employers): average time to hire 41 days in 2024, up from 33 in 2021; engineering 53, IT/infosec 58, customer service 26, operations 25; financial services 63 — [Gem 2025 report PDF](https://lp.gem.com/rs/972-IVV-330/images/2025%20Recruiting%20Benchmarks%20-%20Gem.pdf?version=0)
- Davis, Faberman, Haltiwanger (QJE 2013; JOLTS micro data, 577,268 establishment-month observations, 2001-2006, plus published JOLTS to March 2010): "The average daily job-filling rate is 5.2% per day", range 4.0% (Feb 2001) to 6.9% (Jul 2009); mean vacancy duration 14-25 days, nonfarm average 20.0 days; daily fill rate 2.8% (health and education) to 12.1% (construction); 6.1% at 0-9 employees versus 2.6% at 5,000+; 42% of hires occur at establishments reporting no vacancy — [NBER working paper 16265](https://www.nber.org/system/files/working_papers/w16265/revisions/w16265.rev0.pdf)
- DHI-DFH measure: average working days (Monday-Saturday, excluding holidays) to fill vacancies, the reciprocal of a JOLTS-based daily filling rate — [DHI methodology](https://www.dice.com/indicators/about/methodologies)
- DHI-DFH national mean vacancy duration was 28.3 working days in June 2016 ("high by historical standards"); Jan-Jun 2016 by industry: financial services 42.2, health services 48.0, information 31.9, professional and business services 26.5, construction 15.3; 59.5 working days at employers with 5,000+ employees (May 2016); annual nonfarm low of 16.6 days in 2009 — [DHI Hiring Indicators report, 2016](https://www.dice.com/indicators/contents/data/2016/10/2016-08-DHI-Hiring-Indicators-FINAL-1.pdf)
- The DHI-DFH series are marked discontinued on FRED — [FRED series DHIDFHDMI10](https://fred.stlouisfed.org/series/DHIDFHDMI10)

### Inferences
- Definitions differ: "time to fill" runs from requisition opening to acceptance; "time to hire" (Gem) is shorter because it is measured on the hired candidate. Ashby's "time to first fill" is the closest match to the simulator clock.
- 28.3 working days is about 33 calendar days (multiply by 7/6).
- Constant-hazard sanity target: at 3.5% per working day (2016) about 84% of single vacancies fill within 60 calendar days; at 5.2% (2001-2010) about 93%. A competent baseline policy in the simulator should therefore fill most, but not always all, of the 3 roles by day 60.

### Gaps
- LinkedIn Talent Solutions, SmartRecruiters, Greenhouse, Workable, iCIMS and Lever time-to-hire reports were not retrieved.
- SHRM's exact 2025 nonexecutive median and the older, widely quoted SHRM figures (36 and 42 days) were not verified in this session.
- No national vacancy-duration series after the DHI-DFH discontinuation was found.

---

## B2. Offer acceptance rates, dependence on time, and reasons for declines

### Takeaway
Aggregate offer acceptance is 76-84% in ATS data (lower for technical roles, about 73-76%). Evidence that acceptance falls with elapsed time is correlational and weak in magnitude; nothing found supports a decay as steep as 3% per day.

### Cited Findings
- Ashby (230K applications reaching offer stage, Jan 2021 - Mar 2024): offer acceptance 77% (2021), 76% (2022), 81% (2023); technical 73% versus business 84% — [Ashby offer acceptance report](https://www.ashbyhq.com/talent-trends-report/reports/2023-trends-report-offer-acceptance-rates)
- Ashby: accepted offers spent about 2-2.8 days in the offer stage; declined offers about 6 days; average time in offer stage fell from about 3 to 2.5 days in 2023; of declined offers, 17% (technical) and 27% (business) were rescinded by the employer — [Ashby offer acceptance report](https://www.ashbyhq.com/talent-trends-report/reports/2023-trends-report-offer-acceptance-rates)
- Ashby (to Mar 2026): offer-stage passthrough 81% for all candidates, 85% for referred; median acceptance for referrals 89% (business) and 84% (technical), sourced technical 76%; companies in the slowest time-to-hire quartile take 3.3 (business) to 4.7 (technical) more days on offer turnaround — [Ashby recruiting operations benchmarks](https://www.ashbyhq.com/talent-trends-report/reports/recruiting-operations-benchmarks-talent-trends)
- Gem 2025 report: offer-to-hire 81% (2021) rising to 84% (2024); 2024 by department: customer service 90%, legal/finance/marketing/sales 88%, data science 79%, engineering 76% — [Gem 2025 report PDF](https://lp.gem.com/rs/972-IVV-330/images/2025%20Recruiting%20Benchmarks%20-%20Gem.pdf?version=0)
- Gem 2026 report summary (165M applications, 1.2M hires, Jun 2021 - May 2025): offer acceptance 82%, "the highest since 2021" — [Gem 2026 takeaways blog](https://app.gem.com/blog/key-takeaways-from-the-2026-recruiting-benchmarks-report); this conflicts with the 84% for 2024 in the [Gem 2025 report](https://lp.gem.com/rs/972-IVV-330/images/2025%20Recruiting%20Benchmarks%20-%20Gem.pdf?version=0), probably because of different data windows; I could not resolve it.
- Cronofy Candidate Expectations Report 2024 (about 12,000 candidates; UK, France, Germany, USA, Spain, Canada, Netherlands): 70% said the smoothness of the process would be a key factor when choosing between multiple offers — [Cronofy 2024](https://www.cronofy.com/reports/candidate-expectations-report-2024)

### Inferences
- Simulator check: with acceptance 0.9*exp(-0.03*d), acceptance is 0.73 at day 7, 0.59 at day 14, 0.37 at day 30 and 0.26 at day 41. Matching an 80% realized acceptance requires offers within about 4 days of arrival.
- If offers typically come 35-41 days after application and realized acceptance is 78-84%, the implied decay with the same functional form is about 0.002-0.0035 per day, roughly a tenth of the current value.
- Caveats: published acceptance rates are conditional on the candidate still being in process (withdrawals are removed earlier), which is also how the simulator separates departure from acceptance, so the comparison is fair in structure. The Ashby 2.8-day versus 6-day gap may be reverse causation (candidates who intend to decline deliberate longer).
- Suggested treatment: keep base acceptance near 0.85-0.90, treat decay as a sensitivity axis with values 0.003 (benchmark-consistent), 0.01 and 0.03 (stress case), and say so explicitly.

### Gaps
- No source found gives acceptance probability as a function of days in process, and none gives a quantified breakdown of candidate-side decline reasons.
- SHRM 2025 does not report offer acceptance.

---

## B3. Candidate withdrawal, share leaving for another offer, and the "top candidates are off the market in 10 days" claim

### Takeaway
The 10-day claim traces to an unsupported assertion by Dr. John Sullivan and should not be cited as data. Survey evidence shows slow processes drive withdrawals, but I found no original source quantifying either a daily departure hazard or the share who withdraw because they accepted another offer.

### Cited Findings
- Origin of the claim: an article republished with permission of Dr. John Sullivan states "My research indicates that the top 10 percent of candidates are often gone from the marketplace within 10 days", with no data, sample or method; a second point says top applicants typically drop out "between 10 and 20 days" — [Drake International republication, 2020](https://ca.drakeintl.com/drakepulse/the-top-12-reasons-why-slow-hiring-severely-damages-recruiting-and-business-results)
- LinkedIn Talent Blog (2016) repeats "The top 10% of candidates are hired within 10 days", credits Dr. John Sullivan, and describes the figure as informal — [LinkedIn Talent Blog 2016](https://business.linkedin.com/talent-solutions/blog/trends-and-research/2016/8-hiring-stats-that-will-change-the-way-you-recruit-today)
- Officevibe/Workleap (2017) states "The best candidates are off the market within 10 days" with no source at all — [Workleap blog 2017](https://workleap.com/blog/12-recruiting-stats)
- Talent Board/ERE CandE 2024, North America withdrawal reasons: "My time was disrespected during the screening and interview process" 31%; salary did not meet expectations 23%; "The recruiting process took too long" 20%. Withdrawn candidates are "usually averaging less than 1% of responding candidates" — [2024 CandE business impact report](https://api.eremedia.com/wp-content/uploads/2024/12/The-2024-Business-Impact-of-Candidate-Experience_FINAL.pdf)
- CandE 2025 (now published by Survale), North America: time disrespected 33%; salary 28%; process took too long 16%. EMEA: took too long 22%; APAC 23%; Latin America 21%. Withdrawers are "usually averaging 1-2% of responding candidates each year" — [2025 Survale CandE report](https://survale.com/wp-content/uploads/2025/12/2025-Survale-CandE-Research_The-Business-Impact-of-Candidate-Experience.pdf)
- Cronofy 2024 (about 12,000 candidates, 7 countries): 42% have left a recruitment process because scheduling took too long (49% the year before); 36% would disengage at the one-month mark and 12% at one week; 31% waited 2-3 weeks for a first interview to be scheduled — [Cronofy 2024](https://www.cronofy.com/reports/candidate-expectations-report-2024)
- New York Fed SCE job search supplement (three October waves from 2013; about 2,300 employed, 165 unemployed): employed workers who search receive the most employer contacts and offers (about 20% of the sample, almost half of all offers); the unemployed send 8.1 applications per month versus 1.2 for the employed; offers to employed workers are better in wages and benefits — [Liberty Street Economics 2017](https://libertystreeteconomics.newyorkfed.org/2017/04/how-do-people-find-jobs), summarizing Faberman, Mueller, Sahin, Topa ([NBER w23731](https://www.nber.org/papers/w23731.pdf))

### Inferences
- If the Sullivan claim is read literally for the top decile (mean quality about 1.755 SD), a mean market time of 10 days is a hazard of 0.10 per day, which the simulator reproduces at h0 of about 0.035; reading it as a 10-day median gives h0 of about 0.024. The simulator's upper bound h0 = 0.10 implies a top-decile hazard of 0.29 per day (mean stay 3.5 days), harsher than even the folk claim.
- Cronofy's stated disengagement points (12% by one week, 36% by one month), read as a survival curve, imply an average hazard of roughly 0.015-0.02 per day. Because the population-average hazard in the simulator is about 1.2*h0, this suggests h0 of about 0.012-0.017. This is weak evidence: stated intentions about scheduling delay, not observed exits.
- Reasonable calibrated centre: h0 about 0.02 (range 0.01-0.05), with h0 = 0.10 presented as an extreme stress scenario and h0 = 0 as the no-competition control.
- CandE withdrawal shares (1-2% of respondents) are not usable as a rate: respondents self-select and most are rejected applicants.

### Gaps
- "Accepted another offer" does not appear among the top withdrawal reasons in the 2024 or 2025 CandE summary reports, and I found no original report giving that share.
- No source quantifies how departure speed varies with candidate quality; the exp(0.6*quality) form is an untested modelling assumption. The NY Fed finding that employed searchers get more and better offers is directionally consistent only.
- Exact offer-arrival rates per job seeker are in table images of the NY Fed post and were not extracted; the Econometrica 2022 paper would provide them.
- Sullivan's original ERE article was not located; the ERE page fetched was a different author's 2014 piece with no body text available.

---

## B4. Applicants and interviews per hire, interview rounds, days between stages

### Takeaway
Recent ATS data show roughly 200-300 applications per hire or role, 6-8% passing the first screen, and about 20 interviews per hire, with the interview loop itself lasting about two weeks.

### Cited Findings
- Gem 2025 report: application-to-hire rate fell from 1.6% (2021) to 0.5% (2024), about 200 applications per hire; inbound applicants reaching pre-onsite fell from 11% to 6%, pre-onsite to onsite 29% to 20%, onsite to offer 48% to 35%; sourced candidates hired at 2.0% versus 0.4% inbound — [Gem 2025 report PDF](https://lp.gem.com/rs/972-IVV-330/images/2025%20Recruiting%20Benchmarks%20-%20Gem.pdf?version=0)
- Gem 2025 report: interviews per hire rose from 14 (2021) to 20 (2024); customer service 9, operations 11, sales 21, engineering 39, product management 42 — [Gem 2025 report PDF](https://lp.gem.com/rs/972-IVV-330/images/2025%20Recruiting%20Benchmarks%20-%20Gem.pdf?version=0)
- Gem 2026 summary: 8% of applicants advance past initial screening; 0.5% receive offers; interviews per hire up 33% since 2021; recruiters carry 13.4 open roles on average — [Gem 2026 takeaways blog](https://app.gem.com/blog/key-takeaways-from-the-2026-recruiting-benchmarks-report)
- Ashby research published 7 May 2026 (100M+ applications, 200K jobs, five years): open roles receive on average more than 300 applications; applications per hire tripled since 2021 — SECONDARY, via [HR Dive](https://www.hrdive.com/news/recruiters-see-job-applications-triple-to-more-than-300-per-role/820096/)
- Ashby: interview loop (first to last interview) for hired candidates has a median of 17.9 days (technical) and 14.4 days (business); passthrough: screen about 35%, onsite 24%, post-onsite 95%, offer 81% — [Ashby recruiting operations benchmarks](https://www.ashbyhq.com/talent-trends-report/reports/recruiting-operations-benchmarks-talent-trends)
- SHRM 2025 names seven stages (approval, posting, screening start, screening, interviews, decision and offer, offer to acceptance) but publishes no day counts; extra-large organizations spend "nearly twice as much time" on approval, posting and interviews — [SHRM 2025 report PDF](https://www.shrm.org/content/dam/en/shrm/research/2025-recruiting-benchmarking-report.pdf)

### Inferences
- Arrival rate: 200-300 raw applications per role over 41-60 days is 3.3-7.3 per day per role; at a 6-8% screen pass rate that is 0.2-0.6 screened-in candidates per day per role. The simulator's 1.5 per day across 3 roles (0.5 per role) fits if an "arrival" is defined as a candidate who has passed résumé screening. As raw applicants it would be about ten times too low.
- Interview capacity: 20 interviews per hire times 3 hires over 60 days averages 1 interview per day, so a cap of 3 per day is slack on average and binding in bursts, which is plausible. For engineering-type roles (about 40 per hire) the average would be 2 per day and the cap would bind often.
- The simulator's instantaneous interview abstracts away a real two-week loop; the report should state this as a limitation, or note that a "day" bundles scheduling delay.

### Gaps
- No original source for the number of interview rounds per candidate or for days between individual stages was retrieved.
- Ashby's 300+ figure is from a news summary; the original Ashby page read did not state it.
- All ATS vendor data come from customer bases skewed to technology and venture-backed firms.

---

## B5. Validity of interviews and résumé information as predictors of job performance, and the mapping to noise SD

### Takeaway
The current best estimates (Sackett et al.) are r = .42 for structured and .19 for unstructured interviews, lower than Schmidt and Hunter's .51 and .38. For a standard-normal quality observed with additive Gaussian noise, these correspond to noise SDs of about 2.2 and 5.2: a single interview is a weak signal.

### Cited Findings
- Sackett, Zhang, Berry, Lievens, "Revisiting meta-analytic estimates of validity in personnel selection: Addressing systematic overcorrection for restriction of range", Journal of Applied Psychology (online 2021, print 2022): structured interviews r = .42 (highest), job knowledge tests .40, empirically keyed biodata .38, work samples .33, cognitive ability .31 — [HumRRO summary](https://www.humrro.org/blog/is-cognitive-ability-the-best-predictor-of-job-performance-new-research-says-its-time-to-think-again/) (SECONDARY)
- Side-by-side values, Schmidt and Hunter 1998 versus Sackett et al. (with Sackett's SD of validity across studies): structured interview .51 versus .42 (SD .19); unstructured interview .38 versus .19 (SD .16); cognitive ability .51 versus .31; conscientiousness .31 versus .19; job experience in years .18 versus .07; assessment centres .37 versus .29; integrity tests .41 versus .31; job knowledge .48 versus .40; biodata .35 versus .38; work samples .54 versus .33 — [Master HR summary table](https://master-hr.com/insights/new-study-providing-updated-validity-estimates/) (SECONDARY)
- A second summary confirms .51 to .42 (structured interview), .51 to .31 (cognitive ability), .54 to .33 (work sample), .48 to .40 (job knowledge) — [Criteria Corp summary](https://www.criteriacorp.com/blog/updates-employee-selection-science) (SECONDARY)

### Inferences
- Mapping: if quality q is N(0,1) and the signal is s = q + e with e ~ N(0, sigma^2), then corr(q, s) = 1/sqrt(1 + sigma^2), so sigma = sqrt(1/r^2 - 1).

| Predictor | r | Implied noise SD |
|---|---|---|
| Structured interview (Sackett) | .42 | 2.16 |
| Structured interview (Schmidt and Hunter 1998) | .51 | 1.69 |
| Unstructured interview (Sackett) | .19 | 5.17 |
| Unstructured interview (Schmidt and Hunter 1998) | .38 | 2.43 |
| Work sample (Sackett) | .33 | 2.86 |
| Biodata, a proxy for a scored résumé (Sackett) | .38 | 2.43 |
| Years of experience, a proxy for a raw résumé (Sackett) | .07 | 14.25 |

- If k interviews had independent noise, combined validity would be 1/sqrt(1 + sigma^2/k): for structured interviews .42, .55, .63, .68 at k = 1-4; for unstructured .19, .26, .32, .36. Real interviewers share error, so these are upper bounds.
- A defensible simulator range is interview noise SD from about 1.7 (optimistic) through 2.2 (central) to 5.2 (unstructured), and a résumé/prior signal with SD of 2.4 or more.
- These validities predict supervisor-rated performance, with large between-study variation (SD .19 for structured interviews), so the mapping justifies an order of magnitude, not a point value. If the simulator's current noise SD is near 1 or below, interviews are far more informative than the evidence supports.

### Gaps
- Neither primary paper was read directly; all values come from three secondary summaries that agree with each other. The primary records are at psycnet.apa.org (2022-17327-001 and 1998-10661-006), which were not fetched.
- No validity estimate specific to résumé screening as practised was found.

---

## B6. Mapping table: statistic, value, source, date, simulator parameter and implied value

### Takeaway
Benchmarks support the horizon, arrival rate (if arrivals are screened-in candidates) and interview cap, but they suggest the acceptance decay is about ten times too steep and that h0 = 0.10 is an extreme case; interview noise should be roughly SD 2 or more.

### Cited Findings

| Statistic | Value | Source, sample, date | Simulator parameter | Implied value (my inference) |
|---|---|---|---|---|
| Time to fill, median | About 45 days; 60-61 days at 5,000+ employees | [SHRM 2025](https://www.shrm.org/content/dam/en/shrm/research/2025-recruiting-benchmarking-report.pdf), 2,371 respondents, Jan-Mar 2025 | Horizon (60 days) | 60 days is realistic, moderately tight for 3 roles |
| Time to first fill, median | 60 days business, 75 technical | [Ashby](https://www.ashbyhq.com/talent-trends-report/reports/recruiting-operations-benchmarks-talent-trends), 93K jobs, 2021 - Mar 2026 | Horizon | 60-75 days |
| Time to hire, average | 41 days (2024) | [Gem 2025](https://lp.gem.com/rs/972-IVV-330/images/2025%20Recruiting%20Benchmarks%20-%20Gem.pdf?version=0), 1.3M hires, 2021-2024 | Horizon; typical candidate time in process | Offers occur about 5-6 weeks after application |
| Daily job-filling rate | 5.2% per day; mean duration 20 days | [Davis, Faberman, Haltiwanger](https://www.nber.org/system/files/working_papers/w16265/revisions/w16265.rev0.pdf), JOLTS 2001-2010 | Baseline fill rate (validation target) | 84-93% of single vacancies filled within 60 days |
| Mean vacancy duration | 28.3 working days (about 33 calendar) | [DHI-DFH](https://www.dice.com/indicators/contents/data/2016/10/2016-08-DHI-Hiring-Indicators-FINAL-1.pdf), June 2016, series discontinued | Horizon | Daily fill hazard about 3.5% |
| Applications per hire | About 200 (0.5% hired) | [Gem 2025](https://lp.gem.com/rs/972-IVV-330/images/2025%20Recruiting%20Benchmarks%20-%20Gem.pdf?version=0), 2024 | Arrivals per day | 3.3-7.3 raw per role per day |
| Share passing first screen | 6% inbound (2024); 8% overall | [Gem 2025](https://lp.gem.com/rs/972-IVV-330/images/2025%20Recruiting%20Benchmarks%20-%20Gem.pdf?version=0); [Gem 2026 summary](https://app.gem.com/blog/key-takeaways-from-the-2026-recruiting-benchmarks-report) | Arrivals per day | 0.2-0.6 screened-in per role per day; simulator uses 0.5 |
| Interviews per hire | 20 (2024); 9 to 42 by department | [Gem 2025](https://lp.gem.com/rs/972-IVV-330/images/2025%20Recruiting%20Benchmarks%20-%20Gem.pdf?version=0) | Interviews per day cap (3) | Mean need 1-2 per day; cap of 3 plausible |
| Interview loop length | Median 14.4 days business, 17.9 technical | [Ashby](https://www.ashbyhq.com/talent-trends-report/reports/recruiting-operations-benchmarks-talent-trends), to Mar 2026 | Interview duration (instant in simulator) | State as a simplification |
| Offer acceptance rate | 76-84% overall; 73-76% technical; up to 90% customer service | [Ashby](https://www.ashbyhq.com/talent-trends-report/reports/2023-trends-report-offer-acceptance-rates), 230K offers, 2021 - Mar 2024; [Gem 2025](https://lp.gem.com/rs/972-IVV-330/images/2025%20Recruiting%20Benchmarks%20-%20Gem.pdf?version=0) | Base acceptance (0.9) and decay (0.03 per day) | Base 0.85-0.90 acceptable; decay about 0.002-0.0035 per day to match; 0.03 is a stress case |
| Time in offer stage | Accepted about 2-2.8 days; declined about 6 days | [Ashby](https://www.ashbyhq.com/talent-trends-report/reports/2023-trends-report-offer-acceptance-rates) | Acceptance decay | Direction only; correlational |
| Disengagement from slow scheduling | 12% at one week; 36% at one month; 42% have ever left | [Cronofy 2024](https://www.cronofy.com/reports/candidate-expectations-report-2024), about 12,000 candidates, 7 countries | Departure hazard h0 | About 0.012-0.02 per day average (weak evidence) |
| "Top 10% gone within 10 days" | Unsupported assertion | [Sullivan, republished 2020](https://ca.drakeintl.com/drakepulse/the-top-12-reasons-why-slow-hiring-severely-damages-recruiting-and-business-results) | h0 and the quality slope 0.6 | If taken literally, h0 about 0.024-0.035; do not cite as data |
| Withdrawal reason "took too long" | 16% (2025), 20% (2024), North America | [CandE 2025](https://survale.com/wp-content/uploads/2025/12/2025-Survale-CandE-Research_The-Business-Impact-of-Candidate-Experience.pdf); [CandE 2024](https://api.eremedia.com/wp-content/uploads/2024/12/The-2024-Business-Impact-of-Candidate-Experience_FINAL.pdf) | Departure hazard | Qualitative support that delay causes exits |
| Structured interview validity | r = .42 (.51 in 1998) | Sackett et al. via [Master HR](https://master-hr.com/insights/new-study-providing-updated-validity-estimates/) (SECONDARY) | Interview noise SD | 2.16 (1.69) |
| Unstructured interview validity | r = .19 (.38 in 1998) | Same (SECONDARY) | Interview noise SD | 5.17 (2.43) |
| Biodata; years of experience | r = .38; r = .07 | Same (SECONDARY) | Résumé/prior signal noise SD | 2.43; 14.25 |

### Inferences
- Suggested calibrated "realistic" scenario for the report: horizon 60 days; arrivals 1.5 per day defined as screened-in candidates; interview cap 3 per day; base acceptance 0.85-0.90 with decay 0.003 per day; h0 = 0.02; interview noise SD 2.2. The existing parameters (decay 0.03, h0 up to 0.10) can be kept as a "tight market" stress scenario, clearly labelled as such.
- Running both scenarios is itself useful for the negative-result analysis: if time pressure is much weaker in the calibrated scenario, simple threshold rules may be near-optimal there, which would be a task-structure explanation for RL offering no gain.
- All vendor benchmarks are averages over heterogeneous roles and over-represent technology employers; the table constrains orders of magnitude rather than exact values.

### Gaps
- No empirical basis was found for the quality slope (0.6) in the departure hazard or for the exponential form of acceptance decay.
- No source gives interview cost in comparable units; Gem's 20 interviews per hire could be converted to hours only with an assumed interview length.
- The share of withdrawals caused by accepting a competing offer remains unquantified.
