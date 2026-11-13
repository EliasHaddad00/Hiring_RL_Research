# Rule-based baselines from theory, an exactly solvable reference instance, and upper bounds for the hiring race

Conventions used throughout these notes.

- "Cited Findings" are statements taken from a source I opened in this session (the link is the page I read). Where I could only see an abstract page or a search listing, the bullet says so.
- "Inferences" are my own derivations, heuristic adaptations and prototype computations. Nothing in an Inferences block is proved by a paper unless it says so. Numbers there come from short numpy prototypes I ran on 2026-11-12 (parameters as in the brief; `h0` is not given in the brief, so I tried 0, 0.01, 0.02, 0.05).
- Notation. Resume signal `s = q + N(0, 1.5^2)`. Gaussian posterior that ignores departures, after `n` interviews `y_1..y_n`: precision `tau_n = 1 + 1/2.25 + n = 1.4444 + n`, variance `sig2_n = 1/tau_n`, mean `mu_n = (s/2.25 + sum y_j)/tau_n`. `phi`, `Phi` are the standard normal pdf and cdf. `Psi(u) = phi(u) - u*(1 - Phi(u))` is the standard normal loss function, `E[(Z - u)^+]`. `p(d) = 0.9*exp(-0.03 d)` is the acceptance probability after `d` days. `h(q) = min(1, h0*exp(0.6 q))` is the daily departure hazard.
- Assumptions I had to make about the simulator, which change formulas if wrong: at most one interview per candidate per day; offers are resolved one at a time within the offer phase; a candidate interviewed today can be offered tomorrow at the earliest (offers precede interviews in the day order).

Sources I could not open (budget or bot protection), cited only bibliographically: Weitzman 1979 (original); Doval 2018 (original); Chick and Frazier 2012 (Management Science version, only the 2009 conference companion was read); Berger, Ezra, Feldman, Fusco AAAI 2024 (PDF blocked, only the anthology abstract page was read); Singla 2018 (only abstract and introduction rendered); Xie et al. 2024 (abstract page only); the Beyhaghi and Cai survey (abstract page only); Gilbert and Mosteller 1966; Lippman and McCall 1976; Whittle 1988; Weber and Weiss 1990; Frazier, Powell, Dayanik 2008; Arlotto and Gurvich 2019; Powell and Ryzhov "Optimal Learning"; Bertsekas "Dynamic Programming and Optimal Control"; Gittins, Glazebrook, Weber "Multi-armed Bandit Allocation Indices".

## Weitzman's Pandora's box rule: reservation value, Gaussian formula, links to Knowledge Gradient and Gittins, and adaptations (partial inspection, k > 1, non-obligatory inspection)

### Takeaway
The reservation value of a box is the `z` solving `E[(X - z)^+] = c`; for `X ~ N(mu, s^2)` this is `z = mu + s*Psi^{-1}(c/s)`, one root-find per candidate. In the hiring model an interview is a partial inspection, so the directly usable version is a three-zone rule around a hiring bar `b` (interview if `|mu_n - b| < delta_n`, offer if above the zone, pass if below), with `delta_0 = 0.50`, `delta_1 = 0.24`, `delta_2 = 0.13` for the brief's parameters; this is a heuristic adaptation, not a theorem.

### Cited Findings
- Reservation value: `r_i` is defined by `c_i = E[(v_i - r_i)^+]` (if `c_i = 0`, `r_i = ess sup v_i`). The "reservation-capped" value is `min{v_i, r_i}`. — [Qin, Zhang, Zhang, Zhu 2026, "A 1/2-Approximation for Pandora's Problem with Deadlines", arXiv:2610.13775v2](https://arxiv.org/html/2610.13775v2)
- The same definition (`r` solves `E[(V - r)^+] = c`, `Y = min(V, r)`) is used per (box, round) pair in the time-dependent model. — [Amanatidis, Fusco, Reiffenhäuser, Tsikiridis, arXiv:2407.15261v3](https://arxiv.org/html/2407.15261v3)
- Classical Pandora's problem: n boxes, known cost and value distribution each, inspect one at a time, payoff is best inspected value minus total cost; Weitzman (1979) gave a simple strategy that is optimal in expectation. — [Berger, Ezra, Feldman, Fusco, "Pandora's Problem with Deadlines", AAAI 2024, pp. 20337-20343, DOI 10.1609/AAAI.V38I18.30015 (abstract page)](https://mlanthology.org/aaai/2024/berger2024aaai-pandora)
- Weitzman's rule can fail badly once boxes have deadlines: a two-box instance gives ratio at most `1/N + 1/N^2` against the immediate-commitment optimum. — [Qin et al. 2026](https://arxiv.org/html/2610.13775v2)
- Link to Gittins: cost-aware Bayesian optimization is connected to Pandora's box, "which has a Bayesian-optimal solution based on the Gittins index", reinterpreted as an acquisition function (the "Pandora's Box Gittins Index"). The abstract page does not show the defining equation. — [Xie, Astudillo, Frazier, Scully, Terenin, "Cost-aware Bayesian Optimization via the Pandora's Box Gittins Index", arXiv:2406.20062v3 (NeurIPS 2024 per the search listing)](https://arxiv.org/abs/2406.20062v3)
- Multi-stage inspection ("Markovian price of information"): each item is a Markov chain with a probing price at each non-terminal state and a reward at terminal states. The "grade" of a state is the critical termination penalty at which the optimal player is indifferent between halting and continuing, described as a variant of the Gittins index. The "prevailing cost" is the minimum grade over states visited so far. The expected optimum of the surrogate problem, in which each item's value is its prevailing cost, upper-bounds the optimal utility (Lemma 1). — [Gupta, Jiang, Scully, Singla, "The Markovian Price of Information", arXiv:1902.07856](https://ar5iv.labs.arxiv.org/html/1902.07856)
- Selecting more than one item: if a greedy-style ("Frugal") algorithm is an alpha-approximation in the free-information setting, there is an alpha-approximation in the Markovian price-of-information setting (Theorem 3.1). The adaptation: each item's proxy value is its minimum grade so far, and instead of selecting the top item the algorithm advances that item's chain one step. Corollary: optimal for matroids (a k-item cap is a uniform matroid; the excerpt does not name the k-item case explicitly). — [Gupta et al.](https://ar5iv.labs.arxiv.org/html/1902.07856)
- Single-stage version: Singla's paper covers additive values under packing constraints and states an optimal algorithm for max-weight matroid rank. Only the abstract and introduction were readable. — [Singla, "The Price of Information in Combinatorial Optimization", arXiv:1711.00405](https://ar5iv.labs.arxiv.org/html/1711.00405)
- Bayesian sequential sampling with a known alternative (the partial-inspection analogue of one box). Model: normal outputs with known sampling variance, conjugate normal prior, per-sample cost `c`, no discounting, choose between the sampled system and a known reward `m` at stopping. The k = 1 problem is standardized to `c = sigma = 1, m = 0` with scalings `alpha = c^(1/3) sigma^(-4/3)`, `beta = c^(-1/3) sigma^(-2/3)`, `gamma = c^(2/3) sigma^(-2/3)`, and solved numerically as a free-boundary heat-equation problem. The continuation region has an upper and a lower boundary around `m`. — [Chick and Frazier, "The Conjunction of the Knowledge Gradient and the Economic Approach to Simulation Selection", Winter Simulation Conference 2009, pp. 528-539](https://www.informs-sim.org/wsc09papers/051.pdf)
- Knowledge gradient as one-stage expected value of information: `EVI_i = sigma_Z,i * Psi(Delta_0i / sigma_Z,i)`, with `Psi` the standard normal loss function. `KG*` uses the best average benefit per unit cost over all sampling budgets `B`: `nu_i = sup_B (1/B) * sigma_Z,i(B) * Psi(Delta_0i / sigma_Z,i(B))`, and sampling continues while `max_i nu_i > 1` (in cost units). For k = 1, `KG*` and the paper's `EOC` rule have the same continuation region, and the diffusion solution is the benchmark. For k > 1 the authors do not have an optimal index; they use one-stage allocation bounds and heuristics. — [Chick and Frazier WSC 2009](https://www.informs-sim.org/wsc09papers/051.pdf)
- The journal version is "Sequential Sampling with Economics of Selection Procedures", Management Science 58(3):550-569, 2012 (search listing only; not opened). — [INFORMS listing](https://pubsonline.informs.org/doi/fpi/10.1287/mnsc.1110.1425)
- Non-obligatory inspection, model and difficulty: a box may be selected without being opened; Doval (2018) introduced the variant; it "cannot be solved optimally by any simple ranking-based policy". Doval gave a sufficient condition under which the optimal policy is committing; a corollary is that committing is optimal when boxes have equal inspection costs and are totally ordered by mean-preserving spread. — [Beyhaghi and Kleinberg, "Pandora's Problem with Nonobligatory Inspection", arXiv:1905.01428](https://arxiv.org/pdf/1905.01428)
- Committing policies: fix in advance a set `S` of boxes that are only ever taken closed, and open the rest before selection. For fixed `S`, the best committing policy runs Weitzman's rule on a modified instance in which boxes in `S` have zero cost and deterministic value `E[v_i]`. The best of Weitzman's policy and the n policies with a single closed box achieves at least `1 - 1/e` of the optimal adaptive utility; for two boxes 4/5, tight for committing policies. — [Beyhaghi and Kleinberg](https://arxiv.org/pdf/1905.01428)
- Hardness and structure: computing an optimal policy is NP-hard; there is a PTAS. Structural theorem: an optimal policy is described by a subset `T*`, an order on it and thresholds; open `T*` in order; if an opened box shows a value at least its threshold, switch to Weitzman's rule on the unopened boxes with the best value so far as outside option; if only one box of `T*` remains, take it without inspection. At most one box is ever taken uninspected. — [Fu, Li, Liu, "Pandora Box Problem with Nonobligatory Inspection: Hardness and Approximation Scheme", arXiv:2207.09545v2](https://arxiv.org/pdf/2207.09545)
- Local hedging: label each item independently, using only its own parameters, as "must inspect" or "never inspect", then run an obligatory-inspection policy (Weitzman for one item, Singla-type algorithms for combinatorial selection). Ratio at most 4/3 for single selection; composing with a beta-approximate obligatory-inspection policy gives (4/3)*beta, and the paper's table includes matroid-basis selection with beta = 1. The paper is written in cost-minimization form (reservation price from `E[(u_rsv - X)^+] = c`, hedging probability `p = max{(mu - u_rsv)/(mu - u_rsv + c*u_rsv/mu), 0}`). — [Scully and Doval, "Local hedging for Pandora's box with nonobligatory inspection", arXiv:2410.19011](https://arxiv.org/pdf/2410.19011)
- The survey named in the brief is "Recent Developments in Pandora's Box Problem: Variants and Applications" by Hedyeh Beyhaghi and Linda Cai, arXiv:2308.12242 (submitted 2023-08-23); only the abstract page was read, which does not list the variants. — [arXiv:2308.12242](https://arxiv.org/abs/2308.12242)

### Inferences
- Gaussian closed form (my derivation from the cited definition). For `X ~ N(mu, s^2)`: `E[(X - z)^+] = s*Psi((z - mu)/s)`, so the reservation value is `z = mu + s*u*`, where `u*` solves `Psi(u*) = c/s`. `Psi` is strictly decreasing from infinity to 0, so the root is unique; `z > mu` exactly when `c/s < Psi(0) = 0.3989`. Implementation: `u* = brentq(lambda u: norm.pdf(u) - u*norm.sf(u) - c/s, -40, 40)`. The left side `s*Psi((z - mu)/s)` is exactly the expected improvement of `X` over an outside option `z`, and exactly the knowledge-gradient value of observing `X` when the alternative is `z`. So Weitzman's index is "the outside option at which KG equals the cost", which is also the Whittle-style retirement calibration that defines a Gittins index. The Chick-Frazier WSC paper does not mention Gittins, so this three-way identification rests on my derivation plus the Xie et al. abstract and the Gupta et al. "grade" definition.
- Partial inspection, one-step version (heuristic adaptation). One interview moves the posterior mean from `mu_n` to `mu_{n+1} ~ N(mu_n, st_n^2)` with `st_n^2 = sig2_n - sig2_{n+1} = 1/(tau_n*(tau_n + 1))`. Treat "the post-interview mean" as the box. One-step index: `z_n = mu_n + delta_n`, `delta_n = st_n * Psi^{-1}(c/st_n)`. Computed values for `c = 0.05`:

  | interviews so far n | posterior var | st_n | delta_n (one more interview) | offset if an interview revealed q exactly |
  |---|---|---|---|---|
  | 0 | 0.6923 | 0.5322 | 0.498 | 0.970 |
  | 1 | 0.4091 | 0.3446 | 0.238 | 0.661 |
  | 2 | 0.2903 | 0.2556 | 0.130 | 0.508 |
  | 3 | 0.2250 | 0.2033 | 0.072 | 0.414 |
  | 4 | 0.1837 | 0.1688 | 0.038 | 0.350 |

  The last column (`sqrt(sig2_n) * Psi^{-1}(c/sqrt(sig2_n))`) bounds from above any multi-interview index at zero delay cost, because no number of interviews reveals more than `q`.
- Three-zone rule (heuristic adaptation of the Chick-Frazier continuation region). Given a hiring bar `b` (see the stopping-theory section for how to set it): interview candidate `i` if `|mu_i - b| < delta_{n_i}`; offer if `mu_i >= b + delta_{n_i}`; pass if `mu_i <= b - delta_{n_i}`. With the table above, a resume estimate more than 0.50 above the bar is fast-tracked, one within 0.50 of the bar is interviewed, and after one interview the zone shrinks to 0.24. This reproduces the structure of the team's "one-interview rule with fast-track" from first principles and predicts that second interviews pay only inside a narrow band. To include the race, replace `c` by `c + L`, where `L` is the expected loss from delaying an offer by a day, roughly `(mu_i - b)^+ * (1 - exp(-0.03)*(1 - h(mu_i)))`. That shrinks the upper half of the zone.
- `KG*` variant (heuristic, following the cited `KG*` definition): `delta_n* = max over B in {1,2,3} of st_{n,B} * Psi^{-1}(B*c/st_{n,B})` with `st_{n,B}^2 = sig2_n - sig2_{n+B}`. Use it to widen the zone when one-step lookahead undervalues information.
- Exact per-candidate index (recommended instead of any closed-form approximation). Because interviews are discrete and few, the exact Gittins-type index of a candidate state `(mu, n, d)` can be computed by a one-candidate dynamic program with a retirement price `z`: `V(mu,n,d;z) = max{0, p(d)*(M - z), -c + E[survive * V(next;z)]}`, and the index is the largest `z` with `V > 0`. This is the same recursion as the Lagrangian bound below, so one piece of code gives the index, the baseline policy and the upper bound.
- k > 1 (adaptation of the cited matroid result): keep one index per candidate, always work on the candidate with the highest current index among those that could still enter the top `m`, and hire in index order. The cited optimality is for obligatory inspection with no deadlines, departures or per-day capacity, so in the hiring model this is a heuristic.
- Non-obligatory inspection matters here. Hiring on the resume alone is "taking the box closed", worth `mu_0` with no cost. In the three-zone rule this is the fast-track branch. The cited results say an optimal policy takes at most one box closed in the single-selection problem and that the best committing policy is within `1 - 1/e` (0.8 for two boxes), so a rule that decides per candidate up front "fast-track or interview first" loses little in the classical model. The local-hedging formula is stated for cost minimization; I did not convert it to the maximization form, so do not implement it from these notes.

### Gaps
- I could not open Weitzman's original paper ("Optimal Search for the Best Alternative", Econometrica 47(3):641-654, 1979, DOI 10.2307/1910412; bibliographic details from memory). The definition above is taken from papers that restate it.
- I could not open the Management Science version of Chick and Frazier (2012). It contains a piecewise closed-form approximation of the standardized stopping boundary; I do not reproduce constants from memory. The per-candidate DP above makes the approximation unnecessary for this project.
- No source read here gives the Gittins index for Gaussian sampling in closed form (Brezzi-Lai or Chick-Gans approximations, Powell and Ryzhov chapter). Not opened.
- Doval (2018), "Whether or not to open Pandora's box", Journal of Economic Theory 175:127-158 (details from memory), was read only through Beyhaghi and Kleinberg's summary.
- No paper found treats partial inspection, departures, a deadline and k > 1 together. Every rule above for the full model is a heuristic.

## What the Pandora-with-deadlines papers prescribe, in implementable terms

### Takeaway
All three papers prescribe the same shape of policy: compute reservation-capped values, fix one threshold in advance, inspect in a pre-committed order (earliest deadline first), and accept the first inspected value above the threshold. The October 2026 paper also yields a one-line upper bound, `OPT <= min_t [t + W(t)]`, which is the single-hire special case of the Lagrangian bound recommended below.

### Cited Findings
- Berger, Ezra, Feldman, Fusco (AAAI 2024): each box has a publicly known deadline, the last round in which its value can be selected; motivating examples are candidate interviews and college admissions; main result is an efficient threshold-based strategy with a constant-factor approximation. The abstract page does not state the constant. — [ML Anthology page](https://mlanthology.org/aaai/2024/berger2024aaai-pandora)
- According to the follow-up, Berger et al.'s algorithm is partially adaptive with immediate commitment and has ratio about 0.158, and Amanatidis et al. improved this to arbitrarily close to 1/4 through a chain of reductions (stochastic probing relaxation, adaptivity gap 1/2, submodular maximization, prophet inequality 1/2). — [Qin et al. 2026](https://arxiv.org/html/2610.13775v2)
- arXiv:2610.13775 is "A 1/2-Approximation for Pandora's Problem with Deadlines" by Shenyu Qin, Chihao Zhang, Yuhao Zhang, Zixuan Zhu, submitted 2026-10-09 (v2 2026-10-12). — [arXiv abstract page](https://arxiv.org/abs/2610.13775)
- Model in that paper: n boxes, value `v_i ~ D_i` independent, cost `c_i`, integer deadline `d_i`; at most one box opened per round, only if `t <= d_i`; benchmark is the best fully adaptive strategy that may select any opened box at stopping. — [Qin et al. 2026](https://arxiv.org/html/2610.13775v2)
- Algorithm in that paper. (1) `w_i(t) = E[(min{v_i, r_i} - t)^+]`. (2) A set is deadline-feasible if, sorted by deadline, its j-th element has `d >= j`. (3) `W(t) = max over feasible sets of sum w_i(t)`. (4) Choose `t*` with `W(t*) = t*` (unique root; binary search). (5) Find the maximizing set `S*` by an `O(n^2)` DP over boxes sorted by deadline: `D(i,j) = max{D(i-1,j), D(i-1,j-1) + w_i(t*)}` when `d_i >= j`. (6) Open the boxes of `S*` with positive `w_i(t*)` in non-decreasing deadline order; accept the first with `v_i > t*`; reject the others irrevocably; the last box is accepted on opening. — [Qin et al. 2026](https://arxiv.org/html/2610.13775v2)
- Results in that paper: Lemma 1, for every deadline-respecting adaptive strategy and every `t >= 0`, `u(pi) <= t + W(t)`, hence `OPT <= min_t [t + W(t)]`; Lemma 3, the threshold strategy earns at least `min{t, sum_{i in S} w_i(t)}`; Theorem 1, the algorithm earns at least `OPT/2`; Theorem 2, 1/2 is tight for immediate-commitment policies against that benchmark; Theorem 3, an instance where the best order-committed policy earns less than 0.778 of the best fully adaptive one. — [Qin et al. 2026](https://arxiv.org/html/2610.13775v2)
- Amanatidis, Fusco, Reiffenhäuser, Tsikiridis, "Pandora's Box Problem Over Time" (later versions titled "Pandora's Box Problem With Time Constraints", arXiv:2407.15261). Model: box `i` has processing time `p_i`; inspection cost and reward distribution may depend on the round; a reward collected `tau` rounds after inspection is discounted by a non-increasing function; deadlines are modelled by infinite cost. — [arXiv:2407.15261v3](https://arxiv.org/html/2407.15261v3)
- Their main policy: for each (box, round) pair compute the reservation value and `Y = min(V, r)`; choose a schedule `M` (a matching of boxes to rounds) approximately maximizing the submodular function `E[max_{e in M} Y(e)]`; set `tau = 0.5 * E[max_{e in M} Y(e)]`; go through the schedule, inspect a box only if its reservation value is at least `tau`, and stop at the first sampled value at least `tau`. Guarantees: `OPT <= 2 * max_M E[max Y]`; about 21.3-approximation in general, `8 + eps` with instant inspection, about 1.37 when only value discounting is present (via a free-order prophet inequality). Computing an optimal strategy is NP-hard. — [arXiv:2407.15261v3](https://arxiv.org/html/2407.15261v3)

### Inferences
- Translation to the hiring model for one hire (heuristic adaptation; the theorems assume one known deadline per box, one inspection per round, a fixed box set and a single full inspection). Treat "one interview" as the inspection, so the box value is `mu_1 ~ N(mu_0, 0.5322^2)` and the capped value is `kappa = min(mu_1, mu_0 + 0.498)`. With about 90 expected arrivals, `mu_0 ~ N(0, 0.3077)` and first-day acceptance 0.9, I get by Monte Carlo:
  - Qin-style balanced threshold: `t*` solving `81 * E[(kappa - t)^+] = t` is `t* = 1.09`. Rule: interview a candidate if `mu_0 + 0.498 > t*`; offer the first candidate whose post-interview mean exceeds `t*`.
  - Amanatidis-style threshold: `tau = 0.5 * E[max kappa] = 0.80`.
  - Both are deliberately conservative constants built for worst-case guarantees. The bound-minimizing price `argmin_t [t + W(t)]` is 1.47, much higher. Expect these two rules to under-perform a tuned falling threshold; they are useful as "theory constants with no tuning".
- "Earliest deadline first" translates to "interview first the candidate most likely to be gone tomorrow", meaning the highest `h(mu)` and longest wait among candidates inside the interview zone.
- The quantity `min_z [m*z + 81 * E[(kappa - z)^+]]` evaluates to 1.686 for `m = 1` and 4.296 for `m = 3` (price `z` = 1.47 and 1.18). It is not a valid bound for the hiring model (it ignores fast-tracking and second interviews), but it lands within 0.3% of the rigorous Lagrangian bound at `h0 = 0` (1.691 and 4.307, see the upper-bound section), so it is a good back-of-envelope check.

### Gaps
- The AAAI PDF of Berger et al. was blocked (bot protection), so their exact threshold construction is not recorded here; the 0.158 figure is as reported by Qin et al.
- arXiv:2610.13775 is a one-month-old preprint; I found no peer-reviewed version.
- None of the three papers handles stochastic arrivals, stochastic departures, partial inspection or more than one selection.

## Kesselheim, Psomas, Vardi, "How to Hire Secretaries with Stochastic Departures": policy shape and translation to noisy values

### Takeaway
With advance notice of departures and rank-only information, the optimal policy never accepts earlier than it must: it decides only when the current best candidate is about to leave, and for large n it reduces to a single time threshold. The hiring model has no departure notice and pays for waiting, so the paper supplies a structural sanity check rather than a strong baseline.

### Cited Findings
- Model: n candidates with a total order (ranks only); arrival times i.i.d. from a known distribution; each stays for an i.i.d. random duration from a known distribution; the algorithm learns when a candidate is about to depart and must then irrevocably accept or reject; goal is the probability of hiring the overall best; single hire only. — [Kesselheim, Psomas, Vardi, arXiv:1909.08660](https://arxiv.org/pdf/1909.08660)
- "The algorithm can never increase its success probability by accepting earlier than it needs to." — [arXiv:1909.08660](https://arxiv.org/pdf/1909.08660)
- Theorem 1: an optimal policy depends only on time `t` and the number of arrivals so far, is monotone in the number of arrivals, and for uniform arrivals is monotone in `t`. Theorem 2: for large n a single time threshold `t*` is asymptotically optimal (accept a departing best-so-far candidate if and only if `t > t*`); for small n time-threshold policies can be far from optimal. — [arXiv:1909.08660](https://arxiv.org/pdf/1909.08660)
- Poisson arrivals with exponential departures: for rate 1 the threshold is about 0.486 with success probability about 0.804; as the rate grows both tend to `1/e`. — [arXiv:1909.08660](https://arxiv.org/pdf/1909.08660)
- Multiple hires are not analysed (matroid version listed as open), and relaxing the advance departure signal is listed as open. — [arXiv:1909.08660](https://arxiv.org/pdf/1909.08660)
- A journal version exists as "Technical Note—On Hiring Secretaries with Stochastic Departures", Operations Research, published online 10 May 2023, DOI 10.1287/opre.2023.2476 (search listing; not opened). — [INFORMS listing](https://pubsonline.informs.org/doi/fpi/10.1287/opre.2023.2476)

### Inferences
- Translation (heuristic). The paper's key lever, waiting for free until the departure signal, does not exist in the hiring model: departures are unannounced, each day of waiting multiplies acceptance by `exp(-0.03)`, and good candidates leave faster. The usable version is a hold-or-offer test for a candidate already above the bar: hold one more day only if the expected gain from what might arrive or be learned tomorrow exceeds the expected loss `mu_i * p(d_i) * (1 - exp(-0.03)*(1 - E[h(q)|info]))`.
- Baseline derived from it ("explore then take best-so-far"): make no offers before day `t0`; afterwards offer the candidate with the highest posterior mean if it exceeds everything seen so far. With noisy values use posterior means in place of ranks and tune `t0`. Because the hiring model has a known quality distribution, this rank-based rule should lose to threshold rules; its value is as a sanity check that distributional knowledge is being used.
- A useful experiment suggested by the paper: add an optional "departure warning" flag to the simulator and measure how much reward the warning is worth. That quantifies the cost of the race.

### Gaps
- The paper has no noisy-value, cardinal-value or multi-hire results. I found no source that extends it to values observed with noise.
- The Operations Research version was not opened; differences from the arXiv version are unknown.

## Classical finite-horizon stopping and search with a known distribution: thresholds versus time left and slots left (sanity checks at h0 = 0)

### Takeaway
With a known distribution the optimal rule is a threshold equal to the marginal value of a slot, `b(n, k) = V(n-1, k) - V(n-1, k-1)`, which falls as the deadline nears and falls as more slots remain. For the brief's numbers and no interviews, the resume-only optimal bars with about 90 arrivals left are roughly 0.98, 1.07, 1.22 on the resume estimate for 3, 2, 1 open roles.

### Cited Findings
- Moser's problem (maximize the expected value of a uniform observation with a finite number of draws, no recall): the optimal rule compares the current draw with a threshold that depends on the number of observations left, defined recursively from `E_0 = 0`. — [Ferguson, "Who Solved the Secretary Problem?", Statistical Science 4(3):282-296, 1989](https://www.or.mist.i.u-tokyo.ac.jp/takeda/FreshmanCourse/Ferguson.pdf)
- Full-information best-choice problem (Gilbert and Mosteller): the largest value can be obtained with probability at least 0.58016 in the limit. — [Ferguson 1989](https://www.or.mist.i.u-tokyo.ac.jp/takeda/FreshmanCourse/Ferguson.pdf)
- Search with a cost per observation and recall is set up in the same article (payoff is the maximum observed value minus `n*c`); the article does not print the reservation equation. — [Ferguson 1989](https://www.or.mist.i.u-tokyo.ac.jp/takeda/FreshmanCourse/Ferguson.pdf)
- Multi-secretary problem with a known distribution (select k of n sequential values): treated by Arlotto and Gurvich, "Uniformly bounded regret in the multi-secretary problem" (search listing only; not opened). — [arXiv:1710.07719](https://arxiv.org/abs/1710.07719v2)

### Inferences
- Recursions (standard dynamic programming; my derivation, checked numerically).
  - One hire, no recall, value `X` with known distribution, `n` draws left: `V_0 = 0`, `V_n = E[max(X, V_{n-1})]`; accept if `X > V_{n-1}`. For uniform(0,1) this is `V_n = (1 + V_{n-1}^2)/2`, giving 0.5, 0.625, 0.695, 0.742, 0.775. (The tool summary of Ferguson's article rendered the recursion with a minus sign; `E[max(U, v)] = (1 + v^2)/2` is the correct form.) For `X ~ N(0, s^2)`: `V_n = V_{n-1} + s*Psi(V_{n-1}/s)`.
  - `k` hires, acceptance probability `a`: `V(n, k) = V(n-1, k) + a * E[(X - b(n,k))^+]`, with `b(n, k) = V(n-1, k) - V(n-1, k-1)`, `V(n, 0) = V(0, k) = 0`. Offer if `X > b(n, k)`. For Gaussian `X`, `E[(X - b)^+] = s*Psi(b/s)`.
  - Poisson arrivals at rate `lam` with time `u` left: `dV_k/du = lam * a * E[(X - (V_k(u) - V_{k-1}(u)))^+]`, `V_k(0) = 0`.
- Table for `X ~ N(0,1)`, acceptance 0.9 (thresholds `b(n,k)`; value `V(n,k)`):

  | draws left n | b, k=1 | b, k=2 | b, k=3 | V, k=1 | V, k=2 | V, k=3 |
  |---|---|---|---|---|---|---|
  | 2 | 0.359 | 0 | 0 | 0.579 | 0.718 | 0.718 |
  | 5 | 0.857 | 0.407 | 0.146 | 0.955 | 1.470 | 1.708 |
  | 10 | 1.221 | 0.820 | 0.548 | 1.270 | 2.146 | 2.755 |
  | 20 | 1.561 | 1.213 | 0.979 | 1.584 | 2.824 | 3.832 |
  | 45 | 1.924 | 1.624 | 1.428 | 1.934 | 3.568 | 5.007 |
  | 90 | 2.207 | 1.937 | 1.763 | 2.211 | 4.153 | 5.921 |

- Scaling to the resume-only problem. If the agent hires on the resume estimate alone, `X = mu_0 ~ N(0, 0.3077)` (sd 0.5547) and the table scales linearly: multiply thresholds and values by 0.5547. With 90 draws left the bars are 1.224, 1.074, 0.978 for `k = 1, 2, 3`, and the value for `m = 3` is about 3.28. This is the exact optimum among no-interview policies at `h0 = 0` with one arrival per step, so the tuned "falling threshold" rule should approach it. Check: my crude simulation of a resume threshold of 1.0 falling linearly over the last 15 days scored 3.14 at `h0 = 0.02` with Poisson arrivals.
- Shape facts to test in the simulator at `h0 = 0`: (1) the bar depends on the expected number of arrivals left, not on calendar time as such; (2) the bar is lower when more roles are open; (3) the bar drops to 0 on the last day (any positive estimate is hired); (4) with free recall and no acceptance decay the optimal policy would simply wait and pick the top `m` at the deadline, so in this model the falling threshold is driven by the 3%-per-day acceptance decay and the deadline.
- Search with a cost per draw and recall (stated from textbook knowledge, not verified in a source opened here): the optimal reservation value is constant in time and solves `E[(X - xi)^+] = c`, the same equation as Weitzman's index. A constant bar is therefore the right benchmark only when there is no deadline.

### Gaps
- Gilbert and Mosteller (1966), "Recognizing the Maximum of a Sequence", JASA 61:35-73, and Lippman and McCall (1976), "The Economics of Job Search: A Survey", Economic Inquiry 14, were not opened; decision-number tables and the reservation-wage statement are not sourced here.
- The Arlotto-Gurvich paper was not opened, so no regret statement is quoted.
- The best-choice (probability of picking the maximum) results use a different objective from this project (sum of hired quality); only the expected-value recursions apply directly.

## Exactly solvable instance: belief-MDP formulation, state design, state counts, and a recommendation

### Takeaway
A one-hire instance with coarse (binned) observations, at most one arrival per day, one interview slot and a pool cap of 2 is solved exactly by memoized backward induction in about 3 seconds in plain Python for T = 12 (about 166,000 start-of-day states). The survival-biased belief is exact because surviving `d` days multiplies the likelihood by `(1 - h(q))^d`, a known function of `d`, so a candidate's belief state is just (observation history counts, days waited).

### Cited Findings
- No source I read prescribes this construction; it is standard finite-horizon dynamic programming on belief states, and everything in this section is my own design and computation. The only related cited fact is that a bound built from the true value function `V_{t+1}` has zero gap, which is why an exactly solved instance is the right yardstick for both policies and bounds. — [Brown, Smith, Sun 2010](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Information_Relaxations_and_Duality_in_Stochastic_Dynamic_Programs.pdf)

### Inferences
- Key observation about the belief (my derivation). Conditional on a candidate's observation history `H` and on still being in the pool after `d` poaching rounds, the posterior is `pi(q | H, d) proportional to phi(q) * L_H(q) * (1 - h(q))^d`. The survival factor depends only on `d`. Consequences:
  - With Gaussian signals, the exact sufficient statistic is `(mu_n, n, d)`: the usual Gaussian posterior times a fixed tilt. No belief grid is needed, only a quadrature for expectations.
  - All transition probabilities come from one table of normalizing constants `Z(H, d) = integral phi(q) L_H(q) (1 - h(q))^d dq`: the posterior mean is `M(H,d) = integral q(...)/Z`; the probability of surviving tonight is `Z(H, d+1)/Z(H, d)`; the probability that the next interview returns outcome `b` is `Z(H+b, d)/Z(H, d)`.
  - Small-hazard approximation for rules that keep a Gaussian belief: `E[q | survived d] ~ mu - 0.6 * d * h0 * sig2 * exp(0.6*mu + 0.18*sig2)` (first-order tilt plus Stein's lemma). Checked against quadrature: for `h0 = 0.03, mu = 1.5, sig2 = 0.409, d = 10` exact 1.312 versus approximation 1.305; it degrades for large `d*h0` (for `h0 = 0.05, d = 30` exact 0.806 versus 0.525), so use quadrature in the DP and the approximation only inside hand-written rules.
- Why coarse observations. With continuous Gaussian signals the statistic `mu_n` is continuous, so a DP needs a grid plus interpolation and is exact only up to discretization. If the test instance reports binned observations, the state space is finite and the DP is exact up to quadrature error (1e-10). Hidden quality stays `q ~ N(0,1)`, hazards and acceptance stay as in the full model; only the observation channel is coarsened.
- Recommended instance ("exact-small").
  - `m = 1`, `T = 12`, arrival with probability 0.6 each day (at most one), `K = 1` interview per day, pool cap `P = 2`, `c = 0.05`, `h0` as in the full model or larger (I used 0.05 so that poaching matters on a short horizon), acceptance `0.9*exp(-0.03 d)`.
  - Resume observation: bin of `q + N(0, 1.5^2)` with cut points `(-0.8, 0.8)`, 3 bins. Interview observation: bin of `q + N(0,1)` with cut points `(0, 1)`, 3 bins. At most 2 interviews per candidate.
  - Candidate type = (resume bin, counts of interview outcomes), 3 x 10 = 30 types; candidate state = (type, d). Pool = sorted tuple of candidate states (candidates are exchangeable, so a multiset). Full state = (day, pool).
  - Day order as in the full model: arrival; if the pool would exceed the cap the agent chooses which candidate to release; offer phase (sequential offers; an acceptance ends the episode with expected reward `M(type, d)`; a decliner leaves); interview phase; poaching.
- Backward induction (all four functions memoized on `(t, pool)`):
  - `W(t, pool)` (before arrival) `= (1 - 0.6)*U(t, pool) + 0.6 * sum_r P(r) * max over allowed pools after adding (r, d=0) of U(t, .)`; `W(T+1, .) = 0`.
  - `U(t, pool)` (offer phase) `= max{ I(t, pool), max_i [ p(d_i)*M_i + (1 - p(d_i)) * U(t, pool - i) ] }`.
  - `I(t, pool)` (interview phase) `= max{ Pc(t, pool), max_i [ -c + sum_b P(b | type_i, d_i) * Pc(t, pool with type_i + b) ] }`.
  - `Pc(t, pool)` (poaching) `= sum over survivor subsets of prod_i (s_i or 1 - s_i) * W(t+1, survivors with d + 1)`, `s_i = Z(type_i, d_i + 1)/Z(type_i, d_i)`.
  - Precompute `Z` and `M` on a fine `q` grid (I used 4001 points on [-8, 8]).
- Measured results of my prototype (plain Python dictionaries, single core; `h0 = 0.05`):

  | T | pool cap | start-of-day states | run time | optimal value |
  |---|---|---|---|---|
  | 10 | 2 | 86,194 | 1.4 s | 0.5838 |
  | 12 | 2 | 165,558 | 3.1 s | 0.6335 |
  | 15 | 2 | 357,279 | 5.9 s | 0.6947 |
  | 10 | 3 | 3,927,376 | 259 s | 0.5844 |

  A Monte Carlo run of the extracted optimal policy in an independent simulation with hidden `q` (40,000 episodes, T = 12, cap 2) gave 0.631 +/- 0.004, matching the DP value 0.6335. Raising the pool cap from 2 to 3 at T = 10 changed the optimum by 0.1% (0.5838 to 0.5844) at 185 times the run time, so cap 2 is the recommendation. For reference the hindsight bound for a comparable instance (top positive `q` among accepted arrivals) is about 1.26, twice the true optimum.
- State-count formula for other designs. With at most one arrival per day the ages in the pool are distinct, so the number of pool configurations on day `t` with `j` candidates is at most `C(t, j) * S^j`, where `S` is the number of per-candidate information states. Upper bounds on totals:

  | design | S | T | cap | bound on states | feasibility |
  |---|---|---|---|---|---|
  | coarse observations (recommended) | 30 | 12 | 2 | 0.26 million | seconds, plain Python |
  | coarse observations | 30 | 12 | 3 | 19.6 million | tens of minutes in plain Python; needs numpy or numba |
  | finer bins (4 resume x 4 interview bins, 3 interviews) | 140 | 12 | 2 | 5.6 million | minutes |
  | Gaussian signals, mean grid of 41 points, up to 2 interviews | 123 | 12 | 2 | 4.3 million | minutes in vectorized numpy; approximate (interpolation) |
  | Gaussian signals, mean grid of 41 points, up to 2 interviews | 123 | 12 | 3 | 1.3 billion | not feasible in a week |
  | full problem (pool of tens, m = 3, T = 60) | - | 60 | none | astronomically large | not solvable exactly |

- Scoring. "Percent of optimal" = policy value / DP value on the same instance, where the policy is evaluated either by Monte Carlo (standard error about 0.004 with 40,000 episodes) or, better, exactly by running the same recursion with the policy's action in place of the max. RL agents must be trained on the coarse-observation instance (observation = bin index, days waited, day).
- What transfers. The small instance tests the mechanics that matter (fast-track versus interview, second interview, holding versus offering under poaching, falling bar), but with one role, one slot and a cap of 2 it does not test slot allocation across many candidates or the `m = 3` slot-value effect. Use the upper bound in the next section for the full problem.

### Gaps
- I did not find a published exactly solved instance of this specific model to cross-check against; validation rests on the Monte Carlo agreement above.
- The run time for T = 12 with cap 3 was not measured to completion (the T = 10 run took 259 s; the count bound is roughly 2.2 times larger at T = 12).
- If the simulator allows several offers to resolve simultaneously, or an offer on the same day after an interview, the recursions for `U` and `I` must be changed to match.
- A Gaussian-signal version of the exact instance is only "exact up to grid error"; I did not measure that error.

## Upper bounds for the full problem: information relaxation, hindsight bounds, Lagrangian and fluid relaxations

### Takeaway
The hindsight (perfect-information) bound is trivial to compute but loose: about 6.34 for the full problem, roughly 70% above the best policy I could construct. A Lagrangian relaxation that prices the role cap and solves a one-candidate dynamic program is rigorous, takes about 25 seconds in numpy, and gives 4.17 at `h0 = 0.02` against a price-directed policy scoring 3.72, so the optimum is pinned to within about 11%.

### Cited Findings
- Information relaxation: a filtration `G` relaxes the natural filtration `F` if `F_t` is contained in `G_t` for all `t`; the perfect-information relaxation reveals everything. A penalty `z` is dual feasible if `E[z(a_F)] <= 0` for every non-anticipative policy. Weak duality: `E[r(a_F)] <= sup over G-adapted policies of E[r(a_G) - z(a_G)]`. Strong duality: the infimum over dual-feasible penalties equals the primal optimum. — [Brown, Smith, Sun, "Information Relaxations and Duality in Stochastic Dynamic Programs", Operations Research 58(4):785-801, 2010](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Information_Relaxations_and_Duality_in_Stochastic_Dynamic_Programs.pdf)
- Computing the perfect-information bound: simulate a full scenario, solve the deterministic inner problem `max_a r(a, omega) - z(a, omega)`, average over scenarios. — [Brown, Smith, Sun 2010](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Information_Relaxations_and_Duality_in_Stochastic_Dynamic_Programs.pdf)
- Good penalties: `z_t = E[v_{t+1} | G_t] - E[v_{t+1} | F_t]` for an approximate value function `v`; the true value function gives a zero gap; unbiased nested-simulation estimates still give valid bounds. — [Brown, Smith, Sun 2010](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Information_Relaxations_and_Duality_in_Stochastic_Dynamic_Programs.pdf)
- Tightness: zero-penalty bounds "may lead to weak upper bounds"; in the paper's option example an unpenalized perfect-information bound of about 22.84 compares with a penalized bound of about 13.68 and a heuristic value of about 13.41. — [Brown, Smith, Sun 2010](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Information_Relaxations_and_Duality_in_Stochastic_Dynamic_Programs.pdf)
- Lagrangian relaxation for dynamic selection problems (finite horizon, select at most `N_t` of `S` independently evolving items each period; examples include applicant screening): relax the cardinality constraint with multipliers `lambda_t >= 0`; the problem decomposes into item-specific dynamic programs; the bound is `sum_t lambda_t N_t` plus the item values; the dual is convex in `lambda`. — [Brown and Smith, "Index Policies and Performance Bounds for Dynamic Selection Problems", Management Science 66(7):3029-3050, 2020](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Dynamic-Selection-Problems.pdf)
- In that paper the gap between the Lagrangian bound and the optimal Lagrangian index policy grows at most like the square root of the number selected; the bound is looser for small problems (with 4 items the Lagrangian gap per product was about 0.88 versus 0.16 for the information-relaxation bound with a Lagrangian-based penalty); information-relaxation bounds with such penalties weakly improve on the Lagrangian bound. — [Brown and Smith 2020](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Dynamic-Selection-Problems.pdf)
- Single-selection analogue with deadlines: `u(pi) <= t + W(t)` for every `t >= 0`. — [Qin et al. 2026](https://arxiv.org/html/2610.13775v2)
- Surrogate bound for multi-stage inspection: expected optimum over prevailing costs upper-bounds the optimum (obligatory inspection, no deadlines). — [Gupta et al.](https://ar5iv.labs.arxiv.org/html/1902.07856)

### Inferences
- Bound A, hindsight (my derivation and computation). Reveal every `q`, arrival and random draw. If acceptance is driven by one uniform draw per candidate, a candidate who would accept at all accepts on arrival day (probability 0.9, the maximum of `p(d)`), needs no interview and cannot be poached first. So the hindsight optimum per scenario is the sum of the top `m` positive `q` among arrivals who accept on day 0, and the bound is `E[sum of top m positive values among N ~ Poisson(1.5*60*0.9 = 81) standard normals]`. Monte Carlo (200,000 scenarios): 6.337 for `m = 3`, 2.430 for `m = 1`. It does not depend on `h0`, `c` or the signal noise, which is exactly why it is loose: it ignores the learning problem.
- Bound B, Lagrangian decoupling (my construction, following the cited Lagrangian approach; rigorous under the stated simulator assumptions).
  - Relax "at most `m` hires" with a price `z >= 0` per hire and "at most `K` interviews per day" with a price `lam >= 0` per interview. Candidates are independent, so the relaxed problem splits into one-candidate problems.
  - One-candidate value with `r` days left, state `(mu, n, d)`: `V_r(mu, n, d) = max{ 0, p(d)*(M(mu,n,d) - z), -(c + lam) + E_q,y[(1 - h(q)) * V_{r-1}(mu', n+1, d+1)] }`, the last term only if `r >= 1`; `M` is the tilted posterior mean; `q` is drawn from the tilted posterior, `y = q + N(0,1)`, `mu' = (mu*tau_n + y)/(tau_n + 1)`.
  - Bound: `UB(z, lam) = m*z + lam*K*T + 1.5 * sum_{a=1..T} E_{mu_0 ~ N(0, 0.3077)}[V_{T-a}(mu_0, 0, 0)]`; minimize over `z` and `lam` (convex).
  - Implementation that ran in about 25 s per `h0`: mean grid of 91 points on [-3, 4], 15-node Gauss-Hermite for `q`, 9-node for `y`, linear interpolation, 41 prices `z` solved simultaneously as an extra array axis.
  - Results (`lam = 0` was optimal in every case: the relaxed policy uses about 14 of 180 interview slots, so the slot constraint is not binding in expectation):

    | h0 | UB, m = 3 | price z | UB, m = 1 | price z |
    |---|---|---|---|---|
    | 0 | 4.31 | 1.20 | 1.69 | 1.50 |
    | 0.01 | 4.24 | 1.15 | 1.66 | 1.50 |
    | 0.02 | 4.17 | 1.15 | 1.64 | 1.45 |
    | 0.05 | 4.00 | 1.15 | 1.56 | 1.40 |

  - The bound did not change when the cap on interviews per candidate was raised from 3 to 6, so the relaxed optimum never uses a fourth interview.
  - Caveats: grid and quadrature error are not rigorously controlled (rerun with a finer grid to confirm three digits); the bound assumes at most one interview per candidate per day.
- How tight is Bound B. I simulated a "price-directed" policy built from the same tables: each day recompute the price `z` that minimizes `k*z + (future-arrival term) + sum of pool candidates' V`, where `k` is the number of open roles, then follow each candidate's one-candidate action (offer, interview or pass) at that price. 3,000 episodes each:

  | h0 | m | price-directed policy | Bound B | gap | hindsight bound A |
  |---|---|---|---|---|---|
  | 0 | 3 | 3.91 +/- 0.03 | 4.31 | 9% | 6.34 |
  | 0.02 | 3 | 3.72 +/- 0.03 | 4.17 | 11% | 6.34 |
  | 0.05 | 3 | 3.52 +/- 0.03 | 4.00 | 12% | 6.34 |
  | 0.02 | 1 | 1.35 +/- 0.02 | 1.64 | 18% | 2.43 |

  So the true optimum for `m = 3, h0 = 0.02` lies between about 3.7 and 4.2. For comparison, a resume-only falling threshold scored 3.14 in my crude run. The gap widens for `m = 1`, consistent with the cited square-root behaviour: pricing the cap in expectation is weakest when few items are selected. The price-directed policy is itself a strong theory-derived baseline (about 14 interviews per episode, all 3 roles filled).
- Bound C, information relaxation with a penalty from Bound B's value function. It is guaranteed to be no worse than Bound B (cited). The inner problem is a deterministic scheduling problem per scenario with known `q`, and the penalty needs conditional expectations of the Lagrangian value function along every action sequence. My assessment: more than a week of student effort and several pitfalls; attempt only if the 10% gap is unacceptable. A cheaper way to judge RL agents is to report both "percent of Bound B" and "percent of the price-directed policy".
- Fluid or LP relaxation. For this model the fluid relaxation is Bound B in another guise: choose a hire-probability rule so that expected hires are at most `m` and expected interviews at most `K*T`. The dual variables of that LP are `z` and `lam`. The closed-form shortcut `min_z [m*z + 81*E[(kappa - z)^+]]` from the deadlines section (4.30 for `m = 3`) approximates it within 0.3% at `h0 = 0` but is not rigorous.
- Summary of effort: Bound A, an hour; Bound B, one to two days including the price-directed baseline; Bound C, not recommended within the week.

### Gaps
- No published bound exists for this exact model; Bounds A and B are my constructions and should be reviewed by the team (especially the day-order assumptions).
- I did not compute Bound C, so its improvement over 4.17 is unknown.
- `h0` is not specified in the brief; all numbers must be regenerated for the simulator's value.
- If acceptance randomness is implemented as a fresh draw per offer with repeated offers allowed, Bound A needs re-derivation.

## Restless bandit view: indexability and whether a Whittle index is a sensible extra baseline

### Takeaway
Candidates are restless arms (they depart and their acceptance probability decays while idle), but the binding constraint is the number of roles, not the daily interview slots, so a Whittle index on interview activation adds little. The Lagrangian price on hires (Bound B and the price-directed policy) is the appropriate index-style baseline.

### Cited Findings
- Brown and Smith compare Lagrangian index policies with Whittle-type indices; they report that the Whittle index policy's gap to the bound can grow linearly where the optimal Lagrangian index policy's gap grows like a square root, and cite a counterexample to Whittle's asymptotic-optimality conjecture. They also show that random tie-breaking among items at the optimal multipliers can hurt, and recommend coordinated tie-breaking from the dual solution. — [Brown and Smith 2020](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Dynamic-Selection-Problems.pdf)
- Weber and Weiss, "On an index policy for restless bandits", Journal of Applied Probability 1990, is the standard reference for asymptotic optimality of Whittle's index policy (search listing only; not opened). — [Weber's abstract page](https://www.dpmms.cam.ac.uk/~rrw1/abstracts/w90c.html)

### Inferences
- Indexability here means: in the one-candidate problem with interview charge `lam`, the set of states where "do not interview" is optimal grows monotonically with `lam`. That is plausible for this model (a higher charge can only shrink the interview zone around the bar) but I found no proof, and the arm has three actions (interview, offer, wait), which needs a multi-action extension of Whittle's definition.
- In my computations the optimal interview price was `lam = 0`: the relaxed optimum uses about 14 interviews against 180 slots. A Whittle index for slot allocation would therefore almost never change a decision. Slots bind only on rare busy days; ranking interview candidates by one-candidate value `V` (or by the index `mu + delta_n`) on those days is enough.
- If the team wants an explicit index number per candidate, use the hire-price index: the largest `z` at which the one-candidate value is positive (interview priority), together with `M` (offer priority). It comes for free from the Bound B tables.

### Gaps
- Whittle (1988), "Restless bandits: activity allocation in a changing world", and the Weber-Weiss paper were not opened; conditions for asymptotic optimality are not quoted.
- No formal indexability check was run on the one-candidate problem (it could be tested numerically by sweeping `lam` and checking monotonicity of the interview set).
