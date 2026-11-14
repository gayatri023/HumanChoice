# Mathematical formulation

This document states, for every model in HumanChoice, exactly what is implemented — including the conventions
(timing, indexing) that are easy to get subtly wrong. Each statement is backed by a test in
`tests/mathematical_validation/` unless marked otherwise.

**Notation.** `S` states, `A` actions, `Z` observations, `γ ∈ [0,1]` discount, `h` = number of steps remaining.

---

## 1. Tabular MDP (`humanchoice.mdp`)

`T[s,a,s']` transition probabilities, `R[s,a]` expected immediate reward.

Bellman optimality: `Q(s,a) = R(s,a) + γ Σ_s' T(s,a,s') max_a' Q(s',a')`.

* `value_iteration` — fixed-point iteration, requires `γ < 1`.
* `policy_evaluation` — solves the linear system `(I − γ P_π) V = r_π` exactly.
* `policy_iteration` — Howard's algorithm; must agree with value iteration (tested).
* `backward_induction` — finite horizon; `V_0 = 0`, `Q_h = R + γ T V_{h−1}`; converges to the infinite-horizon
  solution as `h → ∞` (tested).

## 2. Bayes-adaptive MDPs (`humanchoice.bamdp`)

A BAMDP augments the physical state with the agent's **belief about the unknown environment parameters**. With
conjugate priors the belief is a table of pseudo-counts and updating is exact and closed form.

### 2.1 Bernoulli bandit (`BetaBernoulliBanditBAMDP`)

Arm `k` pays 1 with unknown probability `θ_k ~ Beta(α_k, β_k)`.

* Update: success on `k` → `α_k += 1`; failure → `β_k += 1`. Posterior predictive `p_k = α_k / (α_k + β_k)`.
* Bayes-adaptive Bellman equation (`h` trials left, belief `b`):

  `Q_h(b,k) = p_k [1 + γ V_{h−1}(b | success on k)] + (1 − p_k) γ V_{h−1}(b | failure on k)`,
  `V_h(b) = max_k Q_h(b,k)`, `V_0 = 0`; `Q_1(b,k) = p_k`.

* Solved by memoised recursion over reachable pseudo-count states — exact, `O(#reachable beliefs)`.
* **Value of information.** For equal posterior means, an uncertain arm has larger `Q_h` than a well-known one
  whenever `h ≥ 2`, because a pull also buys information that improves later choices (tested; Experiment 8).

Checks: `Q_1 = posterior mean`; with 2 trials left from a `Beta(1,1)` prior the optimal value is exactly `13/12`
(hand calculation, tested); Bayes-optimal value ≥ greedy value (tested); batch posterior = sequential posterior.

### 2.2 General BAMDP (`BAMDP`)

Unknown transitions `T(·|s,a) ~ Dirichlet(tc[s,a,:])` and unknown Bernoulli rewards
`R(s,a) ~ Beta(rc[s,a,:])` (or known rewards). Planning is exact finite-horizon expectimax over belief states;
the branching factor is `|S|·2` per `(s,a)`, so it is suitable only for **small problems and short horizons**.
It is validated against the specialised bandit planner (a 1-state BAMDP *is* a bandit) and collapses to ordinary
backward induction as the prior strength → ∞ (both tested).

## 3. POMDPs (`humanchoice.pomdp`)

Joint kernel `W[a,s,s',z] = P(s', z | s, a)`. The factored form `W = T[a,s,s'] · O[a,s',z]` is the main
constructor (observation after the transition, Cassandra convention); `POMDP.from_joint` accepts `W` directly,
which is what lets I-POMDPs reuse this class. Reward `R[s,a]` depends on the state **before** the transition.

* **Bayes filter:** `b'(s') = Σ_s b(s) W[a,s,s',z] / P(z | b,a)`, `P(z|b,a) = Σ_{s,s'} b(s) W[a,s,s',z]`.
* **Belief-tree expectimax** (`q_values`): `Q_h(b,a) = b·R[:,a] + γ Σ_z P(z|b,a) V_{h−1}(b'_{a,z})`.
  Exact for any belief; cost `(|A||Z|)^h`.
* **Alpha-vector value iteration** (`solve_finite_horizon`): `V_h(b) = max_α α·b` is piecewise-linear and convex.
  Backup: for every `a`, `α_{a,z,i}(s) = γ Σ_s' W[a,s,s',z] α_i(s')` for `α_i ∈ Γ_{h−1}`, and
  `Γ_h = ⋃_a { R[:,a] + Σ_z α_{a,z,i_z} }`. The cross-sum is built **incrementally**, pruning after each
  observation (incremental pruning, Cassandra, Littman & Zhang 1997). Pruning removes duplicates,
  pointwise-dominated vectors and vectors dominated only by combinations (a linear program per vector).
* **QMDP** (`qmdp_q_values`): `Q(b,a) = Σ_s b(s) Q_MDP(s,a)`. An *optimistic upper bound* that implicitly assumes
  the state becomes observable after one step, so it does not price information gathering (tested: the fully observable MDP value, which is the idea
  behind QMDP, upper-bounds the exact POMDP value at every horizon and belief tried).

Two independent exact solvers (alpha-vectors vs belief-tree expectimax) must agree on every belief and horizon
(tested). Tiger values derived by hand are reproduced exactly (undiscounted, uniform belief: `V_1 = −1`, `V_2 = −2`, `V_3 ≈ 2.72`).

## 4. Interactive POMDPs (`humanchoice.ipomdp`)

In an I-POMDP (Gmytrasiewicz & Doshi, 2005) agent *i* maintains a belief over the **interactive state**
`is = (s, m_j)`: the physical state and a *model* of the other agent *j*. Here (level 1):

1. **Model space.** `m_j` is a belief `b_j` of a level-0 POMDP agent. The finite set `M` is the breadth-first
   closure of `b_j` under `j`'s own belief updates up to `depth` steps. Beliefs outside the closure are snapped
   to the nearest model (L1).
2. **Other agent's policy.**
   `P(a_j | m) = (1−ε) Uniform(argmax_a Q⁰(m,a)) + ε Uniform(A_j)`, with `Q⁰` from belief-tree planning to
   `horizon0`. `ε` is a tremble/noise probability. (A custom `j_policy` can be supplied.)
3. **Interactive kernel.** With `SE` the level-0 agent's state-estimation function (its own Bayes filter):

   `W[a_i,(s,m),(s',m'),z_i] = Σ_{a_j} P(a_j|m) T(s'|s,a_i,a_j) Σ_{z_j} O_j(z_j|s',a_i,a_j) O_i(z_i|s',a_i,a_j) 1[m' = SE(m,a_j,z_j)]`

   (observations are conditionally independent given `(s',a_i,a_j)`), and
   `R[(s,m),a_i] = Σ_{a_j} P(a_j|m) R_i(s,a_i,a_j)`.
4. The result is an ordinary POMDP over interactive states and is solved with the code in §3.

**Reductions (tested).** If *j* always listens, the I-POMDP's state marginal and Q-values equal the single-agent
Tiger POMDP's. The interactive filter *infers `j`'s model from its audible actions*, and the creak that reveals an
opened door resets *i*'s stale belief (Experiment 8).

**What this is not.** It is a *finite, level-1* approximation: other agents are level-0 and non-strategic, `j`'s
frame (reward/observation model) is a single known POMDP, `M` is a truncated closure, and `j`'s policy is the
optimal finite-horizon level-0 policy plus noise. Recursive (level ≥ 2) reasoning and uncertainty over `j`'s frame
are **not implemented**.

## 5. Behavioural choice models (`humanchoice.models`)

### 5.1 Conventions

* **K options.** Actions are indices `0..K-1`; every model takes `n_actions` (≥ 2). Two-option-only models
  (`POMDPReversalModel`, `HybridModel`) declare `fixed_n_actions = 2` and reject other sizes.
* **Missed trials.** A trial with no response is the index `-1`. It carries no outcome: the model state is **not** updated
  and the trial contributes nothing to the likelihood. Consequence (tested): a sequence with a missed trial has exactly the
  likelihood of the sequence with that trial deleted.
* **Availability.** An optional `(T, K)` boolean mask restricts the choice set per trial. A masked softmax gives
  unavailable options probability exactly 0; a chosen option must be available.
* **Reward support.** Models declare `reward_support` ∈ {`any`, `binary`}. Beta–Bernoulli models (`BayesBanditModel`,
  `POMDPReversalModel`, `HybridModel`) accept only 0/1 rewards and raise on anything else - continuous rewards are never silently
  treated as Bernoulli observations.
* A model is a *stateless* description; learned quantities live in an explicit state threaded through `update`, so
  likelihood, simulation and plotting share one code path (tested: simulated data have exactly the likelihood the model assigns).

### 5.2 Decision rule

`P(a_t = a) = (1 − ε) · softmax_a(β V_t(a) + κ·1[a = a_{t−1}]) + ε / K_avail`, restricted to available options, where the
perseveration bonus `κ` and lapse `ε` are present only in the models that have them (`κ = 0`, `ε = 0` otherwise).

### 5.3 The ladder

| Model | Parameters θ | Mechanism |
|---|---|---|
| `RandomChoice` | - | uniform over available options (chance baseline) |
| `WinStayLoseShift` | `p_win_stay`, `p_lose_shift` ∈ [0,1] | after a win (`r ≥ win_threshold`) repeat w.p. `p_win_stay`; after a loss switch w.p. `p_lose_shift`, uniformly among the other available options |
| `QLearning` | `α ∈ [0,1]`, `β ∈ [0,20]` | `Q(a) ← Q(a) + α (r − Q(a))`, `Q₀ = 0.5`, chosen option only |
| `AsymmetricQLearning` | `α⁺, α⁻, β` | rate `α⁺` if `r − Q(a) ≥ 0` else `α⁻` (nests `QLearning`) |
| `StickyQLearning` | `α, β, κ ∈ [−2,5]` | adds `κ` to the logit of the previously chosen option (nests `QLearning`) |
| `LapseQLearning` | `α, β, ε ∈ [0,0.5]` | with probability `ε` the choice ignores value (nests `QLearning`) |
| `FullQLearning` | `α⁺, α⁻, β, κ, ε` | all three extensions |
| `KalmanBandit` | `β ∈ [0,40]`, `φ ∈ [−2,5]`, `drift_ratio ∈ [10⁻³,10]` | Bayesian tracking, see below |
| `BayesBanditModel` | `β` | softmax over exact BAMDP Q-values (binary rewards) |
| `POMDPReversalModel` | `β`, `hazard` | hidden-state tracking of a reversing good option (two options, binary rewards) |
| `HybridModel` | `α, hazard, w, β` | `V = w·V_POMDP + (1−w)·V_Q` (two options, binary rewards) |

Fixed (not fitted) hyper-parameters are explicit constructor arguments: `q0`, `win_threshold`, Beta prior, BAMDP lookahead, assumed
`p_good`, and the Kalman constants below.

Perseveration and slow learning can mimic each other, as can lapses and low inverse temperature; whether each is *separately
identifiable* is an empirical question answered by `experiments/identifiability` (error correlations and leakage), not assumed.

### 5.4 Kalman-filter learner (conjugate Gaussian tracking)

Each option's mean payoff `μ_a` is tracked by a Gaussian belief `N(m_a, v_a)` that is propagated as a mean-reverting random walk
and updated by the observed reward `r` (observation variance `σ²`, fixed by `obs_sd`):

* update (chosen `a`): `k = v_a / (v_a + σ²)`, `m_a ← m_a + k (r − m_a)`, `v_a ← (1 − k) v_a`;
* propagate (all options): `m ← λ m + (1 − λ) θ`, `v ← λ² v + drift_ratio · σ²`;
* choice: softmax of `β (m_a + φ √v_a)`; `φ > 0` is a directed-exploration bonus for uncertain options.

The gain `k` is not a free learning rate: it adapts to uncertainty (large for options not sampled recently) and to the believed
volatility `drift_ratio`. This is the Gaussian analogue of the Beta–Bernoulli belief updating of §2. The defaults (`obs_sd = 0.1`,
`λ = 1`, `m₀ = 0.5`, `v₀ = 0.1`) are conventional choices for rewards rescaled to [0,1] and are **not** taken from any particular
dataset; the scale of `φ` depends on them, so they are fixed in the analysis plan and varied in the sensitivity analyses.

### 5.5 Environments

`BernoulliBandit` (K options, stationary), `ReversalBandit` (two options, hidden good option reverses), `DriftingBernoulliBandit`
(binary rewards, logit random walk, tunable volatility), `RestlessBandit` (continuous rewards; `μ(t+1) = λ μ(t) + (1−λ) θ + N(0, σ_d²)`,
reward `clip(μ + N(0, σ_o²), 0, 1)`; defaults follow the design commonly cited for Daw et al. (2006), payoffs divided by 100 - check the paper before relying on exact equivalence). `RestlessBandit` simulates that
*design*; it is not any published dataset's schedule.

## 6. Estimation, recovery, comparison and checking

* **Likelihood.** `NLL(θ) = −Σ_{t observed} log P(c_t | history_{<t}, θ)` (probabilities clipped at `1e−12`).
* **Optimiser.** Bounded L-BFGS-B, multi-start from each model's `sample_ranges`, **plus a nested warm start**: a model that contains a
  simpler one (e.g. `asym_q ⊃ q`) is also started from the simpler model's solution, which guarantees it cannot end with a worse
  likelihood. (Regression test: random multi-start alone left `asym_q` up to ~40 nats worse than `q`, which would silently
  corrupt model comparisons.) `FitResult` reports optimiser success, `start_agreement` and parameters on a bound.
* **Likelihood profiles.** Fix one parameter on a grid, re-optimise the rest; the interval where the profile stays within
  `χ²₁(0.95)/2 ≈ 1.92` of its minimum is an approximate likelihood interval. An interval that does not close inside the parameter
  bounds is flagged **open** (parameter not identified from that participant's data).
* **Information criteria.** `AIC = 2k − 2 log L`, `BIC = k log n − 2 log L` (`n` = observed choices); Akaike weights.
* **Held-out prediction.** Fit on the first fraction of trials; roll the model through the test trials using the *observed*
  choices and rewards (one-step-ahead forecasts). Metrics over observed test trials: log-likelihood per trial, accuracy, and the
  multi-class **Brier score** `mean_t Σ_k (p_tk − 1[c_t=k])²` (a uniform predictor scores `1 − 1/K`). Pooled **calibration**
  (reliability) tables bin predicted probabilities of all (trial, option) pairs against observed frequencies; ECE is their
  weighted absolute gap. Population summaries use participant-level percentile bootstrap CIs, and model contrasts use the
  **paired** difference (not overlapping separate CIs).
* **Predictive checks.** *One-step-ahead:* replicate each choice from the model's predictive distribution given the observed
  history; "stay" statistics compare each replicate with the **observed** previous choice. *Free-running:* simulate whole sessions
  in the task. A statistic whose observed value falls in the replicate tails is a systematic misfit; across participants about 5 %
  are expected by chance (`misfit_rate`). (A first implementation compared each replicate with the previous *replicate* and flagged
  the true model - a bug caught by a statistical test and fixed.)
* **Parameter recovery.** MAE, median absolute error, RMSE, Pearson `r`, Spearman `ρ`, bias, error variance, bootstrap CIs, fraction
  on a bound; stratified by noise level; plus **error correlations** (do two parameters trade off?) and **leakage** (does the true value
  of one parameter bias the estimate of another?).
* **Model recovery.** Confusion `P(selected | generating)` and inversion `P(generating | selected)`; generators may differ from
  candidates (mis-specification); per-generator accuracy with Wilson score intervals.

## 7. Identifiability - what to expect

* `β` is hard to recover when choices are near-deterministic (likelihood flat in `β` once the best action
  already has probability ≈ 1) — this appears as estimates piling up on the upper bound.
* `α` and `β` trade off: a small `α` with a large `β` can mimic a larger `α` with a smaller `β` over short
  sequences. Recovery improves with trial count (tested).
* Separate positive/negative learning rates need more data than a single rate (Experiment 4).
* Bayesian (BAMDP / POMDP) and Q-learning models can produce similar choice sequences in some regimes; model
  recovery (Experiment 5) quantifies this rather than assuming it away.

## References

* Duff, M. (2002). *Optimal Learning: Computational Procedures for Bayes-Adaptive Markov Decision Processes.* PhD thesis, UMass Amherst.
* Ghavamzadeh, M., Mannor, S., Pineau, J., & Tamar, A. (2015). Bayesian reinforcement learning: a survey. *Foundations and Trends in Machine Learning*, 8(5–6).
* Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains. *Artificial Intelligence*, 101.
* Cassandra, A., Littman, M., & Zhang, N. (1997). Incremental pruning: a simple, fast, exact method for partially observable Markov decision processes. *UAI*.
* Littman, M. L., Cassandra, A. R., & Kaelbling, L. P. (1995). Learning policies for partially observable environments: scaling up. *ICML*. (QMDP)
* Gmytrasiewicz, P. J., & Doshi, P. (2005). A framework for sequential planning in multi-agent settings. *JAIR*, 24.
* Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8.
* Daw, N. D., O'Doherty, J. P., Dayan, P., Seymour, B., & Dolan, R. J. (2006). Cortical substrates for exploratory decisions in humans. *Nature*, 441.
* Wilson, R. C., & Collins, A. G. E. (2019). Ten simple rules for the computational modeling of behavioral data. *eLife*, 8, e49547.
* Palminteri, S., Wyart, V., & Koechlin, E. (2017). The importance of falsification in computational cognitive modeling. *Trends in Cognitive Sciences*, 21(6).
