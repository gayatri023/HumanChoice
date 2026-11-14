# Research notes: assumptions, limitations and status

## 1. Scientific status - read this first

* **Every dataset in this repository is synthetic.** Results show that the *inference pipeline works on data whose generating process is
  known*. They say nothing about human cognition. Every `results/*_config.json` carries `"synthetic_data": true`, and the report of
  the empirical-pipeline rehearsal carries a "SYNTHETIC REHEARSAL - NOT human data" banner.
* **No human dataset has been analysed.** What exists is the machinery to do so correctly: a dataset-spec loader with provenance and file
  hashing, pre-specified exclusions, missed-trial handling, K-option models, a pre-specified analysis plan
  ([`docs/analysis_plan.md`](docs/analysis_plan.md)) and a one-command pipeline that has been rehearsed end to end on simulated participants.
  The candidate dataset named in the project brief (Bahrami & Navajas, 2020; OpenCogData catalogue) is hosted on OSF; it could not be
  retrieved in the environment used to build this version, and its licence and exact schema were not verified. The spec template therefore
  marks every such field `VERIFY` and `humanchoice fit` refuses to run until they are confirmed.
* Parameters are interpreted only as *model parameters*. No psychological or clinical interpretation is claimed.

## 2. Status against the review recommendations

| Recommendation | Status | Notes |
|---|---|---|
| Generalise from 2 to K actions (data, models, simulation, fitting) | **Done** | any K ≥ 2; two-option-only models declare and enforce it; tests for K = 2, 3, 4, 7 |
| Reward types: do not treat continuous rewards as Bernoulli | **Done** | `reward_support` on models; binary-only models raise on other rewards; `skip_incompatible` option |
| Missing / unavailable actions | **Done** | missed trials (`-1`) and per-trial availability masks; equivalence tests |
| Canonical schema with block / condition / RT / availability | **Mostly** | optional columns retained; blocks handled by splitting into independent sequences (parameters are *not* shared across blocks); no multi-block likelihood |
| Baselines (random, WSLS) | **Done** | |
| Asymmetric, sticky (perseveration), lapse Q-learning | **Done** | nested warm start fixes an optimisation failure found during this work |
| Volatility-sensitive / Bayesian belief-state model | **Done (one model)** | `KalmanBandit` (drift ratio = believed volatility, uncertainty bonus); not validated on real data |
| Held-out prediction, Brier, calibration, bootstrap CIs, paired contrasts, repeated splits | **Done** | |
| Posterior predictive / systematic-error checks | **Done (v0.2.1)** | choice-only one-step checks + outcome checks in a supplied environment; declared 30 % flag with Wilson intervals; seeded, order-independent participant selection |
| Parameter uncertainty / likelihood profiles | **Done** | profile intervals with an "open interval" flag |
| Parameter-pair correlations / leakage; recovery vs trials, volatility, noise, K | **Done** | Experiment 10 |
| Model recovery with uncertainty, nested and mis-specified generators | **Done** | Experiment 11 (Wilson intervals; `full_q` mis-specified generator) |
| Real human dataset analysed | **NOT done** | blocked: data not retrievable here, licence unverified; pipeline + plan ready |
| Separate planning code (BeliefLab) from behavioural code | **Partly** | dependency direction **enforced by a test** (planning never imports behavioural code; importing it does not load them); the code was *not moved* because the target internship asks for exactly this planning library and the other repository was not available to coordinate with |
| CLI | **Done** | `humanchoice validate-data / fit / reproduce / list-models` |
| CITATION.cff, CHANGELOG | **Done** | author field is a placeholder |
| Dataset hashes, seeds, versions in outputs | **Done** | |
| Benchmarks at realistic scale | **Partial** | `experiments/benchmarks.py` times fits and extrapolates; no large-scale run |
| Hierarchical estimation | **Not done** | deliberately deferred (see the analysis plan and specification §13) |
| Sensitivity analyses | **Runner done, real-data run pending** | `humanchoice sensitivity` (WSLS threshold, starts, Kalman noise, reward-magnitude scaling [continuous data only], 10 % exclusion) |
| Timeouts / availability / schedules in full-session checks and recovery | **Done (v0.2.2)** | observed timeout pattern reproduced; condition-aware environments; pooled + per-schedule runs; `--env` task-matched recovery. **A task-matched environment for the real dataset still has to be written from its task specification** |
| Calibration of the primary hypothesis test | **Done (v0.2.2)** | paired sign-flip permutation p-values; null calibration test |
| Statistical design (primary split, multiplicity, recovery gate with CIs) | **Amended (v0.2.1)** | analysis plan Amendment 2; the 200-participant recovery run is a command to execute (`--participants 200`), not yet run |
| Technical report | **Partial** | `docs/technical_report.md` documents methods and *synthetic* results; the empirical section is to be written after the real analysis |
| "Freeze the alpha" | **Not done here** | no git history in this workspace: run `git tag v0.1.0` on the earlier snapshot; v0.1.0 is also kept as a separate zip |

## 3. Modelling assumptions

| Assumption | Where | Consequence |
|---|---|---|
| Softmax decision rule with one inverse temperature (plus optional `κ`, `ε`) | all learning models | no choice-history effects beyond one-back perseveration; no reaction-time model |
| Fixed hyper-parameters: initial value 0.5, Kalman `obs_sd / decay / m₀ / v₀`, WSLS threshold, Beta prior, BAMDP lookahead | models | results are conditional on them; sensitivity analyses are declared in the plan |
| `BayesBanditModel` assumes a stationary bandit; its lookahead is fixed, not the true remaining horizon | `models/bayes_bandit.py` | deliberately mis-specified on reversal tasks; exact planning is practical only for small K |
| "Hybrid" = Q-learning + POMDP hidden-state tracker | `models/hybrid.py` | **not** the classical model-free / model-based two-step hybrid; no two-step-task model is implemented |
| Per-participant maximum-likelihood point estimates | `fitting` | no shrinkage; weakly identified parameters pile up on bounds (reported as `frac_at_bound`) |
| Multi-start L-BFGS-B with finite-difference gradients + nested warm start | `fitting` | a global optimum is not guaranteed; `start_agreement` exposes rugged surfaces |
| Chronological held-out splits within one session | `comparison` | tests generalisation to *later* trials of the same participant, not to new participants or tasks |
| One-step-ahead predictive checks share the observed history | `evaluation/predictive_checks.py` | they test the decision/updating rules, not free-running behaviour (free-running check available) |
| Rewards rescaled using the task's *stated* range | data layer | no information leaks across participants; a wrong stated range would mis-scale `β` and `φ` |

## 4. Planning-model limitations (unchanged)

* **BAMDP** - exact belief-state enumeration; cost explodes with horizon, arms and states. A reference implementation, not a scalable
  planner (no BAMCP-style sampling).
* **POMDP** - exact alpha-vector solving is worst-case exponential; no point-based solver (PBVI/SARSOP).
* **I-POMDP** - a *finite level-1 approximation*: non-strategic level-0 other agent, a single known frame, truncated model closure, noisy-optimal
  level-0 policy; no recursive (level ≥ 2) reasoning and no uncertainty over the other agent's frame.
* The two-agent Tiger of Experiment 8 is a simplified variant of the Gmytrasiewicz & Doshi (2005) benchmark; its numbers are not comparable to
  published values. The level-1 advantage assumes the other agent is exactly as modelled (an upper bound) plus one mis-specified condition.

## 5. Validation strategy (and its limits)

1. **Hand-derived values** (Tiger horizons 1-3, BAMDP horizon 2 = 13/12, Q-learning and WSLS and Kalman updates, Wilson interval, profile arithmetic).
2. **Independent implementations must agree** (alpha-vectors vs belief-tree; generic vs bandit BAMDP; value vs policy iteration; I-POMDP vs POMDP under a
   degenerate other agent; **fast Q-family likelihood vs generic replay** for every variant, with missed trials).
3. **Mathematical invariants** (probabilities valid for any K; nesting identities; missed trial = deleted trial; unavailable options get probability 0).
4. **Statistical validation** (`tests/statistical_validation/`): estimator consistency, calibration of the true model, false-positive rate of the predictive
   checks, information-criterion selection with enough data, and a nesting-invariant regression test.
5. **Regression tests for bugs found by the tests themselves** (two are recorded in `CHANGELOG.md`): a predictive check that flagged the true model, and
   an optimiser that left a richer model worse than the model it contains.
6. **Architecture test**: the planning layer cannot import the behavioural layer.
7. **README and CLI are executed by the test suite.**

Limits: tests check agreement with theory and with each other, not agreement with published numbers for full benchmark problems; statistical tests use small
seeded samples (regression tests, not proofs); the empirical pipeline has not met real data, so unforeseen real-data problems (e.g. participants who never switch,
quirky timeout coding) will only appear then.

## 6. Reproducibility

* Every random stream is spawned from `numpy.random.SeedSequence`; results do not depend on participant order.
* `sh experiments/run_all.sh` regenerates all tables and figures; `--quick` writes to `_quick_output/` and never touches `results/`.
* Every analysis output records seed, versions of Python / NumPy / SciPy / pandas / Matplotlib / HumanChoice, platform, UTC time, git commit (if available), the SHA-256 of the data file, of the dataset spec and of the analysis plan, and the selected participant ids; the spec, the plan and an `environment.txt` are copied next to the results.

## 7. Next steps (in order of value)

1. Obtain the real dataset legitimately (check licence), fill in `configs/bahrami_navajas_2020.template.json`, run `humanchoice validate-data`, then `humanchoice fit`.
2. Run the declared sensitivity analyses and write the empirical section of the technical report, including a deviations log.
3. Hierarchical estimation, *after* the individual-level analysis is validated on real data.
4. Point-based POMDP solving and sampling-based BAMDP planning so the planning layer scales; extract it into its own package if desired (the boundary test keeps this cheap).
5. A classical two-step task with model-free / model-based weighting; multi-block likelihoods with shared parameters.
