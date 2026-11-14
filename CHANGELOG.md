# Changelog

## 0.2.2 - fixes from the third review (all verified by tests; see docs/analysis_plan.md, active sections 5-8)

**Fixed**
- Full-session predictive checks now reproduce the participant's **actual timeout pattern** (`missed="observed"`, default) or an explicit timeout model (`missed=<p>`), and respect
  the **availability schedule**; simulated and observed statistics are computed over the same trials. `simulate()` gained `missed_mask`.
- Sensitivity analysis no longer corrupts **binary** data: the reward-magnitude scaling is reported as *not applicable* for binary rewards (it would break the Bernoulli observation model). It is documented as a magnitude scaling.
- **Provenance can no longer be bypassed by omission**: a non-synthetic spec must state `source_url`, `citation`, `licence` (or `UNAVAILABLE: <documented reason>`).
- Model recovery accepts `missed_prob`; both recovery experiments take `--env/--condition/--trials/--missed-prob` for **task-matched** runs.
- The primary p-values are now **paired sign-flip permutation** tests (null calibration verified: ~5 % false positives, heavy-tailed data included); predictive-check p-values use the finite-simulation
  correction `2(k+1)/(n+1)` and are never exactly 0.
- `--max-participants` help text; `configs/` path resolution documented (paths are relative to the JSON file; template corrected); the **analysis plan now ships inside the wheel** (`humanchoice/resources/`).

**Added**
- Condition-aware environments (`env_factory(condition)`), `run_empirical_by_condition` and automatic **pooled + per-schedule** analysis in `humanchoice fit` when the data have a `condition` column.
- Second-seed repeat of the predictive checks (rates and flag changes reported); `parameter_interpretability.csv` + report section labelling every parameter `interpretable / unreliable / not assessed`
  from a machine-readable gate (`recovery_gate_report`, `--recovery-gate`); sensitivity runs save their configuration, seed, data hash and package versions.
- CI: clean wheel build + install + CLI smoke test. Slow pipeline tests now use a small model ladder (the full suite is ~4 min; `pytest -m "not slow"` is the fast subset).


## 0.2.1 - analysis-pipeline corrections (found by an independent review; see docs/analysis_plan.md Amendment 2)

**Fixed**
- Predictive checks: `mean_reward` was reported as NaN by the one-step check. Choice-only statistics (7) and outcome statistics (4) are now separate families; outcome
  statistics are evaluated only by the free-running check in a supplied environment, and the report says "not run" otherwise. Asking the one-step check for reward statistics raises.
- The declared 30 % misfit threshold was never applied. `misfit_rate` now reports Wilson intervals and `population_flag`; 23-27 % is "elevated, below flag threshold".
- Predictive checks / profiles used the first participants in file order; selection is now a seeded draw over sorted ids (stratified by schedule), stored in `config.json`.
- `participant_summary` called any reward > 0 a win while the checks used `>= 0.5`; both now use one configurable threshold.
- Saved figures are closed after writing (a long analysis leaked one matplotlib figure per call).

**Changed (pre-specified, before any real data)**
- Decision rule: primary analysis at the 70 % split with Holm adjustment over three primary contrasts; other splits are sensitivity analyses (they overlap).
- Recovery gate: lower bootstrap CI of Spearman rho >= 0.7, headline runs with >= 200 simulated participants.

**Added**
- `humanchoice sensitivity` (declared sensitivity analyses + 10 % exclusion), `--env` option, provenance artefacts (`dataset_spec.json`, `environment.txt`, `analysis_plan.md`, hashes, git commit),
  `--participants` / `--datasets` experiment flags, option-bias and strategy-mixture generators, `docs/NEXT_STEPS.md`, `docs/published_result_reproduction.md` (template).

## 0.2.0 - behavioural-modelling upgrade

**Added**
- **K-option models and data.** Every choice model, environment, simulator, fitter and dataset class now supports any number of options
  (previously two); two-option-only models declare and enforce it. Models declare whether they need binary rewards and refuse continuous ones.
- **Missed trials** (timeouts) as first-class data: kept in the sequence, no learning, excluded from likelihoods, simulated via `missed_prob`.
  Optional per-trial **action availability** masks.
- **Model ladder:** random choice, win-stay/lose-shift, sticky (perseverative) Q-learning, lapse Q-learning, full Q-learning, and a
  Kalman-filter Bayesian learner for restless bandits with continuous rewards. Q-family likelihood is ~20x faster (validated against the generic path).
- **Environments:** K-armed Bernoulli, drifting-probability Bernoulli, Gaussian restless bandit.
- **Data layer:** declarative dataset specs (column mapping, pre-specified exclusions, provenance, SHA-256), data-quality report, optional-column
  retention, block splitting, reward rescaling from a *stated* range.
- **Evaluation:** Brier score, calibration / ECE, Wilson and bootstrap intervals, paired model contrasts, repeated temporal splits, one-step-ahead and
  free-running predictive checks, likelihood profiles, parameter error-correlation and leakage diagnostics, recovery with mis-specified generators.
- **Pipeline + CLI:** `humanchoice validate-data | fit | reproduce | list-models`; one call runs the pre-specified analysis and writes a report.
- Documentation: `docs/analysis_plan.md` (pre-specified), `configs/` (dataset spec template), updated theory and research notes.
- Tests: 120 -> see README (K-option, data layer, diagnostics, statistical validation, architecture boundary, CLI/pipeline).

**Fixed**
- Optimisation: random multi-start could leave a model that nests another (e.g. asymmetric Q-learning) with a *worse* likelihood than the simpler model;
  nested warm starts now prevent this (regression-tested).
- Predictive checks: replicate "stay" statistics were computed against replicate rather than observed history (flagged the true model); fixed.
- Datasets re-read from CSV now recognise the explicit `<missed>` label.

**Unchanged:** MDP / BAMDP / POMDP / I-POMDP planning code and its mathematical validation tests.

## 0.1.0
Initial alpha: planning layer, two-option behavioural models, recovery experiments on synthetic data.
