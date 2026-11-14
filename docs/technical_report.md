# Identifiability and prediction in K-option bandit learning models: a design analysis on synthetic data

*HumanChoice v0.2 technical report. All results are from simulated participants with known generating models; none concerns human behaviour.
The empirical section is deliberately left to be written after the pre-specified analysis ([`analysis_plan.md`](analysis_plan.md)) is run on real data.*

## Abstract

Fitting learning models to bandit choices yields parameters that are easy to over-interpret. Before analysing human data we asked, on simulated four-option restless-bandit
data with timeouts (150 trials per participant): (i) which mechanisms of a seven-model ladder can be recovered, (ii) which models can be told apart, and (iii) whether a
pre-specified analysis, rehearsed end to end, reaches correct conclusions. Perseveration is cleanly identifiable (Spearman rho = 0.94 between generating and recovered values);
the lapse rate (0.03), the asymmetry of learning (alpha-: 0.25) and the believed volatility of a Kalman learner (0.49) are not, and even the learning rate is weak (0.40).
Information criteria recover perseverative participants (12/18) but none of the asymmetric or lapsing ones, and held-out prediction ranks the perseverative model first while rating the
asymmetric model *worse* than plain Q-learning. We draw the implication for the planned real-data study and amend its analysis plan accordingly.

## 1. Question and design

*How well do reward-learning models explain choices in a changing multi-armed bandit, and does asymmetric learning or perseveration improve prediction of future choices?*
The task design assumed here (4 options, continuous payoffs drifting as a mean-reverting random walk, ~150 trials, ~3 % missed responses) is a simulation of the design commonly cited
for Daw et al. (2006) with payoffs rescaled to [0, 1]; it is not the schedule of any specific dataset. A candidate public dataset (Bahrami & Navajas, 2020, via the OpenCogData catalogue)
could not be retrieved or licence-checked in the build environment and has not been analysed.

## 2. Models

Seven models, ordered by added mechanism (equations: [`theory.md`](theory.md)): random choice; win-stay/lose-shift; Q-learning; and Q-learning extended with asymmetric learning rates, perseveration (`kappa`),
or a lapse rate (`epsilon`); and a Kalman-filter learner with an uncertainty bonus and a believed volatility (`drift ratio`). All use a softmax rule. Missed trials produce no learning and no likelihood term.

## 3. Methods

* **Estimation.** Per-participant maximum likelihood (bounded L-BFGS-B, 4 random starts plus a *nested warm start* from the simpler model; without the warm start a model could end with a worse likelihood than the model it contains).
* **Recovery.** 40 participants per cell, generating parameters drawn from each model's sampling ranges; Spearman rho (robust to the heavy-tailed errors of the inverse temperature), Pearson r, median absolute error, bias, error correlations and leakage between parameters.
* **Model recovery.** 20 datasets per generator, every candidate fitted to every dataset, BIC/AIC selection, Wilson 95 % intervals; one generator (`full_q`) matches no candidate.
* **Rehearsal of the empirical pipeline.** 60 participants from a known mixture, run through the full pre-specified analysis: held-out log-likelihood per trial at four chronological split points with participant-level bootstrap CIs and the pre-specified contrasts;
  Brier score and calibration; one-step-ahead predictive checks (eight behavioural statistics); likelihood-profile intervals; audit against the truth.

## 4. Results

### 4.1 Identifiability (4 options, 150 trials)
See the table in the [README](../README.md#2-which-parameters-are-identifiable-experiment-10) and [`results/exp10_identifiability_summary.md`](../results/exp10_identifiability_summary.md). Parameters meeting rho >= 0.7: inverse temperature (Q, Kalman), perseveration, learning rate and
inverse temperature of the lapse model, and (barely) the learning rate of the perseverative model and the Kalman uncertainty bonus. Failing: the Q-learning learning rate (0.40), alpha+ (0.60) and alpha- (0.25), the
lapse rate (0.03), the Kalman drift ratio (0.49). Recovery improves with volatility (drift SD 0.01 -> 0.06), with more options (2 -> 8) and with more trials (75 -> 300), but alpha- (0.53) and the lapse rate (0.29) are still below the gate at 300 trials.
Errors in the lapse model's learning rate and inverse temperature are correlated at -0.40, and those of inverse temperature and lapse rate at +0.31: lapses are partly absorbed by apparent noise.

### 4.2 Model recovery
BIC accuracy by generator (chance 1/7): random 95 %, WSLS 100 %, Kalman 80 %, Q-learning 65 %, perseverative 55 %, asymmetric 5 %, lapse 5 %; the mis-specified `full_q` data are attributed to the perseverative model in 65 % of datasets.
Asymmetric and lapse data are mostly assigned to plain Q-learning (60 % and 70 %): their extra parameters are not worth their cost at this sample size.

### 4.3 Rehearsal of the pre-specified analysis
Held-out log-likelihood per trial (fit on the first 70 %): perseverative Q -1.208 > Kalman -1.232 ~ Q -1.235 > lapse -1.240 > asymmetric -1.257 > WSLS -1.325 > random -1.386. Pre-specified decisions: H1 (value learning beats heuristics) *supported* (Holm-adjusted p = 0.010), H2 (perseveration helps) *supported* (0.010), H3 (asymmetry helps) *contradicted* (0.003), H4 and H5 *not supported*; the other split points agree
(they overlap with the primary split and are sensitivity checks, not replications). Predictive checks: no learning model exceeds the declared 30 % flag threshold; models without perseveration show an *elevated* stay-rate misfit (23-27 % of participants,
lower Wilson bounds 14-17 % vs 5 % chance) and the perseverative model none; full-session reward checks in the correctly specified environment show chance-level misfit (3-10 %). Profile intervals for the learning rate are open for 33-50 % of the profiled
participants; the perseveration interval is closed for all. The pipeline's conclusions agree with the generating truth wherever
the truth is identifiable, and *fail informatively* where it is not.

### 4.4 Consequences for the real study
* **Perseveration (H2) is testable** at the planned design, by prediction, by information criteria and by parameter (rho = 0.94).
* **Asymmetry (H3), lapses (H5) and believed volatility (H4) are not.** A null or negative result for them at ~150 trials says little about the mechanism (Amendment 1 of the analysis plan makes this an explicit interpretation rule).
* To test asymmetry one needs more informative data: more trials, a more volatile payoff process, more options, or pooling across participants (hierarchical estimation, not yet implemented).

## 5. Software validation (summary)
Hand-derived values and independent implementations (alpha-vector vs belief-tree POMDP solvers, generic vs specialised BAMDP, fast vs generic likelihood, I-POMDP vs POMDP under a degenerate other agent); invariants (valid probabilities for any K, missed trial = deleted trial, nesting identities);
statistical tests (estimator consistency, calibration of the true model, false-positive rate of the checks); regression tests for two bugs the tests exposed (a predictive check that flagged the true model; an optimiser that left a nested model worse than its parent). See the README.

## 6. Limitations
Simulated generating models are drawn from the same family that is fitted (the mis-specified generator is the only exception); real behaviour will violate more assumptions. Per-participant ML ignores population structure. Fixed hyper-parameters (Kalman noise, initial values, WSLS threshold) were not varied.
Sample sizes in the reported runs (40 per cell, 20 per generator) leave sampling errors of roughly +/-0.2 on single recovery correlations; the amended plan requires >= 200 and a CI-based gate, which has not yet been run. The simulated payoff design may differ from a real dataset's.

## 7. Empirical section (to be written)
Dataset provenance and licence; exclusions and data-quality report; behavioural description; the ladder's held-out results and the five pre-specified contrasts with the amended interpretation rules; recovery gate re-run at the dataset's design; predictive checks; sensitivity analyses; deviations log.
