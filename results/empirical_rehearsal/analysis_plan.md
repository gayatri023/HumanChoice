# Analysis plan (pre-specified before any real data are fitted)

**Status:** fixed (amendments are listed in §11-§12 and were made before any real data were inspected; where an amendment changes a rule above, the amendment governs). The pipeline in `humanchoice.pipelines.run_empirical_analysis` implements exactly this plan and has been
rehearsed only on **synthetic** data (`results/empirical_rehearsal/`). No human dataset has been analysed. Any change made after
real data have been inspected must be recorded in a *deviations log* at the end of the final report, with the reason.

## 1. Research question

> How well do reward-learning models explain human choices in a changing multi-armed bandit, and does accounting for
> **asymmetric learning from gains and losses** or for **choice perseveration** improve predictions of *future* choices?

Target data: a public human restless-bandit dataset (four options, continuous rewards, ~150 trials per participant). The
candidate named in the project brief is Bahrami & Navajas (2020), catalogued by OpenCogData; **its licence, exact schema and
task details must be confirmed from the source before use** (`configs/bahrami_navajas_2020.template.json` marks every such
field `VERIFY`, and `humanchoice fit` refuses to run until they are filled in).

## 2. Hypotheses

Let `Δ(X, Y)` be the mean paired difference in held-out log-likelihood per trial between model X and model Y, across participants.

| | Hypothesis | Contrast | Status |
|---|---|---|---|
| H1 | Value learning explains choices beyond simple heuristics | `Δ(Q, WSLS) > 0` (primary); `Δ(Q, random) > 0` (H1b, sanity) | primary |
| H2 | Choice perseveration improves prediction | `Δ(sticky Q, Q) > 0` | primary |
| H3 | Asymmetric learning rates improve prediction | `Δ(asym Q, Q) > 0` | primary |
| H4 | Uncertainty-sensitive (Bayesian/Kalman) tracking improves prediction | `Δ(Kalman, Q) > 0` | exploratory |
| H5 | Occasional value-independent lapses improve prediction | `Δ(lapse Q, Q) > 0` | exploratory |

## 3. Data handling (all rules fixed in advance)

* **Missed trials** (no response) are kept in the sequence, receive no learning update and do not enter any likelihood
  (`missing="mark"`). They are never imputed.
* **Rewards** are rescaled linearly to [0, 1] using the **task's stated range** (not the observed min/max).
* **Exclusions** (participant level, nothing else): more than **20 %** missed trials, or fewer than **100** valid trials. No
  exclusion may depend on model fit, and every excluded participant is listed with its reason (`excluded_participants.csv`).
* The dataset file's SHA-256 and the spec (column mapping, ranges, provenance) are stored with the results.

## 4. Models (the ladder; each row adds one mechanism)

`random` -> `wsls` (baseline heuristics) -> `q` -> `asym_q` / `sticky_q` / `lapse_q` (one added mechanism each) and `kalman`
(Bayesian tracking). Fixed hyper-parameters: initial value 0.5; Kalman `obs_sd = 0.1`, `decay = 1`, `m0 = 0.5`, `v0 = 0.1`;
WSLS win threshold 0.5. Parameters, bounds and equations: `docs/theory.md`. The ladder is not extended after seeing data.

## 5. Primary outcome and decision rule *(as amended; Amendment 2 is kept below as history)*

* **Primary outcome:** held-out log-likelihood per trial. Fit on the **first 70 %** of each participant's trials and predict the remainder one step ahead.
* **Primary contrasts (one Holm family):** H1 (`Δ(Q, WSLS)`), H2 (`Δ(sticky Q, Q)`), H3 (`Δ(asym Q, Q)`). Paired participant-level difference with a bootstrap 95 % CI and a
  **paired sign-flip permutation p-value**; p-values are Holm-adjusted over the three. *Supported* / *contradicted* if the adjusted p < .05 in that direction, otherwise *not supported*.
  `Δ(Q, random)` (H1b, sanity) and H4, H5 (exploratory) are reported unadjusted.
* **Sensitivity (not replications):** the 50 %, 60 % and 80 % splits reuse the same participants and overlapping data; they are reported only as the number of splits agreeing with the primary verdict.
* **Secondary outcomes (reported, not decisive):** multi-class Brier score, calibration (reliability diagram, ECE), AIC/BIC wins and Akaike weights, choice accuracy.
* **Schedules:** if the data carry reward-drift schedules, every result is reported pooled **and** per schedule (schedules with fewer than 30 participants are listed as skipped, not hidden).

## 6. Gates before interpreting parameters *(as amended)*

Parameter estimates are interpreted **only** for parameters whose recovery was assessed on simulations matching the dataset (same options, trials, missingness, and a task-matched
reward process) with **>= 200 simulated participants per cell** and whose **lower 95 % bootstrap bound of Spearman rho is >= 0.7**. The empirical report labels every fitted parameter
`interpretable`, `unreliable` or `not assessed` from a machine-readable gate table (`parameter_interpretability.csv`); without a gate table every parameter is labelled *do not interpret*.
For individual participants, likelihood-profile intervals that stay *open* within the bounds are flagged as not identified.

## 7. Model checking *(as amended)*

* **One-step-ahead checks (choice-only statistics, 7):** overall stay rate, stay after high / low reward, stay in the first / last third, choice entropy, option concentration.
* **Full-session checks (outcome statistics, 4):** mean reward, total reward, reward spread, improvement over the session - simulated whole sessions, **only** in an environment that
  reproduces the dataset's reward process for that participant's schedule; simulated sessions reproduce the participant's *actual timeout pattern* and availability schedule so the
  statistics are computed over the same trials. If no such environment exists the report says "not run".
* A model is **flagged** on a statistic only if more than **30 %** of participants are outside the replicate interval (p < .05); the report gives the rate, a Wilson 95 % interval and the chance
  level (5 %). Rates credibly above chance but not above 30 % are *elevated, below flag threshold*. The checks are repeated with a second seed (new participant draw and replicates) and the
  change in rates and flags is reported. Replicate p-values use the finite-simulation correction `2(k+1)/(n+1)`.
* "Win" is `reward >= win_threshold` (default 0.5 on the [0, 1] scale) in both descriptive and predictive statistics.

## 8. Sensitivity analyses *(as amended; run with `humanchoice sensitivity`)*

1. WSLS / stay-statistic threshold: 0.4 and 0.6.
2. Optimiser starts: 4 vs 10.
3. Kalman `obs_sd`: 0.05 and 0.2.
4. Reward-magnitude scaling (x 1/1.1, x 1/0.9) - **continuous-reward data only**; not applicable to binary rewards (reported as such).
5. Stricter exclusion: more than 10 % missed trials (baseline 20 %).
The report states whether each primary verdict changes under each alternative.

## 9. What would count against the models

* `Δ(Q, WSLS) ≤ 0`: value learning no better than a heuristic.
* Held-out calibration far from the diagonal, or all models flagged on stay statistics: the model class is missing a mechanism.
* Recovery gate failing for the parameter a claim depends on.

## 10. Reporting

The report (`report.md`) is generated by the pipeline and always contains: data quality, exclusions, the full ladder with
optimisation diagnostics, held-out results with CIs, calibration, predictive-check misfit rates, profile-interval
summaries, and a deviations log. Negative and null results are reported with the same prominence as positive ones.

## 11. Amendment 1 - interpretation rules from the synthetic design analysis (10 Nov 2026; before any real data were inspected)

**Why.** The rehearsal (Experiment 9) and the identifiability study (Experiment 10) showed that at the planned design (4 options, 150 trials,
~3 % timeouts) the lapse rate (rho = 0.03), the negative learning rate (rho = 0.25) and the believed volatility (rho = 0.49) are **not recoverable**, and that
the asymmetric-learning contrast came out *contradicted* on simulated data in which a fifth of participants were genuinely asymmetric. A non-supported
result on such a mechanism would therefore carry little information about the mechanism itself.

**Change.** The hypotheses, models, exclusions, primary outcome and decision rule above are unchanged. Added:

1. A verdict of *not supported* or *contradicted* for H3, H4 or H5 is reported as **inconclusive about the mechanism** unless the recovery gate of §6 is met for that
   mechanism's parameters at the dataset's actual number of trials and missingness (Experiment 10 re-run for the dataset). A *supported* verdict stands on its own: it
   is a statement about prediction, and the gate only limits parameter interpretation.
2. Before analysing the real data, Experiment 10 is re-run with the dataset's actual trials per participant, timeouts and (if documented) payoff schedule; its output
   is stored with the report.
3. If the real task has more trials, more volatility or more options than assumed, the gate is re-evaluated with the new design rather than reusing the numbers above.

No other aspect of the plan was changed by looking at synthetic results.

## 12. Amendment 2 - statistical and reporting corrections (10 Nov 2026; before any real data were inspected)

Made after an independent review of the pipeline. Items 1-4 correct defects found in the code; 5-8 tighten the statistical design. The
hypotheses, models and primary outcome (held-out log-likelihood per trial) are unchanged.

1. **Two families of predictive checks (replaces §7's single list of eight statistics).**
   *One-step-ahead checks* evaluate seven **choice-only** statistics: overall stay rate, stay after high / low reward, stay in the first / last third,
   choice entropy and option concentration. *Full-session checks* evaluate the **outcome** statistics - mean reward, total reward, reward spread and
   improvement over the session - by simulating whole sessions, and are run **only** in an environment that reproduces the dataset's reward process
   (the stated drift schedule). If no such environment exists the report says "not run"; no generic bandit is substituted. No statistic may disappear
   silently (previously `mean_reward` returned NaN in the one-step check).
2. **Misfit flag.** Per model and statistic the report gives the misfit rate (p < .05), a Wilson 95 % interval and the chance level (5 %). A model is
   **flagged** only if the rate exceeds **30 %**. Rates credibly above 5 % but not above 30 % are reported as *elevated, below flag threshold* - not as
   flags.
3. **Participant selection.** Participants for predictive checks and for profile intervals are chosen by a **seeded random draw over sorted participant ids**
   (stratified by reward-drift schedule when the data carry one), so shuffling the data file cannot change the selection. Profile participants are spread over
   the range of fit quality. Selected ids are stored in `config.json`; the checks are repeated with a second seed as a sensitivity analysis.
4. **One definition of a "win".** Descriptive summaries and predictive checks both use `reward >= win_threshold` (default 0.5 on the [0, 1] scale). For
   continuous rewards these are stay rates after *high* / *low* reward, not after "gains" / "losses".
5. **Primary split and multiplicity (replaces the "3 of 4 splits" rule of §5).** The four split points overlap and are not independent replications. The
   **primary analysis is the 70 % training split only**; the 50 %, 60 % and 80 % splits are **sensitivity analyses**. Primary contrasts are H1 (Q vs WSLS),
   H2 and H3; their bootstrap p-values are **Holm-adjusted** as one family. `Q vs random` (H1b) is a sanity check and H4, H5 are exploratory; both are
   reported unadjusted. A verdict is *supported* / *contradicted* if the adjusted p < .05 in that direction, otherwise *not supported*.
6. **Recovery gate with uncertainty (replaces §6's point-estimate rule).** The headline recovery study uses **>= 200 simulated participants per cell** and
   bootstrap CIs. A parameter is interpretable only if the **lower 95 % bound of Spearman rho is >= 0.7** at the dataset's design; a point estimate just above 0.7 is
   not sufficient. The 40-participant results in the README are preliminary and carry sampling error of about +/- 0.2.
7. **Model recovery.** The headline run uses >= 200 datasets per generator and includes out-of-family generators: asymmetric + perseverative + lapsing
   (`full_q`), an unmodelled option bias, and a mixture of strategies.
8. **Sensitivity analyses to be run** (`humanchoice sensitivity`): WSLS threshold 0.4 / 0.6; 10 optimiser starts; Kalman `obs_sd` 0.05 / 0.2; reward range
   +/- 10 %; **stricter exclusion at 10 % missed trials** (vs the 20 % in §3). The report must show whether each primary verdict changes.
9. **Reward-drift schedules.** If the data have several predefined schedules, every analysis is reported **pooled and by schedule** (sample sizes permitting),
   with the schedule stored as a participant-level `condition`. The synthetic design analysis (Experiment 10) is re-run with an environment that reproduces each
   schedule *after the task specification has been verified from the source*; information about unchosen options, if available, may be used to build the
   environment but must never be given to a participant model as if it had been observed.
10. **Reproduction of a published result** (`docs/published_result_reproduction.md`) precedes the extension: identify one specific figure / table / result from a
    published analysis of the same data, reproduce it, and document every methodological difference.
