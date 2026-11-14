# What changed in v0.2.1 -> v0.2.2, and what to do next (simple steps)

## Already done in v0.2.1 (code + tests; nothing for you to do)

| # | Change | Where |
|---|---|---|
| 1 | One-step checks now cover only the 7 **choice-only** statistics; reward statistics moved to a separate **full-session** check (no silent NaN; asking the wrong check for them raises an error) | `evaluation/predictive_checks.py` |
| 2 | The declared **30 % misfit flag** is implemented with Wilson intervals; 23-27 % is reported as "elevated, below flag threshold", not "flagged" | `misfit_rate()` |
| 3 | Participants for checks / profiles are a **seeded random draw over sorted ids** (stratified by schedule if present); shuffling the CSV cannot change them; ids saved in `config.json` | `pipelines/empirical.py` |
| 4 | One **win definition** (`reward >= threshold`) in descriptive summaries and predictive checks | `data/ingestion.py` |
| 5 | **Primary split = 70 %** + **Holm** across 3 primary contrasts; other splits = sensitivity (they overlap, so they are not replications) | `comparison/model_comparison.py` |
| 6 | **Sensitivity runner** for the declared alternatives + a stricter 10 % exclusion: `humanchoice sensitivity` | `pipelines/sensitivity.py` |
| 7 | Outputs now include `dataset_spec.json`, `environment.txt`, `analysis_plan.md` and hashes of spec / plan / git commit | `pipelines/empirical.py`, `cli.py` |
| 8 | Experiments scale with `--participants 200` / `--datasets 200`; recovery has bootstrap CIs and a strict gate (lower CI >= 0.7); two extra out-of-family generators (option bias, strategy mixture) | `experiments/` |
| 9 | `humanchoice fit --env pkg.mod:factory` plugs in the task-matched environment | `cli.py` |
| 10 | Amendment 2 of the analysis plan records all of this (before any real data) | `docs/analysis_plan.md` §12 |

### Added in v0.2.2 (third review; all tested)
| # | Change |
|---|---|
| 11 | Full-session checks reproduce each participant's **timeouts and availability**; model recovery has `missed_prob`; recovery experiments take `--env --condition --trials --missed-prob` (task-matched) |
| 12 | **Schedules:** env factories can take `condition`; `humanchoice fit` analyses pooled **and** per schedule automatically when the data have a `condition` column |
| 13 | Binary-data-safe sensitivity analysis; sensitivity runs saved with config + versions |
| 14 | Provenance enforced by **presence**, not only by "VERIFY" placeholders |
| 15 | Calibrated sign-flip p-values; finite-simulation correction for predictive-check p-values |
| 16 | `parameter_interpretability.csv` + `--recovery-gate`: each parameter labelled interpretable / unreliable / not assessed |
| 17 | Second-seed predictive-check repeat; analysis plan bundled in the wheel; CI builds and installs the wheel |

## Steps only YOU can do (in this order)

**Step 1 - 2 minutes: your name.** Replace `HumanChoice contributors` in `pyproject.toml` (`authors`) and `CITATION.cff` (`authors`). I cannot invent your details.

**Step 2 - get the data legitimately.** Open the catalogue entry (OpenCogData: *bahrami-navajas-2020*), follow its link to the OSF project, download the CSV, and **read the data-use terms**
(the catalogue's own MIT licence is not the dataset's licence). Note the published sample size you will compare against (a previous paper reports 965 of the 975 participants).

**Step 3 - fill the spec.** `cp configs/bahrami_navajas_2020.template.json configs/bahrami_navajas_2020.json`, then replace every `MAP_ME` (real column names) and `VERIFY`
(source, citation, licence, consent, task description incl. number of drift schedules and the response deadline). If the data say which of the three drift schedules a participant had, add it
as a `condition` column in `columns`. Set `path` and `sha256`. Check: `humanchoice validate-data --spec configs/bahrami_navajas_2020.json`.

**Step 4 - make the simulation match the task** (this is the one step that needs reading the task description). Write `src/humanchoice/envs/bahrami_navajas.py` with a class that
reproduces the real initial payoffs, the **three drift schedules**, reward noise, 4 options and 150 trials. Then re-run recovery on *that* environment with a bigger sample:

```bash
# your environment must be a zero-argument callable, or take a parameter named `condition` / `schedule` (the reward-drift schedule)
for S in 1 2 3; do   # one run per verified schedule
  python experiments/identifiability/run.py --participants 200 --env humanchoice.envs.bahrami_navajas:make_env --condition $S --trials 150 --missed-prob <observed rate>
  python experiments/model_recovery_ladder/run.py --datasets 200 --env humanchoice.envs.bahrami_navajas:make_env --condition $S --trials 150 --missed-prob <observed rate>
done
```
Rule: a parameter is interpretable only if the **lower CI bound of rho is >= 0.7**. Each run writes `results/exp10_recovery_gate.csv` (model, param, status) - keep one per schedule and pass it to `fit` with `--recovery-gate`.

**Step 5 - run the analysis.**

```bash
humanchoice fit         --spec configs/bahrami_navajas_2020.json --out results/empirical_bandit --jobs 4 --env humanchoice.envs.bahrami_navajas:make_env --recovery-gate results/exp10_recovery_gate.csv
humanchoice sensitivity --spec configs/bahrami_navajas_2020.json --out results/empirical_bandit_sensitivity --jobs 4
```
With a `condition` column the output has `pooled/` and `condition_<schedule>/` folders. Read each `report.md` (parameters not passing the gate are labelled "do not interpret"); report **pooled and by schedule**; do not change the plan after looking (record any deviation in the report's deviations section).

**Step 6 - reproduce one published result** with the same data (fill `docs/published_result_reproduction.md`). Name the exact figure/table, compare methods, explain differences.

**Step 7 - write the empirical section** of `docs/technical_report.md`. Only then change the README's status banner from "no human dataset analysed".

**Step 8 - release hygiene.** `git tag v0.2.1`; build from a clean checkout (no `__pycache__`, `_quick_output/`, `_run_state/`); CI green; `python -m build` and install the wheel in a fresh venv.

## Deliberately later (do not start before Step 5 is done)
Hierarchical estimation; multi-block fitting with shared parameters; reaction-time models; PyPI release; moving the planning layer into a separate package (decide this against your application: the
internship brief asks for exactly that planning library, so keep it visible in *that* application).
