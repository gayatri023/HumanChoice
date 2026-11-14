"""The empirical-analysis pipeline: data quality -> model ladder -> held-out prediction -> checks -> report.

One call, :func:`run_empirical_analysis`, executes a *pre-specified* analysis (see ``docs/analysis_plan.md``) on any
:class:`~humanchoice.data.BehaviouralDataset` and writes every table, figure and a markdown report to a directory.
Because the analysis is a function of the dataset only, running it on a rehearsal dataset with known generating models
(:func:`simulate_restless_rehearsal`) tests the *whole* pipeline, including missed-trial handling, before real data are
touched; running it on a real dataset needs no code changes.

Nothing is hidden: failed convergence, parameters on bounds and misfitting statistics are all reported.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata as md
import importlib.resources as ires
import inspect
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import humanchoice
from humanchoice.comparison import planned_contrasts, repeated_temporal_splits, summarise_comparison, summarise_held_out
from humanchoice.comparison.model_comparison import held_out_evaluation
from humanchoice.data import BehaviouralDataset
from humanchoice.envs import RestlessBandit
from humanchoice.evaluation import visualization as V
from humanchoice.evaluation.predictive_checks import (
    free_running_check,
    make_one_step_statistics,
    make_session_statistics,
    misfit_rate,
    one_step_ahead_check,
)
from humanchoice.fitting import fit_dataset, likelihood_profile, profile_interval
from humanchoice.fitting.estimation import check_compatible
from humanchoice.models import (
    AsymmetricQLearning,
    BayesBanditModel,
    ChoiceModel,
    KalmanBandit,
    LapseQLearning,
    POMDPReversalModel,
    QLearning,
    RandomChoice,
    StickyQLearning,
    WinStayLoseShift,
)
from humanchoice.simulation import simulate_dataset
from humanchoice.simulation.participants import sample_participants
from humanchoice.utils import md_table

BASELINES = ("random", "wsls")


def default_ladder(n_actions: int, binary_rewards: bool, win_threshold: float = 0.5) -> dict[str, ChoiceModel]:
    """The pre-specified model ladder, ordered from simplest to richest.

    ``random`` and ``wsls`` are baselines; Q-learning and its asymmetric / perseverative / lapse extensions each add one
    mechanism so that every mechanism's contribution to held-out prediction can be isolated. Continuous-reward tasks
    add the Kalman-filter (Bayesian) learner; binary two-option tasks add the POMDP hidden-state tracker, and small
    binary tasks the Bayes-adaptive planner.
    """
    ladder: dict[str, ChoiceModel] = {
        "random": RandomChoice(n_actions),
        "wsls": WinStayLoseShift(n_actions, win_threshold),
        "q": QLearning(n_actions),
        "asym_q": AsymmetricQLearning(n_actions),
        "sticky_q": StickyQLearning(n_actions),
        "lapse_q": LapseQLearning(n_actions),
    }
    if not binary_rewards:
        ladder["kalman"] = KalmanBandit(n_actions)
    if binary_rewards and n_actions == 2:
        ladder["pomdp"] = POMDPReversalModel()
    if binary_rewards and n_actions <= 3:
        ladder["bamdp"] = BayesBanditModel(n_actions=n_actions, horizon=4)
    return ladder


# ------------------------------------------------------------------------------------------------ rehearsal data
REHEARSAL_MIXTURE = {"q": 0.30, "asym_q": 0.20, "sticky_q": 0.30, "lapse_q": 0.20}


def simulate_restless_rehearsal(
    n_participants: int = 60, n_trials: int = 150, seed: int = 2026, n_actions: int = 4, missed_prob: float = 0.03
) -> BehaviouralDataset:
    """SYNTHETIC rehearsal dataset: a mixed population on a Gaussian restless bandit with timeouts.

    Generating models and parameters are known (``dataset.truth`` has a ``generating_model`` column) so the pipeline's
    conclusions can be audited. The environment follows the design commonly cited for Daw et al. (2006), rescaled to [0, 1]; it is *not*
    the schedule of any specific published dataset. Never present rehearsal output as an empirical finding.
    """
    gens = {"q": QLearning(n_actions), "asym_q": AsymmetricQLearning(n_actions), "sticky_q": StickyQLearning(n_actions),
            "lapse_q": LapseQLearning(n_actions)}
    counts = {k: round(w * n_participants) for k, w in REHEARSAL_MIXTURE.items()}
    counts["q"] += n_participants - sum(counts.values())
    frames, truths = [], []
    for k, (name, model) in enumerate(gens.items()):
        if counts[name] == 0:
            continue
        table = sample_participants(model, counts[name], seed=seed + k, prefix=f"{name}_")
        ds = simulate_dataset(model, table, lambda: RestlessBandit(n_actions), n_trials, seed=seed + 100 * k, missed_prob=missed_prob)
        frames.append(ds.df.drop(columns="missed"))
        truths.append(table.assign(generating_model=name))
    out = BehaviouralDataset(pd.concat(frames, ignore_index=True), ds.choice_categories, reward_values=None, missing="mark")
    out.truth = pd.concat(truths, ignore_index=True)
    out.config = {"synthetic": True, "kind": "restless_bandit_rehearsal", "seed": seed, "missed_prob": missed_prob,
                  "mixture": counts, "n_trials": n_trials, "environment": f"RestlessBandit({n_actions}) [Daw et al. 2006 design, rescaled]"}
    return out


# ----------------------------------------------------------------------------------------------------- pipeline
# ----------------------------------------------------------------------------------- seeded, order-independent selection
def select_participants(dataset: BehaviouralDataset, n: int, seed: int, stratify_by: str | None = "condition") -> list[str]:
    """Random subset of participants that does **not** depend on the row order of the data file.

    Participant ids are sorted first, then sampled with a generator seeded by ``seed`` (so shuffling the CSV cannot change the
    selection). If the data have a participant-level ``stratify_by`` column (e.g. the reward-drift schedule) the sample is
    stratified proportionally (largest-remainder allocation). Returns the selected ids, sorted.
    """
    ids = sorted(dataset.participants)
    if n >= len(ids):
        return ids
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7919]))
    if stratify_by and stratify_by in dataset.df.columns:
        strata = dataset.df.groupby("participant_id")[stratify_by].first().astype(str)
        groups = {k: sorted(v.index) for k, v in strata.groupby(strata)}
    else:
        groups = {"all": ids}
    total = sum(len(g) for g in groups.values())
    quota = {k: n * len(g) / total for k, g in groups.items()}
    alloc = {k: int(np.floor(q)) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    chosen: list[str] = []
    for k in sorted(groups):
        pool = groups[k]
        take = min(alloc[k], len(pool))
        chosen += [pool[i] for i in rng.choice(len(pool), size=take, replace=False)]
    return sorted(chosen)


def select_profile_participants(fits_one_model: pd.DataFrame, n: int) -> list[str]:
    """Participants spread evenly over the range of fit quality (negative log-likelihood per observed trial), ties by id.

    Spans good and poor fits instead of taking the first few rows; deterministic and independent of file order.
    """
    d = fits_one_model.assign(q=fits_one_model["nll"] / fits_one_model["n_obs"]).sort_values(["q", "participant_id"])
    if n >= len(d):
        return sorted(d["participant_id"])
    idx = np.unique(np.round(np.linspace(0, len(d) - 1, n)).astype(int))
    return sorted(d["participant_id"].iloc[idx])


def call_env_factory(factory, condition):
    """Build an environment; the factory receives the participant's ``condition`` (reward-drift schedule) iff it has a parameter
    named ``condition`` or ``schedule`` - otherwise it is called with no arguments."""
    try:
        names = set(inspect.signature(factory).parameters)
    except (TypeError, ValueError):
        names = set()
    for key in ("condition", "schedule"):
        if key in names and condition is not None:
            return factory(**{key: condition})
    return factory()


def packaged_analysis_plan() -> Path | None:
    """Path of the analysis plan bundled with the package (works from a wheel), or ``None``."""
    res = ires.files("humanchoice") / "resources" / "analysis_plan.md"
    path = Path(str(res))
    return path if path.exists() else None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info() -> dict:
    """Versions of the interpreter, platform and the packages the analysis depends on."""
    pkgs = {"humanchoice": humanchoice.__version__}  # the running code's version (installed-package metadata can be stale)
    for name in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = "unknown"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": pkgs}


def run_empirical_analysis(
    dataset: BehaviouralDataset,
    outdir: str | Path,
    models: dict[str, ChoiceModel] | None = None,
    win_threshold: float = 0.5,
    n_starts: int = 4,
    fractions: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8),
    n_jobs: int = 1,
    max_participants: int | None = None,
    ppc_participants: int = 60,
    ppc_sims: int = 200,
    profile_participants: int = 6,
    seed: int = 0,
    title: str = "Empirical analysis",
    banner: str = "",
    excluded: pd.DataFrame | None = None,
    env_factory=None,
    session_sims: int = 100,
    misfit_flag_threshold: float = 0.30,
    spec_json: str | None = None,
    analysis_plan_path: str | Path | None = None,
    recovery_gate: pd.DataFrame | None = None,
    second_seed_check: bool = True,
) -> dict:
    """Run the full pre-specified analysis and write everything to ``outdir``. Returns the key summary tables.

    ``env_factory`` (optional) must reproduce the task's reward process; only then are the full-session outcome checks run.
    Participants for the predictive checks / profiles / ``max_participants`` are chosen by a seeded random draw that does not
    depend on file order (:func:`select_participants`); the chosen ids are recorded in ``config.json``.
    """
    out = Path(outdir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if max_participants is not None and max_participants < len(dataset):
        dataset = dataset.subset(select_participants(dataset, max_participants, seed))  # seeded draw, independent of file order
    binary = bool(dataset.df.loc[dataset.df["choice"] != "<missed>", "reward"].isin([0.0, 1.0]).all())
    models = models or default_ladder(dataset.n_actions, binary, win_threshold)
    for m in models.values():
        check_compatible(m, dataset)
    K = dataset.n_actions
    log: dict = {}

    # 1. data quality & behaviour -------------------------------------------------------------------------------
    quality = dataset.quality_report()
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2))
    psum = dataset.participant_summary(win_threshold)
    psum.to_csv(out / "participant_summary.csv", index=False)
    if excluded is not None and len(excluded):
        excluded.to_csv(out / "excluded_participants.csv", index=False)
    V.plot_behavioural_overview(psum, dataset.trial_summary(), dataset.choice_categories, out / "figures" / "behavioural_overview.png")

    # 2. fit the ladder -------------------------------------------------------------------------------------------
    frames = []
    for key, model in models.items():
        f = fit_dataset(model, dataset, n_starts=n_starts, seed=seed, n_jobs=n_jobs)
        f.insert(1, "model_key", key)
        frames.append(f)
    fits = pd.concat(frames, ignore_index=True)
    fits.to_csv(out / "fits_all.csv", index=False)
    per = pd.DataFrame(
        {"participant_id": fits["participant_id"], "model": fits["model_key"], "log_likelihood": -fits["nll"], "k": fits["n_params"],
         "n": fits["n_obs"], "aic": fits["aic"], "bic": fits["bic"]}
    )
    comparison = summarise_comparison(per, list(models))
    diag = fits.groupby("model_key", sort=False).agg(
        frac_converged=("converged", "mean"), mean_start_agreement=("start_agreement", "mean"),
        frac_any_param_at_bound=("at_bound", lambda s: float((s.astype(str) != "").mean())),
    )
    comparison = comparison.join(diag)
    comparison.reset_index().to_csv(out / "model_comparison_summary.csv", index=False)

    # 3. held-out prediction -------------------------------------------------------------------------------------
    held = repeated_temporal_splits(models, dataset, fractions, n_starts=n_starts, seed=seed, n_jobs=n_jobs)
    held.to_csv(out / "held_out.csv", index=False)
    held_summary = summarise_held_out(held, reference="chance", seed=seed)
    held_summary.to_csv(out / "held_out_summary.csv", index=False)
    contrast_splits, decisions = planned_contrasts(held, seed=seed, primary_fraction=min(fractions, key=lambda f: abs(f - 0.7)))
    contrast_splits.to_csv(out / "planned_contrasts_by_split.csv", index=False)
    decisions.to_csv(out / "planned_contrast_decisions.csv", index=False)
    V.plot_heldout_summary(held_summary, "ll_diff_vs_chance", out / "figures" / "held_out_vs_chance.png")
    ref_fraction = min(fractions, key=lambda f: abs(f - 0.7))
    _, calib = held_out_evaluation(models, dataset, ref_fraction, n_starts=n_starts, seed=seed, return_calibration=True, n_jobs=n_jobs)
    pd.concat([t.assign(model=k) for k, t in calib.items()]).to_csv(out / "calibration.csv", index=False)
    V.plot_reliability({k: v for k, v in calib.items() if k not in BASELINES}, out / "figures" / "reliability.png")

    # 4. predictive checks: do the models reproduce the features of behaviour? -------------------------------------------
    one_step_stats = make_one_step_statistics(win_threshold)

    def one_step_checks(sd: int):
        ids = select_participants(dataset, ppc_participants, sd)
        rows = []
        for key, model in models.items():
            sub = fits[(fits["model_key"] == key) & (fits["participant_id"].isin(ids))]
            for _, row in sub.iterrows():
                pid = row["participant_id"]
                c, r = dataset.get(pid)
                theta = [row[f"fit_{p}"] for p in model.param_names]
                chk = one_step_ahead_check(model, theta, c, r, dataset.get_available(pid), n_sims=ppc_sims, statistics=one_step_stats, rng=sd)
                chk.insert(0, "participant_id", pid)
                chk.insert(1, "model", key)
                rows.append(chk)
        return ids, pd.concat(rows)

    ppc_ids, ppc = one_step_checks(seed)
    ppc.to_csv(out / "predictive_checks_one_step.csv", index=False)
    mis = misfit_rate(ppc, flag_threshold=misfit_flag_threshold)
    mis.to_csv(out / "predictive_check_misfit_rates.csv", index=False)
    seed_check = None
    if second_seed_check:  # Monte-Carlo / sampling sensitivity: a second draw of participants and replicates
        _, ppc2 = one_step_checks(seed + 1)
        mis2 = misfit_rate(ppc2, flag_threshold=misfit_flag_threshold)
        mis2.to_csv(out / "predictive_check_misfit_rates_seed2.csv", index=False)
        both = mis.merge(mis2, on=["model", "statistic"], suffixes=("", "_seed2"))
        seed_check = {"second_seed": seed + 1, "max_abs_rate_difference": float((both["misfit_rate"] - both["misfit_rate_seed2"]).abs().max()),
                      "n_flag_changes": int((both["population_flag"] != both["population_flag_seed2"]).sum()), "n_cells": len(both)}
    V.plot_misfit_rates(
        {str(k): g[["statistic", "misfit_rate"]] for k, g in mis.groupby("model", sort=False) if k != "random"},
        out / "figures" / "predictive_check_misfit.png", threshold=misfit_flag_threshold,
    )

    # 4b. full-session outcome checks (reward statistics): only with an environment that reproduces the task's reward process
    session = None
    session_mis = None
    conditions = (dataset.df.groupby("participant_id")["condition"].first().astype(str).to_dict()
                  if "condition" in dataset.df.columns else {})
    if env_factory is not None:
        session_stats = make_session_statistics(win_threshold)
        srows = []
        for key, model in models.items():
            sub = fits[(fits["model_key"] == key) & (fits["participant_id"].isin(ppc_ids))]
            for _, row in sub.iterrows():
                pid = row["participant_id"]
                c, r = dataset.get(pid)
                theta = [row[f"fit_{p}"] for p in model.param_names]
                cond = conditions.get(pid)
                chk = free_running_check(model, theta, functools.partial(call_env_factory, env_factory, cond), c, r,
                                         n_sims=session_sims, statistics=session_stats, rng=seed, available=dataset.get_available(pid),
                                         missed="observed")
                chk.insert(0, "participant_id", pid)
                chk.insert(1, "model", key)
                srows.append(chk)
        session = pd.concat(srows)
        session.to_csv(out / "predictive_checks_session.csv", index=False)
        session_mis = misfit_rate(session, flag_threshold=misfit_flag_threshold)
        session_mis.to_csv(out / "session_check_misfit_rates.csv", index=False)

    # 5. parameter uncertainty (likelihood profiles) for the best two non-baseline models ----------------------------------
    cand = [k for k in comparison.sort_values("total_bic").index if k not in BASELINES and models[k].n_params >= 2][:2]
    prof_rows, example_profiles, profile_ids = [], {}, {}
    for key in cand:
        ids = select_profile_participants(fits[fits["model_key"] == key], profile_participants)
        profile_ids[key] = ids
        for pid in ids:
            c, r = dataset.get(pid)
            row = fits[(fits["model_key"] == key) & (fits["participant_id"] == pid)].iloc[0]
            for j, pname in enumerate(models[key].param_names):
                prof = likelihood_profile(models[key], c, r, pname, n_grid=15, n_starts=1, seed=seed, available=dataset.get_available(pid))
                lo, hi, is_open = profile_interval(prof)
                span = models[key].bounds[j][1] - models[key].bounds[j][0]
                prof_rows.append({"model": key, "participant_id": pid, "param": pname, "estimate": row[f"fit_{pname}"], "ci_lo": lo,
                                  "ci_hi": hi, "relative_width": (hi - lo) / span, "open_interval": is_open})
                if pid == ids[len(ids) // 2]:  # the median-fit participant as the example
                    example_profiles[f"{key}:{pname}"] = prof
    profiles = pd.DataFrame(prof_rows)
    profiles.to_csv(out / "profile_intervals.csv", index=False)
    if example_profiles:
        V.plot_profiles(example_profiles, out / "figures" / "example_profiles.png")
    prof_summary = (profiles.groupby(["model", "param"]).agg(median_relative_width=("relative_width", "median"),
                                                             frac_open=("open_interval", "mean"), n=("participant_id", "count")).reset_index()
                    if len(profiles) else pd.DataFrame())
    prof_summary.to_csv(out / "profile_summary.csv", index=False)

    # 6. audit against the truth (only for synthetic data) --------------------------------------------------------------
    truth_check = None
    if dataset.truth is not None and "generating_model" in dataset.truth.columns:
        tr = dataset.truth.set_index("participant_id")["generating_model"]
        rows = []
        for crit in ("aic", "bic"):
            w = per.loc[per.groupby("participant_id")[crit].idxmin()]
            w = w.assign(generating=w["participant_id"].map(tr))
            rows.append(pd.crosstab(w["generating"], w["model"]).reindex(columns=list(models), fill_value=0).assign(criterion=crit))
        truth_check = pd.concat(rows)
        truth_check.to_csv(out / "winner_vs_truth.csv")

    # 7. configuration, provenance artefacts and report -----------------------------------------------------------------------
    plan_path = Path(analysis_plan_path) if analysis_plan_path else packaged_analysis_plan()
    interp = _parameter_interpretability(models, fits, recovery_gate)
    interp.to_csv(out / "parameter_interpretability.csv", index=False)
    plan_hash = _sha256_text(plan_path.read_text()) if plan_path is not None and plan_path.exists() else None
    if plan_path is not None and plan_path.exists():
        (out / "analysis_plan.md").write_text(plan_path.read_text())
    if spec_json is not None:
        (out / "dataset_spec.json").write_text(spec_json)
    env = environment_info()
    (out / "environment.txt").write_text("\n".join([f"python {env['python']}", f"platform {env['platform']}",
                                                    *[f"{k}=={v}" for k, v in env["packages"].items()]]) + "\n")
    config = {
        "title": title, "banner": banner, "package_version": humanchoice.__version__, "environment": env,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": _git_commit(),
        "analysis_plan_sha256": plan_hash, "dataset_spec_sha256": None if spec_json is None else _sha256_text(spec_json),
        "seed": seed, "n_starts": n_starts, "train_fractions": list(fractions), "win_threshold": win_threshold, "n_jobs": n_jobs,
        "misfit_flag_threshold": misfit_flag_threshold, "ppc_sims": ppc_sims, "session_sims": session_sims if env_factory else None,
        "selected_participants": {"predictive_checks": ppc_ids, "profiles": profile_ids}, "second_seed_check": seed_check,
        "recovery_gate_supplied": recovery_gate is not None,
        "n_participants": len(dataset), "n_options": K, "binary_rewards": binary, "max_participants": max_participants,
        "models": {k: {"class": type(m).__name__, "params": list(m.param_names), "bounds": [list(b) for b in m.bounds]} for k, m in models.items()},
        "dataset": {k: v for k, v in dataset.config.items() if k != "mixture"} | {"synthetic": bool(dataset.config.get("synthetic", False))},
        "preprocessing_steps": dataset.report.steps,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    _write_report(out, title, banner, quality, comparison, held_summary, mis, session_mis, prof_summary, truth_check, config, ref_fraction, decisions, contrast_splits, interp, seed_check)
    log.update(fits=fits, decisions=decisions, comparison=comparison, held_out=held, held_out_summary=held_summary, predictive_checks=ppc, session_checks=session,
               misfit_rates=mis, session_misfit_rates=session_mis, interpretability=interp, seed_check=seed_check, profiles=profiles, truth_check=truth_check, quality=quality)
    return log


def _write_report(out, title, banner, quality, comparison, held_summary, mis, session_mis, prof_summary, truth_check, config, ref_fraction, decisions, contrast_splits, interp, seed_check):
    best_ll = held_summary[held_summary["train_fraction"] == ref_fraction].sort_values("test_ll_per_trial", ascending=False)
    cmp_tab = comparison.reset_index().round(3)
    lines = [f"# {title}\n"]
    if banner:
        lines += [f"> **{banner}**\n"]
    lines += [
        f"Generated {config['generated_utc']} by humanchoice {config['package_version']} (seed {config['seed']}).\n",
        "## 1. Data\n",
        (f"{quality['n_participants']} participants, {quality['n_trials_total']} trials "
        f"({quality['trials_per_participant_min']}-{quality['trials_per_participant_max']} per participant), "
        f"{config['n_options']} options. Missed trials: {quality['n_missed_total']} ({quality['missed_fraction_overall']:.1%}); "
        f"largest per-participant fraction {quality['missed_fraction_per_participant_max']:.1%}. "
        f"Rewards: mean {quality['reward_mean']:.3f}, SD {quality['reward_sd']:.3f}, range [{quality['reward_min']:.3f}, {quality['reward_max']:.3f}] "
        f"({quality['n_distinct_reward_values']} distinct values).\n"),
        "Preprocessing steps:\n" + "\n".join(f"- {s}" for s in config["preprocessing_steps"]) + "\n",
        "![behaviour](figures/behavioural_overview.png)\n",
        "## 2. Model ladder: fit and information criteria\n",
        ("Totals over participants (lower AIC/BIC better). `frac_any_param_at_bound` and `mean_start_agreement` are identifiability/"
        "optimisation diagnostics, not goodness of fit.\n"),
        md_table(cmp_tab), "",
        "## 3. Held-out prediction (primary criterion)\n",
        (f"Fit on the first {ref_fraction:.0%} of each participant's trials, predict the rest one step ahead. Log-likelihood per trial "
        "with participant-level bootstrap 95% CI; `ll_diff_vs_chance` is the paired difference from uniform choice.\n"),
        md_table(best_ll[["model", "n_participants", "test_ll_per_trial", "ll_lo", "ll_hi", "test_brier", "test_accuracy",
                          "ll_diff_vs_chance", "diff_lo", "diff_hi"]].round(4)), "",
        "![held-out](figures/held_out_vs_chance.png)\n",
        "### Pre-specified contrasts (decision rule of `docs/analysis_plan.md`)\n",
        ("**Primary analysis** = the single split at 70% training: paired participant-level bootstrap difference in held-out log-likelihood per trial; "
         "p-values of the three `primary` contrasts are Holm-adjusted, `sanity`/`exploratory` contrasts are not. The other split points overlap with "
         "the primary one and are **sensitivity analyses, not independent replications**: `sensitivity_agree` counts how many of them give a CI in the "
         "same direction as the verdict.\n"),
        md_table(decisions[["hypothesis", "status", "contrast", "mean_diff", "ci_lo", "ci_hi", "p_value", "p_adjusted", "decision",
                            "sensitivity_agree", "n_sensitivity_splits"]].round(4)), "",
        "Per split point (primary split flagged):\n",
        md_table(contrast_splits[["hypothesis", "model", "reference", "train_fraction", "is_primary_split", "mean_diff", "ci_lo", "ci_hi"]].round(4)), "",
        "![calibration](figures/reliability.png)\n",
        "## 4. Predictive checks (systematic misfit)\n",
        "### 4a. One-step-ahead checks (choice-only statistics)\n",
        (f"Statistics evaluated: {', '.join(sorted(set(mis['statistic'])))}. For each, the fraction of participants whose observed value falls "
         f"outside the model's replicate interval (p < .05); about 5% is expected for a model that captures the feature. A model is **flagged** "
         f"on a statistic only if the rate exceeds the pre-declared {config['misfit_flag_threshold']:.0%} threshold. Rates credibly above chance "
         "but below the threshold are called *elevated, below flag threshold* - they are not formal flags.\n"),
        md_table(mis.pivot(index="statistic", columns="model", values="misfit_rate").round(2).reset_index()), "",
        "**Flagged (rate above threshold):**\n",
        md_table(_misfit_rows(mis, mis["population_flag"])) if mis["population_flag"].any() else "None.", "",
        "**Elevated but below the flag threshold:**\n",
        md_table(_misfit_rows(mis, mis["interpretation"] == "elevated, below flag threshold"))
        if (mis["interpretation"] == "elevated, below flag threshold").any() else "None.", "",
        "![misfit](figures/predictive_check_misfit.png)\n",
        (f"Second-seed repeat (seed {seed_check['second_seed']}, new participant draw and replicates): largest change in any misfit rate "
         f"{seed_check['max_abs_rate_difference']:.3f}; {seed_check['n_flag_changes']} of {seed_check['n_cells']} flag decisions changed.\n")
        if seed_check else "Second-seed repeat: not run.\n",
        "### 4b. Full-session outcome checks (reward statistics)\n",
        (("Mean reward, total reward, reward spread and improvement over the session were checked by simulating whole sessions of each fitted "
          "model in the environment supplied for this dataset (it must reproduce the task's reward process). Misfit rates:\n") if session_mis is not None else
         ("**Not run.** Outcome statistics (mean / total reward, reward spread, improvement over the session) cannot be checked from "
          "one-step-ahead replicates, and no environment reproducing this dataset's reward process was supplied; no substitute environment "
          "is used. They are therefore *not evaluated* in this report.\n")),
        md_table(session_mis.pivot(index="statistic", columns="model", values="misfit_rate").round(2).reset_index()) if session_mis is not None else "", "",
        "## 5. Which parameters may be interpreted?\n",
        ("A parameter is interpretable only if the recovery gate (`docs/analysis_plan.md` §6: lower 95% bound of Spearman rho >= 0.7 at this "
         "dataset's design) was passed. Parameters not assessed or failing the gate must **not** be interpreted.\n"),
        md_table(interp.round(3)), "",
        "### 5b. Likelihood profiles\n",
        "`frac_open` is the fraction of profile intervals the data could not close within the parameter bounds (weakly identified).\n",
        md_table(prof_summary.round(3)) if len(prof_summary) else "No profiles computed.", "",
    ]
    if truth_check is not None:
        lines += ["## 6. Audit against known generating models (synthetic data only)\n",
                  "Rows: generating model; columns: selected model.\n", md_table(truth_check.reset_index().rename(columns={"generating": "generating model"})), ""]
    lines += ["## Deviations from `docs/analysis_plan.md`\n",
              "None recorded by the pipeline. Any manual deviation made after inspecting real data must be listed here with its reason.\n"]
    lines += ["## Limitations\n",
              ("Per-participant maximum-likelihood point estimates; fixed model hyper-parameters; held-out prediction uses chronological splits "
              "of one session; see `RESEARCH_NOTES.md`.\n")]
    (out / "report.md").write_text("\n".join(lines))


def _misfit_rows(mis: pd.DataFrame, mask) -> pd.DataFrame:
    cols = ["model", "statistic", "n_participants", "misfit_rate", "ci_lower", "ci_upper"]
    return mis.loc[mask, cols].round(3)


def _parameter_interpretability(models, fits: pd.DataFrame, gate: pd.DataFrame | None) -> pd.DataFrame:
    """One row per fitted parameter: median estimate, share on a bound, and its recovery-gate status."""
    lookup = {}
    if gate is not None:
        lookup = {(r["model"], r["param"]): r["status"] for _, r in gate.iterrows()}
    rows = []
    for key, model in models.items():
        sub = fits[fits["model_key"] == key]
        for pname in model.param_names:
            at_bound = sub["at_bound"].astype(str).apply(lambda x, n=pname: n in x.split(",")).mean()
            status = lookup.get((key, pname), "not assessed - do not interpret" if gate is not None else "NO recovery gate supplied - do not interpret")
            rows.append({"model": key, "param": pname, "median_estimate": float(sub[f"fit_{pname}"].median()), "frac_at_bound": float(at_bound),
                         "recovery_status": status})
    return pd.DataFrame(rows)


def run_empirical_by_condition(
    dataset: BehaviouralDataset,
    outdir: str | Path,
    condition_col: str = "condition",
    min_participants: int = 30,
    **kwargs,
) -> dict:
    """Pooled analysis plus one analysis per reward-drift schedule (``condition``) with at least ``min_participants`` participants.

    Writes ``<outdir>/pooled`` and ``<outdir>/condition_<level>``; smaller schedules are listed as skipped, never silently dropped.
    ``kwargs`` go to :func:`run_empirical_analysis` (so a condition-aware ``env_factory`` is honoured per participant).
    """
    out = Path(outdir)
    if condition_col not in dataset.df.columns:
        raise ValueError(f"dataset has no '{condition_col}' column")
    results = {"pooled": run_empirical_analysis(dataset, out / "pooled", title="Pooled analysis (all schedules)", **kwargs)}
    levels = dataset.df.groupby("participant_id")[condition_col].first().astype(str)
    rows = [{"scope": "pooled", "n_participants": len(dataset), "status": "analysed"}]
    for level, ids in levels.groupby(levels):
        if len(ids) < min_participants:
            rows.append({"scope": f"condition_{level}", "n_participants": len(ids), "status": f"skipped (n < {min_participants})"})
            continue
        sub = dataset.subset(list(ids.index))
        results[level] = run_empirical_analysis(sub, out / f"condition_{level}", title=f"Schedule '{level}'", **kwargs)
        rows.append({"scope": f"condition_{level}", "n_participants": len(ids), "status": "analysed"})
    summary = pd.concat([r["decisions"].assign(scope=name if name == "pooled" else f"condition_{name}") for name, r in results.items()], ignore_index=True)
    summary.to_csv(out / "by_condition_decisions.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "by_condition_scopes.csv", index=False)
    results["scopes"] = pd.DataFrame(rows)
    results["decisions_by_scope"] = summary
    return results
