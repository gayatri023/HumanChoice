"""Compare candidate models on observed (or synthetic) behavioural data."""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Literal, overload

import numpy as np
import pandas as pd

from humanchoice.data.ingestion import BehaviouralDataset
from humanchoice.evaluation.metrics import (
    akaike_weights,
    brier_score,
    calibration_table,
    choice_accuracy,
    mean_ci,
)
from humanchoice.fitting.estimation import check_compatible, fit_model
from humanchoice.models.base import ChoiceModel
from humanchoice.models.heuristics import RandomChoice
from humanchoice.utils import EPS, MISSED

CALIBRATION_BINS = 10


def _usable_models(models: dict[str, ChoiceModel], dataset: BehaviouralDataset, skip_incompatible: bool) -> dict[str, ChoiceModel]:
    usable = {}
    for key, model in models.items():
        try:
            check_compatible(model, dataset)
        except ValueError as exc:
            if not skip_incompatible:
                raise
            warnings.warn(f"skipping model '{key}': {exc}", stacklevel=3)
            continue
        usable[key] = model
    if not usable:
        raise ValueError("no candidate model is compatible with this dataset")
    return usable


def compare_models(
    models: dict[str, ChoiceModel],
    dataset: BehaviouralDataset,
    n_starts: int = 4,
    seed: int = 0,
    skip_incompatible: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every model to every participant.

    Returns ``(per_participant, summary)``. ``summary`` has one row per model with summed log-likelihood,
    summed AIC/BIC, the number of participants for which the model wins under each criterion and the mean
    Akaike weight. Information criteria complement, and do not replace, held-out evaluation.
    With ``skip_incompatible=True`` models that cannot be applied to the data (option count, binary-only rewards) are
    skipped with a warning instead of raising.
    """
    models = _usable_models(models, dataset, skip_incompatible)
    rows = []
    for pid, choices, rewards in dataset:
        avail = dataset.get_available(pid)
        for key, model in models.items():
            f = fit_model(model, choices, rewards, n_starts=n_starts, seed=seed, available=avail)
            rows.append({"participant_id": pid, "model": key, "log_likelihood": -f.nll, "k": f.n_params,
                         "n": f.n_obs, "aic": f.aic, "bic": f.bic, "converged": f.converged,
                         "start_agreement": f.start_agreement, "at_bound": ",".join(f.at_bound),
                         **{f"fit_{k}": v for k, v in f.params.items()}})
    per = pd.DataFrame(rows)
    return per, summarise_comparison(per, list(models))


def summarise_comparison(per: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    """Population summary of a per-participant fit table (adds ``aic_weight`` to ``per`` in place).

    One row per model: summed log-likelihood / AIC / BIC, mean Akaike weight and the number of participants for which
    the model wins under AIC and BIC.
    """
    per["aic_weight"] = per.groupby("participant_id")["aic"].transform(lambda s: akaike_weights(s.to_numpy()))
    win_aic = per.loc[per.groupby("participant_id")["aic"].idxmin(), "model"].value_counts()
    win_bic = per.loc[per.groupby("participant_id")["bic"].idxmin(), "model"].value_counts()
    summary = per.groupby("model").agg(
        total_log_likelihood=("log_likelihood", "sum"),
        total_aic=("aic", "sum"),
        total_bic=("bic", "sum"),
        mean_aic_weight=("aic_weight", "mean"),
    )
    summary["n_wins_aic"] = [int(win_aic.get(m, 0)) for m in summary.index]
    summary["n_wins_bic"] = [int(win_bic.get(m, 0)) for m in summary.index]
    return summary.reindex([m for m in model_order if m in summary.index])


def _held_out_participant(args):
    """Worker: fit every model on one participant's training trials and score the test trials."""
    models, chance, pid, choices, rewards, avail, train_fraction, n_starts, seed = args
    n_train = round(train_fraction * len(choices))
    c_test = choices[n_train:]
    if (c_test != MISSED).sum() == 0 or (choices[:n_train] != MISSED).sum() == 0:
        return [], {}
    obs_test = c_test != MISSED
    a_tr = None if avail is None else avail[:n_train]
    idx = np.flatnonzero(obs_test)

    def mean_ll(probs):
        p = probs[n_train:]
        return p, float(np.log(np.clip(p[idx, c_test[idx]], EPS, None)).mean())

    rows, preds = [], {}
    p_c, ll_c = mean_ll(chance.trajectory([], choices, rewards, avail)["probs"])
    rows.append({"participant_id": pid, "model": "chance", "test_ll_per_trial": ll_c,
                 "test_accuracy": float(p_c[idx, c_test[idx]].mean()), "test_brier": brier_score(p_c, c_test),
                 "n_test": int(obs_test.sum())})
    for key, model in models.items():
        f = fit_model(model, choices[:n_train], rewards[:n_train], n_starts=n_starts, seed=seed, available=a_tr)
        traj = model.trajectory(list(f.params.values()), choices, rewards, avail)
        p_test, ll = mean_ll(traj["probs"])
        rows.append({"participant_id": pid, "model": key, "test_ll_per_trial": ll,
                     "test_accuracy": choice_accuracy(p_test, c_test), "test_brier": brier_score(p_test, c_test),
                     "n_test": int(obs_test.sum())})
        preds[key] = (p_test, c_test)
    return rows, preds


@overload
def held_out_evaluation(
    models: dict[str, ChoiceModel],
    dataset: BehaviouralDataset,
    train_fraction: float = ...,
    n_starts: int = ...,
    seed: int = ...,
    return_calibration: Literal[False] = ...,
    skip_incompatible: bool = ...,
    n_jobs: int = ...,
) -> pd.DataFrame: ...
@overload
def held_out_evaluation(
    models: dict[str, ChoiceModel],
    dataset: BehaviouralDataset,
    train_fraction: float = ...,
    n_starts: int = ...,
    seed: int = ...,
    *,
    return_calibration: Literal[True],
    skip_incompatible: bool = ...,
    n_jobs: int = ...,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]: ...
def held_out_evaluation(
    models: dict[str, ChoiceModel],
    dataset: BehaviouralDataset,
    train_fraction: float = 0.7,
    n_starts: int = 4,
    seed: int = 0,
    return_calibration: bool = False,
    skip_incompatible: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit on the first ``train_fraction`` of each participant's trials and predict the rest.

    The latent state keeps updating through the test trials using the *observed* choices and rewards, so test
    predictions are genuine one-step-ahead forecasts. Per participant and model the table reports the mean held-out
    log-likelihood per trial, choice-prediction accuracy and multi-class Brier score over the observed test trials.
    A ``chance`` row (uniform choice among the available options) is included as the reference.

    With ``return_calibration=True`` also returns, per model, a reliability table pooled over all participants.
    ``n_jobs > 1`` evaluates participants in parallel processes.
    """
    if not 0.2 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must lie in [0.2, 0.9]")
    models = _usable_models(models, dataset, skip_incompatible)
    chance = RandomChoice(dataset.n_actions)
    jobs = [(models, chance, pid, c, r, dataset.get_available(pid), train_fraction, n_starts, seed) for pid, c, r in dataset]
    if n_jobs == 1:
        results = [_held_out_participant(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_held_out_participant, jobs, chunksize=max(1, len(jobs) // (4 * n_jobs))))
    out = pd.DataFrame([row for rows, _ in results for row in rows])
    if not return_calibration:
        return out
    calib = {}
    for key in models:
        parts = [preds[key] for _, preds in results if key in preds]
        if parts:
            calib[key] = calibration_table(np.vstack([p for p, _ in parts]), np.concatenate([c for _, c in parts]), CALIBRATION_BINS)
    return out, calib


def repeated_temporal_splits(
    models: dict[str, ChoiceModel],
    dataset: BehaviouralDataset,
    fractions: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8),
    n_starts: int = 4,
    seed: int = 0,
    skip_incompatible: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """:func:`held_out_evaluation` at several chronological split points (adds a ``train_fraction`` column).

    Agreement of the model ranking across split points is evidence the ranking is not an artefact of one arbitrary cut.
    """
    frames = []
    for f in fractions:
        ho = held_out_evaluation(models, dataset, f, n_starts, seed, skip_incompatible=skip_incompatible, n_jobs=n_jobs)
        ho.insert(1, "train_fraction", f)
        frames.append(ho)
    return pd.concat(frames, ignore_index=True)


def summarise_held_out(
    held_out: pd.DataFrame, reference: str = "chance", n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Population summary of a held-out table with participant-level bootstrap 95 % CIs.

    One row per (``train_fraction``, model): mean held-out log-likelihood per trial, Brier score and accuracy with CIs,
    plus the *paired* difference in log-likelihood per trial against ``reference`` (positive = better than the
    reference) with its CI. The paired difference, not the overlap of two separate CIs, is the right comparison.
    """
    df = held_out.copy()
    if "train_fraction" not in df.columns:
        df["train_fraction"] = np.nan
    if reference not in set(df["model"]):
        raise ValueError(f"reference model '{reference}' not in the table")
    ref = df[df["model"] == reference].set_index(["train_fraction", "participant_id"])["test_ll_per_trial"]
    rows = []
    for (frac, model), g in df.groupby(["train_fraction", "model"], sort=False, dropna=False):
        ll, ll_lo, ll_hi = mean_ci(g["test_ll_per_trial"], n_boot=n_boot, seed=seed)
        br, br_lo, br_hi = mean_ci(g["test_brier"], n_boot=n_boot, seed=seed)
        ac, ac_lo, ac_hi = mean_ci(g["test_accuracy"], n_boot=n_boot, seed=seed)
        key = list(zip(g["train_fraction"], g["participant_id"]))
        diff = g["test_ll_per_trial"].to_numpy() - ref.reindex(key).to_numpy()
        d, d_lo, d_hi = mean_ci(diff, n_boot=n_boot, seed=seed)
        rows.append({"train_fraction": frac, "model": model, "n_participants": len(g),
                     "test_ll_per_trial": ll, "ll_lo": ll_lo, "ll_hi": ll_hi,
                     "test_brier": br, "brier_lo": br_lo, "brier_hi": br_hi,
                     "test_accuracy": ac, "acc_lo": ac_lo, "acc_hi": ac_hi,
                     f"ll_diff_vs_{reference}": d, "diff_lo": d_lo, "diff_hi": d_hi})
    return pd.DataFrame(rows)


#: the contrasts pre-specified in ``docs/analysis_plan.md``: (hypothesis, better-if-positive model, reference model, status).
#: "primary" contrasts form the Holm family; "sanity" and "exploratory" contrasts are reported unadjusted.
PLANNED_CONTRASTS: tuple[tuple[str, str, str, str], ...] = (
    ("H1", "q", "wsls", "primary"),
    ("H2", "sticky_q", "q", "primary"),
    ("H3", "asym_q", "q", "primary"),
    ("H1b", "q", "random", "sanity"),
    ("H4", "kalman", "q", "exploratory"),
    ("H5", "lapse_q", "q", "exploratory"),
)


def holm_adjust(pvalues: list[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def _signflip_p(diff: np.ndarray, n_perm: int, seed: int) -> float:
    """Two-sided paired sign-flip permutation p-value for a mean difference of 0 (exact null calibration for symmetric differences).

    Used instead of reading a p-value off the percentile bootstrap, whose calibration under the null is not guaranteed
    (see ``tests/statistical_validation``). The +1 correction keeps p > 0.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    perm = np.abs((rng.choice([-1.0, 1.0], size=(n_perm, len(d))) * d).mean(axis=1))
    return float((1 + (perm >= abs(d.mean()) - 1e-15).sum()) / (n_perm + 1))


def planned_contrasts(
    held_out: pd.DataFrame,
    contrasts: tuple[tuple[str, str, str, str], ...] = PLANNED_CONTRASTS,
    primary_fraction: float = 0.7,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the pre-specified decision rule to a held-out table from :func:`repeated_temporal_splits`.

    **Primary analysis:** the paired (per-participant) difference in held-out log-likelihood per trial at the single split
    ``primary_fraction`` (default 70 % training), with a participant-level bootstrap CI and a paired sign-flip permutation p-value. The ``primary`` contrasts form
    one family and their p-values are **Holm-adjusted**; ``sanity`` / ``exploratory`` contrasts are reported unadjusted. A hypothesis
    is ``supported`` if its (adjusted) p < ``alpha`` with ``X`` better, ``contradicted`` if p < ``alpha`` with ``X`` worse,
    otherwise ``not supported``.

    **Sensitivity:** the other split points reuse the same participants and overlapping data, so they are *not* independent
    replications; they are reported only as ``sensitivity_agree`` (how many other splits give a CI in the same direction as the
    primary verdict). Contrasts whose models are absent are skipped.

    Returns ``(per_split, decisions)``.
    """
    if "train_fraction" not in held_out.columns:
        raise ValueError("held_out must come from repeated_temporal_splits (needs a train_fraction column)")
    wide = held_out.pivot_table(index=["train_fraction", "participant_id"], columns="model", values="test_ll_per_trial")
    fractions = sorted(wide.index.get_level_values(0).unique())
    if primary_fraction not in fractions:
        raise ValueError(f"primary_fraction {primary_fraction} not among the evaluated splits {fractions}")
    rows, decisions = [], []
    for hyp, x, y, status in contrasts:
        if x not in wide.columns or y not in wide.columns:
            continue
        per = {}
        for f in fractions:
            d = (wide.loc[f, x] - wide.loc[f, y]).dropna().to_numpy()
            m, lo, hi = mean_ci(d, n_boot=n_boot, seed=seed)
            per[f] = (m, lo, hi, len(d), _signflip_p(d, max(n_boot, 2000), seed) if f == primary_fraction else np.nan)
            rows.append({"hypothesis": hyp, "status": status, "model": x, "reference": y, "train_fraction": f, "is_primary_split": f == primary_fraction,
                         "n_participants": len(d), "mean_diff": m, "ci_lo": lo, "ci_hi": hi})
        m, lo, hi, n, pval = per[primary_fraction]
        decisions.append({"hypothesis": hyp, "status": status, "contrast": f"{x} vs {y}", "primary_split": primary_fraction, "n_participants": n,
                          "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "p_value": pval,
                          "_per": per})
    dec = pd.DataFrame(decisions)
    if dec.empty:
        return pd.DataFrame(rows), dec
    family = dec["status"] == "primary"
    dec["p_adjusted"] = dec["p_value"]
    dec["holm_family"] = family
    if family.any():
        dec.loc[family, "p_adjusted"] = holm_adjust(dec.loc[family, "p_value"].to_numpy())
    verdicts, agree = [], []
    for _, r in dec.iterrows():
        sig = r["p_adjusted"] < alpha
        verdict = ("supported" if r["mean_diff"] > 0 else "contradicted") if sig else "not supported"
        others = [v for f, v in r["_per"].items() if f != primary_fraction]
        if verdict == "supported":
            n_agree = sum(v[1] > 0 for v in others)
        elif verdict == "contradicted":
            n_agree = sum(v[2] < 0 for v in others)
        else:
            n_agree = sum(v[1] <= 0 <= v[2] for v in others)
        verdicts.append(verdict)
        agree.append(int(n_agree))
    dec["decision"] = verdicts
    dec["sensitivity_agree"] = agree
    dec["n_sensitivity_splits"] = len(fractions) - 1
    dec = dec.drop(columns="_per")
    return pd.DataFrame(rows), dec
