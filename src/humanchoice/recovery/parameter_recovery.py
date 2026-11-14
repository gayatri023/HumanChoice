"""Parameter recovery: can the fitting pipeline recover parameters it generated?"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from humanchoice.evaluation import metrics as M
from humanchoice.fitting.estimation import fit_model
from humanchoice.models.base import ChoiceModel


def run_parameter_recovery(
    model: ChoiceModel,
    env_factory: Callable[[], object],
    n_participants: int = 50,
    n_trials: int = 100,
    seed: int = 0,
    n_starts: int = 4,
    true_params: pd.DataFrame | None = None,
    missed_prob: float = 0.0,
) -> pd.DataFrame:
    """Simulate, refit and compare. One row per synthetic participant.

    ``missed_prob`` simulates timeouts (trials with no response) so recovery is assessed under the same missingness as
    the data it is meant to qualify.

    Columns: ``sim_id, n_trials, true_<param>, fit_<param>, nll, converged, start_agreement, at_bound``.
    Generating parameters are drawn uniformly from ``model.sample_ranges`` unless ``true_params`` is given
    (a dataframe with one column per parameter).
    """
    ss = np.random.SeedSequence(seed)
    draw_rng = np.random.default_rng(ss.spawn(1)[0])
    children = ss.spawn(n_participants)
    rows = []
    for i in range(n_participants):
        if true_params is not None:
            theta = np.array([true_params.iloc[i][n] for n in model.param_names], dtype=float)
        else:
            theta = model.sample_params(draw_rng)
        sim = model.simulate(theta, env_factory(), n_trials, np.random.default_rng(children[i]), missed_prob=missed_prob)
        fit = fit_model(model, sim["choices"], sim["rewards"], n_starts=n_starts, seed=seed + i)
        row: dict[str, object] = {"sim_id": i, "n_trials": n_trials}
        row.update({f"true_{n}": v for n, v in zip(model.param_names, theta)})
        row.update({f"fit_{n}": v for n, v in fit.params.items()})
        row.update(
            nll=fit.nll, converged=fit.converged, start_agreement=fit.start_agreement, at_bound=",".join(fit.at_bound)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_recovery_grid(
    model: ChoiceModel,
    env_factory: Callable[[], object],
    trial_counts: list[int],
    n_participants: int = 40,
    seed: int = 0,
    n_starts: int = 4,
    missed_prob: float = 0.0,
) -> pd.DataFrame:
    """Repeat :func:`run_parameter_recovery` over several trial counts (same generating parameters)."""
    frames = [
        run_parameter_recovery(model, env_factory, n_participants, n_trials=t, seed=seed, n_starts=n_starts, missed_prob=missed_prob)
        for t in trial_counts
    ]
    return pd.concat(frames, ignore_index=True)


def recovery_metrics(df: pd.DataFrame, param_names: tuple[str, ...], ci: bool = True) -> pd.DataFrame:
    """Recovery metrics per parameter (and per ``n_trials`` if several are present).

    MAE, median absolute error, RMSE, Pearson ``r`` and Spearman ``rho`` (rank correlation, robust to the few
    fits that run to a parameter bound), bias (mean signed error), error variance and 95% bootstrap CIs for
    RMSE, ``r`` and ``rho``. **Poor recovery is reported, not filtered.**
    """
    rows = []
    for n_trials, g in df.groupby("n_trials"):
        for p in param_names:
            t, e = g[f"true_{p}"].to_numpy(), g[f"fit_{p}"].to_numpy()
            row = {
                "n_trials": n_trials,
                "param": p,
                "n": len(g),
                "mae": M.mean_absolute_error(t, e),
                "rmse": M.rmse(t, e),
                "r": M.correlation(t, e),
                "rho": M.spearman(t, e),
                "median_ae": M.median_absolute_error(t, e),
                "bias": M.bias(t, e),
                "error_var": M.error_variance(t, e),
                "frac_at_bound": float(g["at_bound"].astype(str).apply(lambda x, name=p: name in x.split(",")).mean()),
            }
            if ci:
                row["rmse_ci_lo"], row["rmse_ci_hi"] = M.bootstrap_ci(t, e, M.rmse)
                row["r_ci_lo"], row["r_ci_hi"] = M.bootstrap_ci(t, e, M.correlation)
                row["rho_ci_lo"], row["rho_ci_hi"] = M.bootstrap_ci(t, e, M.spearman)
            rows.append(row)
    return pd.DataFrame(rows)


def stratified_recovery_metrics(
    df: pd.DataFrame,
    param_names: tuple[str, ...],
    by: str = "beta",
    n_bins: int = 3,
    ci: bool = False,
) -> pd.DataFrame:
    """Recovery metrics within quantile bins of a *generating* parameter.

    The inverse temperature ``beta`` controls how noisy choices are, so ``by="beta"`` answers
    "does recovery degrade as behaviour becomes noisier?" (low ``beta`` = high choice noise). Bins are
    quantiles of ``true_<by>`` computed separately for every ``n_trials`` so that strata are equally populated.

    Returns one row per ``(n_trials, stratum, param)`` with the same columns as :func:`recovery_metrics`
    plus ``stratum`` (``"Q1 (lowest)"`` ... ) and the ``by_min`` / ``by_max`` range of the stratum.
    """
    col = f"true_{by}"
    if col not in df.columns:
        raise KeyError(f"{col} not found; available: {sorted(c for c in df.columns if c.startswith('true_'))}")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    d = df.copy()
    labels = [f"Q{k + 1}" + (" (lowest)" if k == 0 else " (highest)" if k == n_bins - 1 else "") for k in range(n_bins)]
    d["stratum"] = d.groupby("n_trials")[col].transform(lambda s: pd.qcut(s, n_bins, labels=labels, duplicates="drop")).astype(str)
    frames = []
    for (_, stratum), g in d.groupby(["n_trials", "stratum"], sort=False):
        m = recovery_metrics(g, param_names, ci=ci)
        m["stratum"] = stratum
        m["by"] = by
        m["by_min"], m["by_max"] = float(g[col].min()), float(g[col].max())
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def error_correlations(df: pd.DataFrame, param_names: tuple[str, ...], n_trials: int | None = None) -> pd.DataFrame:
    """Correlation matrix of *estimation errors* ``fit - true`` between parameters.

    Large off-diagonal values mean two parameters trade off (an over-estimate of one tends to come with an
    under-/over-estimate of the other), i.e. they are partly confounded even if each looks recoverable on its own.
    """
    d = df if n_trials is None else df[df["n_trials"] == n_trials]
    err = pd.DataFrame({p: d[f"fit_{p}"] - d[f"true_{p}"] for p in param_names})
    return err.corr()


def cross_parameter_leakage(df: pd.DataFrame, param_names: tuple[str, ...], n_trials: int | None = None) -> pd.DataFrame:
    """Spearman correlation between each *true* parameter (rows) and the *error* in each estimate (columns).

    A diagonal entry measures regression to the mean; an off-diagonal entry measures *leakage*: the true value of one
    mechanism (say perseveration) biasing the estimate of another (say the learning rate).
    """
    from scipy.stats import spearmanr

    d = df if n_trials is None else df[df["n_trials"] == n_trials]
    out = pd.DataFrame(index=list(param_names), columns=list(param_names), dtype=float)
    for tp in param_names:
        for ep in param_names:
            err = d[f"fit_{ep}"] - d[f"true_{ep}"]
            out.loc[tp, ep] = float(spearmanr(d[f"true_{tp}"], err)[0]) if d[f"true_{tp}"].nunique() > 2 else np.nan
    return out


def recovery_gate_report(metrics: pd.DataFrame, gate: float = 0.7, n_trials: int | None = None, condition: str | None = None) -> pd.DataFrame:
    """Machine-readable map from every (model, parameter) to whether it may be interpreted.

    Input: the table from :func:`recovery_metrics` computed with ``ci=True`` (needs ``model``, ``param``, ``rho``, ``rho_ci_lo``,
    ``rho_ci_hi``; optionally ``condition`` / ``n_trials``). A parameter is ``interpretable`` only if the **lower** 95 % bootstrap bound of
    Spearman's rho is >= ``gate`` at the chosen design; otherwise ``unreliable``; without a CI it is ``not assessed``.
    """
    d = metrics
    if condition is not None:
        d = d[d["condition"] == condition]
    if n_trials is not None:
        d = d[d["n_trials"] == n_trials]
    keep = [c for c in ("model", "param", "n_trials", "rho", "rho_ci_lo", "rho_ci_hi") if c in d.columns]
    out = d[keep].copy()
    out["gate"] = gate
    out["status"] = np.where(out["rho_ci_lo"].isna(), "not assessed (no CI)", np.where(out["rho_ci_lo"] >= gate, "interpretable", "unreliable"))
    return out.reset_index(drop=True)
