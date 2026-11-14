"""Reproducible figures. Every function returns the Matplotlib figure and optionally saves it."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _finish(fig, save_path):
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)  # saved figures are released (long analyses would otherwise leak one figure per call)
    return fig


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    k = np.ones(window) / window
    return np.convolve(np.pad(x, (window // 2, window - 1 - window // 2), mode="edge"), k, mode="valid")


def plot_learning_curves(curves: dict[str, ArrayLike], window: int = 10, ylabel: str = "P(choose better option)",
                         save_path=None):
    """Mean learning curves. ``curves[label]`` is an array ``(n_participants, n_trials)`` of 0/1 indicators."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, arr in curves.items():
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        m = arr.mean(axis=0)
        ax.plot(np.arange(1, len(m) + 1), _smooth(m, window), label=label)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set(xlabel="trial", ylabel=ylabel, ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, save_path)


def plot_action_values(values: ArrayLike, choices: ArrayLike | None = None, labels=("A", "B"), title="Action values",
                       save_path=None):
    """Latent action-value trajectories (and the participant's choices as ticks)."""
    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for a in range(values.shape[1]):
        ax.plot(values[:, a], label=f"Q({labels[a]})")
    if choices is not None:
        c = np.asarray(choices)
        for a in range(values.shape[1]):
            t = np.where(c == a)[0]
            ax.plot(t, np.full(len(t), values.min() - 0.05 - 0.03 * a), "|", ms=6, color=f"C{a}")
    ax.set(xlabel="trial", ylabel="value", title=title)
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, save_path)


def plot_choice_probabilities(probs: ArrayLike, choices: ArrayLike | None = None, hidden_state: ArrayLike | None = None,
                              title="Model choice probability", save_path=None):
    """P(choose option A) over trials, with observed choices and (if known) the hidden good option."""
    probs = np.asarray(probs)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(probs[:, 0], color="C0", label="model P(A)")
    if choices is not None:
        ax.plot(1 - np.asarray(choices), "|", color="k", alpha=0.5, ms=8, label="observed choice (1 = A)")
    if hidden_state is not None:
        ax.step(np.arange(len(hidden_state)), 1 - np.asarray(hidden_state), where="mid", color="C3", alpha=0.5,
                lw=1, label="A is good (1) / B is good (0)")
    ax.set(xlabel="trial", ylabel="probability", ylim=(-0.05, 1.05), title=title)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _finish(fig, save_path)


def plot_observed_vs_predicted(probs: ArrayLike, choices: ArrayLike, n_bins: int = 8, save_path=None):
    """Calibration plot: predicted P(A) (binned) against the observed frequency of choosing A."""
    p = np.asarray(probs)[:, 0]
    y = 1 - np.asarray(choices)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        if (idx == b).sum() >= 5:
            xs.append(p[idx == b].mean())
            ys.append(y[idx == b].mean())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], color="grey", ls=":")
    ax.plot(xs, ys, "o-")
    ax.set(xlabel="predicted P(choose A)", ylabel="observed frequency", xlim=(0, 1), ylim=(0, 1), title="Observed vs predicted")
    return _finish(fig, save_path)


def plot_parameter_distributions(fits: pd.DataFrame, param_names, save_path=None):
    """Histograms of participant-level parameter estimates (individual differences)."""
    fig, axes = plt.subplots(1, len(param_names), figsize=(3.2 * len(param_names), 3), squeeze=False)
    for ax, p in zip(axes[0], param_names):
        ax.hist(fits[f"fit_{p}"], bins=12, color="C0", alpha=0.8)
        ax.set(xlabel=p, ylabel="participants")
    return _finish(fig, save_path)


def plot_parameter_recovery(df: pd.DataFrame, param_names, n_trials: int | None = None, save_path=None):
    """Scatter of generating vs recovered parameters with the identity line and Pearson r."""
    d = df if n_trials is None else df[df["n_trials"] == n_trials]
    fig, axes = plt.subplots(1, len(param_names), figsize=(3.6 * len(param_names), 3.6), squeeze=False)
    for ax, p in zip(axes[0], param_names):
        t, e = d[f"true_{p}"], d[f"fit_{p}"]
        lim = (min(t.min(), e.min()), max(t.max(), e.max()))
        ax.plot(lim, lim, color="grey", ls=":")
        ax.scatter(t, e, s=14, alpha=0.7)
        r = np.corrcoef(t, e)[0, 1] if len(t) > 2 else np.nan
        ax.set(xlabel=f"true {p}", ylabel=f"recovered {p}", title=f"{p}  (r = {r:.2f})")
    return _finish(fig, save_path)


def plot_recovery_vs_trials(metrics: pd.DataFrame, metric: str = "r", save_path=None):
    """Recovery metric as a function of the number of trials, one line per parameter."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for p, g in metrics.groupby("param"):
        ax.plot(g["n_trials"], g[metric], "o-", label=p)
    ax.set(xlabel="trials per participant", ylabel=metric)
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, save_path)


def plot_confusion_matrix(confusion: pd.DataFrame, title="Model recovery", xlabel="selected model",
                          ylabel="generating model", save_path=None):
    """Heat-map confusion matrix annotated with proportions."""
    fig, ax = plt.subplots(figsize=(1.3 * len(confusion.columns) + 2, 1.1 * len(confusion.index) + 1.5))
    im = ax.imshow(confusion.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(confusion.columns)), confusion.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(confusion.index)), confusion.index)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            v = confusion.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.5 else "black")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _finish(fig, save_path)


def plot_model_comparison(summary: pd.DataFrame, columns=("total_aic", "total_bic"), save_path=None):
    """Summed information criteria relative to the best model (lower is better)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    w = 0.8 / len(columns)
    for k, c in enumerate(columns):
        delta = summary[c] - summary[c].min()
        ax.bar(np.arange(len(summary)) + k * w, delta, w, label=f"Δ{c.replace('total_', '').upper()}")
    ax.set_xticks(np.arange(len(summary)) + w * (len(columns) - 1) / 2, summary.index, rotation=20, ha="right")
    ax.set(ylabel="difference from best (lower = better)")
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, save_path)


def plot_predictive_performance(held_out: pd.DataFrame, save_path=None):
    """Held-out log-likelihood per trial (mean ± s.e.m. over participants), chance as reference."""
    g = held_out.groupby("model")["test_ll_per_trial"]
    m, se = g.mean(), g.sem()
    order = [x for x in m.index if x != "chance"] + (["chance"] if "chance" in m.index else [])
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(order, m[order], yerr=se[order].fillna(0), color=["C0" if o != "chance" else "grey" for o in order])
    ax.set(ylabel="held-out log-likelihood per trial")
    ax.tick_params(axis="x", rotation=20)
    return _finish(fig, save_path)


def plot_pomdp_value_function(alpha_sets, horizons, xlabel="P(state 0)", title="POMDP value function", save_path=None):
    """Piecewise-linear convex value functions of a 2-state POMDP for several horizons."""
    xs = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(6, 4))
    for h in horizons:
        v = [alpha_sets[h].value(np.array([x, 1 - x])) for x in xs]
        ax.plot(xs, v, label=f"{h} steps left")
    ax.set(xlabel=xlabel, ylabel="V(b)", title=title)
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, save_path)


# ------------------------------------------------------------------ empirical-analysis figures
def plot_behavioural_overview(summary: pd.DataFrame, trial_table: pd.DataFrame, choice_categories, save_path=None):
    """Four panels describing the *data* before any model: choice shares, reward over time, stay behaviour, missed trials.

    ``summary`` is ``BehaviouralDataset.participant_summary()`` and ``trial_table`` is ``trial_summary()``.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    shares = [summary[f"p_choose_{c}"].mean() for c in choice_categories]
    sem = [summary[f"p_choose_{c}"].std() / np.sqrt(len(summary)) for c in choice_categories]
    axes[0, 0].bar(range(len(shares)), shares, yerr=1.96 * np.array(sem), capsize=3)
    axes[0, 0].axhline(1 / len(shares), color="grey", ls=":", lw=0.8)
    axes[0, 0].set_xticks(range(len(shares)), list(choice_categories))
    axes[0, 0].set(title="Choice share by option", ylabel="mean proportion of choices")
    obs = trial_table[trial_table["choice_index"] >= 0]
    by_t = obs.groupby("trial")["reward"].agg(["mean", "sem"])
    axes[0, 1].plot(by_t.index, by_t["mean"])
    axes[0, 1].fill_between(by_t.index, by_t["mean"] - 1.96 * by_t["sem"], by_t["mean"] + 1.96 * by_t["sem"], alpha=0.25)
    axes[0, 1].set(title="Mean reward across trials", xlabel="trial", ylabel="reward")
    stay_t = obs.groupby("trial")["stay"].mean()
    axes[1, 0].plot(stay_t.index, stay_t.values)
    axes[1, 0].axhline(1 / len(shares), color="grey", ls=":", lw=0.8)
    axes[1, 0].set(title="P(repeat previous choice)", xlabel="trial", ylabel="proportion", ylim=(0, 1))
    axes[1, 1].hist(summary["n_missed"] / summary["n_trials"], bins=20)
    axes[1, 1].set(title="Missed trials per participant", xlabel="fraction of trials missed", ylabel="participants")
    return _finish(fig, save_path)


def plot_reliability(calibration: dict[str, pd.DataFrame], save_path=None):
    """Reliability diagram: predicted probability vs observed choice frequency for each model (diagonal = calibrated)."""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    for name, t in calibration.items():
        t = t[t["n"] > 0]
        ax.plot(t["mean_predicted"], t["observed_frequency"], marker="o", ms=4, label=name)
    ax.set(xlabel="predicted probability", ylabel="observed frequency", xlim=(0, 1), ylim=(0, 1),
           title="Calibration on held-out trials")
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, save_path)


def plot_heldout_summary(summary: pd.DataFrame, diff_column: str, save_path=None):
    """Paired held-out log-likelihood difference vs the reference model with bootstrap CIs, one line per model.

    ``summary`` comes from :func:`humanchoice.comparison.summarise_held_out`; points above 0 beat the reference.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in summary[summary["model"] != diff_column.removeprefix("ll_diff_vs_")].groupby("model", sort=False):
        g = g.sort_values("train_fraction")
        ax.errorbar(g["train_fraction"], g[diff_column], yerr=[g[diff_column] - g["diff_lo"], g["diff_hi"] - g[diff_column]],
                    marker="o", ms=4, capsize=2, label=name)
    ax.axhline(0, color="grey", ls=":", lw=1)
    ax.set(xlabel="fraction of trials used for fitting", ylabel="held-out log-lik. per trial\n(difference from reference)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    return _finish(fig, save_path)


def plot_predictive_check(check: pd.DataFrame, title: str = "Predictive check", save_path=None):
    """Observed vs replicate-interval for each statistic (long table from ``one_step_ahead_check`` or pooled z-scores)."""
    c = check.dropna(subset=["replicate_mean"])
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(c) + 1.4))
    y = np.arange(len(c))
    ax.hlines(y, c["lo"], c["hi"], color="C0", lw=5, alpha=0.4, label="95% of replicates")
    ax.plot(c["replicate_mean"], y, "|", color="C0", ms=10)
    ax.plot(c["observed"], y, "o", color="C3", label="observed")
    ax.set_yticks(y, c["statistic"])
    ax.invert_yaxis()
    ax.set(title=title)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _finish(fig, save_path)


def plot_misfit_rates(rates: dict[str, pd.DataFrame], save_path=None, threshold: float | None = None):
    """Fraction of participants with predictive-check p < .05, per statistic and model (dotted = 5% chance level, dashed red = flag threshold)."""
    stats = list(next(iter(rates.values()))["statistic"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.8 / len(rates)
    for k, (name, t) in enumerate(rates.items()):
        t = t.set_index("statistic").reindex(stats)
        ax.bar(np.arange(len(stats)) + k * w, t["misfit_rate"], w, label=name)
    ax.axhline(0.05, color="grey", ls=":", lw=1)
    if threshold is not None:
        ax.axhline(threshold, color="C3", ls="--", lw=1)
    ax.set_xticks(np.arange(len(stats)) + 0.4 - w / 2, stats, rotation=35, ha="right", fontsize=8)
    ax.set(ylabel="fraction of participants with p < .05", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=7, ncol=2)
    return _finish(fig, save_path)


def plot_profiles(profiles: dict[str, pd.DataFrame], save_path=None):
    """Likelihood profiles (delta NLL vs parameter value); the dashed line is the 95% likelihood-interval threshold."""
    fig, axes = plt.subplots(1, len(profiles), figsize=(3.2 * len(profiles), 3), sharey=True, squeeze=False)
    for ax, (name, prof) in zip(axes[0], profiles.items()):
        ax.plot(prof["value"], prof["delta_nll"])
        ax.axhline(1.92, color="grey", ls="--", lw=0.8)
        ax.set(xlabel=name, ylim=(0, 8))
    axes[0][0].set_ylabel("profile delta NLL")
    return _finish(fig, save_path)


def plot_error_correlations(matrix: pd.DataFrame, title="Correlation of estimation errors", save_path=None):
    """Heatmap of a parameter x parameter matrix (error correlations or leakage); off-diagonal entries reveal confounds."""
    fig, ax = plt.subplots(figsize=(0.9 * len(matrix) + 2, 0.9 * len(matrix) + 1.2))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.index)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            v = matrix.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _finish(fig, save_path)
