"""Model-selection criteria and recovery metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

from humanchoice.utils import MISSED, as_rng


def aic(log_likelihood: float, n_params: int) -> float:
    """Akaike information criterion ``2k - 2 log L``."""
    return 2.0 * n_params - 2.0 * log_likelihood


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Bayesian information criterion ``k log n - 2 log L``."""
    return n_params * np.log(n_obs) - 2.0 * log_likelihood


def akaike_weights(aics: ArrayLike) -> NDArray[np.float64]:
    """Relative model probabilities ``exp(-delta/2) / sum exp(-delta/2)``."""
    a = np.asarray(aics, dtype=float)
    w = np.exp(-0.5 * (a - a.min()))
    return w / w.sum()


def _observed(probs: ArrayLike, choices: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Restrict ``probs`` (T, K) and ``choices`` (T,) to trials with an observed choice (``choice != -1``)."""
    p, c = np.asarray(probs, dtype=float), np.asarray(choices, dtype=int)
    obs = c != MISSED
    return p[obs], c[obs]


def choice_accuracy(probs: ArrayLike, choices: ArrayLike) -> float:
    """Fraction of observed trials where the most probable action was chosen (ties: lowest index)."""
    p, c = _observed(probs, choices)
    return float(np.mean(np.argmax(p, axis=1) == c)) if len(c) else float("nan")


def brier_score(probs: ArrayLike, choices: ArrayLike) -> float:
    """Multi-class Brier score ``mean_t sum_k (p_tk - 1[c_t = k])^2`` over observed trials (0 = perfect; lower is better).

    For ``K`` options a uniformly random predictor scores ``1 - 1/K``. Unlike accuracy it rewards well-calibrated
    probabilities rather than only the most likely option.
    """
    p, c = _observed(probs, choices)
    if not len(c):
        return float("nan")
    onehot = np.zeros_like(p)
    onehot[np.arange(len(c)), c] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def calibration_table(probs: ArrayLike, choices: ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table over all (trial, option) pairs: predicted probability bin vs observed choice frequency.

    Columns ``bin_lo, bin_hi, n, mean_predicted, observed_frequency``. A calibrated model has
    ``observed_frequency ~ mean_predicted`` in every bin.
    """
    p, c = _observed(probs, choices)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(c)), c] = 1.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p.ravel(), edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": n,
                "mean_predicted": float(p.ravel()[m].mean()) if n else np.nan,
                "observed_frequency": float(onehot.ravel()[m].mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """``sum_b (n_b / N) |observed_b - predicted_b|`` from a :func:`calibration_table`."""
    t = table[table["n"] > 0]
    return float((t["n"] * (t["observed_frequency"] - t["mean_predicted"]).abs()).sum() / t["n"].sum())


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well behaved near 0 and 1 and for small ``n``)."""
    if n <= 0:
        return float("nan"), float("nan")
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + level / 2))
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def mean_ci(values: ArrayLike, level: float = 0.95, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """``(mean, lo, hi)``: percentile-bootstrap CI of the mean, resampling participants (NaNs ignored)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    rng = as_rng(seed)
    boots = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(v.mean()), float(lo), float(hi)


def mean_absolute_error(true: ArrayLike, est: ArrayLike) -> float:
    return float(np.mean(np.abs(np.asarray(est) - np.asarray(true))))


def rmse(true: ArrayLike, est: ArrayLike) -> float:
    return float(np.sqrt(np.mean((np.asarray(est) - np.asarray(true)) ** 2)))


def bias(true: ArrayLike, est: ArrayLike) -> float:
    """Mean signed error ``E[est - true]``."""
    return float(np.mean(np.asarray(est) - np.asarray(true)))


def error_variance(true: ArrayLike, est: ArrayLike) -> float:
    """Variance of the estimation error ``Var[est - true]`` (spread around the bias)."""
    return float(np.var(np.asarray(est) - np.asarray(true), ddof=1)) if len(np.asarray(est)) > 1 else float("nan")


def correlation(true: ArrayLike, est: ArrayLike) -> float:
    """Pearson correlation between generating and estimated values (NaN if either is constant)."""
    t, e = np.asarray(true, dtype=float), np.asarray(est, dtype=float)
    if len(t) < 3 or np.std(t) == 0 or np.std(e) == 0:
        return float("nan")
    return float(np.corrcoef(t, e)[0, 1])


def median_absolute_error(true: ArrayLike, est: ArrayLike) -> float:
    """Median of ``|est - true|``: robust to the occasional fit that runs to a parameter bound."""
    return float(np.median(np.abs(np.asarray(est) - np.asarray(true))))


def spearman(true: ArrayLike, est: ArrayLike) -> float:
    """Spearman rank correlation (NaN if either input is constant); robust to heavy-tailed estimation errors."""
    t, e = np.asarray(true, dtype=float), np.asarray(est, dtype=float)
    if len(t) < 3 or np.std(t) == 0 or np.std(e) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(t), rankdata(e))[0, 1])  # ranks with ties averaged


def bootstrap_ci(
    true: ArrayLike,
    est: ArrayLike,
    statistic: Callable[[ArrayLike, ArrayLike], float],
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a paired statistic over participants."""
    t, e = np.asarray(true), np.asarray(est)
    rng = as_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(t), len(t))
        v = statistic(t[idx], e[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
