"""Likelihood profiles: a parameter-uncertainty diagnostic that exposes weak identifiability.

A *profile* fixes one parameter on a grid and re-optimises all the others. A flat profile means the data barely
constrain that parameter; a narrow, parabolic profile means it is well determined. The interval where the profile
rises less than ``chi2(1, level)/2`` above its minimum is an approximate likelihood-based confidence interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.stats import chi2

from humanchoice.fitting.likelihood import FAILURE_NLL, make_objective
from humanchoice.models.base import ChoiceModel
from humanchoice.utils import as_rng


def likelihood_profile(
    model: ChoiceModel,
    choices: ArrayLike,
    rewards: ArrayLike,
    param: str,
    grid: ArrayLike | None = None,
    n_grid: int = 25,
    n_starts: int = 2,
    seed: int = 0,
    available: ArrayLike | None = None,
) -> pd.DataFrame:
    """Profile negative log-likelihood of ``param`` over ``grid`` (default: ``n_grid`` points across its bounds).

    Returns columns ``value``, ``nll``, ``delta_nll`` (relative to the profile minimum).
    """
    if param not in model.param_names:
        raise ValueError(f"{param} is not a parameter of {model.name}: {model.param_names}")
    j = model.param_names.index(param)
    lo, hi = model.bounds[j]
    grid = np.linspace(lo, hi, n_grid) if grid is None else np.asarray(grid, dtype=float)
    obj = make_objective(model, choices, rewards, available)
    rng = as_rng(seed)
    others = [i for i in range(model.n_params) if i != j]
    rows = []
    for g in grid:
        def reduced(x, g=g):
            full = np.empty(model.n_params)
            full[j] = g
            full[others] = x
            return obj(full)

        if not others:
            rows.append((float(g), reduced(np.array([]))))
            continue
        best = FAILURE_NLL
        for _ in range(n_starts):
            x0 = np.array([model.sample_params(rng)[i] for i in others])
            res = minimize(reduced, x0, method="L-BFGS-B", bounds=[model.bounds[i] for i in others])
            best = min(best, res.fun)
        rows.append((float(g), float(best)))
    out = pd.DataFrame(rows, columns=["value", "nll"])
    out["delta_nll"] = out["nll"] - out["nll"].min()
    return out


def profile_interval(profile: pd.DataFrame, level: float = 0.95) -> tuple[float, float, bool]:
    """Approximate likelihood-based interval ``(lo, hi, open)`` from a profile.

    ``open`` is ``True`` when the profile does not rise past the threshold on at least one side within the grid
    (the data cannot rule out the edge of the parameter range): the interval is then truncated at the grid edge.
    """
    cut = chi2.ppf(level, df=1) / 2.0
    v, d = profile["value"].to_numpy(), profile["delta_nll"].to_numpy()
    imin = int(np.argmin(d))

    def crossing(idx_range):
        for a, b in idx_range:
            if d[b] > cut >= d[a]:
                return v[a] + (cut - d[a]) / (d[b] - d[a]) * (v[b] - v[a])
        return None

    left = crossing([(i, i - 1) for i in range(imin, 0, -1)])
    right = crossing([(i, i + 1) for i in range(imin, len(v) - 1)])
    return (
        float(v[0] if left is None else left),
        float(v[-1] if right is None else right),
        left is None or right is None,
    )
