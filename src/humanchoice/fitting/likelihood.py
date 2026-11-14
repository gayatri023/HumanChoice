"""Likelihood objectives for maximum-likelihood estimation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from humanchoice.models.base import ChoiceModel

#: value returned if the likelihood cannot be evaluated (keeps optimisers from crashing)
FAILURE_NLL = 1e10


def negative_log_likelihood(
    model: ChoiceModel,
    params: ArrayLike,
    choices: ArrayLike,
    rewards: ArrayLike,
    available: ArrayLike | None = None,
) -> float:
    """``NLL(theta) = -sum_t log P(choice_t | history_t, theta)`` over observed trials."""
    return -model.log_likelihood(params, choices, rewards, available)


def make_objective(
    model: ChoiceModel, choices: ArrayLike, rewards: ArrayLike, available: ArrayLike | None = None
) -> Callable[[NDArray], float]:
    """Return a robust scalar objective ``theta -> NLL`` for ``scipy.optimize``."""
    choices = np.asarray(choices, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    lo = np.array([b[0] for b in model.bounds])
    hi = np.array([b[1] for b in model.bounds])

    def objective(theta: NDArray) -> float:
        try:
            val = -model.log_likelihood(np.clip(theta, lo, hi), choices, rewards, available)
        except (ValueError, FloatingPointError):
            return FAILURE_NLL
        return val if np.isfinite(val) else FAILURE_NLL

    return objective
