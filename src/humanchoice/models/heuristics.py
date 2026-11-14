"""Baseline choice models: uniform random choice and (probabilistic) win-stay / lose-shift.

Any learning model that cannot beat these on held-out data has not earned its parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from humanchoice.models.base import ChoiceModel
from humanchoice.utils import MISSED


class RandomChoice(ChoiceModel):
    """Uniform choice among the available options. No parameters; the chance baseline."""

    name = "random"

    def init_state(self, params: NDArray) -> None:
        return None

    def action_values(self, params: NDArray, state) -> NDArray[np.float64]:
        return np.zeros(self.n_actions)

    def choice_probs(self, params: NDArray, state, available: ArrayLike | None = None) -> NDArray[np.float64]:
        avail = np.ones(self.n_actions) if available is None else np.asarray(available, dtype=float)
        return avail / avail.sum()

    def update(self, params: NDArray, state, action: int, reward: float) -> None:
        return None


class WinStayLoseShift(ChoiceModel):
    """Probabilistic win-stay / lose-shift.

    After a *win* (``reward >= win_threshold``) repeat the previous choice with probability ``p_win_stay``; after a
    *loss* switch with probability ``p_lose_shift``. A switch picks uniformly among the other available options.
    The first trial is uniform. Parameters: ``p_win_stay``, ``p_lose_shift`` in [0, 1].

    ``win_threshold`` is a fixed modelling choice (default 0.5, natural for binary rewards in {0, 1}); for continuous
    rewards choose it deliberately and report it.
    """

    name = "win_stay_lose_shift"
    param_names = ("p_win_stay", "p_lose_shift")
    bounds = ((0.0, 1.0), (0.0, 1.0))
    sample_ranges = ((0.5, 1.0), (0.3, 1.0))

    def __init__(self, n_actions: int = 2, win_threshold: float = 0.5) -> None:
        super().__init__(n_actions)
        self.win_threshold = float(win_threshold)

    def init_state(self, params: NDArray) -> tuple[int, bool]:
        return (MISSED, False)  # (previous choice, previous trial was a win)

    def action_values(self, params: NDArray, state) -> NDArray[np.float64]:
        return np.zeros(self.n_actions)

    def choice_probs(self, params: NDArray, state, available: ArrayLike | None = None) -> NDArray[np.float64]:
        last, win = state
        avail = np.ones(self.n_actions, dtype=bool) if available is None else np.asarray(available, dtype=bool)
        uniform = avail / avail.sum()
        if last == MISSED or not avail[last]:
            return uniform
        p_stay = params[0] if win else 1.0 - params[1]
        others = avail.copy()
        others[last] = False
        probs = np.zeros(self.n_actions)
        if not others.any():  # nothing to switch to
            probs[last] = 1.0
            return probs
        probs[last] = p_stay
        probs[others] = (1.0 - p_stay) / others.sum()
        return probs

    def update(self, params: NDArray, state, action: int, reward: float) -> tuple[int, bool]:
        return (action, bool(reward >= self.win_threshold))
