"""POMDP (hidden-state inference) choice model for reversal-learning tasks."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from humanchoice.models.base import ChoiceModel
from humanchoice.pomdp.pomdp import POMDP


@lru_cache(maxsize=512)
def reversal_pomdp(hazard: float, p_good: float, gamma: float = 1.0) -> POMDP:
    """POMDP for a two-option reversal task.

    Hidden state ``s`` = index of the currently good option. Actions = choose option 0/1. Observation =
    reward in {0, 1}. The good option flips with probability ``hazard`` before each outcome;
    ``P(reward=1 | choose a, state s') = p_good`` if ``a == s'`` else ``1 - p_good``.
    """
    T = np.empty((2, 2, 2))
    T[:] = [[1 - hazard, hazard], [hazard, 1 - hazard]]
    p1 = np.array([[p_good if a == s2 else 1 - p_good for s2 in range(2)] for a in range(2)])  # (a, s')
    O = np.stack([1 - p1, p1], axis=-1)  # (a, s', z)
    # expected reward uses the pre-transition state: R[s, a] = sum_s' T[a, s, s'] P(r=1 | a, s')
    R = np.einsum("asx,ax->sa", T, p1)
    return POMDP(T, O, R, gamma)


class POMDPReversalModel(ChoiceModel):
    """Bayesian hidden-state tracker for reversal-learning tasks.

    The agent knows (or assumes) that one option is good with reliability ``p_good`` and that the good option
    may switch with per-trial probability ``hazard``. It maintains the exact belief ``b = P(option 0 is good)``
    with a Bayes filter and values each action by belief-tree planning to ``lookahead`` trials.

    Parameters: ``beta`` (inverse temperature) and ``hazard`` (the agent's *believed* reversal rate).
    Fixed hyper-parameters: ``p_good`` (assumed reliability) and ``lookahead``.
    """

    name = "pomdp_reversal"
    reward_support = "binary"
    fixed_n_actions = 2
    param_names = ("beta", "hazard")
    bounds = ((0.0, 40.0), (1e-3, 0.5))
    sample_ranges = ((2.0, 15.0), (0.02, 0.3))

    def __init__(self, p_good: float = 0.8, lookahead: int = 1, gamma: float = 1.0) -> None:
        super().__init__(2)
        if not 0.5 < p_good < 1.0:
            raise ValueError("p_good must lie in (0.5, 1)")
        if lookahead < 1:
            raise ValueError("lookahead must be >= 1")
        self.p_good, self.lookahead, self.gamma = float(p_good), int(lookahead), float(gamma)

    def _pomdp(self, params: NDArray) -> POMDP:
        return reversal_pomdp(round(float(params[1]), 10), self.p_good, self.gamma)

    def init_state(self, params: NDArray) -> NDArray[np.float64]:
        return np.array([0.5, 0.5])

    def action_values(self, params: NDArray, state: NDArray) -> NDArray[np.float64]:
        return self._pomdp(params).q_values(state, self.lookahead)

    def update(self, params: NDArray, state: NDArray, action: int, reward: float) -> NDArray[np.float64]:
        if reward not in (0, 1):
            raise ValueError("POMDPReversalModel requires binary rewards")
        return self._pomdp(params).belief_update(state, action, int(reward))
