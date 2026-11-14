"""Hybrid model mixing model-free and model-based action values."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from humanchoice.models.base import ChoiceModel
from humanchoice.models.pomdp_belief import POMDPReversalModel
from humanchoice.models.q_learning import QLearning


class HybridModel(ChoiceModel):
    """``Q_hybrid = w * Q_model-based + (1 - w) * Q_model-free`` with a shared softmax.

    Combines :class:`QLearning` (habitual, model-free) with :class:`POMDPReversalModel` (model-based inference).
    Parameters: ``alpha``, ``hazard``, ``w`` (weight on the model-based values) and ``beta``.
    """

    name = "hybrid"
    reward_support = "binary"
    fixed_n_actions = 2
    param_names = ("alpha", "hazard", "w", "beta")
    bounds = ((0.0, 1.0), (1e-3, 0.5), (0.0, 1.0), (0.0, 40.0))
    sample_ranges = ((0.05, 0.95), (0.02, 0.3), (0.1, 0.9), (2.0, 15.0))

    def __init__(self, mf: QLearning | None = None, mb: POMDPReversalModel | None = None) -> None:
        super().__init__(2)
        self.mf = mf or QLearning()
        self.mb = mb or POMDPReversalModel()

    def _split(self, params: NDArray) -> tuple[NDArray, NDArray]:
        alpha, hazard, _, beta = params
        return np.array([alpha, beta]), np.array([beta, hazard])

    def init_state(self, params: NDArray):
        p_mf, p_mb = self._split(params)
        return (self.mf.init_state(p_mf), self.mb.init_state(p_mb))

    def action_values(self, params: NDArray, state) -> NDArray[np.float64]:
        p_mf, p_mb = self._split(params)
        w = params[2]
        return w * self.mb.action_values(p_mb, state[1]) + (1 - w) * self.mf.action_values(p_mf, state[0])

    def update(self, params: NDArray, state, action: int, reward: float):
        p_mf, p_mb = self._split(params)
        return (
            self.mf.update(p_mf, state[0], action, reward),
            self.mb.update(p_mb, state[1], action, reward),
        )
