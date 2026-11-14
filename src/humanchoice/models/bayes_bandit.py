"""Bayes-adaptive (BAMDP) choice model for Bernoulli-reward bandit tasks (exact planning; practical for small K and horizons)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from humanchoice.bamdp.bernoulli import BetaBernoulliBanditBAMDP, Posterior
from humanchoice.models.base import ChoiceModel


class BayesBanditModel(ChoiceModel):
    """An agent that plans in belief space with a Beta-Bernoulli BAMDP, then chooses by softmax.

    The agent assumes a **stationary** bandit, maintains conjugate Beta posteriors over each option's
    reward probability, and computes Bayes-optimal values for a ``horizon``-trial lookahead. Its action
    values therefore include a value-of-information (exploration) term absent from Q-learning.

    Parameters: ``beta`` only. Hyper-parameters (fixed, not fitted): Beta ``prior``, ``horizon`` and
    ``gamma``. Values are cached across calls so likelihood evaluation is fast.
    """

    name = "bamdp_bandit"
    param_names = ("beta",)
    bounds = ((0.0, 30.0),)
    sample_ranges = ((1.0, 12.0),)

    reward_support = "binary"

    def __init__(
        self, prior: tuple[float, float] = (1.0, 1.0), horizon: int = 6, gamma: float = 1.0, n_actions: int = 2
    ) -> None:
        super().__init__(n_actions)
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = int(horizon)
        self.planner = BetaBernoulliBanditBAMDP(n_arms=self.n_actions, prior=prior, gamma=gamma)

    def init_state(self, params: NDArray) -> Posterior:
        return self.planner.initial_posterior()

    def action_values(self, params: NDArray, state: Posterior) -> NDArray[np.float64]:
        return self.planner.q_values(state, self.horizon)

    def update(self, params: NDArray, state: Posterior, action: int, reward: float) -> Posterior:
        if reward not in (0, 1):
            raise ValueError("BayesBanditModel requires binary rewards")
        return self.planner.update(state, action, int(reward))
