"""Bayesian (Kalman-filter) learner for restless bandits with continuous rewards."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from humanchoice.models.base import ChoiceModel
from humanchoice.utils import masked_softmax


class KalmanBandit(ChoiceModel):
    """Conjugate Gaussian tracking of drifting reward means (cf. Daw et al., 2006).

    Each option's mean payoff is tracked by a Kalman filter. Per trial the *prior* is propagated by a mean-reverting
    random walk (``decay`` towards ``theta``, innovation variance ``drift_ratio * obs_var``) and the chosen option's
    posterior is updated with Kalman gain ``k = v / (v + obs_var)``. The Kalman gain adapts automatically (large when an
    option has not been sampled for a while) - a Bayesian account of why learning rates should depend on uncertainty.

    Choice: softmax over ``m_a + phi * sqrt(v_a)`` - ``phi > 0`` adds a directed-exploration bonus for uncertain options.

    Fitted parameters: ``beta`` (inverse temperature), ``phi`` (uncertainty bonus) and ``drift_ratio``
    (innovation variance / observation variance; larger = the agent believes the world changes faster).
    Fixed hyper-parameters (constructor): ``obs_sd`` (assumed observation noise, in reward units), ``decay``, ``theta``,
    initial mean ``m0`` and initial variance ``v0``. They fix the *scale* of ``phi`` and must be justified from the
    task description; the defaults are conventional choices for rewards rescaled to [0, 1], **not** taken from any
    specific dataset.
    """

    name = "kalman_bandit"
    param_names = ("beta", "phi", "drift_ratio")
    bounds = ((0.0, 40.0), (-2.0, 5.0), (1e-3, 10.0))
    sample_ranges = ((2.0, 15.0), (0.0, 2.0), (0.05, 2.0))

    def __init__(
        self,
        n_actions: int = 4,
        obs_sd: float = 0.1,
        decay: float = 1.0,
        theta: float = 0.5,
        m0: float = 0.5,
        v0: float = 0.1,
    ) -> None:
        super().__init__(n_actions)
        if obs_sd <= 0 or v0 <= 0 or not 0.0 < decay <= 1.0:
            raise ValueError("obs_sd and v0 must be > 0 and decay in (0, 1]")
        self.obs_var, self.decay, self.theta, self.m0, self.v0 = obs_sd**2, decay, theta, m0, v0

    def init_state(self, params: NDArray) -> tuple[NDArray, NDArray]:
        return np.full(self.n_actions, self.m0), np.full(self.n_actions, self.v0)

    def action_values(self, params: NDArray, state) -> NDArray[np.float64]:
        m, v = state
        return m + params[1] * np.sqrt(v)

    def choice_probs(self, params: NDArray, state, available: ArrayLike | None = None) -> NDArray[np.float64]:
        return masked_softmax(self.action_values(params, state), params[0], available)

    def update(self, params: NDArray, state, action: int, reward: float):
        m, v = state[0].copy(), state[1].copy()
        gain = v[action] / (v[action] + self.obs_var)
        m[action] += gain * (reward - m[action])
        v[action] *= 1.0 - gain
        # propagate every option's prior to the next trial
        m = self.decay * m + (1.0 - self.decay) * self.theta
        v = self.decay**2 * v + params[2] * self.obs_var
        return m, v
