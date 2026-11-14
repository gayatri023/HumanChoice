"""Choice environments for bandit tasks.

All environments share a minimal protocol: attribute ``n_actions``, ``reset(rng)`` and ``step(action, rng) -> reward``,
plus ``good_arm_history`` (the best option *at the time of each trial*, used for accuracy / regret analyses).
Actions are integer indices ``0 .. n_actions-1``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class BernoulliBandit:
    """Stationary bandit: option ``k`` pays 1 with fixed probability ``probs[k]`` (any number of options >= 2)."""

    def __init__(self, probs: Sequence[float] = (0.8, 0.2)) -> None:
        if len(probs) < 2 or any(not 0.0 <= p <= 1.0 for p in probs):
            raise ValueError("probs must contain at least two probabilities in [0, 1]")
        self.probs = tuple(float(p) for p in probs)
        self.n_actions = len(self.probs)
        self.good_arm_history: list[int] = []

    def reset(self, rng: np.random.Generator) -> None:
        self.good_arm_history = []

    def step(self, action: int, rng: np.random.Generator) -> int:
        self.good_arm_history.append(int(np.argmax(self.probs)))
        return int(rng.random() < self.probs[action])


class ReversalBandit:
    """Probabilistic reversal learning task (two options).

    One option is "good" (pays 1 with probability ``p_good``), the other pays with ``1 - p_good``.
    At the start of every trial the identity of the good option flips with probability ``hazard``
    (including trial 1, where the initial good option is uniformly random, so the process is symmetric).
    ``hazard = 0`` recovers a stationary bandit with a random good arm.

    This timing (transition, then outcome) matches the POMDP convention used by
    :class:`humanchoice.models.POMDPReversalModel`.
    """

    n_actions = 2

    def __init__(self, p_good: float = 0.8, hazard: float = 0.05) -> None:
        if not 0.5 <= p_good <= 1.0:
            raise ValueError("p_good must lie in [0.5, 1]")
        if not 0.0 <= hazard <= 1.0:
            raise ValueError("hazard must lie in [0, 1]")
        self.p_good, self.hazard = float(p_good), float(hazard)
        self.good_arm = 0
        self.good_arm_history: list[int] = []

    def reset(self, rng: np.random.Generator) -> None:
        self.good_arm = int(rng.integers(2))
        self.good_arm_history = []

    def step(self, action: int, rng: np.random.Generator) -> int:
        if rng.random() < self.hazard:
            self.good_arm = 1 - self.good_arm
        self.good_arm_history.append(self.good_arm)
        p = self.p_good if action == self.good_arm else 1.0 - self.p_good
        return int(rng.random() < p)


class DriftingBernoulliBandit:
    """Restless bandit with *binary* rewards: reward probabilities follow a random walk on the logit scale.

    ``logit p_k(t+1) = logit p_k(t) + N(0, drift_sd^2)``; probabilities start uniformly in ``[0.2, 0.8]``. ``drift_sd``
    controls volatility (``0`` = stationary), which makes this the natural task for volatility sweeps with models that
    require binary rewards.
    """

    def __init__(self, n_arms: int = 4, drift_sd: float = 0.15) -> None:
        if n_arms < 2 or drift_sd < 0:
            raise ValueError("n_arms must be >= 2 and drift_sd >= 0")
        self.n_actions, self.drift_sd = int(n_arms), float(drift_sd)
        self.logits = np.zeros(self.n_actions)
        self.good_arm_history: list[int] = []

    def reset(self, rng: np.random.Generator) -> None:
        p0 = rng.uniform(0.2, 0.8, self.n_actions)
        self.logits = np.log(p0 / (1.0 - p0))
        self.good_arm_history = []

    def step(self, action: int, rng: np.random.Generator) -> int:
        self.logits = self.logits + rng.normal(0.0, self.drift_sd, self.n_actions)
        self.good_arm_history.append(int(np.argmax(self.logits)))
        p = 1.0 / (1.0 + np.exp(-self.logits[action]))
        return int(rng.random() < p)


class RestlessBandit:
    """Gaussian restless bandit with continuous rewards in [0, 1] (the design of Daw et al., 2006, rescaled).

    The latent mean payoff of each option follows a mean-reverting random walk,
    ``mu(t+1) = decay * mu(t) + (1 - decay) * theta + N(0, drift_sd^2)``, and the reward is
    ``clip(mu + N(0, obs_sd^2), 0, 1)``. Defaults follow the design commonly cited for Daw et al. (2006) with payoffs
    of 0-100 divided by 100 (drift SD 2.8, observation SD 4, decay 0.9836, mean 50) - check the paper before relying on
    exact equivalence. This is a *simulation* of such a design, not of any specific published dataset's schedules.
    """

    def __init__(
        self,
        n_arms: int = 4,
        drift_sd: float = 0.028,
        obs_sd: float = 0.04,
        decay: float = 0.9836,
        theta: float = 0.5,
    ) -> None:
        if n_arms < 2 or drift_sd < 0 or obs_sd < 0 or not 0.0 < decay <= 1.0:
            raise ValueError("invalid RestlessBandit parameters")
        self.n_actions = int(n_arms)
        self.drift_sd, self.obs_sd, self.decay, self.theta = drift_sd, obs_sd, decay, theta
        self.mu = np.full(self.n_actions, theta)
        self.good_arm_history: list[int] = []

    def reset(self, rng: np.random.Generator) -> None:
        self.mu = rng.uniform(0.2, 0.8, self.n_actions)
        self.good_arm_history = []

    def step(self, action: int, rng: np.random.Generator) -> float:
        self.mu = self.decay * self.mu + (1.0 - self.decay) * self.theta + rng.normal(0.0, self.drift_sd, self.n_actions)
        self.good_arm_history.append(int(np.argmax(self.mu)))
        return float(np.clip(self.mu[action] + rng.normal(0.0, self.obs_sd), 0.0, 1.0))
