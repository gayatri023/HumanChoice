"""Bayes-optimal planning for Bernoulli bandits (the simplest BAMDP).

A ``K``-armed Bernoulli bandit with independent ``Beta(alpha_k, beta_k)`` priors over each arm's
reward probability is a *Bayes-adaptive MDP* whose augmented state is the vector of posterior
pseudo-counts. Because the Beta prior is conjugate to the Bernoulli likelihood, the belief update
is exact and closed form::

    success on arm k:  alpha_k <- alpha_k + 1
    failure on arm k:  beta_k  <- beta_k  + 1

and the posterior predictive probability of reward is ``alpha_k / (alpha_k + beta_k)``.

The Bayes-optimal action values over a horizon ``h`` satisfy the *Bayes-adaptive Bellman equation*::

    Q_h(b, k) = p_k * (1 + gamma * V_{h-1}(b | success on k)) + (1 - p_k) * gamma * V_{h-1}(b | failure on k)
    V_h(b)    = max_k Q_h(b, k)

which is solved here by memoised recursion over reachable belief states. The resulting Q-values
contain an *exploration bonus* (value of information) that purely greedy Q-learning lacks.

References: Duff (2002); Ghavamzadeh et al. (2015), "Bayesian Reinforcement Learning: A Survey".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Posterior = tuple[float, ...]  # flat (alpha_1, beta_1, ..., alpha_K, beta_K)


class BetaBernoulliBanditBAMDP:
    """Exact finite-horizon Bayes-optimal planner for a Bernoulli bandit with Beta priors."""

    def __init__(self, n_arms: int = 2, prior: tuple[float, float] = (1.0, 1.0), gamma: float = 1.0) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be >= 2")
        if prior[0] <= 0 or prior[1] <= 0:
            raise ValueError("Beta prior pseudo-counts must be positive")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        self.n_arms = n_arms
        self.prior = (float(prior[0]), float(prior[1]))
        self.gamma = float(gamma)
        self._cache: dict[tuple[Posterior, int], float] = {}

    # ------------------------------------------------------------------ belief
    def initial_posterior(self) -> Posterior:
        """Posterior before any data: the prior on every arm."""
        return self.prior * self.n_arms

    @staticmethod
    def update(posterior: Posterior, arm: int, reward: int) -> Posterior:
        """Conjugate Beta update after pulling ``arm`` and observing ``reward`` in {0, 1}."""
        if reward not in (0, 1):
            raise ValueError("Bernoulli rewards must be 0 or 1")
        idx = 2 * arm + (0 if reward == 1 else 1)
        return posterior[:idx] + (posterior[idx] + 1.0,) + posterior[idx + 1 :]

    @staticmethod
    def posterior_mean(posterior: Posterior) -> NDArray[np.float64]:
        p = np.asarray(posterior, dtype=float).reshape(-1, 2)
        return p[:, 0] / p.sum(axis=1)

    # ---------------------------------------------------------------- planning
    def _q(self, post: Posterior, h: int, arm: int) -> float:
        a, b = post[2 * arm], post[2 * arm + 1]
        p = a / (a + b)
        if h == 1:
            return p
        g = self.gamma
        succ = self.update(post, arm, 1)
        fail = self.update(post, arm, 0)
        return p * (1.0 + g * self._v(succ, h - 1)) + (1.0 - p) * g * self._v(fail, h - 1)

    def _v(self, post: Posterior, h: int) -> float:
        if h == 0:
            return 0.0
        key = (post, h)
        hit = self._cache.get(key)
        if hit is None:
            hit = max(self._q(post, h, k) for k in range(self.n_arms))
            self._cache[key] = hit
        return hit

    def q_values(self, posterior: Posterior, horizon: int) -> NDArray[np.float64]:
        """Bayes-optimal action values with ``horizon`` trials remaining."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        return np.array([self._q(posterior, horizon, k) for k in range(self.n_arms)])

    def value(self, posterior: Posterior, horizon: int) -> float:
        """Bayes-optimal expected total (discounted) reward with ``horizon`` trials remaining."""
        return self._v(posterior, horizon)

    def clear_cache(self) -> None:
        self._cache.clear()
