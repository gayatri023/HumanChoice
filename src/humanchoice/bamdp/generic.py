"""General tabular BAMDP with unknown transitions (Dirichlet) and unknown rewards (Beta-Bernoulli).

The *Bayes-adaptive MDP* augments the physical state ``s`` with the agent's belief about the
environment's parameters. With conjugate priors the belief is a table of pseudo-counts:

* ``tc[s, a, s']``  Dirichlet pseudo-counts over transitions
* ``rc[s, a, 0/1]`` Beta pseudo-counts (success / failure) for Bernoulli rewards

Planning is exact finite-horizon expectimax over belief states (memoised), so it is only suitable for
*small* problems and short horizons; its purpose here is clarity and as a ground-truth reference for
faster specialised planners.

Rewards may alternatively be *known* (``known_rewards[s, a]``), in which case only transitions are learned.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Belief = tuple[tuple[float, ...], tuple[float, ...]]  # (flat transition counts, flat reward counts)


class BAMDP:
    """Bayes-adaptive MDP with Dirichlet / Beta conjugate beliefs."""

    def __init__(
        self,
        prior_transitions: NDArray,
        prior_rewards: NDArray | None = None,
        known_rewards: NDArray | None = None,
        gamma: float = 1.0,
    ) -> None:
        pt = np.asarray(prior_transitions, dtype=float)
        if pt.ndim != 3 or pt.shape[0] != pt.shape[2]:
            raise ValueError("prior_transitions must have shape (S, A, S)")
        if np.any(pt <= 0):
            raise ValueError("Dirichlet pseudo-counts must be strictly positive")
        if (prior_rewards is None) == (known_rewards is None):
            raise ValueError("provide exactly one of prior_rewards (Beta counts) or known_rewards")
        self.S, self.A, _ = pt.shape
        self.gamma = float(gamma)
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        self._tc0 = tuple(pt.ravel().tolist())
        if prior_rewards is not None:
            pr = np.asarray(prior_rewards, dtype=float)
            if pr.shape != (self.S, self.A, 2) or np.any(pr <= 0):
                raise ValueError("prior_rewards must have shape (S, A, 2) with positive entries")
            self._rc0 = tuple(pr.ravel().tolist())
            self._known = None
        else:
            kr = np.asarray(known_rewards, dtype=float)
            if kr.shape != (self.S, self.A):
                raise ValueError("known_rewards must have shape (S, A)")
            self._rc0 = ()
            self._known = kr
        self._cache: dict[tuple[int, Belief, int], float] = {}

    def initial_belief(self) -> Belief:
        return (self._tc0, self._rc0)

    def update(self, belief: Belief, s: int, a: int, s_next: int, reward: int | None = None) -> Belief:
        """Conjugate update after observing the transition ``(s, a) -> s_next`` (and Bernoulli reward)."""
        tc, rc = belief
        tcl = list(tc)
        tcl[(s * self.A + a) * self.S + s_next] += 1.0
        if rc:
            if reward not in (0, 1):
                raise ValueError("reward must be 0 or 1 when rewards are unknown")
            rcl = list(rc)
            rcl[(s * self.A + a) * 2 + (0 if reward == 1 else 1)] += 1.0
            rc = tuple(rcl)
        return (tuple(tcl), rc)

    def expected_model(self, belief: Belief) -> tuple[NDArray, NDArray]:
        """Posterior-mean transition tensor ``(S, A, S)`` and reward matrix ``(S, A)``."""
        T = np.asarray(belief[0]).reshape(self.S, self.A, self.S)
        T = T / T.sum(axis=2, keepdims=True)
        if self._known is not None:
            return T, self._known.copy()
        rc = np.asarray(belief[1]).reshape(self.S, self.A, 2)
        return T, rc[..., 0] / rc.sum(axis=2)

    def _q(self, s: int, a: int, belief: Belief, h: int) -> float:
        base = (s * self.A + a) * self.S
        counts = belief[0][base : base + self.S]
        total = sum(counts)
        branches: tuple[tuple[float, float, int | None], ...]
        if self._known is not None:
            branches = ((float(self._known[s, a]), 1.0, None),)
        else:
            al, be = belief[1][(s * self.A + a) * 2], belief[1][(s * self.A + a) * 2 + 1]
            p1 = al / (al + be)
            branches = ((1.0, float(p1), 1), (0.0, 1.0 - float(p1), 0))
        if h == 1:
            return float(sum(r * pr for r, pr, _ in branches))
        q = 0.0
        for s2 in range(self.S):
            p_s2 = counts[s2] / total
            for r, pr, obs in branches:
                nb = self.update(belief, s, a, s2, obs)
                q += p_s2 * pr * (r + self.gamma * self._v(s2, nb, h - 1))
        return q

    def _v(self, s: int, belief: Belief, h: int) -> float:
        if h == 0:
            return 0.0
        key = (s, belief, h)
        hit = self._cache.get(key)
        if hit is None:
            hit = max(self._q(s, a, belief, h) for a in range(self.A))
            self._cache[key] = hit
        return hit

    def q_values(self, s: int, belief: Belief, horizon: int) -> NDArray[np.float64]:
        """Bayes-optimal action values in physical state ``s`` with ``horizon`` steps remaining."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        return np.array([self._q(s, a, belief, horizon) for a in range(self.A)])

    def value(self, s: int, belief: Belief, horizon: int) -> float:
        return self._v(s, belief, horizon)
