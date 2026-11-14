"""Tabular Markov decision processes and classical dynamic-programming solvers.

Conventions
-----------
* ``T[s, a, s']``  transition probabilities
* ``R[s, a]``      expected immediate reward
* Bellman optimality:  ``Q(s,a) = R(s,a) + gamma * sum_s' T(s,a,s') max_a' Q(s',a')``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from humanchoice.utils import check_stochastic


@dataclass(frozen=True)
class TabularMDP:
    """A finite MDP ``(S, A, T, R, gamma)``."""

    T: NDArray[np.float64]
    R: NDArray[np.float64]
    gamma: float = 0.95

    def __post_init__(self) -> None:
        T = np.asarray(self.T, dtype=float)
        R = np.asarray(self.R, dtype=float)
        if T.ndim != 3 or T.shape[0] != T.shape[2]:
            raise ValueError("T must have shape (S, A, S)")
        if R.shape != T.shape[:2]:
            raise ValueError(f"R must have shape (S, A) = {T.shape[:2]}, got {R.shape}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        check_stochastic(T, "T")
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "R", R)

    @property
    def n_states(self) -> int:
        return self.T.shape[0]

    @property
    def n_actions(self) -> int:
        return self.T.shape[1]


def bellman_q(mdp: TabularMDP, V: NDArray[np.float64]) -> NDArray[np.float64]:
    """One-step lookahead ``Q(s,a) = R(s,a) + gamma * E[V(s')]``."""
    return mdp.R + mdp.gamma * mdp.T @ V


def value_iteration(
    mdp: TabularMDP, tol: float = 1e-10, max_iter: int = 100_000
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Solve the discounted MDP by value iteration. Returns ``(V, Q, n_iter)``."""
    if mdp.gamma >= 1.0:
        raise ValueError("value_iteration requires gamma < 1; use backward_induction for finite horizons")
    V = np.zeros(mdp.n_states)
    for i in range(1, max_iter + 1):
        Q = bellman_q(mdp, V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            return V, bellman_q(mdp, V), i
        V = V_new
    raise RuntimeError("value iteration did not converge")


def policy_evaluation(mdp: TabularMDP, policy: NDArray) -> NDArray[np.float64]:
    """Exact evaluation of a (deterministic or stochastic) policy by solving a linear system.

    ``policy`` may have shape ``(S,)`` (deterministic action indices) or ``(S, A)``.
    """
    pi = np.asarray(policy)
    if pi.ndim == 1:
        onehot = np.zeros((mdp.n_states, mdp.n_actions))
        onehot[np.arange(mdp.n_states), pi.astype(int)] = 1.0
        pi = onehot
    check_stochastic(pi, "policy")
    P = np.einsum("sa,sat->st", pi, mdp.T)
    r = np.einsum("sa,sa->s", pi, mdp.R)
    return np.linalg.solve(np.eye(mdp.n_states) - mdp.gamma * P, r)


def policy_iteration(mdp: TabularMDP, max_iter: int = 1000) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Howard's policy iteration. Returns ``(policy, V)``."""
    pi = np.zeros(mdp.n_states, dtype=int)
    for _ in range(max_iter):
        V = policy_evaluation(mdp, pi)
        new_pi = bellman_q(mdp, V).argmax(axis=1)
        if np.array_equal(new_pi, pi):
            return pi, V
        pi = new_pi
    raise RuntimeError("policy iteration did not converge")


def backward_induction(
    mdp: TabularMDP, horizon: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Finite-horizon optimal values.

    Returns ``V`` with shape ``(horizon + 1, S)`` and ``Q`` with shape ``(horizon, S, A)`` where
    index ``h`` means *h steps remaining* (``V[0] = 0``; ``Q[h-1]`` uses ``V[h-1]``).
    """
    V = np.zeros((horizon + 1, mdp.n_states))
    Q = np.zeros((horizon, mdp.n_states, mdp.n_actions))
    for h in range(1, horizon + 1):
        Q[h - 1] = bellman_q(mdp, V[h - 1])
        V[h] = Q[h - 1].max(axis=1)
    return V, Q
