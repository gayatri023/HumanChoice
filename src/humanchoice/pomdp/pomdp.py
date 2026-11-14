"""Tabular partially observable Markov decision processes (POMDPs).

Model
-----
A POMDP is specified by hidden states ``S``, actions ``A``, observations ``Z`` and the *joint
kernel* ``W[a, s, s', z] = P(s', z | s, a)``. The usual factored form ``W = T * O`` with

* ``T[a, s, s']``  state transition probabilities,
* ``O[a, s', z]``  observation probabilities **after** the transition (Cassandra convention),
* ``R[s, a]``      expected immediate reward (depends on the state *before* the transition),

is provided by the main constructor; :meth:`POMDP.from_joint` accepts ``W`` directly. The joint form
is what makes interactive POMDPs (see :mod:`humanchoice.ipomdp`) an instance of this class.

Belief update (Bayes filter)::

    b'(s') = sum_s b(s) W[a, s, s', z] / P(z | b, a)

Planning
--------
* :meth:`POMDP.q_values`  exact finite-horizon *belief-tree* expectimax (any belief, any small horizon).
* :meth:`POMDP.solve_finite_horizon`  exact value iteration over *alpha-vectors* with incremental
  pruning (Cassandra, Littman & Zhang 1997): ``V_h(b) = max_alpha alpha . b`` (piecewise linear convex).
* :meth:`POMDP.qmdp_q_values`  the QMDP approximation (Littman et al. 1995), an upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from humanchoice.mdp.tabular import TabularMDP, value_iteration
from humanchoice.utils import check_stochastic


@dataclass(frozen=True)
class AlphaSet:
    """A set of alpha-vectors ``(N, S)`` with the action each vector prescribes."""

    vectors: NDArray[np.float64]
    actions: NDArray[np.int_]

    def value(self, belief: NDArray) -> float:
        return float((self.vectors @ np.asarray(belief)).max())

    def action(self, belief: NDArray) -> int:
        return int(self.actions[int(np.argmax(self.vectors @ np.asarray(belief)))])

    def __len__(self) -> int:
        return len(self.vectors)


class POMDP:
    """Finite POMDP with exact Bayes filtering and planning."""

    def __init__(self, T: NDArray, O: NDArray, R: NDArray, gamma: float = 0.95) -> None:
        T = np.asarray(T, dtype=float)
        O = np.asarray(O, dtype=float)
        if T.ndim != 3 or T.shape[1] != T.shape[2]:
            raise ValueError("T must have shape (A, S, S)")
        if O.ndim != 3 or O.shape[:2] != (T.shape[0], T.shape[1]):
            raise ValueError("O must have shape (A, S, Z)")
        check_stochastic(T, "T")
        check_stochastic(O, "O")
        W = T[:, :, :, None] * O[:, None, :, :]
        self._init(W, np.asarray(R, dtype=float), gamma)

    @classmethod
    def from_joint(cls, W: NDArray, R: NDArray, gamma: float = 0.95) -> POMDP:
        """Construct from the joint kernel ``W[a, s, s', z]`` (observation may depend on ``s``)."""
        obj = cls.__new__(cls)
        obj._init(np.asarray(W, dtype=float), np.asarray(R, dtype=float), gamma)
        return obj

    def _init(self, W: NDArray, R: NDArray, gamma: float) -> None:
        if W.ndim != 4 or W.shape[1] != W.shape[2]:
            raise ValueError("W must have shape (A, S, S, Z)")
        A, S, _, Z = W.shape
        if R.shape != (S, A):
            raise ValueError(f"R must have shape (S, A) = {(S, A)}, got {R.shape}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if np.any(W < -1e-12) or not np.allclose(W.sum(axis=(2, 3)), 1.0, atol=1e-8):
            raise ValueError("joint kernel W must be non-negative and sum to 1 over (s', z)")
        self.W, self.R, self.gamma = W, R, float(gamma)
        self.n_actions, self.n_states, self.n_obs = A, S, Z
        self.T = W.sum(axis=3)  # marginal transitions (A, S, S)

    # ----------------------------------------------------------------- filtering
    def predict(self, belief: NDArray, action: int) -> NDArray[np.float64]:
        """Belief over the next state before seeing the observation."""
        return np.asarray(belief) @ self.T[action]

    def observation_probs(self, belief: NDArray, action: int) -> NDArray[np.float64]:
        """``P(z | b, a)`` for every observation ``z``."""
        return np.einsum("s,sxz->z", np.asarray(belief), self.W[action])

    def belief_update(self, belief: NDArray, action: int, obs: int) -> NDArray[np.float64]:
        """Exact Bayes filter. Raises ``ValueError`` for invalid indices or zero-probability observations."""
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} out of range [0, {self.n_actions})")
        if not 0 <= obs < self.n_obs:
            raise ValueError(f"observation {obs} out of range [0, {self.n_obs})")
        nb = np.asarray(belief) @ self.W[action][:, :, obs]
        p = nb.sum()
        if p <= 0.0:
            raise ValueError("observation has zero probability under the current belief")
        return nb / p

    # ------------------------------------------------------------------ planning
    def expected_reward(self, belief: NDArray, action: int) -> float:
        return float(np.asarray(belief) @ self.R[:, action])

    def q_values(self, belief: NDArray, horizon: int) -> NDArray[np.float64]:
        """Optimal action values at ``belief`` with ``horizon`` steps remaining (belief-tree expectimax).

        ``horizon = 1`` is the myopic expected immediate reward. Cost grows as ``(|A||Z|)^horizon``.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        b = np.asarray(belief, dtype=float)
        q = b @ self.R
        if horizon == 1:
            return q
        nb = np.einsum("s,asxz->azx", b, self.W)  # unnormalised next beliefs
        pz = nb.sum(axis=2)
        for a in range(self.n_actions):
            future = 0.0
            for z in range(self.n_obs):
                if pz[a, z] > 1e-14:
                    future += pz[a, z] * self.value(nb[a, z] / pz[a, z], horizon - 1)
            q[a] += self.gamma * future
        return q

    def value(self, belief: NDArray, horizon: int) -> float:
        if horizon <= 0:
            return 0.0
        return float(np.max(self.q_values(belief, horizon)))

    def qmdp_q_values(self, belief: NDArray) -> NDArray[np.float64]:
        """QMDP approximation: solve the underlying fully observable MDP, then average over the belief."""
        mdp = TabularMDP(T=np.transpose(self.T, (1, 0, 2)), R=self.R, gamma=self.gamma)
        _, Q, _ = value_iteration(mdp)
        return np.asarray(belief) @ Q

    def solve_finite_horizon(self, horizon: int, prune: bool = True) -> list[AlphaSet]:
        """Exact alpha-vector value iteration. ``result[h]`` is the value function with ``h`` steps left."""
        sets = [AlphaSet(np.zeros((1, self.n_states)), np.array([-1]))]
        for _ in range(horizon):
            sets.append(self._backup(sets[-1], prune))
        return sets

    def _backup(self, prev: AlphaSet, prune: bool) -> AlphaSet:
        vecs, acts = [], []
        G = prev.vectors
        for a in range(self.n_actions):
            acc = self.R[:, a][None, :]
            for z in range(self.n_obs):
                per_z = self.gamma * G @ self.W[a][:, :, z].T  # (N, S): row_i = gamma * W[a,:,:,z] @ alpha_i
                acc = (acc[:, None, :] + per_z[None, :, :]).reshape(-1, self.n_states)
                if prune:
                    acc = prune_vectors(acc)
            vecs.append(acc)
            acts.append(np.full(len(acc), a))
        V, A = np.vstack(vecs), np.concatenate(acts)
        if prune:
            V, A = prune_vectors(V, A)
        return AlphaSet(V, A)


# ---------------------------------------------------------------------- pruning
def _is_useful(alpha: NDArray, others: NDArray, tol: float = 1e-9) -> bool:
    """LP test: is there a belief where ``alpha`` strictly beats every vector in ``others``?"""
    S = len(alpha)
    diffs = alpha[None, :] - others  # (M, S)
    c = np.zeros(S + 1)
    c[-1] = -1.0  # maximise delta
    A_ub = np.hstack([-diffs, np.ones((len(others), 1))])
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=np.zeros(len(others)),
        A_eq=np.hstack([np.ones((1, S)), np.zeros((1, 1))]),
        b_eq=[1.0],
        bounds=[(0, 1)] * S + [(None, None)],
        method="highs",
    )
    return bool(res.status == 0 and -res.fun > tol)


@overload
def prune_vectors(vectors: NDArray, actions: None = None) -> NDArray: ...
@overload
def prune_vectors(vectors: NDArray, actions: NDArray) -> tuple[NDArray, NDArray]: ...
def prune_vectors(vectors: NDArray, actions: NDArray | None = None) -> NDArray | tuple[NDArray, NDArray]:
    """Remove duplicate, pointwise-dominated and LP-dominated alpha-vectors (Lark's algorithm)."""
    V = np.asarray(vectors, dtype=float)
    A = None if actions is None else np.asarray(actions)
    # 1. exact duplicates
    _, first = np.unique(np.round(V, 10), axis=0, return_index=True)
    first.sort()
    V = V[first]
    A = None if A is None else A[first]
    # 2. pointwise dominance
    ge = np.all(V[:, None, :] >= V[None, :, :] - 1e-12, axis=2)  # ge[i, j]: V_i >= V_j everywhere
    np.fill_diagonal(ge, False)
    keep = ~ge.any(axis=0)
    V = V[keep]
    A = None if A is None else A[keep]
    # 3. LP dominance
    alive = list(range(len(V)))
    if len(V) > 1:
        for i in range(len(V)):
            others = [j for j in alive if j != i]
            if others and not _is_useful(V[i], V[others]):
                alive.remove(i)
    V = V[alive]
    return V if A is None else (V, A[alive])
