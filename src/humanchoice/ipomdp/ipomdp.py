"""Level-1 interactive POMDP (I-POMDP) with a finite set of level-0 models of the other agent.

Background
----------
In an I-POMDP (Gmytrasiewicz & Doshi, 2005) agent *i* holds a belief over the **interactive state**
``is = (s, m_j)``: the physical state ``s`` *and* a model ``m_j`` of the other agent *j* (here a
level-0 POMDP agent with its own belief ``b_j``). Agent *i* therefore reasons about what *j* will do,
what *j* believes, and how *j*'s beliefs change as *j* acts and observes.

Construction used here
----------------------
1. **Model space.** The closure of *j*'s belief states reachable under *j*'s level-0 POMDP (breadth-first
   from ``b0_j``, up to ``depth`` steps) gives a finite model set ``M``. Beliefs outside the closure are
   snapped to the nearest model (L1 distance).
2. **Other agent's policy.** For each model ``m`` the level-0 agent's action distribution is
   ``P(a_j | m) = (1 - eps) * Uniform(argmax_a Q0(m, a)) + eps * Uniform(A_j)`` with ``Q0`` computed by
   belief-tree planning to ``horizon0``.
3. **Interactive transition/observation kernel.**  With ``SE`` the level-0 state-estimation function
   (*j*'s own Bayes filter, using *j*'s own model of the world)::

       W[a_i, (s,m), (s',m'), z_i] = sum_{a_j} P(a_j|m) T(s'|s,a_i,a_j)
                                      sum_{z_j} O_j(z_j|s',a_i,a_j) O_i(z_i|s',a_i,a_j) 1[m' = SE(m,a_j,z_j)]

   (observations of the two agents are conditionally independent given ``(s', a_i, a_j)``).
4. **Reward** ``R[(s,m), a_i] = sum_{a_j} P(a_j|m) R_i(s, a_i, a_j)``.

Because ``M`` is finite, the resulting object is an ordinary POMDP over interactive states; it is solved
with the machinery in :mod:`humanchoice.pomdp`. This is an *approximation* of the full I-POMDP only through
the finite model set and the level-0 assumptions above.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from humanchoice.pomdp.pomdp import POMDP


def belief_closure(pomdp: POMDP, b0: NDArray, depth: int, decimals: int = 8) -> NDArray[np.float64]:
    """Breadth-first set of beliefs reachable from ``b0`` within ``depth`` action-observation steps."""
    b0 = np.asarray(b0, dtype=float)
    models = [b0]
    seen = {tuple(np.round(b0, decimals))}
    frontier = [b0]
    for _ in range(depth):
        nxt = []
        for b in frontier:
            for a in range(pomdp.n_actions):
                pz = pomdp.observation_probs(b, a)
                for z in range(pomdp.n_obs):
                    if pz[z] > 1e-12:
                        nb = pomdp.belief_update(b, a, z)
                        key = tuple(np.round(nb, decimals))
                        if key not in seen:
                            seen.add(key)
                            models.append(nb)
                            nxt.append(nb)
        frontier = nxt
    return np.array(models)


class Level1IPOMDP:
    """Agent *i*'s level-1 I-POMDP over ``(physical state, model of j)``.

    Parameters
    ----------
    T:   ``(S, A_i, A_j, S)`` joint transition probabilities.
    O_i: ``(A_i, A_j, S, Z_i)`` agent *i*'s observation model.
    R_i: ``(S, A_i, A_j)`` agent *i*'s reward.
    O_j: ``(A_i, A_j, S, Z_j)`` the other agent's (true) observation model.
    level0: single-agent POMDP used as *i*'s model of *j* (states ``S``, actions ``A_j``, obs ``Z_j``).
    b0_j: initial belief of *j* (default uniform).
    epsilon_j: probability that *j* acts uniformly at random.
    horizon0: planning horizon attributed to *j*.
    depth: depth of the breadth-first model closure.
    j_policy: optional callable ``belief_j -> action probabilities`` overriding the planned policy.
    """

    def __init__(
        self,
        T: NDArray,
        O_i: NDArray,
        R_i: NDArray,
        O_j: NDArray,
        level0: POMDP,
        b0_j: NDArray | None = None,
        gamma: float = 0.95,
        epsilon_j: float = 0.05,
        horizon0: int = 3,
        depth: int = 6,
        j_policy=None,
    ) -> None:
        T, O_i, R_i, O_j = (np.asarray(x, dtype=float) for x in (T, O_i, R_i, O_j))
        S, Ai, Aj, _ = T.shape
        Zi, Zj = O_i.shape[-1], O_j.shape[-1]
        if (level0.n_states, level0.n_actions, level0.n_obs) != (S, Aj, Zj):
            raise ValueError("level0 POMDP must have (S, A_j, Z_j) states/actions/observations")
        if not 0.0 <= epsilon_j <= 1.0:
            raise ValueError("epsilon_j must lie in [0, 1]")
        self.S, self.Ai, self.Aj, self.Zi, self.Zj = S, Ai, Aj, Zi, Zj
        b0_j = np.full(S, 1.0 / S) if b0_j is None else np.asarray(b0_j, dtype=float)
        self.level0 = level0
        self.models = belief_closure(level0, b0_j, depth)
        M = self.M = len(self.models)

        # other agent's action distribution for every model
        P = np.zeros((M, Aj))
        for m, b in enumerate(self.models):
            if j_policy is not None:
                P[m] = np.asarray(j_policy(b), dtype=float)
            else:
                q = level0.q_values(b, horizon0)
                greedy = np.isclose(q, np.max(q), atol=1e-9).astype(float)
                P[m] = (1 - epsilon_j) * greedy / greedy.sum() + epsilon_j / Aj
        self.j_action_probs = P

        # state estimation SE[m, a_j, z_j] -> model index
        SE = np.zeros((M, Aj, Zj), dtype=int)
        for m, b in enumerate(self.models):
            for aj in range(Aj):
                pz = level0.observation_probs(b, aj)
                for zj in range(Zj):
                    if pz[zj] > 1e-12:
                        nb = level0.belief_update(b, aj, zj)
                        SE[m, aj, zj] = int(np.argmin(np.abs(self.models - nb).sum(axis=1)))
                    else:
                        SE[m, aj, zj] = m
        self.SE = SE

        # joint interactive kernel and reward
        IS = S * M
        W = np.zeros((Ai, IS, IS, Zi))
        R = np.zeros((IS, Ai))
        for m in range(M):
            for s in range(S):
                i0 = s * M + m
                for ai in range(Ai):
                    for aj in range(Aj):
                        w = P[m, aj]
                        R[i0, ai] += w * R_i[s, ai, aj]
                        for s2 in range(S):
                            pt = w * T[s, ai, aj, s2]
                            if pt == 0.0:
                                continue
                            for zj in range(Zj):
                                i1 = s2 * M + SE[m, aj, zj]
                                W[ai, i0, i1, :] += pt * O_j[ai, aj, s2, zj] * O_i[ai, aj, s2, :]
        self.pomdp = POMDP.from_joint(W, R, gamma)

    # ------------------------------------------------------------------ beliefs
    def initial_belief(
        self, state_belief: NDArray | None = None, model_prior: NDArray | None = None
    ) -> NDArray[np.float64]:
        """Product belief over (state, model). Default: uniform state, ``j`` surely at its initial belief."""
        sb = np.full(self.S, 1.0 / self.S) if state_belief is None else np.asarray(state_belief, dtype=float)
        mp = np.zeros(self.M) if model_prior is None else np.asarray(model_prior, dtype=float)
        if model_prior is None:
            mp[0] = 1.0
        return (sb[:, None] * mp[None, :]).ravel()

    def state_marginal(self, belief: NDArray) -> NDArray[np.float64]:
        return np.asarray(belief).reshape(self.S, self.M).sum(axis=1)

    def model_marginal(self, belief: NDArray) -> NDArray[np.float64]:
        return np.asarray(belief).reshape(self.S, self.M).sum(axis=0)

    def update(self, belief: NDArray, action_i: int, obs_i: int) -> NDArray[np.float64]:
        """Interactive Bayes filter after taking ``action_i`` and observing ``obs_i``."""
        return self.pomdp.belief_update(belief, action_i, obs_i)

    def q_values(self, belief: NDArray, horizon: int) -> NDArray[np.float64]:
        """Optimal action values for agent *i* at an interactive belief."""
        return self.pomdp.q_values(belief, horizon)
