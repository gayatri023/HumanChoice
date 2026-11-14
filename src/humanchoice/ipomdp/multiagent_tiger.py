"""A simplified two-agent Tiger game for I-POMDP experiments.

Adapted from the multi-agent tiger problem of Gmytrasiewicz & Doshi (2005). This is a *simplified variant*,
designed so that every quantity can be checked by hand:

* Two agents each choose ``listen`` / ``open-left`` / ``open-right`` simultaneously.
* If **either** agent opens a door the tiger's location is re-drawn uniformly (a "reset"); otherwise it is unchanged.
* Payoffs are independent: each agent receives the single-agent Tiger payoff for its own action
  (listen -1, open the safe door +10, open the tiger door -100). Agents are coupled only through the dynamics
  (resets) and through what they can observe.
* Growl observations: correct with ``growl_accuracy`` if nobody opened a door, otherwise uninformative (0.5).
* Agent *i* additionally hears a *creak* that reveals *j*'s action with ``creak_accuracy`` (CL / CR / silence).
  Agent *j* (the level-0 agent) only observes growls.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from humanchoice.utils import as_rng

LISTEN, OPEN_LEFT, OPEN_RIGHT = 0, 1, 2
CREAK_LEFT, CREAK_RIGHT, SILENCE = 0, 1, 2


def _growl_probs(s_next: int, a_i: int, a_j: int, accuracy: float) -> NDArray[np.float64]:
    if a_i == LISTEN and a_j == LISTEN:
        p = np.array([1 - accuracy, 1 - accuracy])
        p[s_next] = accuracy
        return p
    return np.array([0.5, 0.5])


def _creak_probs(a_j: int, accuracy: float) -> NDArray[np.float64]:
    wrong = (1 - accuracy) / 2
    p = np.full(3, wrong)
    p[{LISTEN: SILENCE, OPEN_LEFT: CREAK_LEFT, OPEN_RIGHT: CREAK_RIGHT}[a_j]] = accuracy
    return p


def _single_agent_reward(s: int, a: int, listen_cost: float, reward: float, penalty: float) -> float:
    if a == LISTEN:
        return listen_cost
    tiger_side = OPEN_LEFT if s == 0 else OPEN_RIGHT
    return penalty if a == tiger_side else reward


def multiagent_tiger(
    growl_accuracy: float = 0.85,
    creak_accuracy: float = 0.9,
    listen_cost: float = -1.0,
    reward_correct: float = 10.0,
    penalty_tiger: float = -100.0,
) -> dict[str, NDArray[np.float64]]:
    """Return the arrays ``T, O_i, O_j, R_i, R_j`` describing the game."""
    S, A = 2, 3
    T = np.zeros((S, A, A, S))
    O_i = np.zeros((A, A, S, 6))  # observation index = growl * 3 + creak
    O_j = np.zeros((A, A, S, 2))
    R_i = np.zeros((S, A, A))
    R_j = np.zeros((S, A, A))
    for s in range(S):
        for ai in range(A):
            for aj in range(A):
                T[s, ai, aj] = np.eye(S)[s] if (ai == LISTEN and aj == LISTEN) else 0.5
                R_i[s, ai, aj] = _single_agent_reward(s, ai, listen_cost, reward_correct, penalty_tiger)
                R_j[s, ai, aj] = _single_agent_reward(s, aj, listen_cost, reward_correct, penalty_tiger)
    for ai in range(A):
        for aj in range(A):
            creak = _creak_probs(aj, creak_accuracy)
            for s2 in range(S):
                g = _growl_probs(s2, ai, aj, growl_accuracy)
                O_j[ai, aj, s2] = g
                O_i[ai, aj, s2] = np.outer(g, creak).ravel()
    return {"T": T, "O_i": O_i, "O_j": O_j, "R_i": R_i, "R_j": R_j}


class MultiAgentTigerEnv:
    """Simulator for the game above (used for I-POMDP vs POMDP experiments)."""

    def __init__(self, seed: int | np.random.Generator | None = None, **kwargs) -> None:
        self.game = multiagent_tiger(**kwargs)
        self.rng = as_rng(seed)
        self.state = int(self.rng.integers(2))

    def reset(self) -> None:
        self.state = int(self.rng.integers(2))

    def step(self, a_i: int, a_j: int) -> tuple[int, int, float, float]:
        """Apply the joint action. Returns ``(obs_i, obs_j, reward_i, reward_j)`` (reward uses the pre-step state)."""
        g = self.game
        r_i, r_j = g["R_i"][self.state, a_i, a_j], g["R_j"][self.state, a_i, a_j]
        s2 = int(self.rng.choice(2, p=g["T"][self.state, a_i, a_j]))
        z_i = int(self.rng.choice(6, p=g["O_i"][a_i, a_j, s2]))
        z_j = int(self.rng.choice(2, p=g["O_j"][a_i, a_j, s2]))
        self.state = s2
        return z_i, z_j, float(r_i), float(r_j)
