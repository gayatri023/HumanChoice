"""Canonical POMDP benchmark problems."""

from __future__ import annotations

import numpy as np

from humanchoice.pomdp.pomdp import POMDP

TIGER_STATES = ("tiger-left", "tiger-right")
TIGER_ACTIONS = ("listen", "open-left", "open-right")
TIGER_OBS = ("growl-left", "growl-right")


def tiger_pomdp(
    accuracy: float = 0.85,
    listen_cost: float = -1.0,
    reward_correct: float = 10.0,
    penalty_tiger: float = -100.0,
    gamma: float = 0.95,
) -> POMDP:
    """The Tiger problem (Kaelbling, Littman & Cassandra, 1998).

    A tiger hides behind one of two doors. Listening is cheap but noisy; opening a door ends the
    episode's current "round" (the state is reset uniformly at random) and pays off or punishes.
    """
    if not 0.5 <= accuracy <= 1.0:
        raise ValueError("accuracy must lie in [0.5, 1]")
    T = np.zeros((3, 2, 2))
    T[0] = np.eye(2)
    T[1] = T[2] = 0.5
    O = np.zeros((3, 2, 2))
    O[0] = [[accuracy, 1 - accuracy], [1 - accuracy, accuracy]]
    O[1] = O[2] = 0.5
    R = np.zeros((2, 3))
    R[:, 0] = listen_cost
    R[0, 1], R[1, 1] = penalty_tiger, reward_correct  # open-left
    R[0, 2], R[1, 2] = reward_correct, penalty_tiger  # open-right
    return POMDP(T, O, R, gamma)
