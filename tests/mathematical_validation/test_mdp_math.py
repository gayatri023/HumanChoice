"""Dynamic-programming identities for tabular MDPs."""
import numpy as np

from humanchoice.mdp import (
    TabularMDP,
    backward_induction,
    bellman_q,
    policy_evaluation,
    policy_iteration,
    value_iteration,
)


def random_mdp(seed=0, S=5, A=3, gamma=0.9):
    rng = np.random.default_rng(seed)
    T = rng.random((S, A, S))
    T /= T.sum(axis=2, keepdims=True)
    return TabularMDP(T, rng.normal(size=(S, A)), gamma)


def test_closed_form_two_state_chain():
    # state 0 --(only action)--> state 1 (absorbing). R(0)=1, R(1)=2, gamma=0.5
    T = np.array([[[0.0, 1.0]], [[0.0, 1.0]]])
    R = np.array([[1.0], [2.0]])
    V, _, _ = value_iteration(TabularMDP(T, R, gamma=0.5))
    assert np.isclose(V[1], 2.0 / (1 - 0.5))  # 4
    assert np.isclose(V[0], 1.0 + 0.5 * 4.0)  # 3


def test_value_iteration_satisfies_bellman_optimality_equation():
    mdp = random_mdp()
    V, Q, _ = value_iteration(mdp)
    assert np.allclose(V, bellman_q(mdp, V).max(axis=1), atol=1e-8)
    assert np.allclose(Q, bellman_q(mdp, V), atol=1e-8)


def test_policy_iteration_agrees_with_value_iteration():
    for seed in range(5):
        mdp = random_mdp(seed)
        V_vi, Q, _ = value_iteration(mdp)
        pi, V_pi = policy_iteration(mdp)
        assert np.allclose(V_vi, V_pi, atol=1e-7)
        assert np.array_equal(pi, Q.argmax(axis=1))


def test_policy_evaluation_solves_linear_bellman_equation():
    mdp = random_mdp(3)
    pi = np.array([0, 1, 2, 0, 1])
    V = policy_evaluation(mdp, pi)
    Q = bellman_q(mdp, V)
    assert np.allclose(V, Q[np.arange(5), pi], atol=1e-9)


def test_optimal_policy_value_is_never_below_any_other_policy():
    mdp = random_mdp(4)
    V_star, _, _ = value_iteration(mdp)
    rng = np.random.default_rng(0)
    for _ in range(20):
        V = policy_evaluation(mdp, rng.integers(0, 3, 5))
        assert np.all(V <= V_star + 1e-9)


def test_finite_horizon_converges_to_infinite_horizon():
    mdp = random_mdp(1, gamma=0.8)
    V_inf, _, _ = value_iteration(mdp)
    V, _ = backward_induction(mdp, horizon=200)
    assert np.allclose(V[-1], V_inf, atol=1e-8)
    # and values are monotone in the horizon when rewards are non-negative
    mdp_pos = TabularMDP(mdp.T, np.abs(mdp.R), 0.8)
    Vp, _ = backward_induction(mdp_pos, 30)
    assert np.all(np.diff(Vp, axis=0) >= -1e-12)
