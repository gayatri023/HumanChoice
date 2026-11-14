"""POMDP belief updates, planning and alpha-vector solutions."""
import numpy as np
import pytest

from humanchoice.pomdp import POMDP, prune_vectors, tiger_pomdp

B0 = np.array([0.5, 0.5])


def test_tiger_belief_update_by_hand():
    p = tiger_pomdp()
    b = p.belief_update(B0, 0, 0)  # listen, growl-left from a uniform belief
    assert np.allclose(b, [0.85, 0.15])
    b2 = p.belief_update(b, 0, 0)  # second growl-left
    assert np.allclose(b2, [0.85**2 / (0.85**2 + 0.15**2), 0.15**2 / (0.85**2 + 0.15**2)])
    # a conflicting growl cancels the evidence
    assert np.allclose(p.belief_update(b, 0, 1), B0)


def test_opening_a_door_resets_belief():
    p = tiger_pomdp()
    assert np.allclose(p.belief_update(np.array([0.99, 0.01]), 1, 0), B0)


def test_law_of_total_probability_over_observations():
    """sum_z P(z|b,a) * b'(.|b,a,z) = predicted belief b T_a (belief is a martingale under the filter)."""
    rng = np.random.default_rng(0)
    T = rng.random((2, 3, 3))
    T /= T.sum(axis=2, keepdims=True)
    O = rng.random((2, 3, 4))
    O /= O.sum(axis=2, keepdims=True)
    p = POMDP(T, O, rng.normal(size=(3, 2)), gamma=0.9)
    b = np.array([0.2, 0.5, 0.3])
    for a in range(2):
        pz = p.observation_probs(b, a)
        assert np.isclose(pz.sum(), 1.0)
        mix = sum(pz[z] * p.belief_update(b, a, z) for z in range(4))
        assert np.allclose(mix, p.predict(b, a))


def test_tiger_known_finite_horizon_values():
    p = tiger_pomdp(gamma=1.0)
    assert np.isclose(p.value(B0, 1), -1.0)  # listen
    assert np.isclose(p.value(B0, 2), -2.0)  # listen twice; opening costs -45
    # horizon 3 by hand: listen, then V_2(0.85) with the best follow-up policy
    b2 = 0.85**2 / (0.85**2 + 0.15**2)
    v1_after_two = 10 * b2 - 100 * (1 - b2)  # open the door away from the likely tiger
    p_same = 0.85**2 + 0.15**2
    v2 = -1 + p_same * v1_after_two + (1 - p_same) * (-1.0)
    assert np.isclose(p.value(B0, 3), -1 + v2)  # = 2.72


def test_immediate_reward_values():
    p = tiger_pomdp()
    assert np.isclose(p.expected_reward(B0, 1), 0.5 * -100 + 0.5 * 10)
    assert np.isclose(p.expected_reward(np.array([0.9, 0.1]), 2), 0.9 * 10 + 0.1 * -100)


@pytest.mark.parametrize("gamma", [1.0, 0.95])
def test_alpha_vector_solution_equals_belief_tree_expectimax(gamma):
    """Two independent exact solvers must agree everywhere on the belief simplex."""
    p = tiger_pomdp(gamma=gamma)
    sets = p.solve_finite_horizon(5)
    for h in range(1, 6):
        for x in np.linspace(0, 1, 21):
            b = np.array([x, 1 - x])
            assert np.isclose(sets[h].value(b), p.value(b, h), atol=1e-9)


def test_value_function_is_convex_and_alpha_action_is_optimal():
    p = tiger_pomdp(gamma=0.95)
    S = p.solve_finite_horizon(4)[4]
    xs = np.linspace(0, 1, 101)
    v = np.array([S.value(np.array([x, 1 - x])) for x in xs])
    assert np.all(v[:-2] - 2 * v[1:-1] + v[2:] >= -1e-9)  # discrete convexity
    for x in (0.1, 0.5, 0.9):
        b = np.array([x, 1 - x])
        assert np.argmax(p.q_values(b, 4)) == S.action(b)


def test_tiger_policy_listens_when_uncertain_and_opens_when_confident():
    S = tiger_pomdp(gamma=0.95).solve_finite_horizon(5)[5]
    assert S.action(B0) == 0  # listen
    assert S.action(np.array([0.995, 0.005])) == 2  # tiger almost surely left -> open right
    assert S.action(np.array([0.005, 0.995])) == 1


def test_full_observability_upper_bounds_the_pomdp_value():
    """V_h^POMDP(b) <= sum_s b(s) V_h^MDP(s): knowing the state can only help (the idea behind QMDP)."""
    from humanchoice.mdp import TabularMDP, backward_induction

    p = tiger_pomdp(gamma=0.95)
    mdp = TabularMDP(np.transpose(p.T, (1, 0, 2)), p.R, gamma=0.95)
    V_mdp, _ = backward_induction(mdp, 6)
    sets = p.solve_finite_horizon(6)
    for h in range(1, 7):
        for x in (0.5, 0.7, 0.9, 0.99):
            b = np.array([x, 1 - x])
            assert sets[h].value(b) <= b @ V_mdp[h] + 1e-9


def test_qmdp_equals_mdp_values_when_state_is_known():
    p = tiger_pomdp(gamma=0.9)
    known = np.array([1.0, 0.0])
    q = p.qmdp_q_values(known)
    assert np.argmax(q) == 2  # tiger left, so open the right door


def test_prune_removes_dominated_vectors_but_keeps_envelope():
    V = np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4], [0.3, 0.3], [0.6, 0.6]])
    kept = prune_vectors(V)
    rows = {tuple(r) for r in kept.tolist()}
    assert (1.0, 0.0) in rows and (0.0, 1.0) in rows
    assert (0.3, 0.3) not in rows and (0.4, 0.4) not in rows  # pointwise dominated by (0.6, 0.6)
    assert (0.6, 0.6) in rows  # needed: beats both corner vectors near b = 0.5
    # envelope value unchanged on a grid
    for x in np.linspace(0, 1, 11):
        b = np.array([x, 1 - x])
        assert np.isclose((V @ b).max(), (kept @ b).max())


def test_prune_removes_vector_dominated_only_by_a_combination():
    # at every belief max(b, 1-b) >= 0.5 > 0.45, so (0.45, 0.45) is never optimal although no single vector dominates it
    V = np.array([[1.0, 0.0], [0.0, 1.0], [0.45, 0.45]])
    kept = prune_vectors(V)
    assert len(kept) == 2


def test_invalid_pomdp_rejected():
    T = np.full((1, 2, 2), 0.5)
    O = np.full((1, 2, 2), 0.5)
    with pytest.raises(ValueError):
        POMDP(T * 0.9, O, np.zeros((2, 1)))
    with pytest.raises(ValueError):
        POMDP(T, O, np.zeros((3, 1)))
    with pytest.raises(ValueError):
        tiger_pomdp().belief_update(B0, 0, 5)  # unknown observation index
