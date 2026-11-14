"""Bayes-adaptive planning: hand calculations, conjugacy, optimality and reduction to the MDP."""
import numpy as np
import pytest

from humanchoice.bamdp import BAMDP, BetaBernoulliBanditBAMDP
from humanchoice.mdp import TabularMDP, backward_induction


def test_horizon_one_is_posterior_mean():
    b = BetaBernoulliBanditBAMDP(2, prior=(1, 1))
    post = (3.0, 2.0, 1.0, 4.0)
    assert np.allclose(b.q_values(post, 1), [3 / 5, 1 / 5])


def test_horizon_two_hand_calculation():
    """Q_2(arm 1) from Beta(1,1) priors: 0.5*(1 + 2/3) + 0.5*(1/2) = 13/12."""
    b = BetaBernoulliBanditBAMDP(2, prior=(1, 1))
    q = b.q_values(b.initial_posterior(), 2)
    assert np.allclose(q, [13 / 12, 13 / 12])
    assert np.isclose(b.value(b.initial_posterior(), 2), 13 / 12)


def test_value_of_information_makes_uncertain_arm_attractive():
    b = BetaBernoulliBanditBAMDP(2, prior=(1, 1))
    post = (1.0, 1.0, 1000.0, 1000.0)  # arm 1 is (almost) known to pay 0.5; arm 0 is unknown with mean 0.5
    q = b.q_values(post, 2)
    assert q[0] > q[1]
    assert np.isclose(q[0], 13 / 12) and np.isclose(q[1], 1.0, atol=1e-3)
    # with a single trial left there is nothing to learn for, so no bonus
    assert np.isclose(*b.q_values(post, 1), atol=1e-3)


def test_bayes_optimal_value_dominates_myopic_value():
    b = BetaBernoulliBanditBAMDP(2, prior=(1, 1))
    post = (2.0, 3.0, 4.0, 2.0)
    for h in range(1, 8):
        assert b.value(post, h) >= h * max(b.posterior_mean(post)) - 1e-12


def test_conjugate_update_equals_batch_posterior():
    b = BetaBernoulliBanditBAMDP(2, prior=(2.0, 3.0))
    post = b.initial_posterior()
    seq = [(0, 1), (0, 0), (1, 1), (0, 1), (1, 0), (1, 0)]
    for arm, r in seq:
        post = b.update(post, arm, r)
    s0, f0 = sum(r for a, r in seq if a == 0), sum(1 - r for a, r in seq if a == 0)
    s1, f1 = sum(r for a, r in seq if a == 1), sum(1 - r for a, r in seq if a == 1)
    assert post == (2.0 + s0, 3.0 + f0, 2.0 + s1, 3.0 + f1)


def test_discounting_lowers_value():
    post = (1.0, 1.0, 1.0, 1.0)
    v1 = BetaBernoulliBanditBAMDP(2, gamma=1.0).value(post, 6)
    v2 = BetaBernoulliBanditBAMDP(2, gamma=0.8).value(post, 6)
    assert v2 < v1


def test_generic_bamdp_matches_specialised_bandit_planner():
    spec = BetaBernoulliBanditBAMDP(2, prior=(1.5, 1.0))
    gen = BAMDP(
        prior_transitions=np.ones((1, 2, 1)),
        prior_rewards=np.tile([1.5, 1.0], (1, 2, 1)),
    )
    belief = gen.initial_belief()
    post = spec.initial_posterior()
    for arm, r in [(0, 1), (1, 0), (0, 0), (1, 1)]:
        belief = gen.update(belief, 0, arm, 0, r)
        post = spec.update(post, arm, r)
        for h in (1, 2, 3, 4):
            assert np.allclose(gen.q_values(0, belief, h), spec.q_values(post, h))


def test_bamdp_collapses_to_mdp_as_prior_strength_grows():
    """With huge pseudo-counts (a known model) Bayes-adaptive values equal ordinary finite-horizon values."""
    rng = np.random.default_rng(0)
    T = rng.random((3, 2, 3))
    T /= T.sum(axis=2, keepdims=True)
    R = rng.random((3, 2))
    strength = 1e9
    bamdp = BAMDP(prior_transitions=T * strength, known_rewards=R, gamma=1.0)
    _, Q = backward_induction(TabularMDP(T, R, gamma=1.0), horizon=4)
    for s in range(3):
        assert np.allclose(bamdp.q_values(s, bamdp.initial_belief(), 4), Q[3, s], atol=1e-5)


def test_bamdp_learns_unknown_transitions():
    """After observing a deterministic transition many times the posterior mean model concentrates on it."""
    bamdp = BAMDP(prior_transitions=np.ones((2, 1, 2)), known_rewards=np.zeros((2, 1)))
    belief = bamdp.initial_belief()
    for _ in range(50):
        belief = bamdp.update(belief, 0, 0, 1)
    T_hat, _ = bamdp.expected_model(belief)
    assert T_hat[0, 0, 1] > 0.95 and np.isclose(T_hat[1, 0].sum(), 1.0)


def test_input_validation():
    with pytest.raises(ValueError):
        BetaBernoulliBanditBAMDP(2, prior=(0.0, 1.0))
    with pytest.raises(ValueError):
        BAMDP(np.ones((1, 2, 1)))  # neither reward specification
    with pytest.raises(ValueError):
        BetaBernoulliBanditBAMDP.update((1.0, 1.0, 1.0, 1.0), 0, 2)


@pytest.mark.slow
def test_bayes_optimal_policy_achieves_dp_value_and_beats_others_monte_carlo():
    """E[total reward] of the Bayes-optimal policy under the prior equals V_H(prior); any other policy is <= V_H."""
    H, N = 8, 6000
    planner = BetaBernoulliBanditBAMDP(2, prior=(1, 1))
    v_star = planner.value(planner.initial_posterior(), H)
    rng = np.random.default_rng(123)

    def run(policy):
        totals = np.empty(N)
        for i in range(N):
            p = rng.beta(1, 1, size=2)  # world drawn from the prior
            post, total = planner.initial_posterior(), 0
            for t in range(H):
                a = policy(post, H - t)
                r = int(rng.random() < p[a])
                total += r
                post = planner.update(post, a, r)
            totals[i] = total
        return totals.mean(), totals.std(ddof=1) / np.sqrt(N)

    m_opt, se_opt = run(lambda post, h: int(np.argmax(planner.q_values(post, h))))
    m_greedy, se_g = run(lambda post, h: int(np.argmax(planner.posterior_mean(post))))
    assert abs(m_opt - v_star) < 4 * se_opt  # DP value matches simulation
    assert m_greedy < v_star + 4 * se_g  # nothing beats the Bayes-optimal value
