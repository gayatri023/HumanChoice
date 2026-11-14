import numpy as np
import pytest

from humanchoice.envs import BernoulliBandit, ReversalBandit
from humanchoice.mdp import (
    TabularMDP,
    value_iteration,
)
from humanchoice.utils import as_rng, check_stochastic, log_softmax, softmax


def test_softmax_is_distribution_and_stable():
    p = softmax([1000.0, 1001.0, 999.0], beta=1.0)
    assert np.isclose(p.sum(), 1.0) and np.all(np.isfinite(p))
    assert p.argmax() == 1


def test_softmax_beta_limits():
    q = np.array([0.2, 0.8])
    assert np.allclose(softmax(q, beta=0.0), [0.5, 0.5])
    assert softmax(q, beta=200.0)[1] > 0.999


def test_softmax_matches_formula_and_logsoftmax():
    q, beta = np.array([0.3, 0.9, -0.1]), 2.5
    expected = np.exp(beta * q) / np.exp(beta * q).sum()
    assert np.allclose(softmax(q, beta), expected)
    assert np.allclose(np.exp(log_softmax(q, beta)), expected)


def test_check_stochastic_rejects_bad_input():
    with pytest.raises(ValueError):
        check_stochastic([0.5, 0.6])
    with pytest.raises(ValueError):
        check_stochastic([-0.1, 1.1])
    with pytest.raises(ValueError):
        check_stochastic([np.nan, 1.0])


def test_as_rng_passthrough_and_seeding():
    g = np.random.default_rng(1)
    assert as_rng(g) is g
    assert as_rng(5).random() == as_rng(5).random()


def test_mdp_validation():
    T = np.full((2, 2, 2), 0.5)
    with pytest.raises(ValueError):
        TabularMDP(T, np.zeros((2, 3)))
    with pytest.raises(ValueError):
        TabularMDP(T * 0.9, np.zeros((2, 2)))
    with pytest.raises(ValueError):
        TabularMDP(T, np.zeros((2, 2)), gamma=1.5)


def test_value_iteration_requires_discounting():
    mdp = TabularMDP(np.full((1, 1, 1), 1.0), np.ones((1, 1)), gamma=1.0)
    with pytest.raises(ValueError):
        value_iteration(mdp)


def test_bernoulli_bandit_validation_and_rates():
    with pytest.raises(ValueError):
        BernoulliBandit((1.2, 0.1))
    env, rng = BernoulliBandit((0.9, 0.1)), np.random.default_rng(0)
    env.reset(rng)
    r = np.mean([env.step(0, rng) for _ in range(5000)])
    assert abs(r - 0.9) < 0.02


def test_reversal_bandit_reversal_rate_and_reward_rates():
    env, rng = ReversalBandit(p_good=0.8, hazard=0.1), np.random.default_rng(0)
    env.reset(rng)
    for _ in range(20000):
        env.step(0, rng)
    h = np.array(env.good_arm_history)
    assert abs(np.mean(h[1:] != h[:-1]) - 0.1) < 0.01
    # reward rate when always choosing arm 0 is 0.5 on average by symmetry
    env.reset(rng)
    assert abs(np.mean([env.step(0, rng) for _ in range(20000)]) - 0.5) < 0.02


def test_reversal_bandit_zero_hazard_is_stationary():
    env, rng = ReversalBandit(p_good=0.9, hazard=0.0), np.random.default_rng(3)
    env.reset(rng)
    for _ in range(200):
        env.step(0, rng)
    assert len(set(env.good_arm_history)) == 1
