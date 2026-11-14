import numpy as np
import pytest

from humanchoice.envs import ReversalBandit
from humanchoice.models import (
    AsymmetricQLearning,
    BayesBanditModel,
    HybridModel,
    POMDPReversalModel,
    QLearning,
)

ALL = [
    (QLearning(), [0.3, 2.0]),
    (AsymmetricQLearning(), [0.4, 0.2, 2.0]),
    (BayesBanditModel(), [3.0]),
    (POMDPReversalModel(), [4.0, 0.1]),
    (HybridModel(), [0.3, 0.1, 0.5, 3.0]),
]


def test_q_learning_update_rule_by_hand():
    m, p = QLearning(q0=0.5), np.array([0.2, 1.0])
    q = m.init_state(p)
    q = m.update(p, q, 0, 1.0)  # 0.5 + 0.2 * (1 - 0.5) = 0.6
    assert np.allclose(q, [0.6, 0.5])
    q = m.update(p, q, 1, 0.0)  # 0.5 + 0.2 * (0 - 0.5) = 0.4
    assert np.allclose(q, [0.6, 0.4])


def test_update_does_not_mutate_previous_state():
    m, p = QLearning(), np.array([0.5, 1.0])
    q0 = m.init_state(p)
    m.update(p, q0, 0, 1.0)
    assert np.allclose(q0, [0.5, 0.5])


def test_asymmetric_uses_correct_rate_by_sign():
    m, p = AsymmetricQLearning(q0=0.5), np.array([0.8, 0.1, 1.0])
    assert np.allclose(m.update(p, m.init_state(p), 0, 1.0), [0.5 + 0.8 * 0.5, 0.5])  # positive PE
    assert np.allclose(m.update(p, m.init_state(p), 0, 0.0), [0.5 - 0.1 * 0.5, 0.5])  # negative PE


def test_asymmetric_nests_q_learning():
    rng = np.random.default_rng(0)
    c, r = rng.integers(0, 2, 60), rng.integers(0, 2, 60).astype(float)
    ll_q = QLearning().log_likelihood([0.37, 2.2], c, r)
    ll_a = AsymmetricQLearning().log_likelihood([0.37, 0.37, 2.2], c, r)
    assert np.isclose(ll_q, ll_a)


def test_choice_probabilities_are_valid_for_all_models():
    rng = np.random.default_rng(1)
    c, r = rng.integers(0, 2, 40), rng.integers(0, 2, 40).astype(float)
    for m, p in ALL:
        probs = m.trajectory(p, c, r)["probs"]
        assert probs.shape == (40, 2)
        assert np.allclose(probs.sum(axis=1), 1.0) and np.all(probs > 0)


def test_beta_zero_gives_uniform_choices():
    c, r = np.array([0, 1, 0, 0]), np.array([1.0, 0.0, 1.0, 1.0])
    probs = QLearning().trajectory([0.5, 0.0], c, r)["probs"]
    assert np.allclose(probs, 0.5)


def test_log_likelihood_equals_sum_of_log_probs():
    m, p = QLearning(), [0.4, 3.0]
    c, r = np.array([0, 0, 1, 0, 1]), np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    probs = m.trajectory(p, c, r)["probs"]
    manual = sum(np.log(probs[t, c[t]]) for t in range(5))
    assert np.isclose(m.log_likelihood(p, c, r), manual)


def test_q_learning_prefers_rewarded_option_after_reward():
    probs = QLearning().trajectory([0.5, 5.0], [0, 0, 0], [1, 1, 1])["probs"]
    assert probs[1, 0] > probs[0, 0] and probs[2, 0] > probs[1, 0]


def test_parameter_validation():
    m = QLearning()
    with pytest.raises(ValueError):
        m.to_array([1.5, 1.0])  # alpha out of bounds
    with pytest.raises(ValueError):
        m.to_array([0.5])  # wrong length
    with pytest.raises(ValueError):
        m.to_array([np.nan, 1.0])
    with pytest.raises(ValueError):
        m.to_array({"alpha": 0.5})  # missing beta
    assert np.allclose(m.to_array({"beta": 2.0, "alpha": 0.3}), [0.3, 2.0])


def test_trajectory_input_validation():
    m = QLearning()
    with pytest.raises(ValueError):
        m.trajectory([0.5, 1.0], [0, 1], [1.0])
    with pytest.raises(ValueError):
        m.trajectory([0.5, 1.0], [0, 2], [1.0, 0.0])


def test_non_binary_rewards_rejected_by_bayesian_models():
    with pytest.raises(ValueError):
        BayesBanditModel().log_likelihood([2.0], [0, 1], [0.5, 1.0])
    with pytest.raises(ValueError):
        POMDPReversalModel().log_likelihood([2.0, 0.1], [0, 1], [0.5, 1.0])


@pytest.mark.parametrize("model,params", ALL)
def test_simulation_is_reproducible_and_consistent_with_likelihood(model, params):
    env = ReversalBandit(0.8, 0.1)
    a = model.simulate(params, env, 30, rng=7)
    b = model.simulate(params, ReversalBandit(0.8, 0.1), 30, rng=7)
    assert np.array_equal(a["choices"], b["choices"]) and np.array_equal(a["rewards"], b["rewards"])
    # replaying the simulated data reproduces the generating probabilities exactly
    replay = model.trajectory(params, a["choices"], a["rewards"])["probs"]
    assert np.allclose(replay, a["probs"])


def test_hybrid_reduces_to_components():
    rng = np.random.default_rng(2)
    c, r = rng.integers(0, 2, 50), rng.integers(0, 2, 50).astype(float)
    h = HybridModel()
    assert np.isclose(h.log_likelihood([0.3, 0.1, 0.0, 3.0], c, r), QLearning().log_likelihood([0.3, 3.0], c, r))
    assert np.isclose(h.log_likelihood([0.3, 0.1, 1.0, 3.0], c, r), POMDPReversalModel().log_likelihood([3.0, 0.1], c, r))


def test_pomdp_model_tracks_reversal_after_surprise():
    """After a run of rewards on A then repeated non-reward on A, belief must move toward B."""
    m, p = POMDPReversalModel(), np.array([5.0, 0.15])
    s = m.init_state(p)
    for _ in range(6):
        s = m.update(p, s, 0, 1.0)
    p_before = m.choice_probs(p, s)[0]
    for _ in range(4):
        s = m.update(p, s, 0, 0.0)
    assert m.choice_probs(p, s)[0] < p_before


def test_bayes_model_explores_more_than_greedy_value_would():
    """Equal posterior means but different certainty: BAMDP values the uncertain option more."""
    m = BayesBanditModel(horizon=5)
    # arm 0: 1 success / 1 failure (uncertain); arm 1: 30 / 30 (nearly certain); both mean 0.5
    posterior = (2.0, 2.0, 31.0, 31.0)
    q = m.action_values(np.array([3.0]), posterior)
    assert q[0] > q[1]


def test_sample_params_within_bounds():
    rng = np.random.default_rng(0)
    for m, _ in ALL:
        for _ in range(20):
            m.to_array(m.sample_params(rng))  # raises if outside bounds
