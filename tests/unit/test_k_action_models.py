"""K-option generalisation, the model ladder, missed trials and unavailable actions."""

import numpy as np
import pytest

from humanchoice.envs import BernoulliBandit, DriftingBernoulliBandit, RestlessBandit, ReversalBandit
from humanchoice.models import (
    AsymmetricQLearning,
    BayesBanditModel,
    FullQLearning,
    HybridModel,
    KalmanBandit,
    LapseQLearning,
    POMDPReversalModel,
    QLearning,
    RandomChoice,
    StickyQLearning,
    WinStayLoseShift,
)
from humanchoice.utils import MISSED, masked_softmax

Q_FAMILY = [QLearning, AsymmetricQLearning, StickyQLearning, LapseQLearning, FullQLearning]


def _theta(model, seed=0):
    return model.sample_params(seed)


def _sequence(K, T=60, seed=0, binary=True):
    rng = np.random.default_rng(seed)
    c = rng.integers(0, K, T)
    r = rng.integers(0, 2, T).astype(float) if binary else rng.uniform(0, 1, T)
    return c, r


# ------------------------------------------------------------------ interface
@pytest.mark.parametrize("K", [2, 3, 4, 7])
@pytest.mark.parametrize("cls", Q_FAMILY)
def test_probabilities_are_valid_for_any_number_of_options(cls, K):
    m = cls(n_actions=K)
    c, r = _sequence(K, seed=K)
    probs = m.trajectory(_theta(m), c, r)["probs"]
    assert probs.shape == (len(c), K)
    assert (probs >= 0).all() and np.allclose(probs.sum(axis=1), 1.0)


@pytest.mark.parametrize("bad", [0, 1, -3, 2.5, True])
def test_invalid_number_of_actions_is_rejected(bad):
    with pytest.raises(ValueError):
        QLearning(n_actions=bad)


def test_two_option_only_models_declare_and_enforce_their_size():
    from humanchoice.models.base import ChoiceModel

    assert POMDPReversalModel().n_actions == 2 and POMDPReversalModel.fixed_n_actions == 2
    assert HybridModel().n_actions == 2 and HybridModel.fixed_n_actions == 2

    class _Probe(POMDPReversalModel):  # the base-class guard rejects any other size
        def __init__(self):
            ChoiceModel.__init__(self, 4)

    with pytest.raises(ValueError, match="exactly 2"):
        _Probe()


def test_invalid_action_indices_are_rejected():
    m = QLearning(n_actions=4)
    th = [0.3, 2.0]
    for bad in ([0, 4], [-2, 1], [0.5, 1.0]):
        with pytest.raises(ValueError):
            m.log_likelihood(th, bad, [1.0, 0.0])
    with pytest.raises(ValueError, match="equal length"):
        m.log_likelihood(th, [0, 1], [1.0])


def test_environment_and_model_must_agree_on_number_of_options():
    with pytest.raises(ValueError, match="offers 4 options"):
        QLearning(n_actions=2).simulate([0.3, 2.0], BernoulliBandit((0.1, 0.2, 0.3, 0.4)), 10, 0)


def test_binary_only_models_reject_continuous_rewards():
    m = BayesBanditModel(n_actions=2)
    with pytest.raises(ValueError, match="binary"):
        m.log_likelihood([2.0], [0, 1, 0], [0.3, 1.0, 0.0])
    with pytest.raises(ValueError, match="binary"):
        POMDPReversalModel().log_likelihood([2.0, 0.1], [0, 1], [0.5, 1.0])
    QLearning().log_likelihood([0.3, 2.0], [0, 1], [0.5, 0.7])  # continuous rewards are fine for Q-learning


# --------------------------------------------------------- simulate == replay
@pytest.mark.parametrize("cls", Q_FAMILY)
def test_simulation_and_likelihood_share_one_code_path(cls):
    m = cls(n_actions=4)
    th = _theta(m, 3)
    sim = m.simulate(th, DriftingBernoulliBandit(4, 0.1), 80, rng=5)
    traj = m.trajectory(th, sim["choices"], sim["rewards"])
    assert np.allclose(traj["probs"], sim["probs"])
    ll = np.log(sim["probs"][np.arange(80), sim["choices"]]).sum()
    assert m.log_likelihood(th, sim["choices"], sim["rewards"]) == pytest.approx(ll)


# ------------------------------------------------------- fast path == generic
@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("cls", Q_FAMILY)
def test_fast_likelihood_matches_generic_replay_including_missed_trials(cls, K):
    m = cls(n_actions=K)
    for seed in range(4):
        th = _theta(m, seed)
        c, r = _sequence(K, T=70, seed=seed, binary=False)
        c[[0, 5, 6, 40]] = MISSED  # includes a missed first trial and consecutive misses
        fast = m.log_likelihood(th, c, r)
        traj = m.trajectory(th, c, r)
        obs = traj["observed"]
        generic = np.log(np.clip(traj["probs"][obs, c[obs]], 1e-12, None)).sum()
        assert fast == pytest.approx(generic, rel=1e-10, abs=1e-10)


def test_fast_path_survives_extreme_reward_scales():
    m = StickyQLearning()
    th = [0.5, 20.0, 1.0]
    ll = m.log_likelihood(th, [0, 1, 0, 1], [1e4, -1e4, 1e4, 1e4])  # exp overflow in the scalar loop
    assert np.isfinite(ll)


# --------------------------------------------------------------- nesting
def test_family_members_nest_exactly():
    c, r = _sequence(3, T=80, seed=9, binary=False)
    q = QLearning(3)
    base = q.log_likelihood([0.4, 3.0], c, r)
    assert AsymmetricQLearning(3).log_likelihood([0.4, 0.4, 3.0], c, r) == pytest.approx(base)
    assert StickyQLearning(3).log_likelihood([0.4, 3.0, 0.0], c, r) == pytest.approx(base)
    assert LapseQLearning(3).log_likelihood([0.4, 3.0, 0.0], c, r) == pytest.approx(base)
    assert FullQLearning(3).log_likelihood([0.4, 0.4, 3.0, 0.0, 0.0], c, r) == pytest.approx(base)


# ------------------------------------------------------------ mathematics
def test_two_option_q_learning_hand_calculation():
    m = QLearning(2, q0=0.5)
    th = [0.5, 2.0]
    # trial 1: Q=(.5,.5) -> p=.5; chose 0, r=1 -> Q0=.75. trial 2: P(1) = e^{2*.5}/(e^{2*.75}+e^{2*.5})
    p2 = np.exp(1.0) / (np.exp(1.5) + np.exp(1.0))
    assert m.log_likelihood(th, [0, 1], [1.0, 0.0]) == pytest.approx(np.log(0.5) + np.log(p2))


def test_four_option_softmax_matches_formula():
    m = QLearning(4)
    probs = m.trajectory([0.3, 3.0], [1, 3], [1.0, 0.0])["probs"]
    assert np.allclose(probs[0], 0.25)
    q = np.array([0.5, 0.5 + 0.3 * 0.5, 0.5, 0.5])
    expected = np.exp(3.0 * q) / np.exp(3.0 * q).sum()
    assert np.allclose(probs[1], expected)


def test_sticky_bonus_multiplies_the_odds_by_exp_kappa():
    m = StickyQLearning(3)
    th = [0.2, 2.0, 0.8]
    p = m.trajectory(th, [1, 0], [0.0, 0.0])["probs"][1]  # after choosing 1 (reward 0): Q = (.5, .4, .5)
    q = np.array([0.5, 0.5 - 0.2 * 0.5, 0.5])
    base = np.exp(2.0 * q) / np.exp(2.0 * q).sum()
    # odds of the repeated option relative to an alternative are inflated by exactly exp(kappa)
    assert (p[1] / p[0]) == pytest.approx((base[1] / base[0]) * np.exp(0.8))
    assert (p[0] / p[2]) == pytest.approx(base[0] / base[2])


def test_lapse_mixture_formula_and_probability_floor():
    m = LapseQLearning(4)
    eps = 0.2
    p = m.trajectory([0.3, 20.0, eps], [0, 1], [1.0, 0.0])["probs"][1]
    q = np.array([0.5 + 0.3 * 0.5, 0.5, 0.5, 0.5])
    soft = np.exp(20.0 * q) / np.exp(20.0 * q).sum()
    assert np.allclose(p, (1 - eps) * soft + eps / 4)
    assert p.min() >= eps / 4 - 1e-12


def test_win_stay_lose_shift_hand_calculation():
    m = WinStayLoseShift(3, win_threshold=0.5)
    th = [0.9, 0.6]  # p_win_stay, p_lose_shift
    probs = m.trajectory(th, [1, 1, 2], [1.0, 0.0, 1.0])["probs"]
    assert np.allclose(probs[0], 1 / 3)
    assert np.allclose(probs[1], [0.05, 0.9, 0.05])  # after a win: stay .9, others share .1
    assert np.allclose(probs[2], [0.3, 0.4, 0.3])  # after a loss: stay 1-.6=.4, others share .6


def test_random_choice_is_the_uniform_baseline():
    m = RandomChoice(5)
    c, r = _sequence(5, T=40, seed=1)
    assert m.n_params == 0
    assert m.log_likelihood([], c, r) == pytest.approx(40 * np.log(1 / 5))


def test_kalman_update_matches_hand_calculation():
    m = KalmanBandit(n_actions=2, obs_sd=0.1, decay=1.0, m0=0.5, v0=0.04)
    beta, phi, ratio = 3.0, 0.0, 0.5
    th = [beta, phi, ratio]
    s = m.update(np.array(th), m.init_state(np.array(th)), 0, 0.9)
    gain = 0.04 / (0.04 + 0.01)  # 0.8
    mean0 = 0.5 + gain * 0.4
    var0 = 0.04 * (1 - gain)
    assert s[0][0] == pytest.approx(mean0)
    assert s[1][0] == pytest.approx(var0 + ratio * 0.01)  # posterior variance + innovation variance
    assert s[1][1] == pytest.approx(0.04 + ratio * 0.01)  # unchosen option only gains uncertainty
    assert s[0][1] == pytest.approx(0.5)


def test_kalman_uncertainty_bonus_favours_the_less_sampled_option():
    m = KalmanBandit(n_actions=3)
    th = np.array([5.0, 1.0, 0.5])
    state = m.init_state(th)
    for _ in range(6):
        state = m.update(th, state, 0, 0.5)  # only option 0 is ever sampled, always with the prior-mean payoff
    p = m.choice_probs(th, state)
    assert p[1] > p[0] and p[2] > p[0]
    # without the bonus (phi = 0) all means are still at the prior mean 0.5, so choice is exactly uniform
    assert np.allclose(m.choice_probs(np.array([5.0, 0.0, 0.5]), state), 1 / 3)


# -------------------------------------------------- unavailable & missed trials
def test_masked_softmax_zeroes_unavailable_actions_and_matches_renormalisation():
    x = np.array([0.3, -0.2, 0.9, 0.1])
    avail = np.array([True, False, True, True])
    p = masked_softmax(x, 2.0, avail)
    full = np.exp(2.0 * x) / np.exp(2.0 * x).sum()
    assert p[1] == 0.0 and p.sum() == pytest.approx(1.0)
    renorm = full * avail / (full * avail).sum()
    assert np.allclose(p, renorm)
    with pytest.raises(ValueError):
        masked_softmax(x, 2.0, np.zeros(4, dtype=bool))


@pytest.mark.parametrize("cls", Q_FAMILY)
def test_unavailable_actions_get_zero_probability_in_every_model(cls):
    K = 4
    m = cls(n_actions=K)
    th = _theta(m, 1)
    T = 30
    rng = np.random.default_rng(0)
    avail = rng.random((T, K)) < 0.7
    avail[np.arange(T), rng.integers(0, K, T)] = True  # at least one available
    c = np.array([rng.choice(np.flatnonzero(a)) for a in avail])
    r = rng.uniform(0, 1, T)
    probs = m.trajectory(th, c, r, available=avail)["probs"]
    assert (probs[~avail] == 0).all()
    assert np.allclose(probs.sum(axis=1), 1.0)
    # simulate with a schedule of availability never picks an unavailable action
    sim = m.simulate(th, DriftingBernoulliBandit(K, 0.1), T, 3, available=avail)
    assert avail[np.arange(T), sim["choices"]].all()


def test_choosing_an_unavailable_action_is_an_error():
    m = QLearning(3)
    avail = np.array([[True, True, False], [True, True, True]])
    with pytest.raises(ValueError, match="unavailable"):
        m.log_likelihood([0.3, 2.0], [2, 0], [1.0, 0.0], available=avail)
    with pytest.raises(ValueError, match="shape"):
        m.log_likelihood([0.3, 2.0], [0, 0], [1.0, 0.0], available=avail[:, :2])
    with pytest.raises(ValueError, match="at least one"):
        m.log_likelihood([0.3, 2.0], [0, 0], [1.0, 0.0], available=np.zeros((2, 3), dtype=bool))


@pytest.mark.parametrize("cls", [*Q_FAMILY, WinStayLoseShift, RandomChoice])
def test_a_missed_trial_is_equivalent_to_the_trial_not_existing(cls):
    K = 3
    m = cls(n_actions=K)
    th = _theta(m, 2)
    c, r = _sequence(K, T=25, seed=4, binary=False)
    with_miss = c.copy()
    with_miss[10] = MISSED
    keep = np.arange(25) != 10
    assert m.log_likelihood(th, with_miss, r) == pytest.approx(m.log_likelihood(th, c[keep], r[keep]))


def test_availability_changes_the_likelihood_only_through_renormalisation():
    m = QLearning(3)
    th = [0.3, 2.0]
    c, r = [0, 1, 0], [1.0, 0.0, 1.0]
    all_av = np.ones((3, 3), dtype=bool)
    assert m.log_likelihood(th, c, r, available=all_av) == pytest.approx(m.log_likelihood(th, c, r))


# ------------------------------------------------------------ environments
def test_environments_expose_consistent_protocol():
    rng = np.random.default_rng(0)
    for env in (BernoulliBandit((0.1, 0.5, 0.9)), ReversalBandit(), DriftingBernoulliBandit(4), RestlessBandit(4)):
        env.reset(rng)
        for t in range(20):
            r = env.step(int(rng.integers(env.n_actions)), rng)
            assert 0.0 <= r <= 1.0
        assert len(env.good_arm_history) == 20
        assert all(0 <= g < env.n_actions for g in env.good_arm_history)


def test_restless_bandit_rewards_are_continuous_and_drift():
    env = RestlessBandit(4)
    rng = np.random.default_rng(1)
    env.reset(rng)
    rewards = [env.step(0, rng) for _ in range(300)]
    assert len(set(np.round(rewards, 6))) > 100
    assert len(set(env.good_arm_history)) > 1  # the best arm changes over time
    with pytest.raises(ValueError):
        RestlessBandit(1)
    with pytest.raises(ValueError):
        DriftingBernoulliBandit(4, drift_sd=-1)
    with pytest.raises(ValueError):
        BernoulliBandit((0.5,))
