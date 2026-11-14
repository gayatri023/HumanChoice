"""Level-1 I-POMDP: structure, reductions to simpler models and qualitative interactive inference."""
import numpy as np
import pytest

from humanchoice.ipomdp import (
    Level1IPOMDP,
    MultiAgentTigerEnv,
    belief_closure,
    multiagent_tiger,
)
from humanchoice.pomdp import POMDP, tiger_pomdp

G = multiagent_tiger()
LEVEL0 = tiger_pomdp(gamma=0.95)


def build(**kw):
    return Level1IPOMDP(G["T"], G["O_i"], G["R_i"], G["O_j"], LEVEL0, gamma=0.95, **kw)


def test_belief_closure_of_tiger_is_a_chain_of_size_2d_plus_1():
    for depth in (1, 3, 5):
        assert len(belief_closure(LEVEL0, np.array([0.5, 0.5]), depth)) == 2 * depth + 1


def test_game_kernels_are_proper_distributions():
    assert np.allclose(G["T"].sum(axis=3), 1)
    assert np.allclose(G["O_i"].sum(axis=3), 1) and np.allclose(G["O_j"].sum(axis=3), 1)


def test_interactive_kernel_is_stochastic_and_other_policy_valid():
    ip = build(depth=4)
    assert np.allclose(ip.pomdp.W.sum(axis=(2, 3)), 1.0)
    assert np.allclose(ip.j_action_probs.sum(axis=1), 1.0)
    assert ip.pomdp.n_states == 2 * ip.M and ip.pomdp.n_obs == 6


def test_other_agent_listens_when_unsure_and_opens_when_confident():
    ip = build(depth=5, epsilon_j=0.0)
    centre = int(np.argmin(np.abs(ip.models[:, 0] - 0.5)))
    assert np.argmax(ip.j_action_probs[centre]) == 0  # listen
    confident_left = int(np.argmax(ip.models[:, 0]))  # j sure the tiger is left -> opens right door (action 2)
    confident_right = int(np.argmin(ip.models[:, 0]))
    assert np.argmax(ip.j_action_probs[confident_left]) == 2
    assert np.argmax(ip.j_action_probs[confident_right]) == 1


def test_reduces_to_single_agent_pomdp_when_other_always_listens():
    """If j is known to always listen, creaks are state-independent noise: i's values equal the single-agent Tiger values."""
    ip = build(depth=3, j_policy=lambda b: np.array([1.0, 0.0, 0.0]))
    for x in (0.5, 0.8, 0.97):
        b_state = np.array([x, 1 - x])
        b_is = ip.initial_belief(state_belief=b_state)
        for h in (1, 2, 3):
            assert np.allclose(ip.q_values(b_is, h), LEVEL0.q_values(b_state, h), atol=1e-9)


def test_filtering_marginal_matches_single_agent_filter_when_other_always_listens():
    ip = build(depth=3, j_policy=lambda b: np.array([1.0, 0.0, 0.0]))
    b_is, b = ip.initial_belief(), np.array([0.5, 0.5])
    for growl, creak in [(0, 2), (0, 2), (1, 2)]:  # listen each time; creak = silence
        b_is = ip.update(b_is, 0, growl * 3 + creak)
        b = LEVEL0.belief_update(b, 0, growl)
        assert np.allclose(ip.state_marginal(b_is), b)


def test_hearing_the_other_agent_open_a_door_weakens_confidence_in_old_evidence():
    ip = build(depth=4)
    b0 = ip.initial_belief(state_belief=np.array([0.9, 0.1]))
    growl_left = 0
    silent = ip.state_marginal(ip.update(b0, 0, growl_left * 3 + 2))[0]  # creak: silence
    creak_left = ip.state_marginal(ip.update(b0, 0, growl_left * 3 + 0))[0]  # creak: j opened left
    assert abs(creak_left - 0.5) < abs(silent - 0.5)


def test_inferring_the_other_agents_model_from_its_action():
    """If i hears j open a door, j's belief must have just been reset: posterior mass moves to the 'uncertain' model."""
    ip = build(depth=4, epsilon_j=0.02)
    extreme = int(np.argmin(ip.models[:, 0]))  # j believes tiger right -> opens left
    prior = np.zeros(ip.M)
    prior[extreme], prior[0] = 0.5, 0.5
    b0 = ip.initial_belief(model_prior=prior)
    posterior = ip.update(b0, 0, 0 * 3 + 0)  # i listens, growl-left, creak: j opened left
    assert ip.model_marginal(posterior)[0] > 0.8


def test_interactive_agent_values_differ_from_single_agent_when_other_may_open_doors():
    """Resets caused by j change what i should expect, so values must differ from the single-agent case."""
    ip = build(depth=4)
    b_state = np.array([0.97, 0.03])
    q_ip = ip.q_values(ip.initial_belief(state_belief=b_state), 3)
    q_single = LEVEL0.q_values(b_state, 3)
    assert not np.allclose(q_ip, q_single)


def test_validation_errors():
    wrong_level0 = POMDP(np.eye(2)[None].repeat(2, 0), np.full((2, 2, 2), 0.5), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        Level1IPOMDP(G["T"], G["O_i"], G["R_i"], G["O_j"], wrong_level0)
    with pytest.raises(ValueError):
        build(epsilon_j=1.5)


def test_multiagent_env_dynamics():
    env = MultiAgentTigerEnv(seed=0)
    s0 = env.state
    env.step(0, 0)  # both listen: state unchanged
    assert env.state == s0
    # reward is computed from the pre-step state
    env.state = 0
    _, _, r_i, r_j = env.step(1, 0)  # i opens left while tiger is left; j listens
    assert r_i == -100.0 and r_j == -1.0
    # opening resets uniformly
    states = []
    for _ in range(2000):
        env.step(2, 0)
        states.append(env.state)
    assert abs(np.mean(states) - 0.5) < 0.05
