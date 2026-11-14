"""BAMDP tutorial: Bayes-optimal exploration in a Bernoulli bandit, and a general BAMDP with unknown transitions.

Run:  python examples/01_bamdp_bandit.py
"""

import numpy as np

from humanchoice.bamdp import BAMDP, BetaBernoulliBanditBAMDP

# --- 1. Conjugate belief updating -----------------------------------------------------------------------------
planner = BetaBernoulliBanditBAMDP(n_arms=2, prior=(1.0, 1.0))
belief = planner.initial_posterior()  # (alpha_1, beta_1, alpha_2, beta_2) = (1, 1, 1, 1)
for arm, reward in [(0, 1), (0, 1), (1, 0)]:
    belief = planner.update(belief, arm, reward)
print("posterior pseudo-counts:", belief)
print("posterior mean reward  :", planner.posterior_mean(belief).round(3))

# --- 2. Bayes-optimal Q-values contain a value-of-information bonus ----------------------------------------------
# arm 0 is completely unknown, arm 1 is well known; both have posterior mean 0.5.
for horizon in (1, 2, 5, 10):
    q = planner.q_values((1.0, 1.0, 30.0, 30.0), horizon)
    print(f"{horizon:2d} trials left: Q(unknown arm) = {q[0]:.3f}, Q(known arm) = {q[1]:.3f}, bonus = {q[0] - q[1]:+.3f}")

# --- 3. Hand check: with 2 trials left from the prior the optimal value is 13/12 ----------------------------------
print("V_2(prior) =", round(planner.value(planner.initial_posterior(), 2), 6), "(13/12 =", round(13 / 12, 6), ")")

# --- 4. A general BAMDP: unknown transitions (Dirichlet) and unknown Bernoulli rewards (Beta) ---------------------
# Two states, two actions. Action 1 in state 0 is a risky shortcut to the rewarding state 1.
S, A = 2, 2
prior_T = np.ones((S, A, S))  # Dirichlet(1, 1) over next states for every (s, a)
prior_R = np.ones((S, A, 2))  # Beta(1, 1) over the reward probability of every (s, a)
bamdp = BAMDP(prior_T, prior_rewards=prior_R, gamma=0.95)
b = bamdp.initial_belief()
b = bamdp.update(b, s=0, a=1, s_next=1, reward=1)  # observe: shortcut worked and paid off
T_hat, R_hat = bamdp.expected_model(b)
print("posterior-mean P(s'|s=0,a=1):", T_hat[0, 1].round(3), " posterior-mean reward:", R_hat[0, 1].round(3))
print("Bayes-optimal Q(s=0, .) with 4 steps left:", bamdp.q_values(0, b, 4).round(3))
