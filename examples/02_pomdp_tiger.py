"""POMDP tutorial: belief tracking and exact planning in the Tiger problem (Kaelbling, Littman & Cassandra, 1998).

Run:  python examples/02_pomdp_tiger.py
"""

import numpy as np

from humanchoice.pomdp import TIGER_ACTIONS, tiger_pomdp

tiger = tiger_pomdp(accuracy=0.85, gamma=0.95)

# --- 1. Bayes filter -------------------------------------------------------------------------------------------
belief = np.array([0.5, 0.5])  # P(tiger left), P(tiger right)
print("belief:", belief.round(3))
for growl in (0, 0, 1, 0):  # growl-left, growl-left, growl-right, growl-left (action 0 = listen)
    belief = tiger.belief_update(belief, action=0, obs=growl)
    print(f"after listening and hearing growl {'LR'[growl]}: P(tiger left) = {belief[0]:.3f}")

# --- 2. Exact finite-horizon solution by alpha-vector value iteration with incremental pruning -----------------------
sets = tiger.solve_finite_horizon(horizon=5)
print("\n#alpha-vectors per horizon:", [len(s) for s in sets[1:]])
for b_left in (0.5, 0.9, 0.98):
    b = np.array([b_left, 1 - b_left])
    print(f"P(tiger left) = {b_left:.2f}: optimal action = {TIGER_ACTIONS[sets[5].action(b)]:10s} V_5 = {sets[5].value(b):7.3f}")

# --- 3. Two independent exact methods agree --------------------------------------------------------------------
b = np.array([0.7, 0.3])
print("\nalpha-vector V_4:", round(sets[4].value(b), 6), "| belief-tree expectimax V_4:", round(tiger.value(b, 4), 6))

# --- 4. QMDP is an optimistic upper bound --------------------------------------------------------------------
# QMDP solves the fully observable MDP and averages its Q-values over the belief. It implicitly assumes the state
# becomes known after one step, so it does not price the cost of *acquiring* information: its values are far above
# what any real POMDP policy can achieve (compare with the exact values above, which are finite-horizon).
print("QMDP action values at the uniform belief [listen, open-L, open-R]:", tiger.qmdp_q_values(np.array([0.5, 0.5])).round(1))
print("-> an upper bound on the optimal discounted values, not an achievable return")
