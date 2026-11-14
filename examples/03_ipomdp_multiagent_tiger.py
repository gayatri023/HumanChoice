"""I-POMDP tutorial: a level-1 agent that models another agent, in a two-agent Tiger game.

Run:  python examples/03_ipomdp_multiagent_tiger.py
"""

import numpy as np

from humanchoice.ipomdp import Level1IPOMDP, multiagent_tiger
from humanchoice.pomdp import tiger_pomdp

game = multiagent_tiger()
level0 = tiger_pomdp(gamma=0.95)  # agent i's model of agent j: a level-0 Tiger POMDP agent
agent = Level1IPOMDP(game["T"], game["O_i"], game["R_i"], game["O_j"], level0,
                     gamma=0.95, epsilon_j=0.05, horizon0=3, depth=8)

print(f"interactive states: {agent.pomdp.n_states} = 2 tiger positions x {agent.M} models of j")
centre = int(np.argmin(np.abs(agent.models[:, 0] - 0.5)))
confident = int(np.argmax(agent.models[:, 0]))
print("modelled P(j acts | j undecided)         [listen, open-L, open-R]:", agent.j_action_probs[centre].round(3))
print("modelled P(j acts | j sure tiger is left) [listen, open-L, open-R]:", agent.j_action_probs[confident].round(3))

# i listens four times; observation index = growl * 3 + creak, creak: 0 = j opened left, 1 = right, 2 = silence
b, naive = agent.initial_belief(), np.array([0.5, 0.5])
for step, (growl, creak) in enumerate([(0, 2), (0, 2), (0, 2), (0, 1)], start=1):
    b = agent.update(b, action_i=0, obs_i=growl * 3 + creak)
    naive = level0.belief_update(naive, 0, growl)
    note = "  <- hears j open the right door" if creak == 1 else ""
    print(f"step {step}: level-1 P(tiger left) = {agent.state_marginal(b)[0]:.3f} | naive level-0 = {naive[0]:.3f}{note}")

print("\nplanned action values for i (listen, open-L, open-R), 3 steps:", agent.q_values(b, 3).round(2))
