"""Experiment 8 - the three planning models of the library, exercised end to end.

(a) POMDP    - exact alpha-vector value functions for the Tiger problem; size of the exact solution per horizon.
(b) BAMDP    - value of information: Bayes-optimal vs greedy value in a two-armed Bernoulli bandit, and the
               exploration bonus of an uncertain arm over an equally-good well-known arm.
(c) I-POMDP  - a level-1 interactive agent (which models the other agent) vs a level-0 POMDP agent (which ignores it)
               in the two-agent Tiger game, when the other agent is exactly as modelled and when it is mis-specified.

All environments are simulated. Planning is exact; the only randomness is in the simulated episodes (seeded).
"""

import sys
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import FIGURES, parse_args, save_config, save_table

from humanchoice.bamdp import BetaBernoulliBanditBAMDP
from humanchoice.evaluation.visualization import plot_pomdp_value_function
from humanchoice.ipomdp import Level1IPOMDP, MultiAgentTigerEnv, multiagent_tiger
from humanchoice.pomdp import tiger_pomdp

CATASTROPHE = -100.0


# ----------------------------------------------------------------------------------------------- (a) POMDP
def pomdp_part(max_h):
    tiger = tiger_pomdp(gamma=0.95)
    sets = tiger.solve_finite_horizon(max_h)
    rows = [{"horizon": h, "n_alpha_vectors": len(sets[h]), "V_at_belief_0.5": sets[h].value(np.array([0.5, 0.5])),
             "action_at_belief_0.5": ("listen", "open-left", "open-right")[sets[h].action(np.array([0.5, 0.5]))] if h else ""}
            for h in range(1, max_h + 1)]
    plot_pomdp_value_function(sets, list(range(1, max_h + 1)), xlabel="P(tiger is left)", title="Tiger: exact value functions",
                              save_path=FIGURES / "exp8_tiger_value_functions.png")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- (b) BAMDP
def bamdp_part(max_h):
    planner = BetaBernoulliBanditBAMDP(2, (1.0, 1.0))

    @cache
    def greedy_value(post, h):  # value of always pulling the arm with the higher posterior mean (ties -> arm 0)
        if h == 0:
            return 0.0
        means = planner.posterior_mean(post)
        k = int(np.argmax(means))
        p = means[k]
        return p * (1 + greedy_value(planner.update(post, k, 1), h - 1)) + (1 - p) * greedy_value(planner.update(post, k, 0), h - 1)

    start = planner.initial_posterior()
    rows = []
    for h in range(1, max_h + 1):
        v_opt, v_greedy = planner.value(start, h), greedy_value(start, h)
        # arm 0: completely unknown Beta(1,1); arm 1: well-known, same mean 0.5 -> Beta(30,30)
        q = planner.q_values((1.0, 1.0, 30.0, 30.0), h)
        rows.append({"horizon": h, "V_bayes_optimal": v_opt, "V_greedy": v_greedy, "advantage": v_opt - v_greedy,
                     "Q_uncertain_arm": q[0], "Q_known_arm": q[1], "exploration_bonus": q[0] - q[1]})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].plot(df["horizon"], df["advantage"], "o-")
    axes[0].axhline(0, color="grey", lw=0.8, ls=":")
    axes[0].set(xlabel="trials remaining", ylabel="V(Bayes-optimal) - V(greedy)",
                title="Advantage of planning for information")
    axes[1].plot(df["horizon"], df["exploration_bonus"], "o-", color="C2")
    axes[1].axhline(0, color="grey", lw=0.8, ls=":")
    axes[1].set(xlabel="trials remaining", ylabel="Q(uncertain) - Q(known)", title="Exploration bonus (equal means)")
    fig.tight_layout()
    fig.savefig(FIGURES / "exp8_bamdp_value_of_information.png", dpi=150)
    return df


# ----------------------------------------------------------------------------------------------- (c) I-POMDP
def eps_greedy(q, eps):
    greedy = np.isclose(q, q.max(), atol=1e-9).astype(float)
    return (1 - eps) * greedy / greedy.sum() + eps / len(q)


def run_episode(kind, seed, n_steps, plan_h, ip, level0, j_true_horizon, eps_j):
    rng = np.random.default_rng(seed)
    env = MultiAgentTigerEnv(rng)
    b = ip.initial_belief() if kind == "level-1 I-POMDP" else np.array([0.5, 0.5])
    b_j = np.array([0.5, 0.5])
    total, catastrophes = 0.0, 0
    for _ in range(n_steps):
        q = ip.q_values(b, plan_h) if kind == "level-1 I-POMDP" else level0.q_values(b, plan_h)
        a_i = int(np.argmax(q))  # ties -> listen
        a_j = int(rng.choice(3, p=eps_greedy(level0.q_values(b_j, j_true_horizon), eps_j)))
        z_i, z_j, r_i, _ = env.step(a_i, a_j)
        total += r_i
        catastrophes += r_i == CATASTROPHE
        b = ip.update(b, a_i, z_i) if kind == "level-1 I-POMDP" else level0.belief_update(b, a_i, z_i // 3)  # level-0 uses only the growl
        b_j = level0.belief_update(b_j, a_j, z_j)
    return total, catastrophes


def ipomdp_part(n_episodes, n_steps, plan_h, seed):
    G, level0, eps_j = multiagent_tiger(), tiger_pomdp(gamma=0.95), 0.05
    ip = Level1IPOMDP(G["T"], G["O_i"], G["R_i"], G["O_j"], level0, gamma=0.95, epsilon_j=eps_j, horizon0=3, depth=8)
    rows, raw = [], []
    for cond, j_h in {"j exactly as modelled (horizon 3)": 3, "j mis-specified (horizon 1)": 1}.items():
        results = {}
        for kind in ("level-0 POMDP", "level-1 I-POMDP"):
            out = np.array([run_episode(kind, int(s), n_steps, plan_h, ip, level0, j_h, eps_j)
                            for s in np.random.SeedSequence(seed).generate_state(n_episodes)])
            results[kind] = out
            raw.append(pd.DataFrame({"condition": cond, "agent": kind, "return": out[:, 0], "catastrophes": out[:, 1]}))
        diff = results["level-1 I-POMDP"][:, 0] - results["level-0 POMDP"][:, 0]
        for kind, out in results.items():
            rows.append({"condition": cond, "agent": kind, "mean_return": out[:, 0].mean(),
                         "ci95": 1.96 * out[:, 0].std(ddof=1) / np.sqrt(n_episodes),
                         "catastrophes_per_episode": out[:, 1].mean(), "episodes": n_episodes, "steps": n_steps})
        rows.append({"condition": cond, "agent": "paired difference (L1 - L0)", "mean_return": diff.mean(),
                     "ci95": 1.96 * diff.std(ddof=1) / np.sqrt(n_episodes), "catastrophes_per_episode": np.nan,
                     "episodes": n_episodes, "steps": n_steps})
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    sub = table[table["agent"] != "paired difference (L1 - L0)"]
    conds = list(sub["condition"].unique())
    for k, kind in enumerate(("level-0 POMDP", "level-1 I-POMDP")):
        s = sub[sub["agent"] == kind]
        ax.bar(np.arange(len(conds)) + (k - 0.5) * 0.38, s["mean_return"], 0.38, yerr=s["ci95"], label=kind, capsize=3)
    ax.set_xticks(range(len(conds)), conds, fontsize=8)
    ax.set(ylabel=f"mean return over {n_steps} steps", title="Two-agent Tiger (95% CI)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "exp8_ipomdp_vs_pomdp.png", dpi=150)

    # What the interactive belief update infers from the other agent's audible actions.
    # i listens four times. Observation = (growl, creak); creak 2 = silence, 0/1 = j opened the left/right door.
    scenarios = {
        "3 x (growl-left, silence)": [(0, 2)] * 3,
        "... then (growl-left, silence)": [(0, 2)] * 4,
        "... then (growl-left, creak-right): j opened right": [(0, 2)] * 3 + [(0, 1)],
        "... then (growl-left, creak-left): j opened left": [(0, 2)] * 3 + [(0, 0)],
    }
    infer = []
    for label, seq in scenarios.items():
        b, b0 = ip.initial_belief(), np.array([0.5, 0.5])
        for growl, creak in seq:
            b = ip.update(b, 0, growl * 3 + creak)
            b0 = level0.belief_update(b0, 0, growl)  # level-0 agent hears the growl but ignores the creak
        confident = np.abs(ip.models[:, 0] - 0.5) > 0.4
        infer.append({"observations (i listens each step)": label, "level-1 P(tiger left)": ip.state_marginal(b)[0],
                      "level-0 P(tiger left)": b0[0], "level-1 P(j is confident)": float(ip.model_marginal(b)[confident].sum())})
    return table, pd.DataFrame(infer), raw


def main():
    args = parse_args(__doc__)
    max_h_pomdp, max_h_bamdp, episodes, steps = (3, 5, 20, 8) if args.quick else (5, 12, 400, 20)

    t_pomdp = pomdp_part(max_h_pomdp)
    save_table(t_pomdp, "exp8_pomdp_tiger")
    print(t_pomdp.round(3).to_string(index=False), "\n")

    t_bamdp = bamdp_part(max_h_bamdp)
    save_table(t_bamdp, "exp8_bamdp_value_of_information")
    print(t_bamdp.round(4).to_string(index=False), "\n")

    t_ip, t_infer, raw = ipomdp_part(episodes, steps, plan_h=3, seed=args.seed)
    save_table(t_ip, "exp8_ipomdp_vs_pomdp")
    save_table(t_infer, "exp8_ipomdp_inference")
    save_table(pd.concat(raw, ignore_index=True), "exp8_ipomdp_vs_pomdp_episodes")
    print(t_ip.round(3).to_string(index=False), "\n")
    print(t_infer.round(3).to_string(index=False))
    save_config("exp8_interactive_models", {"pomdp_max_horizon": max_h_pomdp, "bamdp_max_horizon": max_h_bamdp, "episodes": episodes,
                                            "steps_per_episode": steps, "planning_horizon": 3, "epsilon_j": 0.05, "model_closure_depth": 8,
                                            "seed": args.seed})


if __name__ == "__main__":
    main()
