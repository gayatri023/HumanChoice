"""Experiment 3 - individual differences.

(a) The four illustrative participants from the specification (alpha, beta) = A (0.2, 2), B (0.7, 2), C (0.4, 0.5),
    D (0.4, 5): learning curves and win-stay / lose-shift behaviour.  These are experimental settings, not
    psychological categories.
(b) A grid over (alpha, beta): how parameter combinations map onto behaviour.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import FIGURES, parse_args, save_config, save_table

from humanchoice.envs import ReversalBandit
from humanchoice.evaluation.visualization import plot_action_values, plot_learning_curves
from humanchoice.models import QLearning
from humanchoice.simulation import EXAMPLE_PARTICIPANTS


def make_env():
    return ReversalBandit(0.8, 0.05)


def descriptives(choices, rewards, good):
    stay = choices[1:] == choices[:-1]
    prev_win = rewards[:-1] == 1
    return {
        "accuracy": float((choices == good).mean()),
        "mean_reward": float(rewards.mean()),
        "p_stay": float(stay.mean()),
        "p_win_stay": float(stay[prev_win].mean()) if prev_win.any() else np.nan,
        "p_lose_shift": float((~stay)[~prev_win].mean()) if (~prev_win).any() else np.nan,
    }


def simulate(model, theta, n_trials, seed_seq):
    env = make_env()
    sim = model.simulate(theta, env, n_trials, np.random.default_rng(seed_seq))
    return sim, np.asarray(env.good_arm_history)


def main():
    args = parse_args(__doc__)
    reps, T = (10, 60) if args.quick else (200, 100)
    model = QLearning()

    # (a) the four specification participants
    rows, curves = [], {}
    for k, row in EXAMPLE_PARTICIPANTS.iterrows():
        theta = [row["alpha"], row["beta"]]
        stats, correct = [], np.zeros((reps, T))
        for i, child in enumerate(np.random.SeedSequence([args.seed, int(k)]).spawn(reps)):
            sim, good = simulate(model, theta, T, child)
            correct[i] = sim["choices"] == good
            stats.append(descriptives(sim["choices"], sim["rewards"], good))
        m = pd.DataFrame(stats).mean()
        rows.append({"participant": row["participant_id"], "alpha": theta[0], "beta": theta[1], **m.to_dict(), "replicates": reps})
        curves[f"{row['participant_id']} (alpha={theta[0]}, beta={theta[1]})"] = correct
    table = pd.DataFrame(rows)
    save_table(table, "exp3_individual_differences")
    plot_learning_curves(curves, window=5, save_path=FIGURES / "exp3_participants_learning_curves.png")

    # example latent value trajectory for participant B
    sim, _ = simulate(model, [0.7, 2.0], T, np.random.SeedSequence([args.seed, 99]))
    plot_action_values(sim["values"], sim["choices"], title="Participant B (alpha 0.7, beta 2): action values",
                       save_path=FIGURES / "exp3_action_values_B.png")

    # (b) alpha x beta grid
    alphas, betas = [0.1, 0.3, 0.6, 0.9], [0.5, 1, 2, 4, 8]
    n_cell = 8 if args.quick else 60
    grid = np.zeros((len(alphas), len(betas)))
    grid_rows = []
    for a_i, a in enumerate(alphas):
        for b_i, b in enumerate(betas):
            accs = []
            for child in np.random.SeedSequence([args.seed, 1000 + a_i * 10 + b_i]).spawn(n_cell):
                sim, good = simulate(model, [a, b], T, child)
                accs.append((sim["choices"] == good).mean())
            grid[a_i, b_i] = np.mean(accs)
            grid_rows.append({"alpha": a, "beta": b, "accuracy": grid[a_i, b_i], "n": n_cell})
    save_table(pd.DataFrame(grid_rows), "exp3_alpha_beta_grid")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(betas)), betas)
    ax.set_yticks(range(len(alphas)), alphas)
    for i in range(len(alphas)):
        for j in range(len(betas)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="w" if grid[i, j] < grid.mean() else "k", fontsize=8)
    ax.set(xlabel="inverse temperature beta", ylabel="learning rate alpha", title="Accuracy in a reversal task")
    fig.colorbar(im, ax=ax, label="P(choose good option)")
    fig.tight_layout()
    fig.savefig(FIGURES / "exp3_alpha_beta_grid.png", dpi=150)

    save_config("exp3_individual_differences", {
        "replicates": reps, "n_trials": T, "grid_alphas": alphas, "grid_betas": betas,
        "n_per_grid_cell": n_cell, "env": "ReversalBandit(0.8, 0.05)", "seed": args.seed,
    })
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
