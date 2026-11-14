"""Experiment 1 - learning-rate sensitivity.

Vary alpha, hold beta fixed, and measure how quickly Q-learners adapt to reward feedback in (a) a stationary
bandit and (b) a reversal bandit. Prediction: large alpha learns fast but is noisy; small alpha learns slowly
but is stable, so the best alpha depends on how volatile the environment is.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from common import FIGURES, parse_args, save_config, save_table

from humanchoice.envs import BernoulliBandit, ReversalBandit
from humanchoice.evaluation.visualization import plot_learning_curves
from humanchoice.models import QLearning

ALPHAS = [0.05, 0.1, 0.2, 0.4, 0.7, 0.95]
BETA = 5.0


def run_condition(label, env_factory, n_participants, n_trials, seed):
    model, rows, curves = QLearning(), [], {}
    for k, alpha in enumerate(ALPHAS):
        children = np.random.SeedSequence([seed, k]).spawn(n_participants)
        correct = np.zeros((n_participants, n_trials))
        reward = np.zeros(n_participants)
        for i, child in enumerate(children):
            env = env_factory()
            sim = model.simulate([alpha, BETA], env, n_trials, np.random.default_rng(child))
            correct[i] = sim["choices"] == np.asarray(env.good_arm_history)
            reward[i] = sim["rewards"].mean()
        curves[f"alpha = {alpha}"] = correct
        rows.append(
            {
                "environment": label,
                "alpha": alpha,
                "beta": BETA,
                "acc_first_20": correct[:, :20].mean(),
                "acc_last_40": correct[:, -40:].mean(),
                "mean_reward": reward.mean(),
                "n_participants": n_participants,
            }
        )
    return curves, rows


def main():
    args = parse_args(__doc__)
    n, T = (20, 60) if args.quick else (300, 120)
    rows = []
    for label, factory in {
        "stationary": lambda: BernoulliBandit((0.8, 0.2)),
        "reversal (hazard 0.05)": lambda: ReversalBandit(0.8, 0.05),
    }.items():
        curves, r = run_condition(label, factory, n, T, args.seed)
        rows += r
        tag = label.split()[0]
        plot_learning_curves(curves, window=5, ylabel="P(choose better option)",
                             save_path=FIGURES / f"exp1_learning_rate_{tag}.png")
    table = pd.DataFrame(rows)
    save_table(table, "exp1_learning_rate")
    save_config("exp1_learning_rate", {"alphas": ALPHAS, "beta": BETA, "n_participants": n, "n_trials": T, "seed": args.seed})
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
