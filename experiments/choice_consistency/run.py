"""Experiment 2 - choice consistency (inverse temperature).

Vary beta, hold alpha fixed, and measure (i) how often the better option is chosen and (ii) how deterministic
the agent's own policy is (mean probability of its most likely action). Both rise with beta; accuracy
saturates because a deterministic agent can still be wrong when its values are.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import FIGURES, parse_args, save_config, save_table

from humanchoice.envs import BernoulliBandit
from humanchoice.models import QLearning

BETAS = [0.25, 0.5, 1, 2, 4, 8, 16]
ALPHA = 0.3


def main():
    args = parse_args(__doc__)
    n, T = (20, 60) if args.quick else (300, 100)
    model, rows = QLearning(), []
    for k, beta in enumerate(BETAS):
        acc, cons, rew = [], [], []
        for child in np.random.SeedSequence([args.seed, k]).spawn(n):
            env = BernoulliBandit((0.8, 0.2))
            sim = model.simulate([ALPHA, beta], env, T, np.random.default_rng(child))
            acc.append((sim["choices"][-50:] == 0).mean())
            cons.append(sim["probs"].max(axis=1).mean())
            rew.append(sim["rewards"].mean())
        rows.append(
            {
                "alpha": ALPHA,
                "beta": beta,
                "p_better_late": np.mean(acc),
                "sem": np.std(acc, ddof=1) / np.sqrt(n),
                "policy_determinism": np.mean(cons),
                "mean_reward": np.mean(rew),
                "n_participants": n,
            }
        )
    table = pd.DataFrame(rows)
    save_table(table, "exp2_choice_consistency")
    save_config("exp2_choice_consistency", {"betas": BETAS, "alpha": ALPHA, "n_participants": n, "n_trials": T, "seed": args.seed})

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.errorbar(table["beta"], table["p_better_late"], yerr=1.96 * table["sem"], fmt="o-",
                label="P(choose better option), last 50 trials")
    ax.plot(table["beta"], table["policy_determinism"], "s--", label="policy determinism (mean max P)")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.set(xscale="log", xlabel="inverse temperature beta", ylabel="probability", ylim=(0.4, 1.02),
           title=f"Choice consistency (alpha = {ALPHA})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "exp2_choice_consistency.png", dpi=150)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
