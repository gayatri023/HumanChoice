"""Experiment 11 - model recovery for the ladder on a 4-option restless bandit (SYNTHETIC).

Candidates (fitted): random, WSLS, Q-learning, asymmetric Q, perseverative Q, lapse Q, Kalman filter.
Generators: every candidate PLUS a deliberately mis-specified `full_q` (asymmetric + perseverative + lapsing) that no
candidate reproduces exactly - the realistic situation in which the true process is not among the models considered.

Reports P(selected | generating) and P(generating | selected) with Wilson 95% intervals on recovery accuracy, under BIC
and AIC. Nested pairs (q vs asym_q / sticky_q / lapse_q) are expected to be hard to separate; this quantifies how hard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from common import FIGURES, RESULTS, load_env_factory, md_table, parse_args, save_config, save_table

from humanchoice.envs import RestlessBandit
from humanchoice.evaluation.visualization import plot_confusion_matrix
from humanchoice.models import (
    AsymmetricQLearning,
    FullQLearning,
    KalmanBandit,
    LapseQLearning,
    QLearning,
    RandomChoice,
    StickyQLearning,
    WinStayLoseShift,
)
from humanchoice.models.base import ChoiceModel
from humanchoice.recovery import recovery_accuracy_table, run_model_recovery, summarise_model_recovery
from humanchoice.utils import as_rng, masked_softmax

K = 4


class BiasedQ(QLearning):
    """Generator only: Q-learning with an unmodelled fixed preference (logit bonus) for option 0 - an action bias no candidate has."""

    name = "biased_q (mis-specified)"

    def __init__(self, n_actions=4):
        super().__init__(n_actions)
        self.param_names = (*self.param_names, "bias")
        self.bounds = (*self.bounds, (-1.5, 1.5))
        self.sample_ranges = (*self.sample_ranges, (0.4, 1.2))

    def choice_probs(self, params, state, available=None):
        logits = params[1] * np.asarray(state, dtype=float)
        logits[0] += params[2]
        return masked_softmax(logits, 1.0, available)

    def log_likelihood(self, params, choices, rewards, available=None):  # never fitted here; use the generic path
        return ChoiceModel.log_likelihood(self, params, choices, rewards, available)


class MixtureOfStrategies:
    """Generator only: each simulated participant uses a randomly chosen strategy (WSLS, Q-learning or perseverative Q-learning)."""

    param_names = ("strategy_u",)

    def __init__(self, n_actions=4):
        self.n_actions = n_actions
        self.parts = [WinStayLoseShift(n_actions), QLearning(n_actions), StickyQLearning(n_actions)]

    def sample_params(self, rng):
        return np.array([as_rng(rng).uniform()])

    def simulate(self, theta, env, n_trials, rng=None, **kwargs):
        rng = as_rng(rng)
        part = self.parts[min(int(theta[0] * len(self.parts)), len(self.parts) - 1)]
        return part.simulate(part.sample_params(rng), env, n_trials, rng, **kwargs)


def main():
    def extra(ap):
        ap.add_argument("--datasets", type=int, default=None, help="datasets per generator (default 20; use 200 for the headline run)")
        ap.add_argument("--env", default=None, help="'package.module:callable' = TASK-MATCHED environment")
        ap.add_argument("--condition", default=None, help="reward-drift schedule passed to the environment factory")
        ap.add_argument("--trials", type=int, default=150, help="trials per dataset of the task-matched design")
        ap.add_argument("--missed-prob", type=float, default=0.0, help="timeout probability per trial of the task-matched design")

    args = parse_args(__doc__, extra)
    n, T, starts = (3, 60, 2) if args.quick else (20, args.trials, 3)
    n = args.datasets or n
    task_env = load_env_factory(args.env, args.condition)
    K = task_env().n_actions if task_env is not None else 4
    make_env = task_env if task_env is not None else (lambda: RestlessBandit(K))
    candidates = {"random": RandomChoice(K), "wsls": WinStayLoseShift(K), "q": QLearning(K), "asym_q": AsymmetricQLearning(K),
                  "sticky_q": StickyQLearning(K), "lapse_q": LapseQLearning(K), "kalman": KalmanBandit(K)}
    generators = {**candidates, "full_q (mis-specified)": FullQLearning(K), "biased_q (mis-specified)": BiasedQ(K),
                  "mixture (mis-specified)": MixtureOfStrategies(K)}
    res = run_model_recovery(candidates, make_env, n_per_model=n, n_trials=T, seed=args.seed, criterion="bic",
                             n_starts=starts, generators=generators, missed_prob=args.missed_prob)
    save_table(res.fits, "exp11_model_recovery_ladder_fits")
    md = ["# Experiment 11 - model recovery on the ladder (SYNTHETIC data)\n", f"{n} datasets per generator, {T} trials, {K} options.\n"]
    rows = []
    for crit in ("bic", "aic"):
        r = summarise_model_recovery(res.fits, list(candidates), crit, generator_names=list(generators))
        acc = recovery_accuracy_table(r)
        acc.insert(0, "criterion", crit)
        rows.append(acc)
        r.confusion.round(3).to_csv(RESULTS / f"exp11_confusion_{crit}.csv")
        r.inversion.round(3).to_csv(RESULTS / f"exp11_inversion_{crit}.csv")
        plot_confusion_matrix(r.confusion, title=f"Model recovery ({crit.upper()}, {T} trials)", save_path=FIGURES / f"exp11_confusion_{crit}.png")
        md += [f"\n## {crit.upper()}: overall accuracy over generators that are candidates = {r.accuracy:.2f} (chance {1 / len(candidates):.2f})\n",
               "P(selected | generating):\n", md_table(r.confusion.round(2).reset_index()), "\n",
               "P(generating | selected):\n", md_table(r.inversion.round(2).reset_index()), "\n",
               "Per-generator accuracy with Wilson 95% interval:\n", md_table(acc.round(3)), "\n"]
    save_table(pd.concat(rows, ignore_index=True), "exp11_recovery_accuracy")
    (RESULTS / "exp11_model_recovery_ladder_summary.md").write_text("\n".join(md) + "\n")
    save_config("exp11_model_recovery_ladder", {"candidates": list(candidates), "generators": list(generators), "datasets_per_generator": n, "task_env": args.env, "task_condition": args.condition, "missed_prob": args.missed_prob,
                                                "n_trials": T, "n_starts": starts, "seed": args.seed})
    print(pd.concat(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
