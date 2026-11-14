"""Experiments 6 and 7 - model comparison and behavioural prediction.

A *synthetic* mixed population with known generating models (so the comparison can be audited against the truth):
equal numbers of Q-learners, POMDP hidden-state trackers and BAMDP planners.

Exp 6: fit all candidate models to every participant; compare AIC / BIC wins and summed criteria.
Exp 7: fit on the first 30 / 50 / 70 % of each participant's trials and predict the rest (held-out
log-likelihood per trial and choice-prediction accuracy), with a chance baseline.

Information criteria complement, and do not replace, held-out predictive validation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from common import FIGURES, RESULTS, md_table, parse_args, save_config, save_table

from humanchoice.comparison import compare_models, held_out_evaluation
from humanchoice.data import BehaviouralDataset
from humanchoice.envs import ReversalBandit
from humanchoice.evaluation.visualization import plot_model_comparison, plot_predictive_performance
from humanchoice.models import AsymmetricQLearning, BayesBanditModel, POMDPReversalModel, QLearning
from humanchoice.simulation import sample_participants, simulate_dataset


def make_env():
    return ReversalBandit(0.8, 0.08)


GENERATORS = {"Q": QLearning(), "P": POMDPReversalModel(), "B": BayesBanditModel(horizon=5)}
TRUTH_NAME = {"Q": "q_learning", "P": "pomdp", "B": "bamdp"}
CANDIDATES = {"q_learning": QLearning(), "asym_q": AsymmetricQLearning(), "bamdp": BayesBanditModel(horizon=5), "pomdp": POMDPReversalModel()}


def main():
    args = parse_args(__doc__)
    n_each, T, starts, fracs = (3, 80, 2, [0.5]) if args.quick else (10, 200, 3, [0.3, 0.5, 0.7])

    frames, truth = [], {}
    for k, (prefix, gen) in enumerate(GENERATORS.items()):
        ds = simulate_dataset(gen, sample_participants(gen, n_each, seed=args.seed + k, prefix=prefix), make_env, T, seed=args.seed + 10 * k)
        frames.append(ds.df)
        truth.update({pid: TRUTH_NAME[prefix] for pid in ds.participants})
    population = BehaviouralDataset(pd.concat(frames, ignore_index=True))

    print("Exp 6: fitting all candidates ...", flush=True)
    per, summary = compare_models(CANDIDATES, population, n_starts=starts, seed=args.seed)
    per["generating_model"] = per["participant_id"].map(truth)
    save_table(per, "exp6_model_comparison_per_participant")
    save_table(summary.reset_index(), "exp6_model_comparison_summary")
    winners = {}
    for crit in ("aic", "bic"):
        w = per.loc[per.groupby("participant_id")[crit].idxmin()]
        w = w.assign(correct=w["model"] == w["generating_model"])
        winners[crit] = w
        print(f"{crit.upper()}: winner equals generating model for {w['correct'].mean():.2f} of participants")
    plot_model_comparison(summary, save_path=FIGURES / "exp6_model_comparison.png")
    wins = pd.concat([w.groupby(["generating_model", "model"]).size().rename(crit.upper()) for crit, w in winners.items()], axis=1).fillna(0).astype(int)
    save_table(wins.reset_index(), "exp6_winners_by_generating_model")

    print("Exp 7: held-out prediction ...", flush=True)
    rows = []
    for f in fracs:
        ho = held_out_evaluation(CANDIDATES, population, train_fraction=f, n_starts=starts, seed=args.seed)
        ho["train_fraction"] = f
        rows.append(ho)
        plot_predictive_performance(ho, save_path=FIGURES / f"exp7_predictive_performance_train{int(f * 100)}.png")
    held = pd.concat(rows, ignore_index=True)
    held["generating_model"] = held["participant_id"].map(truth)
    save_table(held, "exp7_held_out")
    agg = held.groupby(["train_fraction", "model"]).agg(test_ll_per_trial=("test_ll_per_trial", "mean"), test_accuracy=("test_accuracy", "mean")).reset_index()
    save_table(agg, "exp7_held_out_summary")
    # pooled scores over a mixed population hide who wins for whom: break the held-out score down by generating model
    by_gen = (held[held["model"] != "chance"].groupby(["train_fraction", "generating_model", "model"])
              .agg(test_ll_per_trial=("test_ll_per_trial", "mean"), test_accuracy=("test_accuracy", "mean")).reset_index())
    best = by_gen.loc[by_gen.groupby(["train_fraction", "generating_model"])["test_ll_per_trial"].idxmax(), ["train_fraction", "generating_model", "model"]]
    best = best.rename(columns={"model": "best_held_out_model"})
    save_table(by_gen, "exp7_held_out_by_generating_model")
    save_table(best, "exp7_best_held_out_model_by_generator")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for m, g in agg.groupby("model"):
        ax.plot(g["train_fraction"], g["test_ll_per_trial"], marker="o", label=m, color="grey" if m == "chance" else None,
                ls=":" if m == "chance" else "-")
    ax.set(xlabel="fraction of trials used for fitting", ylabel="held-out log-likelihood per trial")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "exp7_prediction_vs_training_size.png", dpi=150)

    md = ["# Experiments 6 & 7 - model comparison and prediction (SYNTHETIC data)\n",
          f"Population: {n_each} Q-learners, {n_each} POMDP trackers, {n_each} BAMDP planners; {T} trials each.\n",
          "## Summary over participants\n", md_table(summary.round(2).reset_index()), "\n",
          "## Winner by generating model (counts)\n", md_table(wins.reset_index()), "\n",
          "## Held-out prediction, pooled over the mixed population (hides who wins for whom)\n", md_table(agg.round(3)), "\n",
          "## Held-out prediction by generating model\n", md_table(by_gen.round(3)), "\n",
          "## Best held-out model for each generating model\n", md_table(best)]
    (RESULTS / "exp6_exp7_summary.md").write_text("\n".join(md) + "\n")
    save_config("exp6_exp7_model_comparison", {"participants_per_generator": n_each, "n_trials": T, "train_fractions": fracs, "n_starts": starts,
                                               "candidates": list(CANDIDATES), "seed": args.seed})
    print(summary.round(2).to_string())
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
