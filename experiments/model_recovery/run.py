"""Experiment 5 - model recovery.

Generate datasets from every candidate model, fit every candidate to every dataset, select the winner (AIC or BIC)
and tabulate the confusion matrix. A model is only *identifiable* if data it generates are best explained by it.
Both ``P(selected | generating)`` (the confusion matrix) and ``P(generating | selected)`` (the inversion matrix,
which is what matters when interpreting a winning model on real data) are reported.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from common import FIGURES, RESULTS, md_table, parse_args, save_config, save_table

from humanchoice.envs import ReversalBandit
from humanchoice.evaluation.visualization import plot_confusion_matrix
from humanchoice.models import AsymmetricQLearning, BayesBanditModel, POMDPReversalModel, QLearning
from humanchoice.recovery import run_model_recovery, summarise_model_recovery


def make_env():
    return ReversalBandit(0.8, 0.08)


MODELS = {
    "q_learning": QLearning(),
    "asym_q": AsymmetricQLearning(),
    "bamdp": BayesBanditModel(horizon=5),
    "pomdp": POMDPReversalModel(),
}


def main():
    args = parse_args(__doc__)
    counts, n, starts = ([60], 4, 2) if args.quick else ([100, 300], 30, 3)
    names = list(MODELS)
    summary_rows, md = [], ["# Experiment 5 - model recovery (SYNTHETIC data)\n"]
    for T in counts:
        print(f"model recovery with {T} trials per dataset ...", flush=True)
        res = run_model_recovery(MODELS, make_env, n_per_model=n, n_trials=T, seed=args.seed, criterion="bic", n_starts=starts)
        save_table(res.fits, f"exp5_model_recovery_fits_T{T}")
        for crit in ("bic", "aic"):
            r = summarise_model_recovery(res.fits, names, crit)
            r.confusion.round(3).to_csv(RESULTS / f"exp5_confusion_{crit}_T{T}.csv")
            r.inversion.round(3).to_csv(RESULTS / f"exp5_inversion_{crit}_T{T}.csv")
            plot_confusion_matrix(r.confusion, title=f"Model recovery ({crit.upper()}, {T} trials)",
                                  save_path=FIGURES / f"exp5_confusion_{crit}_T{T}.png")
            plot_confusion_matrix(r.inversion, title=f"P(generating | selected) ({crit.upper()}, {T} trials)",
                                  xlabel="selected model", ylabel="generating model",
                                  save_path=FIGURES / f"exp5_inversion_{crit}_T{T}.png")
            summary_rows.append({"n_trials": T, "criterion": crit, "accuracy": round(r.accuracy, 3), "chance": round(1 / len(names), 3),
                                 "datasets": int(r.counts.to_numpy().sum())})
            md += [f"\n## {crit.upper()}, {T} trials per dataset - accuracy {r.accuracy:.2f} (chance {1 / len(names):.2f})\n",
                   "Confusion matrix P(selected | generating):\n", md_table(r.confusion.round(2).reset_index()), "\n",
                   "Inversion matrix P(generating | selected):\n", md_table(r.inversion.round(2).reset_index()), "\n"]
    summary = pd.DataFrame(summary_rows)
    save_table(summary, "exp5_model_recovery_summary")
    (RESULTS / "exp5_model_recovery_summary.md").write_text("\n".join(md) + "\n")
    save_config("exp5_model_recovery", {"models": names, "trial_counts": counts, "datasets_per_model": n, "n_starts": starts,
                                        "env": "ReversalBandit(p_good=0.8, hazard=0.08)", "seed": args.seed})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
