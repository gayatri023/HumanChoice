"""Experiment 4 - parameter recovery.

For each model: draw generating parameters, simulate participants, refit by maximum likelihood and compare.
Repeated across trial counts and stratified by noise level (low / mid / high inverse temperature).

Poor recovery is REPORTED, not hidden: any (model, parameter, trial count) with Pearson r < 0.6 is flagged in
``results/exp4_parameter_recovery_summary.md``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from common import FIGURES, RESULTS, md_table, parse_args, save_config, save_table

from humanchoice.envs import ReversalBandit
from humanchoice.evaluation.visualization import plot_parameter_recovery, plot_recovery_vs_trials
from humanchoice.models import AsymmetricQLearning, BayesBanditModel, POMDPReversalModel, QLearning
from humanchoice.recovery import recovery_metrics, run_recovery_grid, stratified_recovery_metrics


def make_env():
    return ReversalBandit(0.8, 0.08)


MODELS = {
    "q_learning": QLearning(),
    "asymmetric_q_learning": AsymmetricQLearning(),
    "pomdp_reversal": POMDPReversalModel(),
    "bamdp_bandit": BayesBanditModel(horizon=5),
}
POOR_R = 0.6


def main():
    args = parse_args(__doc__)
    counts, n, starts = ([40, 100], 8, 2) if args.quick else ([50, 100, 200, 400], 60, 4)
    all_metrics, all_strat = [], []
    for name, model in MODELS.items():
        print(f"recovering {name} ...", flush=True)
        df = run_recovery_grid(model, make_env, counts, n_participants=n, seed=args.seed, n_starts=starts)
        save_table(df, f"exp4_parameter_recovery_{name}")
        m = recovery_metrics(df, model.param_names, ci=True)
        m.insert(0, "model", name)
        all_metrics.append(m)
        if "beta" in model.param_names:
            s = stratified_recovery_metrics(df, model.param_names, by="beta", n_bins=3, ci=False)
            s.insert(0, "model", name)
            all_strat.append(s)
        ref = counts[min(2, len(counts) - 1)]
        plot_parameter_recovery(df, model.param_names, n_trials=ref, save_path=FIGURES / f"exp4_recovery_scatter_{name}.png")
        plot_recovery_vs_trials(m, "r", save_path=FIGURES / f"exp4_recovery_vs_trials_{name}.png")

    metrics = pd.concat(all_metrics, ignore_index=True)
    strat = pd.concat(all_strat, ignore_index=True)
    save_table(metrics, "exp4_parameter_recovery_metrics")
    save_table(strat, "exp4_parameter_recovery_by_noise_level")
    save_config("exp4_parameter_recovery", {
        "models": list(MODELS), "trial_counts": counts, "participants_per_cell": n, "n_starts": starts,
        "env": "ReversalBandit(p_good=0.8, hazard=0.08)", "poor_recovery_threshold_r": POOR_R, "seed": args.seed,
    })

    cols = ["model", "param", "n_trials", "r", "r_ci_lo", "r_ci_hi", "rho", "median_ae", "rmse", "bias", "frac_at_bound"]
    lines = ["# Experiment 4 - parameter recovery summary (SYNTHETIC data)\n",
             ("r = Pearson correlation (95% bootstrap CI) between generating and recovered parameters; "
              "rho = Spearman rank correlation; median_ae = median absolute error. Beta estimates have heavy-tailed "
              "errors (a few fits run to the upper bound), which depress r and inflate rmse/bias much more than "
              "rho and median_ae - compare them.\n"),
             md_table(metrics[cols].round(3)), "\n"]
    poor = metrics[metrics["r"] < POOR_R]
    lines.append(f"\n## Poorly recovered (r < {POOR_R}) - reported, not hidden\n")
    lines.append(md_table(poor[cols].round(3)) if len(poor) else "None.")
    lines.append("\n## Recovery by noise level (inverse-temperature tercile)\n")
    scols = ["model", "n_trials", "stratum", "by_min", "by_max", "param", "r", "rho", "median_ae", "rmse", "bias"]
    lines.append(md_table(strat[scols].round(3)))
    (RESULTS / "exp4_parameter_recovery_summary.md").write_text("\n".join(lines) + "\n")
    print(metrics[cols].round(3).to_string(index=False))
    print(f"\n{len(poor)} of {len(metrics)} (model, parameter, trial-count) cells below r = {POOR_R}")


if __name__ == "__main__":
    main()
