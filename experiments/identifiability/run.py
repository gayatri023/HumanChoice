"""Experiment 10 - identifiability of the model ladder on a 4-option restless bandit (SYNTHETIC).

Parameter recovery for the Q-learning family and the Kalman learner under realistic conditions (4 options, continuous
rewards, 3% timeouts), then three sensitivity sweeps:

  (a) number of trials     75 / 150 / 300
  (b) environmental volatility   drift SD 0.01 / 0.028 / 0.06 per trial
  (c) number of options    K = 2 / 4 / 8

and two confound diagnostics: correlation of estimation errors between parameters, and *leakage* of a true parameter
into the estimate of another (e.g. does perseveration masquerade as a learning rate? does lapsing masquerade as noise?).
Poor recovery is reported, not hidden: any cell with Pearson r < 0.6 is flagged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from common import FIGURES, RESULTS, load_env_factory, md_table, parse_args, save_config, save_table

from humanchoice.envs import RestlessBandit
from humanchoice.evaluation.visualization import plot_error_correlations, plot_parameter_recovery
from humanchoice.models import AsymmetricQLearning, KalmanBandit, LapseQLearning, QLearning, StickyQLearning
from humanchoice.recovery import (
    cross_parameter_leakage,
    error_correlations,
    recovery_gate_report,
    recovery_metrics,
    run_recovery_grid,
    stratified_recovery_metrics,
)

MISSED = 0.03
POOR_R = 0.6


def models_for(K):
    return {"q": QLearning(K), "asym_q": AsymmetricQLearning(K), "sticky_q": StickyQLearning(K), "lapse_q": LapseQLearning(K),
            "kalman": KalmanBandit(K)}


def main():
    def extra(ap):
        ap.add_argument("--participants", type=int, default=None, help="simulated participants per cell (default 40; use 200 for the headline run)")
        ap.add_argument("--env", default=None, help="'package.module:callable' = TASK-MATCHED environment; runs only the headline condition in it")
        ap.add_argument("--condition", default=None, help="reward-drift schedule passed to the environment factory")
        ap.add_argument("--trials", type=int, default=150, help="trials per participant of the task-matched design")
        ap.add_argument("--missed-prob", type=float, default=MISSED, help="timeout probability per trial of the task-matched design")

    args = parse_args(__doc__, extra)
    missed = args.missed_prob
    task_env = load_env_factory(args.env, args.condition)
    n = args.participants or (6 if args.quick else 40)
    if args.quick and args.env:
        args.trials = min(args.trials, 60)
    starts = 2 if args.quick else 4
    conditions = []  # (label, K, drift_sd, trial_counts)
    if task_env is not None:  # task-matched: ONE condition, in the supplied environment, with its trial count and timeout rate
        K_task = task_env().n_actions
        conditions.append(("trials", K_task, None, [args.trials]))
    else:
        base_T = [40, 80] if args.quick else [75, 150, 300]
        conditions.append(("trials", 4, 0.028, base_T))
        for sd in ([0.01] if args.quick else [0.01, 0.06]):
            conditions.append((f"volatility drift_sd={sd}", 4, sd, [150 if not args.quick else 80]))
        for K in ([2] if args.quick else [2, 8]):
            conditions.append((f"options K={K}", K, 0.028, [150 if not args.quick else 80]))

    metric_frames, strat_frames, raw = [], [], {}
    for label, K, sd, counts in conditions:
        for name, model in models_for(K).items():
            if name == "kalman" and K != 4 and not label.startswith("trials"):
                continue  # Kalman cost: only the headline condition and the volatility sweep
            print(f"[{label}] {name} ...", flush=True)
            make_env = task_env if task_env is not None else (lambda K=K, sd=sd: RestlessBandit(K, drift_sd=sd))
            df = run_recovery_grid(model, make_env, counts, n_participants=n, seed=args.seed, n_starts=starts, missed_prob=missed)
            df.insert(0, "condition", label)
            df.insert(1, "model", name)
            raw[(label, name)] = df
            m = recovery_metrics(df, model.param_names, ci=True)
            m.insert(0, "condition", label)
            m.insert(1, "model", name)
            metric_frames.append(m)
            if "beta" in model.param_names and label == "trials":
                s = stratified_recovery_metrics(df, model.param_names, by="beta", n_bins=3, ci=False)
                s.insert(0, "model", name)
                strat_frames.append(s)
    metrics = pd.concat(metric_frames, ignore_index=True)
    GATE = 0.7
    metrics["gate_point_estimate"] = metrics["rho"] >= GATE  # the original rule: rho >= 0.7
    metrics["gate_lower_ci"] = metrics["rho_ci_lo"] >= GATE  # the stricter rule: the lower 95% bootstrap bound must reach 0.7
    save_table(metrics, "exp10_identifiability_metrics")
    save_table(pd.concat(strat_frames, ignore_index=True), "exp10_identifiability_by_noise_level")
    save_table(pd.concat(raw.values(), ignore_index=True), "exp10_identifiability_raw")

    # confounds at the headline condition (4 options, 150 trials)
    ref_T = conditions[0][3][-1] if task_env is not None else (80 if args.quick else 150)
    K_ref = conditions[0][1]
    lines = ["# Experiment 10 - identifiability (SYNTHETIC data)\n",
             f"{n} simulated participants per cell" + (f" in the task-matched environment `{args.env}` (condition {args.condition}), {ref_T} trials" if task_env else "") + f" (rho and r with 95% bootstrap CIs); gate = rho >= {GATE} (point estimate) or its lower CI bound >= {GATE} (strict); 4 options unless stated; {missed:.0%} timeouts; {starts} optimiser starts (+ nested warm start).\n"]
    for name in ("sticky_q", "lapse_q", "asym_q", "kalman"):
        model = models_for(K_ref)[name]
        df = raw[("trials", name)]
        ec = error_correlations(df, model.param_names, n_trials=ref_T)
        lk = cross_parameter_leakage(df, model.param_names, n_trials=ref_T)
        ec.round(3).to_csv(RESULTS / f"exp10_error_correlation_{name}.csv")
        lk.round(3).to_csv(RESULTS / f"exp10_leakage_{name}.csv")
        plot_error_correlations(ec, f"{name}: correlation of estimation errors (T={ref_T})", FIGURES / f"exp10_error_corr_{name}.png")
        plot_parameter_recovery(df, model.param_names, n_trials=ref_T, save_path=FIGURES / f"exp10_recovery_{name}.png")
        lines += [f"## {name}: error correlations and leakage at T={ref_T}\n", "Correlation of estimation errors:\n", md_table(ec.round(2).reset_index()),
                  "\nSpearman(true parameter row, error in parameter column) - off-diagonal = leakage:\n", md_table(lk.round(2).reset_index()), ""]
    gate = recovery_gate_report(metrics, gate=GATE, n_trials=ref_T, condition="trials")  # machine-readable: which parameters may be interpreted
    save_table(gate, "exp10_recovery_gate")
    (RESULTS / "exp10_recovery_gate.json").write_text(gate.to_json(orient="records", indent=2))
    cols = ["condition", "model", "param", "n_trials", "rho", "rho_ci_lo", "rho_ci_hi", "r", "r_ci_lo", "r_ci_hi", "median_ae", "rmse", "bias",
            "frac_at_bound", "gate_point_estimate", "gate_lower_ci"]
    lines += ["## Recovery metrics\n", md_table(metrics[cols].round(3)), "\n"]
    poor = metrics[metrics["r"] < POOR_R]
    lines += [f"## Poorly recovered (r < {POOR_R}) - reported, not hidden\n", md_table(poor[cols].round(3)) if len(poor) else "None.", ""]
    (RESULTS / "exp10_identifiability_summary.md").write_text("\n".join(lines))
    save_config("exp10_identifiability", {"participants_per_cell": n, "missed_prob": missed, "task_env": args.env, "task_condition": args.condition, "conditions": [c[0] for c in conditions], "gate_rho": GATE,
                                          "n_starts": starts, "seed": args.seed, "poor_threshold_r": POOR_R})
    print(f"{len(poor)} of {len(metrics)} cells below r = {POOR_R}")


if __name__ == "__main__":
    main()
