"""Experiment 9 - the empirical-analysis pipeline, rehearsed on SYNTHETIC restless-bandit data.

The pipeline (`humanchoice.pipelines.run_empirical_analysis`) is the pre-specified analysis of `docs/analysis_plan.md`.
Here it is run on simulated participants with *known* generating models (30% Q-learning, 20% asymmetric, 30%
perseverative, 20% lapsing; 4 options; 150 trials; 3% timeouts) so that every step - missed-trial handling, model
ladder, held-out prediction, calibration, predictive checks, profile intervals - can be audited against the truth
before any human data are touched. NOTHING here is a finding about human behaviour.

To run the same analysis on real data:  humanchoice fit --spec configs/<your_dataset>.json --out results/<name>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import RESULTS, parse_args, save_config

from humanchoice.envs import RestlessBandit
from humanchoice.pipelines import run_empirical_analysis, simulate_restless_rehearsal


def main():
    args = parse_args(__doc__)
    n, T = (8, 60) if args.quick else (60, 150)
    ds = simulate_restless_rehearsal(n_participants=n, n_trials=T, seed=args.seed)
    kw = {"n_starts": 2, "fractions": (0.5, 0.7), "ppc_participants": 8, "ppc_sims": 60, "profile_participants": 2} if args.quick else {}
    out = RESULTS / "empirical_rehearsal"
    res = run_empirical_analysis(ds, out, seed=args.seed, env_factory=lambda: RestlessBandit(4), session_sims=50, title="Empirical-analysis pipeline: synthetic rehearsal",
                                 banner="SYNTHETIC REHEARSAL - simulated participants with known generating models. NOT human data.", **kw)
    save_config("exp9_empirical_rehearsal", {"participants": n, "n_trials": T, "mixture": ds.config["mixture"], "missed_prob": ds.config["missed_prob"],
                                             "seed": args.seed, "output": str(out)})
    print(res["comparison"].round(2).to_string())
    print(res["held_out_summary"].query("train_fraction == 0.7")[["model", "test_ll_per_trial", "ll_diff_vs_chance", "diff_lo", "diff_hi"]].round(3).to_string(index=False))
    print(res["decisions"][["hypothesis", "contrast", "mean_diff", "p_adjusted", "decision", "sensitivity_agree"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
