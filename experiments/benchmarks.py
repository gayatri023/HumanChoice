"""Benchmark: simulation and fitting cost per model at a realistic task size (4 options, 150 trials, 3% timeouts).

Run on an otherwise idle machine (timings are wall-clock, single process). The extrapolation to a ~1000-participant
dataset assumes one fit per participant per model; the full pipeline additionally refits each model at every held-out split
point (4 by default), so multiply by ~5 for `humanchoice fit`. Use `--jobs` to divide by the number of cores.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from common import RESULTS, md_table, parse_args, save_config, save_table

from humanchoice.envs import RestlessBandit
from humanchoice.fitting import fit_model
from humanchoice.pipelines import default_ladder

N_PARTICIPANTS_REAL = 975


def main():
    args = parse_args(__doc__)
    reps = 3 if args.quick else 15
    rows = []
    for key, model in default_ladder(4, binary_rewards=False).items():
        t_sim, t_fit, t_ll = [], [], []
        for i in range(reps):
            th = model.sample_params(i) if model.n_params else []
            t0 = time.perf_counter()
            sim = model.simulate(th, RestlessBandit(4), 150, i, missed_prob=0.03)
            t_sim.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            model.log_likelihood(th, sim["choices"], sim["rewards"])
            t_ll.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            fit_model(model, sim["choices"], sim["rewards"], n_starts=4, seed=i)
            t_fit.append(time.perf_counter() - t0)
        rows.append({"model": key, "n_params": model.n_params, "simulate_ms": 1e3 * np.median(t_sim), "likelihood_ms": 1e3 * np.median(t_ll),
                     "fit_ms_4_starts": 1e3 * np.median(t_fit),
                     f"est_minutes_{N_PARTICIPANTS_REAL}_participants_one_fit": np.median(t_fit) * N_PARTICIPANTS_REAL / 60})
    df = pd.DataFrame(rows)
    total = df[f"est_minutes_{N_PARTICIPANTS_REAL}_participants_one_fit"].sum()
    save_table(df, "benchmarks")
    (RESULTS / "benchmarks.md").write_text(
        "# Benchmarks (single process, 4 options, 150 trials, 3% timeouts)\n\n" + md_table(df.round(2)) +
        f"\n\nWhole ladder, one fit per participant for {N_PARTICIPANTS_REAL} participants: about {total:.0f} CPU-minutes; "
        "the full pipeline refits at 4 split points, so roughly 5x that, divided by `--jobs`.\n")
    save_config("benchmarks", {"repetitions": reps, "n_trials": 150, "n_options": 4, "participants_extrapolated": N_PARTICIPANTS_REAL})
    print(df.round(2).to_string(index=False), f"\nwhole ladder x {N_PARTICIPANTS_REAL} participants: ~{total:.0f} CPU-min (one fit each)")


if __name__ == "__main__":
    main()
