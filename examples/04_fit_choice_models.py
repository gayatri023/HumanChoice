"""Behavioural modelling tutorial: simulate synthetic participants, save/load CSV, fit models, check recovery.

Run:  python examples/04_fit_choice_models.py
All data here are SYNTHETIC; never report them as human findings.
"""

import tempfile
from pathlib import Path

import numpy as np

from humanchoice.comparison import compare_models
from humanchoice.data import BehaviouralDataset
from humanchoice.envs import ReversalBandit
from humanchoice.fitting import fit_dataset
from humanchoice.models import POMDPReversalModel, QLearning
from humanchoice.simulation import EXAMPLE_PARTICIPANTS, save_dataset, simulate_dataset

q = QLearning()
env = lambda: ReversalBandit(p_good=0.8, hazard=0.08)

# 1. simulate the four example participants (alpha, beta) from the specification
data = simulate_dataset(q, EXAMPLE_PARTICIPANTS, env, n_trials=300, seed=7)
with tempfile.TemporaryDirectory() as tmp:
    save_dataset(data, tmp, "example")  # example.csv + example_truth.csv + example_config.json (provenance)
    data = BehaviouralDataset.from_csv(Path(tmp) / "example.csv")  # re-validates on import
print(data.report)
print(data.participant_summary().round(2).to_string(index=False))

# 2. maximum-likelihood fit per participant, next to the truth
fits = fit_dataset(q, data, n_starts=5, seed=0)
table = fits[["participant_id", "fit_alpha", "fit_beta", "converged", "at_bound"]].copy()
table["true_alpha"], table["true_beta"] = EXAMPLE_PARTICIPANTS["alpha"].to_numpy(), EXAMPLE_PARTICIPANTS["beta"].to_numpy()
print("\nrecovered vs generating parameters:\n", table.round(2).to_string(index=False))
print("(a parameter 'at_bound' is weakly identified; large beta is hard to recover from near-deterministic choices)")

# 3. which model explains the data best?
per, summary = compare_models({"q_learning": q, "pomdp": POMDPReversalModel()}, data, n_starts=3)
print("\nmodel comparison (lower AIC/BIC is better):\n", summary.round(1).to_string())
print("mean AIC weights:", np.round(summary["mean_aic_weight"].to_numpy(), 2))
