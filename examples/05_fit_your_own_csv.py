"""Fit the library's models to YOUR behavioural CSV.

Expected columns: participant_id, trial, choice, reward   (choice in {A, B}; reward in {0, 1}; trials 1..N per participant)

Run:  python examples/05_fit_your_own_csv.py path/to/data.csv
"""

import sys

from humanchoice.comparison import compare_models, held_out_evaluation
from humanchoice.data import BehaviouralDataset, DataValidationError
from humanchoice.models import AsymmetricQLearning, POMDPReversalModel, QLearning

if len(sys.argv) != 2:
    sys.exit(__doc__)
try:
    data = BehaviouralDataset.from_csv(sys.argv[1], missing="drop", sort=True)  # every repair is logged below
except DataValidationError as err:
    sys.exit(f"Invalid data: {err}")

print(data.report)  # exactly what was checked and what (if anything) was changed
models = {"q_learning": QLearning(), "asym_q": AsymmetricQLearning(), "pomdp": POMDPReversalModel()}
per, summary = compare_models(models, data, n_starts=5)
print(summary.round(2).to_string())
held = held_out_evaluation(models, data, train_fraction=0.7)
print(held.groupby("model")[["test_ll_per_trial", "test_accuracy"]].mean().round(3).to_string())
