"""Regenerate the example dataset in data/synthetic/ (4 synthetic participants from the specification, 300 trials each).

Run:  python examples/00_make_synthetic_dataset.py
The files are SYNTHETIC: the generating (alpha, beta) of every participant is stored in the *_truth.csv file.
"""

from pathlib import Path

from humanchoice.envs import ReversalBandit
from humanchoice.models import QLearning
from humanchoice.simulation import EXAMPLE_PARTICIPANTS, save_dataset, simulate_dataset

out = Path(__file__).resolve().parents[1] / "data" / "synthetic"
data = simulate_dataset(QLearning(), EXAMPLE_PARTICIPANTS, lambda: ReversalBandit(p_good=0.8, hazard=0.08), n_trials=300, seed=2026)
save_dataset(data, out, "example_q_learning", extra_config={"environment": "ReversalBandit(p_good=0.8, hazard=0.08)"})
print("wrote", sorted(p.name for p in out.glob("example_q_learning*")))
