"""Parameter tables for synthetic participants."""

from __future__ import annotations

import pandas as pd

from humanchoice.models.base import ChoiceModel
from humanchoice.utils import as_rng

#: The four illustrative configurations from the project specification (experimental settings, not psychological types).
EXAMPLE_PARTICIPANTS = pd.DataFrame(
    {
        "participant_id": ["A", "B", "C", "D"],
        "alpha": [0.20, 0.70, 0.40, 0.40],
        "beta": [2.0, 2.0, 0.5, 5.0],
    }
)


def sample_participants(model: ChoiceModel, n: int, seed: int | None = None, prefix: str = "S") -> pd.DataFrame:
    """Draw ``n`` synthetic participants with parameters uniform over ``model.sample_ranges``."""
    rng = as_rng(seed)
    rows = [model.sample_params(rng) for _ in range(n)]
    table = pd.DataFrame(rows, columns=list(model.param_names))
    table.insert(0, "participant_id", [f"{prefix}{i + 1:03d}" for i in range(n)])
    return table
