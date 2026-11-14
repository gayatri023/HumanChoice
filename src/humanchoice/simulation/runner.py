"""Simulate synthetic behavioural datasets with known generating parameters."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

import humanchoice
from humanchoice.data.ingestion import BehaviouralDataset
from humanchoice.models.base import ChoiceModel


def default_categories(n_actions: int) -> tuple[str, ...]:
    """Option labels ``A, B, C, ...`` (up to 26 options)."""
    if not 2 <= n_actions <= 26:
        raise ValueError("n_actions must be between 2 and 26")
    return tuple(chr(ord("A") + k) for k in range(n_actions))


def simulate_dataset(
    model: ChoiceModel,
    param_table: pd.DataFrame,
    env_factory: Callable[[], object],
    n_trials: int,
    seed: int = 0,
    choice_categories: tuple[str, ...] | None = None,
    missed_prob: float = 0.0,
) -> BehaviouralDataset:
    """Simulate one participant per row of ``param_table``.

    Each participant gets an independent random stream spawned from ``seed`` so results are reproducible
    and do not depend on participant order. The generating parameters are stored in ``dataset.truth``.
    """
    if choice_categories is None:
        choice_categories = default_categories(model.n_actions)
    if len(choice_categories) != model.n_actions:
        raise ValueError(f"{len(choice_categories)} choice categories but the model has n_actions={model.n_actions}")
    children = np.random.SeedSequence(seed).spawn(len(param_table))
    frames = []
    env = None
    for (_, row), child in zip(param_table.iterrows(), children):
        params = [row[n] for n in model.param_names]
        env = env_factory()
        sim = model.simulate(params, env, n_trials, np.random.default_rng(child), missed_prob=missed_prob)
        frames.append(
            pd.DataFrame(
                {
                    "participant_id": row["participant_id"],
                    "trial": np.arange(1, n_trials + 1),
                    "choice": [choice_categories[c] if c >= 0 else np.nan for c in sim["choices"]],
                    "reward": np.where(sim["choices"] >= 0, sim["rewards"], np.nan),
                }
            )
        )
    long = pd.concat(frames, ignore_index=True)
    binary = bool(long["reward"].dropna().isin([0.0, 1.0]).all())
    ds = BehaviouralDataset(
        long,
        choice_categories=choice_categories,
        reward_values=(0, 1) if binary else None,
        missing="mark" if missed_prob > 0 else "error",
    )
    ds.truth = param_table.reset_index(drop=True)
    ds.config = {
        "package_version": humanchoice.__version__,
        "model": model.name,
        "n_trials": n_trials,
        "seed": seed,
        "n_participants": len(param_table),
        "n_actions": model.n_actions,
        "missed_prob": missed_prob,
        "environment": type(env).__name__,
    }
    return ds


def save_dataset(dataset: BehaviouralDataset, directory: str | Path, name: str, extra_config: dict | None = None) -> None:
    """Write ``<name>.csv`` (behaviour), ``<name>_truth.csv`` (generating parameters) and ``<name>_config.json``."""
    if dataset.truth is None:
        raise ValueError("save_dataset needs a synthetic dataset with known generating parameters (dataset.truth)")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(directory / f"{name}.csv")
    dataset.truth.to_csv(directory / f"{name}_truth.csv", index=False)
    cfg = {**dataset.config, **(extra_config or {}), "synthetic": True}
    (directory / f"{name}_config.json").write_text(json.dumps(cfg, indent=2))
