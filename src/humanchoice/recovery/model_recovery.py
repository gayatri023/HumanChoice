"""Model recovery: can the pipeline tell which model generated the data?"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from humanchoice.fitting.estimation import fit_model
from humanchoice.models.base import ChoiceModel


@dataclass
class ModelRecoveryResult:
    """Long-format fits plus confusion matrices."""

    fits: pd.DataFrame  # one row per (generating model, dataset, fitted model)
    confusion: pd.DataFrame  # rows: generating model; columns: selected model; entries: row proportions
    counts: pd.DataFrame  # same, raw counts
    inversion: pd.DataFrame  # P(generating model | selected model) assuming equal prior
    accuracy: float
    criterion: str


def run_model_recovery(
    models: dict[str, ChoiceModel],
    env_factory: Callable[[], object],
    n_per_model: int = 20,
    n_trials: int = 100,
    seed: int = 0,
    criterion: str = "bic",
    n_starts: int = 3,
    generators: dict[str, ChoiceModel] | None = None,
    missed_prob: float = 0.0,
) -> ModelRecoveryResult:
    """Generate data from every generator, fit every candidate model to every dataset and select by ``criterion``.

    ``models`` are the *candidates* (fitted). ``generators`` (default: the same models) are the data-generating
    processes; supplying different ones tests robustness to mis-specification, e.g. lapse-contaminated or
    perseverative generators that no candidate exactly matches. The confusion matrix is then rectangular
    (generators x candidates). ``missed_prob`` adds timeouts to the generated data so recovery matches the task's missingness.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError("criterion must be 'aic' or 'bic'")
    names = list(models)
    gens = generators if generators is not None else models
    ss = np.random.SeedSequence(seed)
    rows = []
    for gen in gens:
        draw_rng = np.random.default_rng(ss.spawn(1)[0])
        sim_seeds = ss.spawn(n_per_model)
        for i in range(n_per_model):
            theta = gens[gen].sample_params(draw_rng)
            sim = gens[gen].simulate(theta, env_factory(), n_trials, np.random.default_rng(sim_seeds[i]), missed_prob=missed_prob)
            for cand in names:
                f = fit_model(models[cand], sim["choices"], sim["rewards"], n_starts=n_starts, seed=seed + i)
                rows.append(
                    {"generating": gen, "dataset": i, "fitted": cand, "nll": f.nll, "aic": f.aic, "bic": f.bic,
                     "converged": f.converged}
                )
    return summarise_model_recovery(pd.DataFrame(rows), names, criterion, generator_names=list(gens))


def summarise_model_recovery(
    fits: pd.DataFrame, names: list[str], criterion: str = "bic", generator_names: list[str] | None = None
) -> ModelRecoveryResult:
    """Select the best model per dataset under ``criterion`` and build the confusion matrices.

    ``fits`` is the long-format table from :func:`run_model_recovery`; re-summarising the same fits under a
    different criterion (AIC vs BIC) needs no refitting. ``accuracy`` is computed over generators that are themselves
    candidates (a contaminated generator has no "correct" answer).
    """
    if criterion not in ("aic", "bic"):
        raise ValueError("criterion must be 'aic' or 'bic'")
    gen_names = list(generator_names) if generator_names is not None else list(names)
    winners = fits.loc[fits.groupby(["generating", "dataset"])[criterion].idxmin()]
    counts = pd.crosstab(winners["generating"], winners["fitted"]).reindex(index=gen_names, columns=names, fill_value=0)
    confusion = counts.div(counts.sum(axis=1), axis=0)
    inversion = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    matched = [g for g in gen_names if g in names]
    total = counts.loc[matched].to_numpy().sum()
    correct = 0
    for name in matched:
        correct += int(counts.loc[name, name])
    accuracy = float(correct / total) if total else float("nan")
    return ModelRecoveryResult(fits, confusion, counts, inversion, accuracy, criterion)


def recovery_accuracy_table(result: ModelRecoveryResult, level: float = 0.95) -> pd.DataFrame:
    """Per-generator recovery accuracy with Wilson score intervals (a handful of datasets gives wide intervals)."""
    from humanchoice.evaluation.metrics import wilson_interval

    rows = []
    for g in result.counts.index:
        n = int(result.counts.loc[g].sum())
        k = int(result.counts.loc[g, g]) if g in result.counts.columns else 0
        lo, hi = wilson_interval(k, n, level)
        rows.append({"generating": g, "n_datasets": n, "n_correct": k, "accuracy": k / n if n else np.nan,
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)
