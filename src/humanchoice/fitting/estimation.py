"""Maximum-likelihood parameter estimation with multi-start optimisation and diagnostics."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from humanchoice.data.ingestion import BehaviouralDataset
from humanchoice.evaluation.metrics import aic, bic
from humanchoice.fitting.likelihood import FAILURE_NLL, make_objective
from humanchoice.models.base import ChoiceModel
from humanchoice.utils import MISSED, as_rng


@dataclass
class FitResult:
    """Outcome of fitting one model to one participant."""

    model: str
    params: dict[str, float]
    nll: float
    n_obs: int  # observed (non-missed) choices
    n_params: int
    aic: float
    bic: float
    converged: bool
    n_starts: int
    start_agreement: float  # fraction of starts reaching (almost) the best NLL; low values flag a rugged surface
    at_bound: tuple[str, ...] = field(default_factory=tuple)  # parameters sitting on a bound (weak identifiability)
    message: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        d.update({f"fit_{k}": v for k, v in d.pop("params").items()})
        d["at_bound"] = ",".join(self.at_bound)
        return d


def check_compatible(model: ChoiceModel, dataset: BehaviouralDataset) -> None:
    """Raise a clear error if ``model`` cannot be applied to ``dataset`` (option count or reward type)."""
    if model.n_actions != dataset.n_actions:
        raise ValueError(f"{model.name} has n_actions={model.n_actions} but the dataset has {dataset.n_actions} options")
    if model.reward_support == "binary":
        obs = dataset.df[dataset.df["choice"] != "<missed>"]["reward"]
        if not obs.isin([0.0, 1.0]).all():
            raise ValueError(f"{model.name} requires binary rewards but the dataset has other reward values")


def fit_model(
    model: ChoiceModel,
    choices: ArrayLike,
    rewards: ArrayLike,
    n_starts: int = 5,
    seed: int | np.random.Generator | None = 0,
    x0: ArrayLike | None = None,
    bound_tol: float = 1e-3,
    available: ArrayLike | None = None,
) -> FitResult:
    """Fit ``model`` by maximum likelihood (L-BFGS-B, bounded, multi-start).

    Starting points are drawn from ``model.sample_ranges`` (plus ``x0`` if given). Convergence information
    includes whether the best optimiser run reported success, how many starts agreed on the optimum, and
    which parameters ended on a bound. Missed trials (``-1``) are ignored. Parameter-free models (e.g. random
    choice) are simply evaluated.
    """
    choices = np.asarray(choices, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    n = int((choices != MISSED).sum())
    if n == 0:
        raise ValueError("no observed choices to fit")
    if model.n_params == 0:
        ll = model.log_likelihood([], choices, rewards, available)
        return FitResult(model.name, {}, -ll, n, 0, aic(ll, 0), bic(ll, 0, n), True, 0, 1.0, (), "no free parameters")
    rng = as_rng(seed)
    obj = make_objective(model, choices, rewards, available)
    starts = [model.sample_params(rng) for _ in range(n_starts)]
    if x0 is not None:
        starts[0] = model.to_array(x0)
    nested = model.nested_start(choices, rewards, available, rng)
    if nested is not None:  # a model can never fit worse than the simpler model it contains
        starts.append(np.clip(nested, [b[0] for b in model.bounds], [b[1] for b in model.bounds]))

    runs = []
    for s in starts:
        res = minimize(obj, s, method="L-BFGS-B", bounds=list(model.bounds))
        runs.append(res)
    best = min(runs, key=lambda r: r.fun)
    if best.fun >= FAILURE_NLL:
        raise RuntimeError(f"{model.name}: likelihood could not be evaluated at any starting point")
    agreement = float(np.mean([r.fun <= best.fun + 1e-2 for r in runs]))
    lo = np.array([b[0] for b in model.bounds])
    hi = np.array([b[1] for b in model.bounds])
    span = hi - lo
    at_bound = tuple(
        nme for nme, v, lw, h, sp in zip(model.param_names, best.x, lo, hi, span) if min(v - lw, h - v) < bound_tol * sp
    )
    ll = -best.fun
    return FitResult(
        model=model.name,
        params=model.to_dict(best.x),
        nll=float(best.fun),
        n_obs=n,
        n_params=model.n_params,
        aic=aic(ll, model.n_params),
        bic=bic(ll, model.n_params, n),
        converged=bool(best.success),
        n_starts=len(starts),
        start_agreement=agreement,
        at_bound=at_bound,
        message=str(best.message),
    )


def _fit_participant(args):
    model, pid, choices, rewards, available, n_starts, seed = args
    return {"participant_id": pid, **fit_model(model, choices, rewards, n_starts=n_starts, seed=seed, available=available).to_row()}


def fit_dataset(
    model: ChoiceModel, dataset: BehaviouralDataset, n_starts: int = 5, seed: int = 0, n_jobs: int = 1
) -> pd.DataFrame:
    """Participant-level fits: one row per participant with parameters, NLL, AIC, BIC and diagnostics.

    ``n_jobs > 1`` fits participants in parallel processes (on platforms that spawn processes, call this from
    inside an ``if __name__ == "__main__":`` block).
    """
    check_compatible(model, dataset)
    jobs = [(model, pid, c, r, dataset.get_available(pid), n_starts, seed) for pid, c, r in dataset]
    if n_jobs == 1:
        rows = [_fit_participant(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            rows = list(pool.map(_fit_participant, jobs, chunksize=max(1, len(jobs) // (4 * n_jobs))))
    return pd.DataFrame(rows)
