"""Small numerical helpers shared across the package."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

EPS = 1e-12
#: choice index used for a trial on which no choice was made (timeout / missing response). Such trials carry
#: no outcome, so models skip them: no learning, and no contribution to the likelihood.
MISSED = -1


def as_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a ``numpy.random.Generator`` from a seed or an existing generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def softmax(x: ArrayLike, beta: float = 1.0) -> NDArray[np.float64]:
    """Numerically stable softmax with inverse temperature ``beta``.

    ``P(a) = exp(beta * x_a) / sum_b exp(beta * x_b)`` along the last axis.
    """
    z = beta * np.asarray(x, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def masked_softmax(x: ArrayLike, beta: float = 1.0, available: ArrayLike | None = None) -> NDArray[np.float64]:
    """Softmax restricted to the ``available`` actions (unavailable actions get probability exactly 0).

    Identical to :func:`softmax` when ``available`` is ``None``. Raises if no action is available.
    """
    z = beta * np.asarray(x, dtype=float)
    if available is not None:
        avail = np.asarray(available, dtype=bool)
        if avail.shape != z.shape:
            raise ValueError(f"available has shape {avail.shape}, expected {z.shape}")
        if not avail.any(axis=-1).all():
            raise ValueError("at least one action must be available on every trial")
        z = np.where(avail, z, -np.inf)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(x: ArrayLike, beta: float = 1.0) -> NDArray[np.float64]:
    """Numerically stable log-softmax along the last axis."""
    z = beta * np.asarray(x, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def check_stochastic(p: ArrayLike, name: str = "p", atol: float = 1e-8) -> NDArray[np.float64]:
    """Validate that the last axis of ``p`` is a probability distribution."""
    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(p < -atol):
        raise ValueError(f"{name} contains negative probabilities")
    sums = p.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=atol):
        raise ValueError(f"{name} rows must sum to 1 (got sums in [{sums.min():.6g}, {sums.max():.6g}])")
    return p


def md_table(df) -> str:
    """Render a dataframe as a GitHub-flavoured markdown table (no extra dependency)."""
    import pandas as pd

    cols = list(df.columns)
    rows = [["" if pd.isna(v) else str(v) for v in r] for r in df.itertuples(index=False)]
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)
