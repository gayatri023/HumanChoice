"""Shared helpers for the experiment scripts: paths, CLI flags and reproducibility records.

Every experiment writes
  * ``results/<name>*.csv``      the numbers behind every figure,
  * ``results/<name>_config.json`` seeds, sizes and library versions (so a run can be repeated exactly),
  * ``figures/<name>*.png``      the figures.

All data produced here are SYNTHETIC. They are never to be reported as empirical human findings.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import humanchoice
from humanchoice.utils import md_table  # noqa: F401  (re-exported for the experiment scripts)

ROOT = Path(__file__).resolve().parents[1]
# `--quick` smoke runs write to a git-ignored scratch folder so they can never overwrite the reported results.
_OUT = ROOT / "_quick_output" if "--quick" in sys.argv else ROOT
RESULTS = _OUT / "results"
FIGURES = _OUT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def parse_args(description: str, extra=None) -> argparse.Namespace:
    """Common flags (``--quick``, ``--seed``); ``extra(ap)`` may add experiment-specific ones (e.g. ``--participants``)."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--quick", action="store_true", help="tiny sizes for a smoke test (numbers are NOT the reported ones)")
    ap.add_argument("--seed", type=int, default=2026, help="master random seed")
    if extra is not None:
        extra(ap)
    return ap.parse_args()


def save_config(name: str, config: dict) -> None:
    """Write the full configuration of an experiment next to its results."""
    record = {
        "experiment": name,
        "synthetic_data": True,
        "package_version": humanchoice.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **config,
    }
    (RESULTS / f"{name}_config.json").write_text(json.dumps(record, indent=2, default=str))


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    return path



def load_env_factory(spec: str | None, condition: str | None = None):
    """Resolve ``package.module:callable`` to a zero-argument environment factory (the task-matched environment).

    If the callable has a ``condition`` / ``schedule`` parameter it receives ``condition`` (the reward-drift schedule).
    """
    if not spec:
        return None
    import functools
    import importlib

    from humanchoice.pipelines import call_env_factory

    module, _, name = spec.partition(":")
    return functools.partial(call_env_factory, getattr(importlib.import_module(module), name), condition)
