"""Sensitivity analyses declared in ``docs/analysis_plan.md`` §8: does any main conclusion change under a reasonable alternative?

Each *variant* re-runs the primary held-out contrasts (single 70 % split, Holm-adjusted; see
:func:`humanchoice.comparison.planned_contrasts`) after changing exactly one analysis choice:

* ``win_threshold``  0.4 / 0.6 (WSLS and stay statistics; baseline 0.5)
* ``starts``         10 optimiser starts (baseline 4)
* ``kalman_obs_sd``  0.05 / 0.2 (baseline 0.1)
* ``reward_range``   *reward-magnitude scaling* of the [0, 1] rewards (x 1/1.1, x 1/0.9), mimicking a mis-stated range; **not applicable to binary
                     rewards** (scaling 0/1 outcomes would break the Bernoulli observation model), where it is reported as such
* ``max_missed``     stricter exclusion of participants with > 10 % missed trials (baseline: the spec's rule)

The output records, for every variant and hypothesis, the verdict and effect, and whether the verdict differs from the baseline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from humanchoice.comparison import planned_contrasts, repeated_temporal_splits
from humanchoice.data import BehaviouralDataset
from humanchoice.models import KalmanBandit
from humanchoice.pipelines.empirical import default_ladder, environment_info
from humanchoice.utils import md_table

DEFAULT_VARIANTS: tuple[dict, ...] = (
    {"name": "baseline"},
    {"name": "win_threshold=0.4", "win_threshold": 0.4},
    {"name": "win_threshold=0.6", "win_threshold": 0.6},
    {"name": "starts=10", "n_starts": 10},
    {"name": "kalman_obs_sd=0.05", "kalman_obs_sd": 0.05},
    {"name": "kalman_obs_sd=0.2", "kalman_obs_sd": 0.2},
    {"name": "reward_range+10%", "reward_scale": 1 / 1.1},
    {"name": "reward_range-10%", "reward_scale": 1 / 0.9},
    {"name": "exclude_missed>10%", "max_missed": 0.10},
)


def _variant_dataset(dataset: BehaviouralDataset, variant: dict) -> BehaviouralDataset:
    ds = dataset
    if "max_missed" in variant:
        ds, _ = ds.apply_exclusions(max_missed_fraction=variant["max_missed"])
    if "reward_scale" in variant:
        df = ds.df.drop(columns="missed", errors="ignore").copy()
        obs = df["choice"] != "<missed>"
        df.loc[obs, "reward"] = df.loc[obs, "reward"] * variant["reward_scale"]
        scaled = BehaviouralDataset(df, ds.choice_categories, reward_values=None, missing="mark")
        scaled.config = dict(ds.config)
        ds = scaled
    return ds


def run_sensitivity_analyses(
    dataset: BehaviouralDataset,
    outdir: str | Path | None = None,
    variants: tuple[dict, ...] = DEFAULT_VARIANTS,
    n_starts: int = 4,
    primary_fraction: float = 0.7,
    win_threshold: float = 0.5,
    n_jobs: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Run every variant; returns a long table (variant x hypothesis) with the verdict and whether it changed vs the baseline."""
    binary = bool(dataset.df.loc[dataset.df["choice"] != "<missed>", "reward"].isin([0.0, 1.0]).all())
    rows = []
    for variant in variants:
        if "reward_scale" in variant and binary:
            rows.append(pd.DataFrame([{"variant": variant["name"], "hypothesis": "(all)", "decision": "not applicable (binary rewards)",
                                       "applicable": False, "n_participants_variant": len(dataset)}]))
            continue
        ds = _variant_dataset(dataset, variant)
        thr = variant.get("win_threshold", win_threshold)
        ladder = default_ladder(ds.n_actions, binary, thr)
        if "kalman" in ladder:
            ladder["kalman"] = KalmanBandit(ds.n_actions, obs_sd=variant.get("kalman_obs_sd", 0.1))
        held = repeated_temporal_splits(ladder, ds, (primary_fraction,), variant.get("n_starts", n_starts), seed, n_jobs=n_jobs)
        _, dec = planned_contrasts(held, primary_fraction=primary_fraction, seed=seed)
        dec.insert(0, "variant", variant["name"])
        dec["n_participants_variant"] = len(ds)
        dec["applicable"] = True
        rows.append(dec)
    out = pd.concat(rows, ignore_index=True)
    base = out[out["variant"] == variants[0]["name"]].set_index("hypothesis")["decision"]
    out["baseline_decision"] = out["hypothesis"].map(base)
    out["decision_changed"] = out["applicable"] & (out["decision"] != out["baseline_decision"])
    if outdir is not None:
        d = Path(outdir)
        d.mkdir(parents=True, exist_ok=True)
        out.to_csv(d / "sensitivity_results.csv", index=False)
        table = out.pivot(index="hypothesis", columns="variant", values="decision")[[v["name"] for v in variants]].reset_index()
        n_changed = int(out["decision_changed"].sum())
        n_compared = int((out["applicable"] & (out["variant"] != variants[0]["name"])).sum())
        (d / "sensitivity_summary.md").write_text(
            "# Sensitivity analyses\n\nVerdict of each pre-specified contrast (primary split, Holm-adjusted for the primary family) under each "
            f"declared alternative. **{n_changed} of {n_compared} variant verdicts differ from the baseline.**\n\n" + md_table(table) + "\n"
        )
        (d / "sensitivity_config.json").write_text(json.dumps({
            "variants": list(variants), "n_starts": n_starts, "primary_fraction": primary_fraction, "win_threshold": win_threshold, "seed": seed,
            "n_participants": len(dataset), "binary_rewards": binary, "dataset_sha256": dataset.config.get("source_sha256"),
            "environment": environment_info(), "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2, default=str))
    return out
