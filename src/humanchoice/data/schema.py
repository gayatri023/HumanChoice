"""Declarative dataset specifications: how a *raw* file maps onto the canonical schema, with provenance.

A spec is a small JSON document so that an analysis can be tied to (i) the exact file (SHA-256), (ii) every
preprocessing decision, and (iii) pre-specified exclusion rules, all fixed *before* any model is fitted.

Example (see ``configs/``)::

    {
      "name": "my_bandit_study",
      "path": "raw/data.csv",
      "columns": {"participant_id": "subject", "trial": "trial_nr", "choice": "arm", "reward": "points"},
      "choice_categories": ["1", "2", "3", "4"],
      "reward_range": [0, 100],
      "rescale_rewards": true,
      "missing": "mark",
      "exclusions": {"max_missed_fraction": 0.2, "min_valid_trials": 100},
      "provenance": {"source_url": "...", "licence": "...", "citation": "..."}
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from humanchoice.data.ingestion import BehaviouralDataset, file_sha256
from humanchoice.data.validation import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    DataValidationError,
    PreprocessingReport,
)

PLACEHOLDER = "VERIFY"
#: provenance a non-synthetic dataset must state (or explain as ``UNAVAILABLE: <documented reason>``)
REQUIRED_PROVENANCE = ("source_url", "citation", "licence")


@dataclass
class DatasetSpec:
    """Mapping from a raw CSV to the canonical schema, plus pre-specified cleaning and provenance."""

    name: str
    path: str
    columns: dict[str, str]
    choice_categories: list[str]
    reward_range: tuple[float, float] | None = None
    rescale_rewards: bool = False
    reward_values: list[float] | None = None
    missing: str = "mark"
    sort: bool = True
    allow_gaps: bool = False
    explode_blocks: bool = False
    exclusions: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    sha256: str | None = None
    synthetic: bool = False

    # ------------------------------------------------------------------ io
    @classmethod
    def from_json(cls, path: str | Path) -> DatasetSpec:
        path = Path(path)
        raw = json.loads(path.read_text())
        raw.pop("_comment", None)
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise DataValidationError(f"unknown keys in dataset spec: {sorted(unknown)}")
        spec = cls(**raw)
        if spec.reward_range is not None:
            spec.reward_range = (float(spec.reward_range[0]), float(spec.reward_range[1]))
        spec.path = str((path.parent / spec.path).resolve()) if not Path(spec.path).is_absolute() else spec.path
        spec.check()
        return spec

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def check(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.columns]
        if missing:
            raise DataValidationError(f"spec.columns must map {list(REQUIRED_COLUMNS)}; missing {missing}")
        bad = set(self.columns) - set(REQUIRED_COLUMNS) - set(OPTIONAL_COLUMNS)
        if bad:
            raise DataValidationError(f"spec.columns has unknown canonical names {sorted(bad)}")
        unknown = set(self.exclusions) - {"max_missed_fraction", "min_valid_trials"}
        if unknown:
            raise DataValidationError(f"unknown exclusion rules {sorted(unknown)}")

    @property
    def unverified_fields(self) -> list[str]:
        """Provenance fields that block analysis: marked ``VERIFY``, **or missing / empty** for a non-synthetic dataset.

        A required field (``source_url``, ``citation``, ``licence``) may be given as ``UNAVAILABLE: <documented reason>``.
        Synthetic datasets (``synthetic: true``) are exempt from the missing-field rule.
        """
        bad = [k for k, v in self.provenance.items() if PLACEHOLDER in str(v).upper()]
        if not self.synthetic:
            for k in REQUIRED_PROVENANCE:
                v = str(self.provenance.get(k, "")).strip()
                if not v or v.upper().startswith("UNAVAILABLE") and len(v.partition(":")[2].strip()) < 10:
                    bad.append(k)
        return list(dict.fromkeys(bad))


@dataclass
class LoadedData:
    """Result of :func:`load_dataset`: the cleaned dataset plus everything needed to document it."""

    dataset: BehaviouralDataset
    spec: DatasetSpec
    sha256: str
    excluded: pd.DataFrame
    raw_report: PreprocessingReport


def load_dataset(spec: DatasetSpec | str | Path, verify_hash: bool = True) -> LoadedData:
    """Load, validate and clean the dataset described by ``spec`` (a :class:`DatasetSpec` or a JSON path)."""
    if not isinstance(spec, DatasetSpec):
        spec = DatasetSpec.from_json(spec)
    path = Path(spec.path)
    if not path.exists():
        raise DataValidationError(
            f"data file not found: {path}. Real datasets are not bundled; download the file from the source listed in "
            f"the spec's provenance and point 'path' at it."
        )
    digest = file_sha256(path)
    if verify_hash and spec.sha256 and spec.sha256.lower() != digest:
        raise DataValidationError(f"SHA-256 mismatch for {path.name}: spec has {spec.sha256}, file has {digest}")
    raw = pd.read_csv(path)
    absent = [src for src in spec.columns.values() if src not in raw.columns]
    if absent:
        raise DataValidationError(f"columns {absent} named in the spec are not in the file (found: {list(raw.columns)})")
    renamed = raw.rename(columns={src: canon for canon, src in spec.columns.items()})
    ds = BehaviouralDataset(
        renamed,
        choice_categories=tuple(spec.choice_categories),
        reward_values=None if spec.reward_values is None else tuple(spec.reward_values),
        missing=spec.missing,
        sort=spec.sort,
        allow_gaps=spec.allow_gaps,
        reward_range=spec.reward_range,
        rescale_rewards=spec.rescale_rewards,
        explode_blocks=spec.explode_blocks,
    )
    raw_report = ds.report
    excluded = pd.DataFrame(columns=["participant_id", "n_trials", "n_missed", "valid_trials", "reason"])
    if spec.exclusions:
        ds, excluded = ds.apply_exclusions(
            max_missed_fraction=spec.exclusions.get("max_missed_fraction"),
            min_valid_trials=None if spec.exclusions.get("min_valid_trials") is None else int(spec.exclusions["min_valid_trials"]),
        )
    ds.config.update(
        {
            "dataset": spec.name,
            "source_file": str(path),
            "source_sha256": digest,
            "provenance": spec.provenance,
            "synthetic": spec.synthetic,
        }
    )
    return LoadedData(ds, spec, digest, excluded, raw_report)
