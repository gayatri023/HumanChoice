"""Validation and transparent cleaning of trial-level behavioural data.

Canonical long format, one row per trial::

    participant_id, trial, choice, reward                      (required)
    block, condition, reaction_time, available_actions         (optional; retained when present)

Nothing is coerced silently: every repair (dropping / marking missed trials, sorting, rescaling rewards, splitting
blocks) is an explicit option and is written to the returned :class:`PreprocessingReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("participant_id", "trial", "choice", "reward")
OPTIONAL_COLUMNS = ("block", "condition", "reaction_time", "available_actions")
#: label given to a trial with no response (timeout); it maps to the action index ``-1`` and is skipped by the models
MISSED_LABEL = "<missed>"
AVAILABLE_SEP = "|"


class DataValidationError(ValueError):
    """Raised when a behavioural dataset violates the expected schema."""


@dataclass
class PreprocessingReport:
    """Human-readable record of every check and every modification applied to a dataset."""

    n_rows_in: int = 0
    n_rows_out: int = 0
    n_participants: int = 0
    steps: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.steps.append(message)

    def __str__(self) -> str:
        head = f"{self.n_rows_in} rows in -> {self.n_rows_out} rows out, {self.n_participants} participants"
        return "\n".join([head, *[f"  - {s}" for s in self.steps]])


def normalise_labels(series: pd.Series) -> pd.Series:
    """Stringify choice labels; integer-valued floats (``1.0``, a CSV artefact when NaNs are present) become ``"1"``."""
    out = series.copy()
    if out.dtype.kind == "f":
        notna = out.notna()
        if (np.mod(out[notna], 1) == 0).all():
            out = out.astype("Int64")
    return out.astype("string").str.strip()


def validate_behaviour(
    df: pd.DataFrame,
    choice_categories: tuple[str, ...] = ("A", "B"),
    reward_values: tuple[float, ...] | None = (0, 1),
    missing: str = "error",
    sort: bool = False,
    allow_gaps: bool = False,
    reward_range: tuple[float, float] | None = None,
    rescale_rewards: bool = False,
    extra_columns: tuple[str, ...] = (),
    explode_blocks: bool = False,
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Validate (and optionally minimally repair) a behavioural dataframe.

    Parameters
    ----------
    choice_categories: permitted values of ``choice`` (any number >= 2).
    reward_values: permitted reward values (``None`` allows any finite number).
    missing: ``"error"`` rejects missing observations; ``"drop"`` removes those rows; ``"mark"`` keeps rows with a
        missing *choice* as missed trials (``choice = "<missed>"``, ``reward = 0``, ``missed = True``) so the trial
        sequence stays intact - the right treatment for timeouts. A missing reward on an observed choice is always an error
        under ``"mark"``.
    sort: sort by ``(participant_id, trial)``; otherwise out-of-order trials are an error.
    allow_gaps: if ``False`` trials must be 1..N for every participant (gaps are an error).
    reward_range: ``(lo, hi)`` that every observed reward must lie within.
    rescale_rewards: map rewards linearly from ``reward_range`` to [0, 1] using the *stated* range (never the data's
        min/max, which would leak information across participants). Requires ``reward_range``.
    extra_columns: additional columns to keep besides the optional schema columns.
    explode_blocks: treat each ``(participant_id, block)`` as its own sequence (id becomes ``"<pid>|<block>"``);
        parameters are then fitted independently per block.
    """
    if missing not in ("error", "drop", "mark"):
        raise ValueError("missing must be 'error', 'drop' or 'mark'")
    if rescale_rewards and reward_range is None:
        raise ValueError("rescale_rewards requires reward_range")
    if len(choice_categories) < 2 or len(set(choice_categories)) != len(choice_categories):
        raise ValueError("choice_categories must contain at least two distinct labels")
    categories = tuple(str(c) for c in choice_categories)
    report = PreprocessingReport(n_rows_in=len(df))
    absent = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if absent:
        raise DataValidationError(f"missing required columns: {absent}")
    if len(df) == 0:
        raise DataValidationError("dataset is empty")
    keep = list(REQUIRED_COLUMNS) + [c for c in (*OPTIONAL_COLUMNS, *extra_columns) if c in df.columns and c not in REQUIRED_COLUMNS]
    d = df[keep].copy()
    report.log("required columns present: " + ", ".join(REQUIRED_COLUMNS))
    if len(keep) > len(REQUIRED_COLUMNS):
        report.log("optional columns retained: " + ", ".join(keep[len(REQUIRED_COLUMNS):]))

    # participants
    if d["participant_id"].isna().any() or (d["participant_id"].astype(str).str.strip() == "").any():
        raise DataValidationError("participant_id contains missing or empty identifiers")
    d["participant_id"] = d["participant_id"].astype(str).str.strip()

    # choices (labels normalised before missing-value handling)
    d["choice"] = normalise_labels(d["choice"])
    # empty strings and the explicit missed-trial label (e.g. a CSV written by this library) both mean "no response"
    d["choice"] = d["choice"].mask((d["choice"] == "") | (d["choice"] == MISSED_LABEL))
    # missing observations
    trial_missing = d["trial"].isna()
    choice_missing = d["choice"].isna()
    reward_missing = d["reward"].isna()
    d["missed"] = False
    if trial_missing.any():
        if missing == "drop":
            d = d[~trial_missing].copy()
            report.log(f"dropped {int(trial_missing.sum())} rows with missing trial index")
            choice_missing, reward_missing = d["choice"].isna(), d["reward"].isna()
        else:
            raise DataValidationError(f"{int(trial_missing.sum())} rows have missing trial/choice/reward (trial index cannot be missing)")
    miss_any = choice_missing | reward_missing
    if miss_any.any():
        if missing == "error":
            raise DataValidationError(f"{int(miss_any.sum())} rows have missing trial/choice/reward (use missing='drop' or 'mark')")
        if missing == "drop":
            d = d[~miss_any].copy()
            report.log(f"dropped {int(miss_any.sum())} rows with missing trial/choice/reward")
        else:  # mark
            bad = reward_missing & ~choice_missing
            if bad.any():
                raise DataValidationError(f"{int(bad.sum())} rows have a choice but a missing reward")
            d.loc[choice_missing, "choice"] = MISSED_LABEL
            d.loc[choice_missing, "reward"] = 0.0
            d.loc[choice_missing, "missed"] = True
            report.log(f"marked {int(choice_missing.sum())} rows without a response as missed trials (no learning, excluded from likelihood)")
    else:
        report.log("no missing observations")

    # trial indices
    try:
        trial = pd.to_numeric(d["trial"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError("trial must be numeric") from exc
    if not np.all(np.equal(np.mod(trial, 1), 0)):
        raise DataValidationError("trial must contain integers")
    d["trial"] = trial.astype(int)
    if (d["trial"] < 1).any():
        raise DataValidationError("trial indices must be >= 1")

    # optional block handling
    if explode_blocks:
        if "block" not in d.columns:
            raise DataValidationError("explode_blocks=True requires a 'block' column")
        if d["block"].isna().any():
            raise DataValidationError("block contains missing values")
        d["participant_id"] = d["participant_id"] + "|" + d["block"].astype(str)
        report.log("treated each (participant, block) as a separate sequence (ids '<participant>|<block>')")
    if d.duplicated(["participant_id", "trial"]).any():
        raise DataValidationError("duplicate (participant_id, trial) pairs")

    # choice categories
    observed_choice = d["choice"] != MISSED_LABEL
    bad_choices = sorted(set(d.loc[observed_choice, "choice"].astype(str)) - set(categories))
    if bad_choices:
        raise DataValidationError(f"invalid choice categories {bad_choices}; allowed: {list(categories)}")
    d["choice"] = d["choice"].astype(str)

    # rewards (observed trials only)
    reward = pd.to_numeric(d["reward"], errors="coerce")
    if reward.isna().any() or not np.all(np.isfinite(reward)):
        raise DataValidationError("reward must be finite numeric")
    obs_reward = reward[observed_choice]
    if reward_values is not None and not obs_reward.isin(reward_values).all():
        raise DataValidationError(f"reward values must lie in {list(reward_values)}")
    if reward_range is not None:
        lo, hi = reward_range
        if not hi > lo:
            raise ValueError("reward_range must satisfy lo < hi")
        if ((obs_reward < lo) | (obs_reward > hi)).any():
            raise DataValidationError(f"rewards outside the stated range [{lo}, {hi}]")
        if rescale_rewards:
            reward = reward.where(~observed_choice, (reward - lo) / (hi - lo))
            report.log(f"rescaled rewards linearly from the stated range [{lo}, {hi}] to [0, 1]")
    d["reward"] = reward.astype(float)

    # available actions
    if "available_actions" in d.columns:
        sets = d["available_actions"].astype("string").str.split(AVAILABLE_SEP)
        for i, (items, ch) in enumerate(zip(sets, d["choice"])):
            if items is pd.NA or items is None:
                raise DataValidationError("available_actions contains missing values (use a full list, or drop the column)")
            items = [x.strip() for x in items]
            unknown = set(items) - set(categories)
            if unknown or not items:
                raise DataValidationError(f"available_actions row {i}: invalid labels {sorted(unknown)}")
            if ch != MISSED_LABEL and ch not in items:
                raise DataValidationError(f"available_actions row {i}: chosen option '{ch}' was not available")
        report.log("available_actions validated (chosen options were always available)")

    # ordering
    ordered = d.groupby("participant_id", sort=False)["trial"].apply(lambda s: bool(s.is_monotonic_increasing))
    if not ordered.all():
        if not sort:
            raise DataValidationError(
                f"trials not in increasing order for participants {list(ordered.index[~ordered])} (use sort=True)"
            )
        report.log("sorted rows by (participant_id, trial)")
    if sort:
        d = d.sort_values(["participant_id", "trial"], kind="stable")
    if not allow_gaps:
        for pid, g in d.groupby("participant_id", sort=False):
            if not np.array_equal(g["trial"].to_numpy(), np.arange(1, len(g) + 1)):
                raise DataValidationError(f"participant {pid}: trials must be consecutive 1..N (allow_gaps=False)")
        report.log("trials are consecutive 1..N for every participant")

    d = d.reset_index(drop=True)
    report.n_rows_out = len(d)
    report.n_participants = d["participant_id"].nunique()
    report.log(f"validated {report.n_participants} participants; choices in {list(categories)}")
    return d, report
