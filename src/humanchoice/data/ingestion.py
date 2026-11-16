"""Import behavioural data and expose it in the representation used by the models."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from humanchoice.data.validation import (
    AVAILABLE_SEP,
    MISSED_LABEL,
    PreprocessingReport,
    validate_behaviour,
)
from humanchoice.utils import MISSED


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes (recorded in result configs so analyses can be tied to an exact data version)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class BehaviouralDataset:
    """Validated trial-level data grouped by participant.

    Choices are mapped to indices by ``choice_categories`` order (default ``A -> 0``, ``B -> 1``; any number of options),
    which is the representation consumed by every ``ChoiceModel``. Trials without a response map to ``-1``
    (:data:`humanchoice.utils.MISSED`) and are skipped by the models.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        choice_categories: tuple[str, ...] = ("A", "B"),
        report: PreprocessingReport | None = None,
        **validation_kwargs,
    ) -> None:
        self.choice_categories = tuple(str(c) for c in choice_categories)
        if report is None:
            df, report = validate_behaviour(df, choice_categories=self.choice_categories, **validation_kwargs)
        self.df = df
        self.report = report
        self._index = {c: i for i, c in enumerate(self.choice_categories)} | {MISSED_LABEL: MISSED}
        self._groups = {str(pid): g for pid, g in df.groupby("participant_id", sort=False)}
        #: generating parameters (set by :func:`humanchoice.simulation.simulate_dataset` for synthetic data)
        self.truth: pd.DataFrame | None = None
        #: provenance record (seed, model, package version, source file hash, ...)
        self.config: dict = {}

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> BehaviouralDataset:
        ds = cls(pd.read_csv(path), **kwargs)
        ds.config["source_file"] = str(path)
        ds.config["source_sha256"] = file_sha256(path)
        return ds

    def to_csv(self, path: str | Path) -> None:
        self.df.drop(columns=["missed"], errors="ignore").to_csv(path, index=False)

    @property
    def n_actions(self) -> int:
        return len(self.choice_categories)

    @property
    def participants(self) -> list[str]:
        return list(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def get(self, participant_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(choices, rewards)`` for one participant; choices are 0-based indices, ``-1`` = missed trial."""
        g = self._groups[participant_id]
        choices = g["choice"].map(self._index).to_numpy(dtype=np.int64)
        rewards = g["reward"].to_numpy(dtype=np.float64)
        return choices, rewards

    def get_available(self, participant_id: str) -> NDArray[np.bool_] | None:
        """``(T, K)`` boolean availability matrix, or ``None`` if the data have no ``available_actions`` column."""
        if "available_actions" not in self.df.columns:
            return None
        g = self._groups[participant_id]
        avail = np.zeros((len(g), self.n_actions), dtype=bool)
        for t, items in enumerate(g["available_actions"].astype(str)):
            for label in items.split(AVAILABLE_SEP):
                avail[t, self._index[label.strip()]] = True
        return avail

    def __iter__(self):
        for pid in self._groups:
            yield (pid, *self.get(pid))

    def subset(self, participant_ids: list[str]) -> BehaviouralDataset:
        """A dataset restricted to ``participant_ids`` (no re-validation; provenance is carried over)."""
        keep = set(participant_ids)
        sub = BehaviouralDataset(
            self.df[self.df["participant_id"].isin(keep)].reset_index(drop=True),
            self.choice_categories,
            report=self.report,
        )
        sub.config, sub.truth = dict(self.config), self.truth
        return sub

    # ---------------------------------------------------------------- summaries
    def trial_summary(self) -> pd.DataFrame:
        """Trial-level table with choice index, previous *observed* reward and stay indicator (missed trials -> NaN)."""
        d = self.df.copy()
        d["choice_index"] = d["choice"].map(self._index)
        obs = d[d["choice_index"] != MISSED]
        grp = obs.groupby("participant_id", sort=False)
        d["prev_reward"] = grp["reward"].shift(1).reindex(d.index)
        stay = (obs["choice_index"] == grp["choice_index"].shift(1)).astype(float)
        stay[grp["reward"].shift(1).isna()] = np.nan
        d["stay"] = stay.reindex(d.index)
        return d

    def participant_summary(self, win_threshold: float = 0.5) -> pd.DataFrame:
        """Participant-level descriptives: trials, missed trials, choice proportions, reward, stay rates after high / low reward.

        A "win" is ``reward >= win_threshold`` on the preceding observed trial - **the same definition the predictive checks
        use** (``humanchoice.evaluation.predictive_checks``). For binary 0/1 rewards the default 0.5 is the natural win/loss split;
        for continuous rewards ``p_win_stay`` / ``p_lose_stay`` are stay rates after *high* / *low* reward relative to the threshold,
        which should be fixed in advance and reported (it is not a statement that low rewards were "losses").
        """
        t = self.trial_summary()
        rows = []
        for pid, g in t.groupby("participant_id", sort=False):
            observed = g[g["choice_index"] != MISSED]
            valid = observed.dropna(subset=["stay"])
            win, lose = valid[valid["prev_reward"] >= win_threshold], valid[valid["prev_reward"] < win_threshold]
            row = {
                "participant_id": pid,
                "n_trials": len(g),
                "n_missed": int(len(g) - len(observed)),
                "mean_reward": float(observed["reward"].mean()) if len(observed) else np.nan,
                "p_stay": float(valid["stay"].mean()) if len(valid) else np.nan,
                "p_win_stay": float(win["stay"].mean()) if len(win) else np.nan,
                "p_lose_stay": float(lose["stay"].mean()) if len(lose) else np.nan,
            }
            for k, cat in enumerate(self.choice_categories):
                row[f"p_choose_{cat}"] = float((observed["choice_index"] == k).mean()) if len(observed) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def quality_report(self) -> dict:
        """Dataset-level data-quality facts needed before any modelling (counts, missingness, reward distribution)."""
        s = self.participant_summary()
        obs = self.df[self.df["choice"] != MISSED_LABEL]
        n_rows = len(self.df)
        return {
            "n_participants": len(self),
            "n_trials_total": n_rows,
            "trials_per_participant_min": int(s["n_trials"].min()),
            "trials_per_participant_max": int(s["n_trials"].max()),
            "n_missed_total": int(s["n_missed"].sum()),
            "missed_fraction_overall": float(s["n_missed"].sum() / n_rows),
            "missed_fraction_per_participant_max": float((s["n_missed"] / s["n_trials"]).max()),
            "participants_with_any_missed": int((s["n_missed"] > 0).sum()),
            "reward_min": float(obs["reward"].min()),
            "reward_max": float(obs["reward"].max()),
            "reward_mean": float(obs["reward"].mean()),
            "reward_sd": float(obs["reward"].std()),
            "n_distinct_reward_values": int(obs["reward"].nunique()),
            "choice_proportions": {c: float((obs["choice"] == c).mean()) for c in self.choice_categories},
            "choice_categories": list(self.choice_categories),
            "columns": list(self.df.columns),
        }

    def apply_exclusions(
        self, max_missed_fraction: float | None = None, min_valid_trials: int | None = None
    ) -> tuple[BehaviouralDataset, pd.DataFrame]:
        """Apply *pre-specified* participant exclusions; returns the reduced dataset and a table of who was excluded and why.

        Thresholds must be fixed before looking at model fits (an exclusion rule tuned to make a model look good is a
        forking path). Every exclusion is also logged in ``dataset.report``.
        """
        s = self.participant_summary()
        s["valid_trials"] = s["n_trials"] - s["n_missed"]
        reasons = pd.Series("", index=s.index, dtype=object)
        if max_missed_fraction is not None:
            reasons[s["n_missed"] / s["n_trials"] > max_missed_fraction] += f"missed>{max_missed_fraction:.0%};"
        if min_valid_trials is not None:
            reasons[s["valid_trials"] < min_valid_trials] += f"valid_trials<{min_valid_trials};"
        excluded = s[reasons != ""].assign(reason=reasons[reasons != ""])[["participant_id", "n_trials", "n_missed", "valid_trials", "reason"]]
        kept = self.subset([p for p in self.participants if p not in set(excluded["participant_id"])])
        kept.report = PreprocessingReport(self.report.n_rows_in, len(kept.df), len(kept), list(self.report.steps))
        kept.report.log(
            f"excluded {len(excluded)} of {len(self)} participants by pre-specified rules "
            f"(max_missed_fraction={max_missed_fraction}, min_valid_trials={min_valid_trials})"
        )
        return kept, excluded.reset_index(drop=True)
