import numpy as np
import pandas as pd
import pytest

from humanchoice.data import BehaviouralDataset, DataValidationError, validate_behaviour


def good_df():
    return pd.DataFrame(
        {
            "participant_id": ["P01", "P01", "P01", "P02", "P02"],
            "trial": [1, 2, 3, 1, 2],
            "choice": ["A", "B", "A", "B", "B"],
            "reward": [1, 0, 1, 1, 1],
        }
    )


def test_valid_data_roundtrip(tmp_path):
    ds = BehaviouralDataset(good_df())
    assert ds.participants == ["P01", "P02"]
    c, r = ds.get("P01")
    assert c.tolist() == [0, 1, 0] and r.tolist() == [1.0, 0.0, 1.0]
    f = tmp_path / "d.csv"
    ds.to_csv(f)
    assert len(BehaviouralDataset.from_csv(f)) == 2
    assert "required columns present" in str(ds.report)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.drop(columns="reward"), "missing required columns"),
        (lambda d: d.iloc[0:0], "empty"),
        (lambda d: d.assign(choice=["A", "C", "A", "B", "B"]), "invalid choice"),
        (lambda d: d.assign(reward=[1, 0, 2, 1, 1]), "reward values"),
        (lambda d: d.assign(participant_id=["P01", None, "P01", "P02", "P02"]), "participant_id"),
        (lambda d: d.assign(trial=[1, 2, 2, 1, 2]), "duplicate"),
        (lambda d: d.assign(trial=[1, 3, 2, 1, 2]), "increasing order"),
        (lambda d: d.assign(trial=[1, 2, 4, 1, 2]), "consecutive"),
        (lambda d: d.assign(trial=[0, 1, 2, 1, 2]), ">= 1"),
        (lambda d: d.assign(reward=[1, np.nan, 1, 1, 1]), "missing"),
    ],
)
def test_invalid_data_is_rejected(mutate, match):
    with pytest.raises(DataValidationError, match=match):
        validate_behaviour(mutate(good_df()))


def test_missing_rows_can_be_dropped_and_are_reported():
    df = good_df().assign(reward=[1, np.nan, 1, 1, 1])
    clean, report = validate_behaviour(df, missing="drop", allow_gaps=True)
    assert len(clean) == 4
    assert any("dropped 1 rows" in s for s in report.steps)


def test_sort_repairs_order_and_is_reported():
    df = good_df().assign(trial=[1, 3, 2, 1, 2])
    clean, report = validate_behaviour(df, sort=True)
    assert clean[clean.participant_id == "P01"]["trial"].tolist() == [1, 2, 3]
    assert any("sorted" in s for s in report.steps)


def test_summaries():
    ds = BehaviouralDataset(good_df())
    t = ds.trial_summary()
    assert np.isnan(t.loc[0, "stay"]) and t.loc[1, "stay"] == 0.0 and t.loc[4, "stay"] == 1.0
    s = ds.participant_summary().set_index("participant_id")
    assert s.loc["P01", "n_trials"] == 3
    assert np.isclose(s.loc["P01", "mean_reward"], 2 / 3)
    assert np.isclose(s.loc["P02", "p_win_stay"], 1.0)
