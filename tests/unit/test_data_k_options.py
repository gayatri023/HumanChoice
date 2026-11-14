"""Data layer for K options: missed trials, blocks, optional columns, rescaling, availability, exclusions, specs."""

import json

import numpy as np
import pandas as pd
import pytest

from humanchoice.data import (
    BehaviouralDataset,
    DatasetSpec,
    DataValidationError,
    file_sha256,
    load_dataset,
    validate_behaviour,
)

CATS = ("1", "2", "3", "4")


def df4():
    return pd.DataFrame(
        {
            "participant_id": ["p1"] * 5 + ["p2"] * 4,
            "trial": [1, 2, 3, 4, 5, 1, 2, 3, 4],
            "choice": [1, 2, 4, np.nan, 3, 2, 2, 1, 4],  # float dtype because of the NaN: labels are "1.0" ...
            "reward": [10, 50, 100, np.nan, 20, 0, 40, 60, 80],
        }
    )


def test_four_options_and_float_labels_are_normalised():
    ds = BehaviouralDataset(df4(), CATS, reward_values=None, missing="mark")
    c, r = ds.get("p1")
    assert c.tolist() == [0, 1, 3, -1, 2]  # "1.0" -> "1" -> index 0; missed trial -> -1
    assert r.tolist() == [10, 50, 100, 0, 20]
    assert ds.n_actions == 4
    assert any("marked 1 rows" in s for s in ds.report.steps)


def test_default_missing_policy_still_rejects_and_drop_still_works():
    with pytest.raises(DataValidationError, match="missing"):
        BehaviouralDataset(df4(), CATS, reward_values=None)
    clean, rep = validate_behaviour(df4(), CATS, reward_values=None, missing="drop", allow_gaps=True)
    assert len(clean) == 8 and any("dropped 1 rows" in s for s in rep.steps)


def test_reward_missing_for_an_observed_choice_is_an_error_even_when_marking():
    d = df4()
    d.loc[1, "reward"] = np.nan
    with pytest.raises(DataValidationError, match="choice but a missing reward"):
        validate_behaviour(d, CATS, reward_values=None, missing="mark")


def test_stay_statistics_skip_missed_trials():
    ds = BehaviouralDataset(df4(), CATS, reward_values=None, missing="mark")
    t = ds.trial_summary().set_index(["participant_id", "trial"])
    assert np.isnan(t.loc[("p1", 4), "stay"])  # the missed trial has no stay value
    # trial 5 (choice 3) is compared with the previous *observed* choice (4, trial 3): a switch
    assert t.loc[("p1", 5), "stay"] == 0.0
    assert t.loc[("p1", 5), "prev_reward"] == 100
    s = ds.participant_summary().set_index("participant_id")
    assert s.loc["p1", "n_missed"] == 1 and s.loc["p1", "p_choose_4"] == pytest.approx(0.25)


def test_reward_range_and_rescaling_use_the_stated_range_only():
    ds = BehaviouralDataset(df4(), CATS, reward_values=None, missing="mark", reward_range=(0, 100), rescale_rewards=True)
    assert ds.get("p1")[1].tolist() == [0.1, 0.5, 1.0, 0.0, 0.2]  # missed trial keeps reward 0 (never rescaled)
    assert any("rescaled" in s for s in ds.report.steps)
    with pytest.raises(DataValidationError, match="outside the stated range"):
        BehaviouralDataset(df4(), CATS, reward_values=None, missing="mark", reward_range=(0, 90))
    with pytest.raises(ValueError, match="requires reward_range"):
        validate_behaviour(df4(), CATS, reward_values=None, missing="mark", rescale_rewards=True)


def test_optional_columns_are_retained_and_extra_columns_on_request():
    d = pd.DataFrame(
        {
            "participant_id": ["p"] * 3,
            "trial": [1, 2, 3],
            "choice": ["1", "2", "1"],
            "reward": [1.0, 0.0, 1.0],
            "condition": ["rich", "rich", "poor"],
            "reaction_time": [0.5, 0.7, 0.6],
            "notes": ["x", "y", "z"],
        }
    )
    ds = BehaviouralDataset(d, ("1", "2"), reward_values=None)
    assert {"condition", "reaction_time"} <= set(ds.df.columns) and "notes" not in ds.df.columns
    ds2 = BehaviouralDataset(d, ("1", "2"), reward_values=None, extra_columns=("notes",))
    assert "notes" in ds2.df.columns


def test_blocks_can_be_split_into_independent_sequences():
    d = pd.DataFrame(
        {
            "participant_id": ["p"] * 4,
            "trial": [1, 2, 1, 2],
            "block": [1, 1, 2, 2],
            "choice": ["A", "B", "B", "A"],
            "reward": [1, 0, 0, 1],
        }
    )
    ds = BehaviouralDataset(d, explode_blocks=True)
    assert ds.participants == ["p|1", "p|2"]
    with pytest.raises(DataValidationError, match="duplicate"):
        BehaviouralDataset(d)  # without splitting, trial numbers repeat within a participant
    with pytest.raises(DataValidationError, match="requires a 'block'"):
        BehaviouralDataset(d.drop(columns="block"), explode_blocks=True)


def test_available_actions_are_parsed_validated_and_exposed():
    d = pd.DataFrame(
        {
            "participant_id": ["p"] * 3,
            "trial": [1, 2, 3],
            "choice": ["1", "3", "2"],
            "reward": [1, 0, 1],
            "available_actions": ["1|2|3", "1|3", "2|3"],
        }
    )
    ds = BehaviouralDataset(d, ("1", "2", "3"))
    assert ds.get_available("p").tolist() == [[True, True, True], [True, False, True], [False, True, True]]
    bad = d.assign(available_actions=["1|2|3", "1|2", "2|3"])  # trial 2 chose "3", which was not available
    with pytest.raises(DataValidationError, match="not available"):
        BehaviouralDataset(bad, ("1", "2", "3"))
    assert BehaviouralDataset(d.drop(columns="available_actions"), ("1", "2", "3")).get_available("p") is None


def test_quality_report_and_prespecified_exclusions():
    d = pd.DataFrame(
        {
            "participant_id": ["a"] * 10 + ["b"] * 10,
            "trial": list(range(1, 11)) * 2,
            "choice": ["1", "2"] * 5 + [np.nan] * 6 + ["1"] * 4,
            "reward": [1.0] * 10 + [np.nan] * 6 + [0.0] * 4,
        }
    )
    ds = BehaviouralDataset(d, ("1", "2"), reward_values=None, missing="mark")
    q = ds.quality_report()
    assert q["n_participants"] == 2 and q["n_missed_total"] == 6
    assert q["missed_fraction_per_participant_max"] == pytest.approx(0.6)
    kept, excluded = ds.apply_exclusions(max_missed_fraction=0.5)
    assert kept.participants == ["a"] and excluded["participant_id"].tolist() == ["b"]
    assert "missed>50%" in excluded["reason"].iloc[0]
    assert any("excluded 1 of 2" in s for s in kept.report.steps)
    kept2, ex2 = ds.apply_exclusions(min_valid_trials=5)
    assert kept2.participants == ["a"] and len(ex2) == 1
    kept3, ex3 = ds.apply_exclusions()
    assert len(kept3) == 2 and ex3.empty


def test_csv_roundtrip_records_a_file_hash(tmp_path):
    ds = BehaviouralDataset(df4(), CATS, reward_values=None, missing="mark")
    f = tmp_path / "d.csv"
    ds.to_csv(f)
    back = BehaviouralDataset.from_csv(f, choice_categories=CATS, reward_values=None, missing="mark")
    assert back.get("p1")[0].tolist() == ds.get("p1")[0].tolist()
    assert back.config["source_sha256"] == file_sha256(f)


# ------------------------------------------------------------------ specs
def _write_raw(tmp_path):
    raw = pd.DataFrame(
        {
            "subj": ["s1"] * 4 + ["s2"] * 4 + ["s3"] * 4,
            "t": [1, 2, 3, 4] * 3,
            "arm": [1, 2, 3, 4, 1, np.nan, np.nan, np.nan, 2, 2, 3, 1],
            "pts": [10, 20, 30, 40, 10, np.nan, np.nan, np.nan, 50, 60, 70, 80],
        }
    )
    f = tmp_path / "raw.csv"
    raw.to_csv(f, index=False)
    return f


def _spec_dict(**over):
    spec = {
        "name": "toy",
        "path": "raw.csv",
        "columns": {"participant_id": "subj", "trial": "t", "choice": "arm", "reward": "pts"},
        "choice_categories": ["1", "2", "3", "4"],
        "reward_range": [0, 100],
        "rescale_rewards": True,
        "missing": "mark",
        "exclusions": {"max_missed_fraction": 0.5},
        "provenance": {"source_url": "https://example.org", "citation": "Toy (2020)", "licence": "VERIFY before use"},
    }
    spec.update(over)
    return spec


def test_spec_loading_maps_columns_applies_exclusions_and_records_provenance(tmp_path):
    f = _write_raw(tmp_path)
    (tmp_path / "spec.json").write_text(json.dumps(_spec_dict(sha256=file_sha256(f))))
    loaded = load_dataset(tmp_path / "spec.json")
    assert loaded.dataset.participants == ["s1", "s3"]  # s2 has 75% missed trials
    assert loaded.excluded["participant_id"].tolist() == ["s2"]
    assert loaded.dataset.get("s1")[1].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert loaded.dataset.config["source_sha256"] == file_sha256(f)
    assert loaded.spec.unverified_fields == ["licence"]


def test_spec_errors_are_loud(tmp_path):
    f = _write_raw(tmp_path)
    (tmp_path / "bad_hash.json").write_text(json.dumps(_spec_dict(sha256="0" * 64)))
    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        load_dataset(tmp_path / "bad_hash.json")
    (tmp_path / "bad_col.json").write_text(json.dumps(_spec_dict(columns={"participant_id": "subject", "trial": "t", "choice": "arm", "reward": "pts"})))
    with pytest.raises(DataValidationError, match="not in the file"):
        load_dataset(tmp_path / "bad_col.json")
    (tmp_path / "bad_key.json").write_text(json.dumps(_spec_dict(typo=1)))
    with pytest.raises(DataValidationError, match="unknown keys"):
        DatasetSpec.from_json(tmp_path / "bad_key.json")
    (tmp_path / "nomap.json").write_text(json.dumps(_spec_dict(columns={"participant_id": "subj"})))
    with pytest.raises(DataValidationError, match="must map"):
        DatasetSpec.from_json(tmp_path / "nomap.json")
    (tmp_path / "rule.json").write_text(json.dumps(_spec_dict(exclusions={"max_rt": 1})))
    with pytest.raises(DataValidationError, match="unknown exclusion"):
        DatasetSpec.from_json(tmp_path / "rule.json")
    f.unlink()
    (tmp_path / "ok.json").write_text(json.dumps(_spec_dict()))
    with pytest.raises(DataValidationError, match="not bundled"):
        load_dataset(tmp_path / "ok.json")


def test_missed_label_in_a_saved_file_is_recognised_as_missing_when_reloaded(tmp_path):
    ds = BehaviouralDataset(df4(), CATS, reward_values=None, missing="mark")
    f = tmp_path / "saved.csv"
    ds.to_csv(f)
    assert "<missed>" in f.read_text()
    back = BehaviouralDataset.from_csv(f, choice_categories=CATS, reward_values=None, missing="mark")
    assert back.get("p1")[0].tolist() == [0, 1, 3, -1, 2]
    assert back.df["missed"].sum() == 1 and any("marked 1 rows" in s for s in back.report.steps)
    with pytest.raises(DataValidationError, match="missing"):  # not silently accepted under the strict default
        BehaviouralDataset.from_csv(f, choice_categories=CATS, reward_values=None)
