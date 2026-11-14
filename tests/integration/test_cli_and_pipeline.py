"""The CLI and the end-to-end empirical pipeline (on synthetic rehearsal data and on a toy 'real' file)."""

import json

import numpy as np
import pytest

from humanchoice.cli import main
from humanchoice.data import file_sha256
from humanchoice.pipelines import default_ladder, run_empirical_analysis, simulate_restless_rehearsal


def test_default_ladders_follow_the_task():
    cont = default_ladder(4, binary_rewards=False)
    assert list(cont) == ["random", "wsls", "q", "asym_q", "sticky_q", "lapse_q", "kalman"]
    two = default_ladder(2, binary_rewards=True)
    assert {"pomdp", "bamdp"} <= set(two) and "kalman" not in two
    four_bin = default_ladder(4, binary_rewards=True)
    assert "bamdp" not in four_bin and "pomdp" not in four_bin  # exact planning is not offered for 4 options
    assert all(m.n_actions == 4 for m in cont.values())


def test_rehearsal_dataset_has_known_truth_timeouts_and_continuous_rewards():
    ds = simulate_restless_rehearsal(n_participants=10, n_trials=50, seed=3)
    assert len(ds) == 10 and ds.n_actions == 4 and ds.config["synthetic"] is True
    assert set(ds.truth["generating_model"]) <= {"q", "asym_q", "sticky_q", "lapse_q"}
    q = ds.quality_report()
    assert q["n_missed_total"] > 0 and q["n_distinct_reward_values"] > 100
    again = simulate_restless_rehearsal(n_participants=10, n_trials=50, seed=3)
    assert ds.df.equals(again.df)  # reproducible


@pytest.mark.slow
def test_pipeline_runs_end_to_end_writes_every_artifact_and_labels_synthetic_data(tmp_path):
    ds = simulate_restless_rehearsal(n_participants=6, n_trials=50, seed=4)
    from humanchoice.models import AsymmetricQLearning, KalmanBandit, QLearning, RandomChoice, StickyQLearning

    small = {"random": RandomChoice(4), "q": QLearning(4), "asym_q": AsymmetricQLearning(4), "sticky_q": StickyQLearning(4), "kalman": KalmanBandit(4)}
    res = run_empirical_analysis(ds, tmp_path, n_starts=2, fractions=(0.5, 0.7), ppc_participants=4, ppc_sims=40,
                                 profile_participants=1, banner="SYNTHETIC TEST", models=small)
    for name in ["report.md", "config.json", "data_quality.json", "fits_all.csv", "model_comparison_summary.csv", "held_out.csv",
                 "held_out_summary.csv", "predictive_checks_one_step.csv", "predictive_check_misfit_rates.csv", "profile_intervals.csv",
                 "winner_vs_truth.csv", "calibration.csv", "participant_summary.csv"]:
        assert (tmp_path / name).exists(), name
    for fig in ["behavioural_overview", "held_out_vs_chance", "reliability", "predictive_check_misfit"]:
        assert (tmp_path / "figures" / f"{fig}.png").exists()
    report = (tmp_path / "report.md").read_text()
    assert "SYNTHETIC TEST" in report and "## 3. Held-out prediction" in report
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["dataset"]["synthetic"] is True and cfg["n_options"] == 4 and "kalman" in cfg["models"]
    # a mechanistically richer model must beat chance on held-out data for at least one ladder member
    hs = res["held_out_summary"]
    assert (hs[hs["model"] != "chance"]["ll_diff_vs_chance"] > 0).any()
    # model comparison reports optimisation diagnostics for every model
    assert {"frac_converged", "mean_start_agreement", "frac_any_param_at_bound"} <= set(res["comparison"].columns)


# ------------------------------------------------------------------ CLI
def _toy_raw(tmp_path, n_participants=5, T=40, seed=0):
    ds = simulate_restless_rehearsal(n_participants=n_participants, n_trials=T, seed=seed)
    raw = ds.df.drop(columns="missed").copy()
    raw["choice"] = raw["choice"].map({"A": 1, "B": 2, "C": 3, "D": 4, "<missed>": np.nan})
    raw["reward"] = np.where(raw["choice"].isna(), np.nan, raw["reward"] * 100)
    raw = raw.rename(columns={"participant_id": "subject", "trial": "t", "choice": "arm", "reward": "points"})
    f = tmp_path / "raw.csv"
    raw.to_csv(f, index=False)
    return f


def _spec(tmp_path, f, **over):
    spec = {
        "name": "toy", "path": f.name, "sha256": file_sha256(f),
        "columns": {"participant_id": "subject", "trial": "t", "choice": "arm", "reward": "points"},
        "choice_categories": ["1", "2", "3", "4"], "reward_range": [0, 100], "rescale_rewards": True, "missing": "mark",
        "exclusions": {"max_missed_fraction": 0.5},
        "provenance": {"source_url": "https://example.org", "citation": "Toy (2020)", "licence": "CC0"},
    }
    spec.update(over)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_cli_validate_data_reports_quality_and_hash(tmp_path, capsys):
    f = _toy_raw(tmp_path)
    assert main(["validate-data", "--spec", str(_spec(tmp_path, f)), "--json", str(tmp_path / "q.json")]) == 0
    out = capsys.readouterr().out
    assert file_sha256(f) in out and "missed trials" in out and "participants kept: 5" in out
    assert json.loads((tmp_path / "q.json").read_text())["n_participants"] == 5


def test_cli_returns_nonzero_with_a_clear_message_for_bad_data(tmp_path, capsys):
    f = _toy_raw(tmp_path)
    bad = _spec(tmp_path, f, sha256="0" * 64)
    assert main(["validate-data", "--spec", str(bad)]) == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err
    missing = _spec(tmp_path, f, columns={"participant_id": "subject", "trial": "t", "choice": "nope", "reward": "points"})
    assert main(["validate-data", "--spec", str(missing)]) == 2
    assert "not in the file" in capsys.readouterr().err


def test_fit_refuses_unverified_provenance_unless_explicitly_allowed(tmp_path, capsys):
    f = _toy_raw(tmp_path)
    spec = _spec(tmp_path, f, provenance={"licence": "VERIFY before use", "source_url": "https://example.org"})
    assert main(["fit", "--spec", str(spec), "--out", str(tmp_path / "out")]) == 3
    assert "Refusing to analyse" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.slow
def test_cli_fit_on_a_raw_file_with_rescaling_and_exclusions_and_reproduce_rehearsal(tmp_path):
    f = _toy_raw(tmp_path, n_participants=4, T=40)
    spec = _spec(tmp_path, f, provenance={"licence": "VERIFY"})
    rc = main(["fit", "--spec", str(spec), "--out", str(tmp_path / "real"), "--starts", "1", "--max-participants", "3", "--allow-unverified"])
    assert rc == 0
    rep = (tmp_path / "real" / "report.md").read_text()
    assert "PROVENANCE NOT VERIFIED" in rep and "rescaled rewards" in rep
    cfg = json.loads((tmp_path / "real" / "config.json").read_text())
    assert cfg["dataset"]["source_sha256"] == file_sha256(f) and cfg["max_participants"] == 3
    assert main(["reproduce", "empirical-bandit", "--quick", "--out", str(tmp_path / "demo")]) == 0
    assert "NOT human data" in (tmp_path / "demo" / "report.md").read_text()


def test_cli_misc(capsys):
    assert main(["list-models"]) == 0
    assert "KalmanBandit" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--version"])


@pytest.mark.slow
def test_sensitivity_runner_reports_every_variant_and_whether_verdicts_change(tmp_path):
    from humanchoice.pipelines import run_sensitivity_analyses

    ds = simulate_restless_rehearsal(n_participants=10, n_trials=60, seed=7)
    variants = ({"name": "baseline"}, {"name": "win_threshold=0.4", "win_threshold": 0.4}, {"name": "reward_range+10%", "reward_scale": 1 / 1.1},
                {"name": "exclude_missed>1%", "max_missed": 0.01})
    out = run_sensitivity_analyses(ds, tmp_path, variants=variants, n_starts=2)
    assert set(out["variant"]) == {v["name"] for v in variants}
    assert not out[out["variant"] == "baseline"]["decision_changed"].any()  # the baseline cannot differ from itself
    assert {"hypothesis", "decision", "baseline_decision", "decision_changed", "p_adjusted"} <= set(out.columns)
    n_by_variant = out.groupby("variant")["n_participants_variant"].first()
    assert n_by_variant["exclude_missed>1%"] < n_by_variant["baseline"]  # the stricter exclusion removed participants
    assert (tmp_path / "sensitivity_results.csv").exists() and "variant verdicts differ" in (tmp_path / "sensitivity_summary.md").read_text()


def test_cli_sensitivity_refuses_unverified_provenance(tmp_path, capsys):
    f = _toy_raw(tmp_path)
    spec = _spec(tmp_path, f, provenance={"licence": "VERIFY"})
    assert main(["sensitivity", "--spec", str(spec), "--out", str(tmp_path / "s")]) == 3
    assert "Refusing to analyse" in capsys.readouterr().err


@pytest.mark.slow
def test_cli_env_option_enables_the_full_session_checks(tmp_path):
    f = _toy_raw(tmp_path, n_participants=4, T=40)
    spec = _spec(tmp_path, f)
    out = tmp_path / "with_env"
    rc = main(["fit", "--spec", str(spec), "--out", str(out), "--starts", "1", "--env", "humanchoice.envs:RestlessBandit"])
    assert rc == 0 and (out / "predictive_checks_session.csv").exists()
    assert "**Not run.**" not in (out / "report.md").read_text()
    assert (out / "dataset_spec.json").exists() and (out / "environment.txt").exists()
    no_env = tmp_path / "no_env"
    assert main(["fit", "--spec", str(spec), "--out", str(no_env), "--starts", "1"]) == 0
    assert "**Not run.**" in (no_env / "report.md").read_text()


def test_cli_env_option_rejects_a_bad_factory(tmp_path, capsys):
    f = _toy_raw(tmp_path)
    assert main(["fit", "--spec", str(_spec(tmp_path, f)), "--out", str(tmp_path / "x"), "--env", "no_such_module:thing"]) == 2
    assert "cannot import environment factory" in capsys.readouterr().err
    assert main(["fit", "--spec", str(_spec(tmp_path, f)), "--out", str(tmp_path / "y"), "--env", "nocolon"]) == 2


def test_cli_refuses_a_non_synthetic_spec_whose_provenance_is_simply_omitted(tmp_path, capsys):
    f = _toy_raw(tmp_path)
    spec = _spec(tmp_path, f, provenance={})
    assert main(["fit", "--spec", str(spec), "--out", str(tmp_path / "o")]) == 3
    err = capsys.readouterr().err
    assert "Refusing to analyse" in err and "citation" in err
