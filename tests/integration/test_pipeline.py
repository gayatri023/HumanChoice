"""End-to-end checks of the research workflow: data -> model -> simulation -> fit -> recovery -> comparison."""
import numpy as np
import pandas as pd
import pytest

from humanchoice.comparison import compare_models, held_out_evaluation
from humanchoice.data import BehaviouralDataset
from humanchoice.envs import ReversalBandit
from humanchoice.models import BayesBanditModel, POMDPReversalModel, QLearning
from humanchoice.recovery import (
    recovery_metrics,
    run_model_recovery,
    run_parameter_recovery,
)
from humanchoice.simulation import sample_participants, save_dataset, simulate_dataset

ENV = lambda: ReversalBandit(0.8, 0.08)


def test_csv_roundtrip_then_fit(tmp_path):
    m = QLearning()
    ds = simulate_dataset(m, sample_participants(m, 4, seed=0), ENV, n_trials=100, seed=1)
    save_dataset(ds, tmp_path, "synthetic_example")
    loaded = BehaviouralDataset.from_csv(tmp_path / "synthetic_example.csv")
    per, summary = compare_models({"q": QLearning(), "pomdp": POMDPReversalModel()}, loaded, n_starts=2)
    assert set(per["model"]) == {"q", "pomdp"} and len(per) == 8
    assert summary["n_wins_aic"].sum() == 4 and summary["n_wins_bic"].sum() == 4


@pytest.mark.slow
def test_parameter_recovery_is_reasonable_for_q_learning():
    m = QLearning()
    df = run_parameter_recovery(m, ENV, n_participants=24, n_trials=200, seed=11, n_starts=3)
    res = recovery_metrics(df, m.param_names, ci=False).set_index("param")
    assert res.loc["alpha", "r"] > 0.6 and res.loc["beta", "r"] > 0.6
    assert res.loc["alpha", "mae"] < 0.2


def test_parameter_recovery_is_reproducible():
    m = BayesBanditModel(horizon=4)
    a = run_parameter_recovery(m, ENV, n_participants=4, n_trials=60, seed=5, n_starts=2)
    b = run_parameter_recovery(m, ENV, n_participants=4, n_trials=60, seed=5, n_starts=2)
    pd.testing.assert_frame_equal(a, b)


def test_more_data_improves_recovery():
    m = QLearning()
    short = run_parameter_recovery(m, ENV, n_participants=20, n_trials=40, seed=2, n_starts=3)
    long = run_parameter_recovery(m, ENV, n_participants=20, n_trials=400, seed=2, n_starts=3)
    r_short = recovery_metrics(short, m.param_names, ci=False).set_index("param").loc["alpha", "rmse"]
    r_long = recovery_metrics(long, m.param_names, ci=False).set_index("param").loc["alpha", "rmse"]
    assert r_long < r_short


@pytest.mark.slow
def test_model_recovery_beats_chance_and_reports_confusion():
    models = {"q": QLearning(), "bamdp": BayesBanditModel(horizon=5), "pomdp": POMDPReversalModel()}
    res = run_model_recovery(models, ENV, n_per_model=6, n_trials=150, seed=4, criterion="bic", n_starts=3)
    assert res.counts.to_numpy().sum() == 18
    assert np.allclose(res.confusion.sum(axis=1), 1.0)
    assert res.accuracy > 1 / 3 + 0.15


def test_true_model_predicts_held_out_trials_better_than_chance():
    m = POMDPReversalModel()
    table = pd.DataFrame({"participant_id": ["a", "b", "c"], "beta": [8.0, 10.0, 12.0], "hazard": [0.05, 0.1, 0.08]})
    ds = simulate_dataset(m, table, ENV, n_trials=200, seed=9)
    ho = held_out_evaluation({"pomdp": POMDPReversalModel()}, ds, train_fraction=0.7, n_starts=3)
    means = ho.groupby("model")["test_ll_per_trial"].mean()
    assert means["pomdp"] > means["chance"]
    assert ho[ho.model == "pomdp"]["n_test"].eq(60).all()


def test_model_recovery_can_be_resummarised_under_another_criterion_without_refitting():
    from humanchoice.recovery import summarise_model_recovery

    models = {"q": QLearning(), "pomdp": POMDPReversalModel()}
    res = run_model_recovery(models, ENV, n_per_model=3, n_trials=60, seed=1, criterion="bic", n_starts=2)
    aic_res = summarise_model_recovery(res.fits, list(models), "aic")
    assert aic_res.criterion == "aic" and aic_res.counts.to_numpy().sum() == 6
    assert aic_res.fits is res.fits
    # re-summarising under the original criterion is idempotent
    assert summarise_model_recovery(res.fits, list(models), "bic").counts.equals(res.counts)
    with pytest.raises(ValueError):
        summarise_model_recovery(res.fits, list(models), "waic")
