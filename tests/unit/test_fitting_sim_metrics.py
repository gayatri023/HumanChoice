import json

import numpy as np
import pandas as pd
import pytest

from humanchoice.envs import BernoulliBandit, ReversalBandit
from humanchoice.evaluation import metrics as M
from humanchoice.fitting import fit_dataset, fit_model
from humanchoice.models import POMDPReversalModel, QLearning
from humanchoice.simulation import (
    EXAMPLE_PARTICIPANTS,
    sample_participants,
    save_dataset,
    simulate_dataset,
)


def test_information_criteria_formulas():
    assert np.isclose(M.aic(-100.0, 3), 206.0)
    assert np.isclose(M.bic(-100.0, 3, 100), 3 * np.log(100) + 200.0)
    w = M.akaike_weights([10.0, 12.0, 20.0])
    assert np.isclose(w.sum(), 1.0) and w[0] > w[1] > w[2]
    assert np.isclose(w[0] / w[1], np.exp(1.0))


def test_recovery_metrics_values():
    t, e = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.5, 2.5, 3.5, 4.5])
    assert np.isclose(M.mean_absolute_error(t, e), 0.5) and np.isclose(M.rmse(t, e), 0.5)
    assert np.isclose(M.bias(t, e), 0.5) and np.isclose(M.correlation(t, e), 1.0)
    assert np.isclose(M.error_variance(t, e), 0.0)
    assert np.isnan(M.correlation([1, 1, 1], [1, 2, 3]))


def test_bootstrap_ci_brackets_statistic_and_is_seeded():
    rng = np.random.default_rng(0)
    t = rng.normal(size=200)
    e = t + rng.normal(scale=0.5, size=200)
    lo, hi = M.bootstrap_ci(t, e, M.rmse, n_boot=300, seed=1)
    assert lo < M.rmse(t, e) < hi
    assert (lo, hi) == M.bootstrap_ci(t, e, M.rmse, n_boot=300, seed=1)


def test_choice_accuracy():
    assert M.choice_accuracy(np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]), np.array([0, 1, 1])) == pytest.approx(2 / 3)


def test_fit_model_returns_consistent_result():
    m = QLearning()
    sim = m.simulate([0.4, 3.0], ReversalBandit(0.8, 0.1), 150, rng=0)
    f = fit_model(m, sim["choices"], sim["rewards"], n_starts=4, seed=1)
    assert f.n_obs == 150 and f.n_params == 2
    assert np.isclose(f.aic, 2 * 2 + 2 * f.nll) and np.isclose(f.bic, 2 * np.log(150) + 2 * f.nll)
    # the fitted likelihood must be at least as good as the generating parameters
    assert f.nll <= -m.log_likelihood([0.4, 3.0], sim["choices"], sim["rewards"]) + 1e-6
    assert 0.0 <= f.start_agreement <= 1.0


def test_fit_is_deterministic_given_seed():
    m = QLearning()
    sim = m.simulate([0.3, 2.0], BernoulliBandit(), 80, rng=2)
    a = fit_model(m, sim["choices"], sim["rewards"], seed=5)
    b = fit_model(m, sim["choices"], sim["rewards"], seed=5)
    assert a.params == b.params and a.nll == b.nll


def test_parameters_on_bounds_are_flagged():
    """A perfectly consistent participant pushes beta to its upper bound; this must be reported."""
    f = fit_model(QLearning(), np.zeros(60, dtype=int), np.ones(60), n_starts=3, seed=0)
    assert "beta" in f.at_bound


def test_fit_dataset_one_row_per_participant():
    m = QLearning()
    table = sample_participants(m, 3, seed=0)
    ds = simulate_dataset(m, table, lambda: BernoulliBandit(), n_trials=60, seed=1)
    fits = fit_dataset(m, ds, n_starts=2)
    assert len(fits) == 3 and {"fit_alpha", "fit_beta", "aic", "bic", "converged"} <= set(fits.columns)


def test_simulate_dataset_reproducible_and_participant_streams_independent():
    m = QLearning()
    table = EXAMPLE_PARTICIPANTS
    a = simulate_dataset(m, table, lambda: ReversalBandit(), 40, seed=3)
    b = simulate_dataset(m, table, lambda: ReversalBandit(), 40, seed=3)
    pd.testing.assert_frame_equal(a.df, b.df)
    only_first = simulate_dataset(m, table.iloc[:1], lambda: ReversalBandit(), 40, seed=3)
    pd.testing.assert_frame_equal(
        a.df[a.df.participant_id == "A"].reset_index(drop=True), only_first.df.reset_index(drop=True)
    )


def test_save_dataset_records_provenance(tmp_path):
    m = QLearning()
    ds = simulate_dataset(m, EXAMPLE_PARTICIPANTS, lambda: ReversalBandit(), 20, seed=0)
    save_dataset(ds, tmp_path, "demo", extra_config={"env": "reversal"})
    cfg = json.loads((tmp_path / "demo_config.json").read_text())
    assert cfg["synthetic"] is True and cfg["seed"] == 0 and cfg["env"] == "reversal"
    assert (tmp_path / "demo.csv").exists() and (tmp_path / "demo_truth.csv").exists()


def test_pomdp_model_recovers_hazard_direction():
    """A volatile world should yield a larger fitted hazard than a stable one (sanity check of identifiability)."""
    m = POMDPReversalModel()
    fits = {}
    for hz in (0.03, 0.3):
        est = []
        for seed in range(4):
            sim = m.simulate([8.0, hz], ReversalBandit(0.8, hz), 200, rng=seed)
            est.append(fit_model(m, sim["choices"], sim["rewards"], n_starts=3, seed=seed).params["hazard"])
        fits[hz] = np.mean(est)
    assert fits[0.3] > fits[0.03]


def test_stratified_recovery_metrics_bins_by_generating_parameter():
    from humanchoice.recovery import stratified_recovery_metrics

    rng = np.random.default_rng(0)
    beta = np.linspace(0.5, 8.0, 60)
    # estimation noise shrinks as beta grows -> RMSE must be larger in the lowest stratum
    est = beta + rng.normal(0, 1.0 / beta)
    df = pd.DataFrame({"n_trials": 100, "true_beta": beta, "fit_beta": est, "at_bound": ""})
    out = stratified_recovery_metrics(df, ("beta",), by="beta", n_bins=3)
    assert list(out["stratum"]) == ["Q1 (lowest)", "Q2", "Q3 (highest)"]
    assert (out["n"] == 20).all()
    assert out["by_max"].iloc[0] <= out["by_min"].iloc[1] <= out["by_max"].iloc[1] <= out["by_min"].iloc[2]
    assert out["rmse"].iloc[0] > out["rmse"].iloc[2]
    with pytest.raises(KeyError):
        stratified_recovery_metrics(df, ("beta",), by="alpha")
    with pytest.raises(ValueError):
        stratified_recovery_metrics(df, ("beta",), by="beta", n_bins=1)


def test_spearman_and_median_error_are_robust_to_outliers():
    t = np.arange(1.0, 21.0)
    e = t.copy()
    e[-1] = 1000.0  # one fit runs away to a bound
    assert M.spearman(t, e) == pytest.approx(1.0)  # ranks unchanged
    assert M.correlation(t, e) < 0.6  # Pearson is wrecked by one outlier
    assert M.median_absolute_error(t, e) == 0.0
    assert M.mean_absolute_error(t, e) > 40
    assert M.spearman([1, 1, 1, 1], [1, 2, 3, 4]) != M.spearman([1, 1, 1, 1], [1, 2, 3, 4])  # NaN for constant input
    assert M.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert M.spearman([1, 2, 2, 4], [1, 2, 3, 4]) < 1.0  # ties handled by average ranks
    out = recovery_metrics_frame()
    assert {"rho", "median_ae", "rho_ci_lo", "rho_ci_hi"} <= set(out.columns)


def recovery_metrics_frame():
    from humanchoice.recovery import recovery_metrics

    rng = np.random.default_rng(1)
    t = rng.uniform(0, 1, 30)
    df = pd.DataFrame({"n_trials": 50, "true_a": t, "fit_a": t + rng.normal(0, 0.1, 30), "at_bound": ""})
    return recovery_metrics(df, ("a",), ci=True)
