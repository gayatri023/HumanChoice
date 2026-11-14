"""Brier / calibration / intervals, predictive checks, likelihood profiles, fitting with missed trials, comparison."""

import numpy as np
import pandas as pd
import pytest

from humanchoice.comparison import held_out_evaluation, repeated_temporal_splits, summarise_held_out
from humanchoice.comparison.model_comparison import compare_models
from humanchoice.data import BehaviouralDataset
from humanchoice.envs import DriftingBernoulliBandit, RestlessBandit
from humanchoice.evaluation import metrics as M
from humanchoice.evaluation.predictive_checks import free_running_check, misfit_rate, one_step_ahead_check
from humanchoice.fitting import fit_dataset, fit_model, likelihood_profile, profile_interval
from humanchoice.models import BayesBanditModel, QLearning, RandomChoice, StickyQLearning
from humanchoice.recovery import (
    cross_parameter_leakage,
    error_correlations,
    recovery_accuracy_table,
    run_model_recovery,
)
from humanchoice.simulation import sample_participants, simulate_dataset
from humanchoice.utils import MISSED


# ---------------------------------------------------------------- metrics
def test_brier_score_known_values_and_missed_trials_ignored():
    onehot = np.eye(4)[[0, 2, 1]]
    assert M.brier_score(onehot, [0, 2, 1]) == 0.0
    uniform = np.full((5, 4), 0.25)
    assert M.brier_score(uniform, [0, 1, 2, 3, 0]) == pytest.approx(1 - 1 / 4)  # uniform predictor scores 1 - 1/K
    wrong = np.eye(3)[[0, 0]]
    assert M.brier_score(wrong, [1, 2]) == pytest.approx(2.0)  # worst possible
    probs = np.array([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8]])
    assert M.brier_score(probs, [0, MISSED, 1]) == pytest.approx(np.mean([0.3**2 * 2, 0.2**2 * 2]))
    assert M.choice_accuracy(probs, [0, MISSED, 1]) == 1.0


def test_calibration_table_of_a_calibrated_predictor_and_ece():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), size=20000)
    c = np.array([rng.choice(3, p=row) for row in p])
    t = M.calibration_table(p, c, n_bins=10)
    assert t["n"].sum() == p.size
    assert M.expected_calibration_error(t) < 0.02
    overconfident = np.eye(3)[np.zeros(20000, dtype=int)] * 0.98 + 0.01  # always claims option 0 w.p. .98
    assert M.expected_calibration_error(M.calibration_table(overconfident, c)) > 0.3


def test_wilson_interval_matches_reference_values():
    lo, hi = M.wilson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=2e-4)
    assert M.wilson_interval(0, 10)[0] == 0.0 and M.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert np.isnan(M.wilson_interval(0, 0)[0])
    narrow = M.wilson_interval(500, 1000)
    assert narrow[1] - narrow[0] < M.wilson_interval(5, 10)[1] - M.wilson_interval(5, 10)[0]


def test_mean_ci_brackets_the_mean_and_handles_nans():
    v = np.r_[np.random.default_rng(1).normal(2.0, 1.0, 200), np.nan]
    m, lo, hi = M.mean_ci(v)
    assert lo < m < hi and abs(m - 2.0) < 0.3
    assert M.mean_ci([3.0, 3.0, 3.0]) == (3.0, 3.0, 3.0)
    assert np.isnan(M.mean_ci([np.nan])[0])


# ---------------------------------------------------- fitting & missed trials
def test_fit_ignores_missed_trials_and_counts_observed_choices():
    m = QLearning(3)
    sim = m.simulate([0.4, 3.0], DriftingBernoulliBandit(3, 0.1), 120, 2)
    c = sim["choices"].copy()
    c[[3, 4, 50, 51, 52]] = MISSED
    f = fit_model(m, c, sim["rewards"], n_starts=3, seed=0)
    keep = c != MISSED
    g = fit_model(m, c[keep], sim["rewards"][keep], n_starts=3, seed=0)
    assert f.n_obs == 115 == g.n_obs
    assert f.nll == pytest.approx(g.nll, rel=1e-8)
    with pytest.raises(ValueError, match="no observed"):
        fit_model(m, [MISSED, MISSED], [0.0, 0.0])


def test_parameter_free_model_is_evaluated_not_optimised():
    r = fit_model(RandomChoice(4), [0, 1, 2, 3, 0], [0, 1, 0, 1, 0])
    assert r.n_params == 0 and r.nll == pytest.approx(5 * np.log(4)) and r.converged
    assert r.aic == pytest.approx(2 * 5 * np.log(4)) and r.bic == pytest.approx(2 * 5 * np.log(4))


def _toy_dataset(K=4, n=4, T=60, seed=0, env=None):
    m = QLearning(K)
    table = sample_participants(m, n, seed=seed)
    return simulate_dataset(m, table, env or (lambda: RestlessBandit(K)), T, seed=seed)


def test_simulated_k_option_dataset_roundtrips_with_continuous_rewards(tmp_path):
    ds = _toy_dataset()
    assert ds.n_actions == 4 and ds.choice_categories == ("A", "B", "C", "D")
    assert ds.df["reward"].nunique() > 50  # continuous rewards survive (no int casting)
    path = tmp_path / "d.csv"
    ds.to_csv(path)
    back = BehaviouralDataset.from_csv(path, choice_categories=ds.choice_categories, reward_values=None)
    assert np.allclose(back.get("S001")[1], ds.get("S001")[1])


def test_parallel_fitting_matches_serial_fitting():
    ds = _toy_dataset(K=3, n=3, T=50, env=lambda: DriftingBernoulliBandit(3, 0.1))
    serial = fit_dataset(QLearning(3), ds, n_starts=2, seed=1)
    parallel = fit_dataset(QLearning(3), ds, n_starts=2, seed=1, n_jobs=2)
    pd.testing.assert_frame_equal(serial.drop(columns="message"), parallel.drop(columns="message"))


def test_incompatible_models_raise_or_are_skipped_with_a_warning():
    ds = _toy_dataset(K=4, n=2, T=40)  # 4 options, continuous rewards
    with pytest.raises(ValueError, match="binary rewards"):
        fit_dataset(BayesBanditModel(n_actions=4), ds)
    with pytest.raises(ValueError, match="n_actions"):
        fit_dataset(QLearning(2), ds)
    with pytest.warns(UserWarning, match="skipping model 'bamdp'"):
        per, _ = compare_models({"q": QLearning(4), "bamdp": BayesBanditModel(n_actions=4)}, ds, n_starts=2, skip_incompatible=True)
    assert set(per["model"]) == {"q"}
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="no candidate"):
        compare_models({"bamdp": BayesBanditModel(n_actions=4)}, ds, skip_incompatible=True)


# ------------------------------------------------------ held-out prediction
def test_held_out_table_columns_chance_row_and_summary():
    K = 4
    ds = _toy_dataset(K=K, n=5, T=80)
    models = {"random": RandomChoice(K), "q": QLearning(K)}
    ho = held_out_evaluation(models, ds, train_fraction=0.6, n_starts=2)
    assert set(ho["model"]) == {"chance", "random", "q"}
    chance = ho[ho["model"] == "chance"]
    assert np.allclose(chance["test_ll_per_trial"], np.log(1 / K))
    assert np.allclose(chance["test_brier"], 1 - 1 / K) and np.allclose(chance["test_accuracy"], 1 / K)
    rnd = ho[ho["model"] == "random"].reset_index(drop=True)
    assert np.allclose(rnd["test_ll_per_trial"], chance["test_ll_per_trial"].to_numpy())  # RandomChoice IS the chance model

    rep = repeated_temporal_splits(models, ds, fractions=(0.5, 0.7), n_starts=2)
    assert sorted(rep["train_fraction"].unique()) == [0.5, 0.7]
    s = summarise_held_out(rep, reference="chance", n_boot=200)
    assert {"ll_diff_vs_chance", "diff_lo", "diff_hi", "ll_lo", "ll_hi"} <= set(s.columns)
    row = s[(s.model == "random") & (s.train_fraction == 0.5)].iloc[0]
    assert row["ll_diff_vs_chance"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="reference"):
        summarise_held_out(rep, reference="nope")


def test_held_out_returns_pooled_calibration_and_skips_participants_without_test_data():
    ds = _toy_dataset(K=3, n=3, T=60, env=lambda: DriftingBernoulliBandit(3, 0.1))
    ho, calib = held_out_evaluation({"q": QLearning(3)}, ds, 0.7, n_starts=2, return_calibration=True)
    assert set(calib) == {"q"} and calib["q"]["n"].sum() == ho[ho.model == "q"]["n_test"].sum() * 3
    with pytest.raises(ValueError):
        held_out_evaluation({"q": QLearning(3)}, ds, 0.95)


# ------------------------------------------------------------ predictive checks
def test_predictive_check_flags_a_model_that_misses_perseveration_but_not_the_true_model():
    K = 3
    true = StickyQLearning(K)
    th = np.array([0.3, 3.0, 2.5])  # strongly perseverative
    sim = true.simulate(th, DriftingBernoulliBandit(K, 0.1), 400, 4)
    good = one_step_ahead_check(true, th, sim["choices"], sim["rewards"], n_sims=300, rng=1).set_index("statistic")
    assert good.loc["p_stay", "p_value"] > 0.05
    plain = QLearning(K)
    fit = fit_model(plain, sim["choices"], sim["rewards"], n_starts=4, seed=0)
    bad = one_step_ahead_check(plain, list(fit.params.values()), sim["choices"], sim["rewards"], n_sims=300, rng=1).set_index("statistic")
    assert bad.loc["p_stay", "p_value"] < 0.02
    assert bad.loc["p_stay", "observed"] > bad.loc["p_stay", "hi"]  # the model under-predicts staying


def test_predictive_check_handles_missed_trials_and_columns():
    K = 3
    m = QLearning(K)
    sim = m.simulate([0.4, 3.0], DriftingBernoulliBandit(K, 0.1), 80, 0)
    c = sim["choices"].copy()
    c[[0, 10, 11]] = MISSED
    out = one_step_ahead_check(m, [0.4, 3.0], c, sim["rewards"], n_sims=50, rng=0)
    assert {"statistic", "observed", "replicate_mean", "lo", "hi", "p_value"} <= set(out.columns)
    from humanchoice.evaluation.predictive_checks import ONE_STEP_STATISTICS

    assert tuple(out["statistic"]) == ONE_STEP_STATISTICS  # choice-only statistics: no statistic silently becomes NaN
    assert out["p_value"].between(0, 1).all()


def test_free_running_check_and_misfit_rate():
    K = 3
    m = QLearning(K)
    th = [0.4, 3.0]
    sim = m.simulate(th, DriftingBernoulliBandit(K, 0.1), 100, 0)
    out = free_running_check(m, th, lambda: DriftingBernoulliBandit(K, 0.1), sim["choices"], sim["rewards"], n_sims=40, rng=0)
    from humanchoice.evaluation.predictive_checks import make_statistics

    assert len(out) == len(make_statistics()) and out["lo"].le(out["hi"]).all()
    table = pd.DataFrame({"statistic": ["a"] * 10 + ["b"] * 10, "p_value": [0.01] * 5 + [0.5] * 5 + [0.5] * 10})
    mr = misfit_rate(table).set_index("statistic")
    assert mr.loc["a", "misfit_rate"] == 0.5 and mr.loc["b", "misfit_rate"] == 0.0


# ------------------------------------------------------------------ profiles
def test_profile_is_minimal_at_the_mle_and_interval_covers_it():
    m = QLearning(2)
    sim = m.simulate([0.4, 3.0], DriftingBernoulliBandit(2, 0.1), 300, 6)
    fit = fit_model(m, sim["choices"], sim["rewards"], n_starts=5, seed=0)
    prof = likelihood_profile(m, sim["choices"], sim["rewards"], "alpha", n_grid=21, n_starts=3)
    assert prof["delta_nll"].min() == 0.0
    best = prof.loc[prof["delta_nll"].idxmin(), "value"]
    assert abs(best - fit.params["alpha"]) < 0.1
    lo, hi, _is_open = profile_interval(prof)
    assert lo <= fit.params["alpha"] <= hi
    with pytest.raises(ValueError, match="not a parameter"):
        likelihood_profile(m, sim["choices"], sim["rewards"], "kappa")


def test_flat_profile_is_flagged_as_an_open_interval():
    m = QLearning(2)
    rng = np.random.default_rng(0)
    c = rng.integers(0, 2, 150)  # choices unrelated to rewards: beta is not identified away from 0
    r = rng.integers(0, 2, 150).astype(float)
    prof = likelihood_profile(m, c, r, "beta", n_grid=15, n_starts=2)
    lo, _hi, is_open = profile_interval(prof)
    assert is_open and lo == 0.0


def test_profile_interval_arithmetic():
    from scipy.stats import chi2

    cut = chi2.ppf(0.95, 1) / 2  # 1.9207
    prof = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0, 4.0], "delta_nll": [5.0, 1.0, 0.0, 1.0, 5.0]})
    lo, hi, is_open = profile_interval(prof, level=0.95)
    assert not is_open
    assert lo == pytest.approx(1.0 - (cut - 1.0) / 4.0)  # linear interpolation between (0, 5) and (1, 1)
    assert hi == pytest.approx(3.0 + (cut - 1.0) / 4.0)


# ------------------------------------------------------ recovery diagnostics
def test_error_correlations_and_leakage_detect_a_planted_tradeoff():
    rng = np.random.default_rng(0)
    n = 300
    true_a, true_k = rng.uniform(0.1, 0.9, n), rng.uniform(0, 2, n)
    err_a = rng.normal(0, 0.05, n)
    df = pd.DataFrame(
        {
            "n_trials": 100,
            "true_alpha": true_a,
            "true_kappa": true_k,
            "fit_alpha": true_a + err_a + 0.1 * true_k,  # perseveration leaks into the learning-rate estimate
            "fit_kappa": true_k - 4 * err_a,  # and the two errors are anti-correlated
        }
    )
    ec = error_correlations(df, ("alpha", "kappa"))
    assert ec.loc["alpha", "kappa"] < -0.5
    leak = cross_parameter_leakage(df, ("alpha", "kappa"))
    assert leak.loc["kappa", "alpha"] > 0.5
    assert abs(leak.loc["alpha", "kappa"]) < 0.3


def test_model_recovery_with_misspecified_generators_gives_a_rectangular_confusion_matrix():
    K = 2
    cands = {"q": QLearning(K), "random": RandomChoice(K)}
    gens = {"q": QLearning(K), "sticky": StickyQLearning(K)}
    res = run_model_recovery(cands, lambda: DriftingBernoulliBandit(K, 0.1), n_per_model=3, n_trials=60, seed=0, n_starts=2, generators=gens)
    assert res.confusion.shape == (2, 2) and list(res.confusion.index) == ["q", "sticky"]
    assert res.counts.to_numpy().sum() == 6
    acc = recovery_accuracy_table(res).set_index("generating")
    assert acc.loc["q", "n_datasets"] == 3 and 0.0 <= acc.loc["q", "ci_lo"] <= acc.loc["q", "ci_hi"] <= 1.0


def test_holm_adjustment_matches_hand_calculation():
    from humanchoice.comparison import holm_adjust

    # sorted p: .01, .03, .04 -> .01*3 = .03 ; .03*2 = .06 ; .04*1 = .04 -> monotone max -> .03, .06, .06
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5]) == pytest.approx([0.5])
    assert holm_adjust([0.6, 0.9]).max() <= 1.0
    adj = holm_adjust([0.2, 0.001, 0.04])
    assert (adj >= np.array([0.2, 0.001, 0.04])).all()


def _held_out_table(shifts, n=40, fractions=(0.5, 0.6, 0.7, 0.8), noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in fractions:
        for i in range(n):
            base = rng.normal(-1.2, 0.1)
            for model, shift in shifts.items():
                rows.append({"participant_id": f"p{i}", "train_fraction": f, "model": model, "test_ll_per_trial": base + shift + rng.normal(0, noise)})
    return pd.DataFrame(rows)


def test_planned_contrasts_primary_split_with_holm_and_sensitivity_splits():
    from humanchoice.comparison import planned_contrasts

    held = _held_out_table({"q": 0.0, "sticky_q": 0.05, "asym_q": 0.0, "wsls": -0.1, "random": -0.19, "lapse_q": -0.05})
    per, dec = planned_contrasts(held, n_boot=500)
    d = dec.set_index("hypothesis")
    assert d.loc["H2", "decision"] == "supported" and d.loc["H2", "primary_split"] == 0.7
    assert d.loc["H3", "decision"] == "not supported"  # asym_q is identical to q
    assert d.loc["H1", "decision"] == "supported" and d.loc["H1b", "decision"] == "supported"
    assert d.loc["H5", "decision"] == "contradicted"  # lapse_q is credibly worse than q
    assert "H4" not in d.index  # kalman absent: skipped
    # Holm family = the three primary contrasts; adjusted p never below raw p; others unadjusted
    assert list(d.index[d["holm_family"]]) == ["H1", "H2", "H3"]
    assert (d["p_adjusted"] >= d["p_value"] - 1e-12).all()
    assert d.loc["H5", "p_adjusted"] == d.loc["H5", "p_value"]
    # the other three splits are sensitivity analyses: all agree here, and are flagged as not independent
    assert d.loc["H2", "sensitivity_agree"] == 3 and d.loc["H2", "n_sensitivity_splits"] == 3
    assert d.loc["H3", "sensitivity_agree"] == 3  # CI includes 0 at every other split too
    assert len(per) == 5 * 4 and per.groupby("hypothesis")["is_primary_split"].sum().eq(1).all()


def test_holm_adjustment_can_turn_a_borderline_primary_contrast_into_not_supported():
    from humanchoice.comparison import planned_contrasts

    # three primary effects of the same borderline size and noise: unadjusted p ~ 0.01-0.05, Holm multiplies by up to 3
    rng = np.random.default_rng(4)
    held = _held_out_table({"q": 0.0, "sticky_q": 0.0, "asym_q": 0.0, "wsls": -0.0, "random": -0.1}, n=60, noise=0.04, seed=1)
    held = held.assign(test_ll_per_trial=held["test_ll_per_trial"] + np.where(held["model"] == "sticky_q", 0.012, 0.0)
                       + rng.normal(0, 0.0, len(held)))
    _, dec = planned_contrasts(held, n_boot=2000)
    d = dec.set_index("hypothesis")
    assert d.loc["H2", "p_adjusted"] >= d.loc["H2", "p_value"]
    if d.loc["H2", "p_value"] < 0.05 <= d.loc["H2", "p_adjusted"]:
        assert d.loc["H2", "decision"] == "not supported"  # significant raw, not after the family correction


def test_planned_contrasts_validate_their_inputs():
    from humanchoice.comparison import planned_contrasts

    held = _held_out_table({"q": 0.0, "sticky_q": 0.05}, n=10)
    with pytest.raises(ValueError, match="train_fraction"):
        planned_contrasts(held.drop(columns="train_fraction"))
    with pytest.raises(ValueError, match="primary_fraction"):
        planned_contrasts(held[held["train_fraction"] == 0.5])
