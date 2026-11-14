"""Statistical properties of the inference machinery (small, seeded samples: regression tests, not proofs)."""

import numpy as np
import pandas as pd
import pytest

from humanchoice.envs import DriftingBernoulliBandit, RestlessBandit
from humanchoice.evaluation import metrics as M
from humanchoice.evaluation.predictive_checks import misfit_rate, one_step_ahead_check
from humanchoice.fitting import fit_model
from humanchoice.models import QLearning, StickyQLearning

pytestmark = pytest.mark.slow


def test_estimator_is_consistent_for_a_four_option_perseverative_model():
    """Mean absolute error of alpha and kappa must fall clearly when the data grow 5-fold.

    (A median over only 10 participants is too noisy to show this reliably - an earlier version of this test failed by
    chance - so 16 participants and the mean absolute error are used.)
    """
    model = StickyQLearning(4)
    rng = np.random.default_rng(0)
    truths = [np.array([rng.uniform(0.2, 0.8), rng.uniform(2.0, 6.0), rng.uniform(0.3, 1.5)]) for _ in range(16)]
    errors = {}
    for T in (100, 500):
        err = []
        for i, th in enumerate(truths):
            sim = model.simulate(th, DriftingBernoulliBandit(4, 0.1), T, rng=1000 + i)
            fit = fit_model(model, sim["choices"], sim["rewards"], n_starts=4, seed=i)
            err.append([abs(fit.params["alpha"] - th[0]), abs(fit.params["kappa"] - th[2])])
        errors[T] = np.mean(err, axis=0)
    assert errors[500][0] < 0.8 * errors[100][0]
    assert errors[500][1] < 0.8 * errors[100][1]
    assert errors[500][0] < 0.1 and errors[500][1] < 0.3


def test_predictions_from_the_true_model_are_calibrated_and_beat_the_uniform_brier_score():
    K = 4
    model = QLearning(K)
    rng = np.random.default_rng(3)
    probs, choices = [], []
    for i in range(25):
        th = model.sample_params(rng)
        sim = model.simulate(th, RestlessBandit(K), 200, rng=50 + i)
        probs.append(model.trajectory(th, sim["choices"], sim["rewards"])["probs"])
        choices.append(sim["choices"])
    P, C = np.vstack(probs), np.concatenate(choices)
    table = M.calibration_table(P, C, n_bins=10)
    assert M.expected_calibration_error(table) < 0.03
    assert M.brier_score(P, C) < 1 - 1 / K - 0.02  # clearly better than a uniform predictor
    # the generating model can only be as good as the task's intrinsic noise allows: it must not be perfect either
    assert M.brier_score(P, C) > 0.2


def test_predictive_check_false_positive_rate_is_near_nominal_for_the_true_model():
    K = 3
    model = StickyQLearning(K)
    rng = np.random.default_rng(5)
    rows = []
    for i in range(30):
        th = model.sample_params(rng)
        sim = model.simulate(th, DriftingBernoulliBandit(K, 0.1), 150, rng=200 + i)
        chk = one_step_ahead_check(model, th, sim["choices"], sim["rewards"], n_sims=200, rng=i)
        chk["participant"] = i
        rows.append(chk)
    mr = misfit_rate(pd.concat(rows)).set_index("statistic")
    # nominal 5%; with 30 participants anything up to ~25% is consistent with calibration, while a real misfit gives ~100%
    assert mr.loc["p_stay", "misfit_rate"] <= 0.25
    assert mr.loc["p_stay_after_win", "misfit_rate"] <= 0.3
    assert mr["misfit_rate"].mean() < 0.2


def test_information_criteria_select_the_generating_structure_with_enough_data():
    """With 4 options and 600 trials, BIC must prefer the perseverative model over plain Q-learning for perseverative data."""
    gen = StickyQLearning(4)
    wins = 0
    for i in range(6):
        th = np.array([0.4, 4.0, 1.5])
        sim = gen.simulate(th, DriftingBernoulliBandit(4, 0.1), 600, rng=300 + i)
        f_sticky = fit_model(gen, sim["choices"], sim["rewards"], n_starts=4, seed=i)
        f_plain = fit_model(QLearning(4), sim["choices"], sim["rewards"], n_starts=4, seed=i)
        wins += f_sticky.bic < f_plain.bic
    assert wins == 6


def test_richer_family_members_never_fit_materially_worse_than_the_model_they_nest():
    """Regression test for a real bug: random multi-start L-BFGS-B left asymmetric Q-learning up to ~40 nats WORSE than
    plain Q-learning (which it contains) because the likelihood has local optima. Nested warm starts fix it."""
    from humanchoice.models import AsymmetricQLearning, FullQLearning, LapseQLearning

    rng = np.random.default_rng(0)
    base = QLearning(4)
    worst = 0.0
    for i in range(12):
        th = base.sample_params(rng)
        sim = base.simulate(th, RestlessBandit(4), 150, rng=i, missed_prob=0.03)
        nq = fit_model(base, sim["choices"], sim["rewards"], n_starts=10, seed=0).nll
        for cls in (AsymmetricQLearning, StickyQLearning, LapseQLearning, FullQLearning):
            worst = max(worst, fit_model(cls(4), sim["choices"], sim["rewards"], n_starts=3, seed=0).nll - nq)
    assert worst < 0.05
