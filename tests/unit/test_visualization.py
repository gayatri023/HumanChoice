"""Smoke tests: every figure function runs on realistic inputs and writes a non-empty file."""
import numpy as np
import pandas as pd
import pytest

from humanchoice.envs import ReversalBandit
from humanchoice.evaluation import visualization as V
from humanchoice.models import POMDPReversalModel, QLearning
from humanchoice.pomdp import tiger_pomdp
from humanchoice.recovery import recovery_metrics, run_parameter_recovery


@pytest.fixture(scope="module")
def sim():
    return POMDPReversalModel().simulate([6.0, 0.1], ReversalBandit(0.8, 0.1), 80, rng=0)


def saved(fig, path):
    assert path.exists() and path.stat().st_size > 1000
    assert len(fig.axes) >= 1


def test_trial_level_plots(tmp_path, sim):
    p = tmp_path / "a.png"
    saved(V.plot_learning_curves({"x": (sim["choices"] == 0).astype(float)[None, :]}, window=5, save_path=p), p)
    p = tmp_path / "b.png"
    saved(V.plot_action_values(sim["values"], sim["choices"], save_path=p), p)
    p = tmp_path / "c.png"
    saved(V.plot_choice_probabilities(sim["probs"], sim["choices"], hidden_state=np.zeros(80), save_path=p), p)
    p = tmp_path / "d.png"
    saved(V.plot_observed_vs_predicted(sim["probs"], sim["choices"], save_path=p), p)


def test_recovery_and_comparison_plots(tmp_path):
    m = QLearning()
    df = run_parameter_recovery(m, lambda: ReversalBandit(), n_participants=6, n_trials=60, seed=0, n_starts=2)
    fits = df.rename(columns={})
    p = tmp_path / "e.png"
    saved(V.plot_parameter_recovery(df, m.param_names, save_path=p), p)
    p = tmp_path / "f.png"
    saved(V.plot_parameter_distributions(fits, m.param_names, save_path=p), p)
    p = tmp_path / "g.png"
    saved(V.plot_recovery_vs_trials(recovery_metrics(df, m.param_names, ci=False), save_path=p), p)
    conf = pd.DataFrame([[0.8, 0.2], [0.3, 0.7]], index=["a", "b"], columns=["a", "b"])
    p = tmp_path / "h.png"
    saved(V.plot_confusion_matrix(conf, save_path=p), p)
    summary = pd.DataFrame({"total_aic": [10.0, 14.0], "total_bic": [12.0, 13.0]}, index=["a", "b"])
    p = tmp_path / "i.png"
    saved(V.plot_model_comparison(summary, save_path=p), p)
    ho = pd.DataFrame({"model": ["a", "a", "b", "b", "chance", "chance"], "test_ll_per_trial": [-0.5, -0.55, -0.6, -0.62, -0.69, -0.69]})
    p = tmp_path / "j.png"
    saved(V.plot_predictive_performance(ho, save_path=p), p)


def test_pomdp_value_function_plot(tmp_path):
    sets = tiger_pomdp().solve_finite_horizon(3)
    p = tmp_path / "k.png"
    saved(V.plot_pomdp_value_function(sets, [1, 2, 3], save_path=p), p)
