"""Corrections to the analysis pipeline: check families, declared misfit flag, order-independent selection, win definition."""

import numpy as np
import pandas as pd
import pytest

from humanchoice.data import BehaviouralDataset
from humanchoice.envs import RestlessBandit
from humanchoice.evaluation.predictive_checks import (
    ONE_STEP_STATISTICS,
    SESSION_STATISTICS,
    _table,
    free_running_check,
    make_one_step_statistics,
    make_session_statistics,
    make_statistics,
    misfit_rate,
    one_step_ahead_check,
)
from humanchoice.models import QLearning
from humanchoice.pipelines import run_empirical_analysis, simulate_restless_rehearsal
from humanchoice.pipelines.empirical import select_participants, select_profile_participants


# ------------------------------------------------------------- A: two families of checks
def test_one_step_check_evaluates_exactly_the_choice_only_statistics_with_no_silent_nan():
    m = QLearning(4)
    sim = m.simulate([0.4, 4.0], RestlessBandit(4), 120, 0)
    chk = one_step_ahead_check(m, [0.4, 4.0], sim["choices"], sim["rewards"], n_sims=40)
    assert tuple(chk["statistic"]) == ONE_STEP_STATISTICS
    assert chk["p_value"].notna().all() and chk["replicate_mean"].notna().all()
    assert not set(SESSION_STATISTICS) & set(chk["statistic"])


def test_asking_the_one_step_check_for_reward_statistics_fails_loudly():
    m = QLearning(4)
    sim = m.simulate([0.4, 4.0], RestlessBandit(4), 60, 0)
    with pytest.raises(ValueError, match="free_running_check"):
        one_step_ahead_check(m, [0.4, 4.0], sim["choices"], sim["rewards"], n_sims=10, statistics=make_session_statistics())


def test_session_statistics_come_from_the_free_running_check_and_are_all_defined():
    m = QLearning(4)
    th = [0.4, 4.0]
    sim = m.simulate(th, RestlessBandit(4), 120, 1)
    chk = free_running_check(m, th, lambda: RestlessBandit(4), sim["choices"], sim["rewards"], n_sims=40,
                             statistics=make_session_statistics(), rng=0)
    assert tuple(chk["statistic"]) == SESSION_STATISTICS and chk["p_value"].notna().all()
    assert set(make_one_step_statistics()) | set(make_session_statistics()) == set(make_statistics())
    # reward statistics are computed from the observed rewards
    tab = _table(sim["choices"], sim["rewards"], 0.5)
    stats = make_session_statistics()
    assert stats["mean_reward"](tab, 4) == pytest.approx(sim["rewards"].mean())
    assert stats["total_reward"](tab, 4) == pytest.approx(sim["rewards"].sum())


# --------------------------------------------------------------- B: declared flag
def _pvals(n, n_low, model="m", stat="p_stay"):
    return pd.DataFrame({"model": model, "statistic": stat, "p_value": [0.01] * n_low + [0.5] * (n - n_low)})


def test_misfit_rate_applies_the_declared_threshold_with_wilson_intervals():
    t = pd.concat([_pvals(100, 25, "elevated"), _pvals(100, 40, "flagged"), _pvals(100, 4, "chance")])
    r = misfit_rate(t, flag_threshold=0.30).set_index("model")
    assert r.loc["elevated", "misfit_rate"] == 0.25 and not r.loc["elevated", "population_flag"]
    assert r.loc["elevated", "interpretation"] == "elevated, below flag threshold"
    assert r.loc["flagged", "population_flag"] and r.loc["flagged", "interpretation"] == "flagged"
    assert not r.loc["chance", "population_flag"] and r.loc["chance", "interpretation"] == "consistent with chance"
    for col in ("n_participants", "expected_by_chance", "ci_lower", "ci_upper", "population_flag", "flag_threshold"):
        assert col in r.columns
    e = r.loc["elevated"]
    assert e["ci_lower"] < 0.25 < e["ci_upper"] and e["ci_lower"] > 0.05  # credibly above chance, but not a formal flag
    assert misfit_rate(t, flag_threshold=0.20).set_index("model").loc["elevated", "population_flag"]


def test_misfit_rate_groups_by_statistic_and_works_without_a_model_column():
    t = pd.concat([_pvals(20, 10, stat="a"), _pvals(20, 0, stat="b")]).drop(columns="model")
    r = misfit_rate(t).set_index("statistic")
    assert r.loc["a", "misfit_rate"] == 0.5 and r.loc["b", "misfit_rate"] == 0.0


# ------------------------------------------------ C: order-independent selection
def _dataset(n=40, T=30, seed=0, with_condition=False):
    ds = simulate_restless_rehearsal(n_participants=n, n_trials=T, seed=seed)
    df = ds.df.drop(columns="missed")
    if with_condition:
        cond = {p: ("s1", "s2", "s3")[i % 3] for i, p in enumerate(sorted(df["participant_id"].unique()))}
        df = df.assign(condition=df["participant_id"].map(cond))
    return BehaviouralDataset(df, ds.choice_categories, reward_values=None, missing="mark")


def test_participant_selection_does_not_depend_on_row_order_but_does_on_the_seed():
    ds = _dataset()
    rng = np.random.default_rng(0)
    ids = np.array(ds.participants)
    shuffled_df = ds.df.drop(columns="missed")
    order = rng.permutation(ids)
    # same rows, participants appear in a different order in the file
    reordered = pd.concat([shuffled_df[shuffled_df.participant_id == p] for p in order], ignore_index=True)
    ds2 = BehaviouralDataset(reordered, ds.choice_categories, reward_values=None, missing="mark")
    assert ds.participants != ds2.participants
    assert select_participants(ds, 10, seed=3) == select_participants(ds2, 10, seed=3)
    assert select_participants(ds, 10, seed=3) != select_participants(ds, 10, seed=4)
    assert select_participants(ds, 100, seed=3) == sorted(ds.participants)  # asking for more than exist: everyone
    first_ten = ds.participants[:10]
    assert select_participants(ds, 10, seed=3) != sorted(first_ten)  # not simply the first ones in file order


def test_selection_is_stratified_by_condition_in_proportion():
    ds = _dataset(n=30, with_condition=True)
    chosen = select_participants(ds, 12, seed=1, stratify_by="condition")
    cond = ds.df.groupby("participant_id")["condition"].first()
    counts = cond[chosen].value_counts()
    assert len(chosen) == 12 and set(counts) == {4}  # 10/10/10 participants -> 4/4/4


def test_profile_participants_span_the_range_of_fit_quality_and_ignore_row_order():
    fits = pd.DataFrame({"participant_id": [f"p{i:02d}" for i in range(20)], "nll": np.linspace(50, 150, 20)[::-1], "n_obs": 100})
    ids = select_profile_participants(fits, 5)
    q = fits.set_index("participant_id").loc[ids, "nll"] / 100
    assert len(ids) == 5 and q.min() == pytest.approx(0.5) and q.max() == pytest.approx(1.5)  # best and worst fit both included
    assert select_profile_participants(fits.sample(frac=1, random_state=0), 5) == ids


# --------------------------------------------------- D: one definition of "win"
def test_descriptive_summary_and_predictive_checks_share_the_same_win_definition():
    rng = np.random.default_rng(0)
    T = 400
    df = pd.DataFrame({"participant_id": "p", "trial": np.arange(1, T + 1), "choice": rng.choice(list("ABCD"), T),
                       "reward": rng.uniform(0, 1, T)})  # continuous: almost every reward is > 0
    ds = BehaviouralDataset(df, tuple("ABCD"), reward_values=None)
    c, r = ds.get("p")
    for thr in (0.3, 0.5, 0.7):
        summary = ds.participant_summary(win_threshold=thr).iloc[0]
        stats = make_statistics(thr)
        tab = _table(c, r, thr)
        assert summary["p_win_stay"] == pytest.approx(stats["p_stay_after_win"](tab, 4))
        assert summary["p_lose_stay"] == pytest.approx(stats["p_stay_after_loss"](tab, 4))
    # the old definition (reward > 0) would call nearly every trial a win and so differ from the threshold-based one
    old_style_win_rate = (r[:-1] > 0).mean()
    assert old_style_win_rate > 0.99 and (r[:-1] >= 0.5).mean() < 0.6
    # binary rewards: threshold 0.5 reproduces the natural win = 1 / loss = 0 split
    b = df.assign(reward=(rng.random(T) < 0.5).astype(float))
    bds = BehaviouralDataset(b, tuple("ABCD"))
    s = bds.participant_summary().iloc[0]
    tab = _table(*bds.get("p"), 0.5)
    assert s["p_win_stay"] == pytest.approx(make_statistics(0.5)["p_stay_after_win"](tab, 4))


# ----------------------------------------------- pipeline: reporting & provenance
@pytest.mark.slow
def test_pipeline_reports_check_families_provenance_and_is_row_order_invariant(tmp_path):
    ds = simulate_restless_rehearsal(n_participants=8, n_trials=50, seed=5)
    from humanchoice.models import AsymmetricQLearning, RandomChoice, StickyQLearning

    small = {"random": RandomChoice(4), "q": QLearning(4), "asym_q": AsymmetricQLearning(4), "sticky_q": StickyQLearning(4)}
    kw = {"n_starts": 2, "fractions": (0.5, 0.7), "ppc_participants": 4, "ppc_sims": 30, "profile_participants": 2, "session_sims": 20, "models": small}
    spec = '{"name": "toy"}'
    plan = tmp_path / "plan.md"
    plan.write_text("# plan v1")
    run_empirical_analysis(ds, tmp_path / "a", spec_json=spec, analysis_plan_path=plan, **kw)  # no environment supplied
    rep = (tmp_path / "a" / "report.md").read_text()
    assert "**Not run.**" in rep and "not evaluated" in rep and "Statistics evaluated:" in rep
    assert "flagged" in rep.lower() and not (tmp_path / "a" / "predictive_checks_session.csv").exists()
    import json

    cfg = json.loads((tmp_path / "a" / "config.json").read_text())
    assert cfg["dataset_spec_sha256"] and cfg["analysis_plan_sha256"] and cfg["environment"]["packages"]["scipy"] != "unknown"
    assert set(cfg["environment"]["packages"]) >= {"numpy", "scipy", "pandas", "matplotlib", "humanchoice"}
    for f in ("dataset_spec.json", "environment.txt", "analysis_plan.md"):
        assert (tmp_path / "a" / f).exists()
    assert len(cfg["selected_participants"]["predictive_checks"]) == 4

    # with the task environment supplied the session checks are run and reported
    run_empirical_analysis(ds, tmp_path / "b", env_factory=lambda: RestlessBandit(4), **kw)
    rep_b = (tmp_path / "b" / "report.md").read_text()
    assert "Not run" not in rep_b and (tmp_path / "b" / "predictive_checks_session.csv").exists()
    sess = pd.read_csv(tmp_path / "b" / "predictive_checks_session.csv")
    assert set(sess["statistic"]) == set(SESSION_STATISTICS) and sess["p_value"].notna().all()

    # shuffling the file order cannot change the selected participants
    df = ds.df.drop(columns="missed")
    order = np.random.default_rng(1).permutation(ds.participants)
    shuffled = pd.concat([df[df.participant_id == p] for p in order], ignore_index=True)
    ds2 = BehaviouralDataset(shuffled, ds.choice_categories, reward_values=None, missing="mark")
    run_empirical_analysis(ds2, tmp_path / "c", **kw)
    cfg_c = json.loads((tmp_path / "c" / "config.json").read_text())
    assert cfg_c["selected_participants"]["predictive_checks"] == cfg["selected_participants"]["predictive_checks"]
