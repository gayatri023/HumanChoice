"""Fixes from the third review: timeouts / availability in full-session checks, schedules, binary-safe sensitivity,
provenance enforcement, calibrated p-values, recovery-gate mapping, packaged plan, second-seed repeat."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from humanchoice.comparison.model_comparison import _signflip_p
from humanchoice.data import DatasetSpec
from humanchoice.envs import DriftingBernoulliBandit, RestlessBandit, ReversalBandit
from humanchoice.evaluation.predictive_checks import free_running_check, one_step_ahead_check
from humanchoice.fitting import fit_model
from humanchoice.models import AsymmetricQLearning, QLearning, RandomChoice, StickyQLearning
from humanchoice.pipelines import (
    call_env_factory,
    run_empirical_analysis,
    run_empirical_by_condition,
    run_sensitivity_analyses,
    simulate_restless_rehearsal,
)
from humanchoice.pipelines.empirical import packaged_analysis_plan
from humanchoice.recovery import recovery_gate_report, run_model_recovery
from humanchoice.simulation import sample_participants, simulate_dataset
from humanchoice.utils import MISSED

ROOT = Path(__file__).resolve().parents[2]


def small_ladder(K=4):
    return {"random": RandomChoice(K), "q": QLearning(K), "asym_q": AsymmetricQLearning(K), "sticky_q": StickyQLearning(K)}


FAST = {"n_starts": 2, "fractions": (0.5, 0.7), "ppc_participants": 4, "ppc_sims": 25, "profile_participants": 2, "session_sims": 15}


def n_observed(tab, k):
    return float((tab["cur"] != MISSED).sum())


# ------------------------------------------------ P0-1: timeouts and availability in the full-session check
def test_observed_timeout_pattern_is_reproduced_in_every_simulated_session():
    m = QLearning(4)
    th = [0.4, 4.0]
    sim = m.simulate(th, RestlessBandit(4), 80, 0, missed_prob=0.15)
    n_missed = int((sim["choices"] == MISSED).sum())
    assert n_missed > 3
    chk = free_running_check(m, th, lambda: RestlessBandit(4), sim["choices"], sim["rewards"], n_sims=15, statistics={"n_obs": n_observed})
    row = chk.iloc[0]
    assert row["observed"] == 80 - n_missed and row["lo"] == row["hi"] == row["replicate_mean"] == 80 - n_missed  # identical in all replicates
    none = free_running_check(m, th, lambda: RestlessBandit(4), sim["choices"], sim["rewards"], n_sims=10, statistics={"n_obs": n_observed}, missed="none")
    assert none.iloc[0]["replicate_mean"] == 80  # no-timeout mode: the mismatch the reviewer found is now opt-in and visible
    prob = free_running_check(m, th, lambda: RestlessBandit(4), sim["choices"], sim["rewards"], n_sims=60, statistics={"n_obs": n_observed}, missed=0.25)
    assert 50 < prob.iloc[0]["replicate_mean"] < 70  # explicit timeout model: about 75 % of 80 trials observed


def test_full_session_check_respects_the_availability_schedule_and_validates_inputs():
    K, T = 3, 40
    m = QLearning(K)
    rng = np.random.default_rng(0)
    avail = rng.random((T, K)) < 0.7
    avail[np.arange(T), rng.integers(0, K, T)] = True
    mask = rng.random(T) < 0.2
    sim = m.simulate([0.4, 3.0], DriftingBernoulliBandit(K, 0.1), T, 1, available=avail, missed_mask=mask)
    assert (sim["choices"][mask] == MISSED).all() and avail[np.arange(T)[~mask], sim["choices"][~mask]].all()
    chk = free_running_check(m, [0.4, 3.0], lambda: DriftingBernoulliBandit(K, 0.1), sim["choices"], sim["rewards"], n_sims=10, available=avail)
    assert chk["p_value"].notna().any()
    with pytest.raises(ValueError, match="missed must be"):
        free_running_check(m, [0.4, 3.0], lambda: DriftingBernoulliBandit(K, 0.1), sim["choices"], sim["rewards"], missed="bogus")
    with pytest.raises(ValueError, match="not both"):
        m.simulate([0.4, 3.0], DriftingBernoulliBandit(K, 0.1), T, 0, missed_prob=0.1, missed_mask=mask)
    with pytest.raises(ValueError, match="shape"):
        m.simulate([0.4, 3.0], DriftingBernoulliBandit(K, 0.1), T, 0, missed_mask=mask[:-1])


def test_replicate_p_values_use_the_finite_simulation_correction():
    m = QLearning(4)
    sim = StickyQLearning(4).simulate([0.3, 6.0, 3.0], DriftingBernoulliBandit(4, 0.1), 300, 2)
    chk = one_step_ahead_check(m, [0.3, 6.0], sim["choices"], sim["rewards"], n_sims=100, rng=0)
    assert (chk["p_value"] >= 2 / 101 - 1e-12).all() and (chk["p_value"] > 0).all()
    assert chk.set_index("statistic").loc["p_stay", "p_value"] == pytest.approx(2 / 101)  # extreme misfit: smallest attainable p, not 0


# ------------------------------------------------ P0-2: condition-aware environments and by-schedule analysis
def test_environment_factory_receives_the_condition_only_if_it_asks_for_it():
    def with_condition(condition):
        return ("cond", condition)

    def with_schedule(schedule):
        return ("sched", schedule)

    assert call_env_factory(with_condition, "s2") == ("cond", "s2")
    assert call_env_factory(with_schedule, "s3") == ("sched", "s3")
    assert isinstance(call_env_factory(RestlessBandit, "s2"), RestlessBandit)  # n_arms must NOT receive the condition
    assert RestlessBandit.__init__ and call_env_factory(lambda: 5, "x") == 5


def _two_schedule_dataset(n_each=5, T=45, seed=2):
    from humanchoice.data import BehaviouralDataset

    frames = []
    for k, sd in enumerate((0.01, 0.08)):
        m = QLearning(4)
        ds = simulate_dataset(m, sample_participants(m, n_each, seed=seed + k, prefix=f"s{k + 1}_"), lambda sd=sd: RestlessBandit(4, drift_sd=sd), T, seed=seed + k)
        frames.append(ds.df.assign(condition=f"s{k + 1}").drop(columns="missed"))
    odd = simulate_dataset(QLearning(4), sample_participants(QLearning(4), 2, seed=9, prefix="s3_"), lambda: RestlessBandit(4), T, seed=9)
    frames.append(odd.df.assign(condition="s3").drop(columns="missed"))
    return BehaviouralDataset(pd.concat(frames, ignore_index=True), tuple("ABCD"), reward_values=None, missing="mark")


@pytest.mark.slow
def test_by_condition_runner_reports_pooled_and_each_schedule_and_lists_small_ones(tmp_path):
    ds = _two_schedule_dataset()
    seen = []

    def factory(condition):
        seen.append(condition)
        return RestlessBandit(4, drift_sd={"s1": 0.01, "s2": 0.08}.get(condition, 0.03))

    res = run_empirical_by_condition(ds, tmp_path, min_participants=5, models=small_ladder(), env_factory=factory, **FAST)
    assert {"pooled", "s1", "s2", "scopes"} <= set(res) and "s3" not in res
    for sub in ("pooled", "condition_s1", "condition_s2"):
        assert (tmp_path / sub / "report.md").exists()
    scopes = pd.read_csv(tmp_path / "by_condition_scopes.csv").set_index("scope")["status"]
    assert scopes["condition_s3"].startswith("skipped") and scopes["pooled"] == "analysed"
    assert {"s1", "s2"} <= set(seen) and set(seen) <= {"s1", "s2", "s3"}  # the factory is asked for each checked participant's own schedule
    assert set(pd.read_csv(tmp_path / "by_condition_decisions.csv")["scope"]) == {"pooled", "condition_s1", "condition_s2"}


# ------------------------------------------------ P0-4: sensitivity on binary data
@pytest.mark.slow
def test_reward_range_sensitivity_is_not_applicable_to_binary_rewards_and_does_not_crash(tmp_path):
    m = QLearning(2)
    ds = simulate_dataset(m, sample_participants(m, 8, seed=1), lambda: ReversalBandit(0.8, 0.08), 60, seed=1)
    variants = ({"name": "baseline"}, {"name": "reward_range+10%", "reward_scale": 1 / 1.1})
    out = run_sensitivity_analyses(ds, tmp_path, variants=variants, n_starts=2)
    na = out[out["variant"] == "reward_range+10%"]
    assert len(na) == 1 and not na.iloc[0]["applicable"] and "not applicable" in na.iloc[0]["decision"]
    assert not out["decision_changed"].any()
    cfg = json.loads((tmp_path / "sensitivity_config.json").read_text())
    assert cfg["binary_rewards"] is True and cfg["seed"] == 0 and cfg["variants"][1]["reward_scale"] == pytest.approx(1 / 1.1)
    assert {"python", "packages"} <= set(cfg["environment"]) and "generated_utc" in cfg


# ------------------------------------------------ P1-5: provenance cannot be bypassed by omission
def _spec(provenance, synthetic=False):
    return DatasetSpec("x", "d.csv", {"participant_id": "a", "trial": "b", "choice": "c", "reward": "d"}, ["A", "B"], provenance=provenance, synthetic=synthetic)


def test_missing_empty_or_unexplained_provenance_blocks_analysis_but_documented_gaps_do_not():
    assert _spec({}).unverified_fields == ["source_url", "citation", "licence"]
    ok = {"source_url": "https://example.org/x", "citation": "Author (2020)", "licence": "CC-BY-4.0"}
    assert _spec(ok).unverified_fields == []
    assert _spec({**ok, "licence": ""}).unverified_fields == ["licence"]
    assert _spec({**ok, "licence": "UNAVAILABLE"}).unverified_fields == ["licence"]  # no reason given
    assert _spec({**ok, "licence": "UNAVAILABLE: the OSF page states no licence; the authors were contacted"}).unverified_fields == []
    assert _spec({**ok, "citation": "VERIFY"}).unverified_fields == ["citation"]
    assert _spec({}, synthetic=True).unverified_fields == []  # synthetic data are exempt from the missing-field rule


# ------------------------------------------------ P1-7: recovery gate -> parameter interpretability
def test_recovery_gate_report_uses_the_lower_confidence_bound():
    m = pd.DataFrame({"model": ["q", "q", "sticky_q", "kalman"], "param": ["alpha", "beta", "kappa", "phi"], "n_trials": 150,
                      "rho": [0.80, 0.72, 0.94, 0.71], "rho_ci_lo": [0.55, 0.70, 0.91, np.nan], "rho_ci_hi": [0.92, 0.80, 0.96, np.nan]})
    g = recovery_gate_report(m).set_index(["model", "param"])["status"]
    assert g[("q", "alpha")] == "unreliable"  # point estimate 0.80 but the lower bound is only 0.55
    assert g[("q", "beta")] == "interpretable" and g[("sticky_q", "kappa")] == "interpretable"
    assert g[("kalman", "phi")] == "not assessed (no CI)"
    assert (recovery_gate_report(m, n_trials=999).empty)


@pytest.mark.slow
def test_pipeline_labels_every_parameter_and_ships_the_plan_and_second_seed_check(tmp_path):
    ds = simulate_restless_rehearsal(n_participants=6, n_trials=50, seed=3)
    run_empirical_analysis(ds, tmp_path / "nogate", models=small_ladder(), **FAST)  # no plan path, no gate
    ip = pd.read_csv(tmp_path / "nogate" / "parameter_interpretability.csv")
    assert ip["recovery_status"].str.startswith("NO recovery gate").all()
    rep = (tmp_path / "nogate" / "report.md").read_text()
    assert "Which parameters may be interpreted?" in rep and "do not interpret" in rep and "Second-seed repeat (seed 1" in rep
    assert (tmp_path / "nogate" / "analysis_plan.md").exists()  # the packaged plan was found without any explicit path
    cfg = json.loads((tmp_path / "nogate" / "config.json").read_text())
    assert cfg["analysis_plan_sha256"] and cfg["second_seed_check"]["second_seed"] == 1 and cfg["second_seed_check"]["n_cells"] > 0
    assert (tmp_path / "nogate" / "predictive_check_misfit_rates_seed2.csv").exists()

    gate = pd.DataFrame({"model": ["q", "sticky_q"], "param": ["alpha", "kappa"], "status": ["unreliable", "interpretable"]})
    run_empirical_analysis(ds, tmp_path / "gate", models=small_ladder(), recovery_gate=gate, **FAST)
    ip2 = pd.read_csv(tmp_path / "gate" / "parameter_interpretability.csv").set_index(["model", "param"])["recovery_status"]
    assert ip2[("q", "alpha")] == "unreliable" and ip2[("sticky_q", "kappa")] == "interpretable"
    assert ip2[("asym_q", "alpha_neg")].startswith("not assessed")  # never silently interpretable


def test_the_analysis_plan_ships_inside_the_package_and_matches_the_docs():
    plan = packaged_analysis_plan()
    assert plan is not None and plan.read_text() == (ROOT / "docs" / "analysis_plan.md").read_text()


# ------------------------------------------------ P0-3: model recovery with timeouts
def test_model_recovery_passes_timeouts_to_the_generators():
    calls = []

    class Spy(QLearning):
        def simulate(self, *args, **kwargs):
            calls.append(kwargs.get("missed_prob"))
            return super().simulate(*args, **kwargs)

    run_model_recovery({"q": QLearning(4), "random": RandomChoice(4)}, lambda: RestlessBandit(4), n_per_model=2, n_trials=40, seed=0,
                       n_starts=1, generators={"spy": Spy(4)}, missed_prob=0.2)
    assert calls and all(c == 0.2 for c in calls)


# ------------------------------------------------ P1-8: the hypothesis test is calibrated
@pytest.mark.slow
def test_signflip_p_value_controls_false_positives_under_the_null_and_has_power():
    rng = np.random.default_rng(0)
    n, reps = 40, 400
    rejections = sum(_signflip_p(rng.normal(0, 1, n), 1000, seed=i) < 0.05 for i in range(reps))
    rate = rejections / reps
    assert 0.02 <= rate <= 0.09, rate  # nominal 5 %
    power = np.mean([_signflip_p(rng.normal(0.8, 1, n), 1000, seed=i) < 0.05 for i in range(60)])
    assert power > 0.9
    # heavy-tailed, skewed-looking-but-symmetric differences (like log-likelihood gaps with outliers) are also controlled
    heavy = sum(_signflip_p(rng.standard_t(2, n), 1000, seed=i) < 0.05 for i in range(reps)) / reps
    assert heavy <= 0.09, heavy
    assert 0 < _signflip_p(np.full(30, 1.0), 500, 0) < 0.01  # a perfectly consistent effect: small but never exactly 0
    assert fit_model is not None
