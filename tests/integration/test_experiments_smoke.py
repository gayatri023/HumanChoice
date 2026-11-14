"""Experiment scripts and tutorial examples must run end to end; quick runs must never touch reported results."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, check=False, capture_output=True, text=True, timeout=300)


@pytest.mark.slow
def test_quick_experiment_runs_and_does_not_overwrite_reported_results():
    reported = ROOT / "results" / "exp2_choice_consistency.csv"
    before = reported.stat().st_mtime if reported.exists() else None
    out = _run("experiments/choice_consistency/run.py", "--quick")
    assert out.returncode == 0, out.stderr
    assert (ROOT / "_quick_output" / "results" / "exp2_choice_consistency.csv").exists()
    assert (ROOT / "_quick_output" / "figures" / "exp2_choice_consistency.png").exists()
    after = reported.stat().st_mtime if reported.exists() else None
    assert before == after


@pytest.mark.parametrize("script", ["01_bamdp_bandit", "02_pomdp_tiger", "03_ipomdp_multiagent_tiger", "06_four_option_bandit_with_timeouts"])
def test_tutorial_examples_run(script):
    out = _run(f"examples/{script}.py")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip()


def test_bamdp_example_reproduces_the_hand_calculation():
    out = _run("examples/01_bamdp_bandit.py")
    assert "V_2(prior) = 1.083333 (13/12 = 1.083333 )" in out.stdout
