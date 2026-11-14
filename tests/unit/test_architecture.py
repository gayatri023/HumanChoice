"""Architecture boundary: the planning layer must not depend on the behavioural-modelling layer.

    planning layer  : mdp, bamdp, pomdp, ipomdp   (general-purpose uncertainty / planning algorithms)
    shared kernel   : utils                       (seeding, softmax helpers)
    behavioural     : envs, models, data, simulation, fitting, recovery, comparison, evaluation, pipelines, cli

Dependencies may only point downwards (behavioural -> planning -> kernel). Because of this, the planning layer can be
extracted into a separate package (e.g. a standalone belief-state planning library) with no code changes beyond moving
files; this test is what keeps that true.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "humanchoice"
PLANNING = ["mdp", "bamdp", "pomdp", "ipomdp"]
ALLOWED_FOR_PLANNING = set(PLANNING) | {"utils"}
BEHAVIOURAL = ["envs", "models", "data", "simulation", "fitting", "recovery", "comparison", "evaluation", "pipelines"]


def _humanchoice_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("humanchoice"):
            parts = node.module.split(".")
            found.add(parts[1] if len(parts) > 1 else (node.names[0].name if node.names else ""))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("humanchoice."):
                    found.add(alias.name.split(".")[1])
    return found - {""}


@pytest.mark.parametrize("package", PLANNING)
def test_planning_packages_do_not_import_behavioural_code(package):
    for path in (SRC / package).rglob("*.py"):
        illegal = _humanchoice_imports(path) - ALLOWED_FOR_PLANNING
        assert not illegal, f"{path.relative_to(SRC)} imports behavioural/other layers: {sorted(illegal)}"


def test_shared_kernel_is_self_contained():
    assert not (_humanchoice_imports(SRC / "utils.py") - {"utils"})


def test_planning_layer_runs_without_the_behavioural_layer_being_importable(monkeypatch):
    """Importing the planning packages must not pull in pandas-based behavioural modules."""
    import subprocess
    import sys

    code = (
        "import sys; import humanchoice.bamdp, humanchoice.pomdp, humanchoice.ipomdp, humanchoice.mdp;"
        "bad = [m for m in sys.modules if m.startswith('humanchoice.') and m.split('.')[1] in "
        f"{BEHAVIOURAL!r}]; print(bad); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stdout + out.stderr
