"""Every ```python block in the README must run as written (documentation that cannot rot)."""
import re
from pathlib import Path

import pytest

README = (Path(__file__).resolve().parents[2] / "README.md").read_text()
BLOCKS = re.findall(r"```python\n(.*?)```", README, flags=re.DOTALL)


def test_readme_has_code_blocks():
    assert len(BLOCKS) >= 4


@pytest.mark.parametrize("code", BLOCKS, ids=[f"block{i}" for i in range(len(BLOCKS))])
def test_readme_block_runs(code):
    exec(compile(code, "<README>", "exec"), {})  # noqa: S102


def test_readme_hand_calculation_claim_is_true():
    from humanchoice.bamdp import BetaBernoulliBanditBAMDP

    planner = BetaBernoulliBanditBAMDP(2, (1.0, 1.0))
    assert planner.value(planner.initial_posterior(), 2) == pytest.approx(13 / 12)
