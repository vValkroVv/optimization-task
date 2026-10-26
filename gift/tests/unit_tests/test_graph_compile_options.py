"""AutoParallel cannot silently override disabled GraphTrainer compilation."""

import pytest

from torchtitan.config import run_pre_submit


@pytest.mark.parametrize(
    "enabled,mode,autoparallel,valid",
    [
        (False, "aot_fx_trace", True, False),
        (True, "jit", True, False),
        (True, "aot_fx_trace", True, True),
        (False, "aot_fx_trace", False, True),
        (False, "jit", False, True),
    ],
)
def test_autoparallel_requires_enabled_aot(enabled, mode, autoparallel, valid):
    result = run_pre_submit(
        "configs/experiments/graph_llama3_debug.yaml",
        [
            ("compile.enable", str(enabled).lower()),
            ("compile.mode", mode),
            ("compile.enable_autoparallel", str(autoparallel).lower()),
        ],
    )
    assert result.ok == valid, result.report.issues
    if not valid:
        assert any(
            issue.path == "compile.enable_autoparallel"
            and "AutoParallel" in issue.message
            for issue in result.report.errors
        )
