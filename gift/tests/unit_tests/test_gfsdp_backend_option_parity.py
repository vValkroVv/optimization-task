"""GFSDP refusals are visible before runtime; hardware checks use mocked devices."""

import importlib.util
import sys
from pathlib import Path
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from torchtitan.config.pre_submit import run_pre_submit


@pytest.mark.parametrize(
    "flag",
    ["parallelism.enable_fsdp_symm_mem", "training.enable_cpu_offload"],
)
def test_gfsdp_rejects_unsupported_options_before_runtime(flag):
    result = run_pre_submit(
        "configs/experiments/gift_moe_debug.yaml",
        [("parallelism.fsdp_backend", "gfsdp2"), (flag, "true")],
    )
    matching = [issue for issue in result.report.errors if issue.path == flag]
    assert len(matching) == 1
    assert "not supported on fsdp_backend='gfsdp2'" in matching[0].message


@pytest.mark.parametrize(
    "flag",
    ["parallelism.enable_fsdp_symm_mem", "training.enable_cpu_offload"],
)
def test_fsdp_keeps_these_options_for_runtime_hardware_validation(flag):
    result = run_pre_submit("configs/experiments/llama3_debug.yaml", [(flag, "true")])
    assert result.ok, result.report.errors


@pytest.mark.parametrize(
    "available,hip,capability,expected,queries_capability",
    [
        (False, None, (9, 0), False, False),
        (True, "rocm", (9, 0), False, False),
        (True, None, (8, 9), False, True),
        (True, None, (9, 0), True, True),
        (True, None, (10, 0), True, True),
    ],
)
def test_shared_symmetric_memory_rule_never_queries_real_cuda(
    monkeypatch, available, hip, capability, expected, queries_capability
):
    # Load the actual leaf helper without initializing distributed package code.
    helper_path = (
        Path(__file__).resolve().parents[2]
        / "torchtitan/distributed/fsdp_capabilities.py"
    )
    spec = importlib.util.spec_from_file_location("fsdp_capability_test", helper_path)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    capability_query = Mock(return_value=capability)
    fake_torch = ModuleType("torch")
    fake_version = ModuleType("torch.version")
    vars(fake_version)["hip"] = hip
    vars(fake_torch).update(
        cuda=SimpleNamespace(
            is_available=Mock(return_value=available),
            get_device_capability=capability_query,
        ),
        version=fake_version,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.version", fake_version)
    check = cast(Callable[[], bool], vars(helper)["fsdp_symmetric_memory_supported"])
    assert check() is expected
    assert capability_query.call_count == int(queries_capability)


@pytest.mark.parametrize("enabled", [False, True])
def test_preflight_uses_shared_hardware_rule_only_when_requested(monkeypatch, enabled):
    from torchtitan.config.preflight import _check_symm_mem
    from torchtitan.config.validation import ValidationReport

    check = Mock(return_value=False)
    helper = ModuleType("torchtitan.distributed.fsdp_capabilities")
    vars(helper)["fsdp_symmetric_memory_supported"] = check
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    report = ValidationReport(stage="in_container")
    _check_symm_mem(SimpleNamespace(enable_fsdp_symm_mem=enabled), report)
    assert check.call_count == int(enabled)
    assert len(report.errors) == int(enabled)
