"""Targeted CPU reproduction; imports no real torch and changes no staged files."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
STAGING = HERE.parent.parent / "PHASE-MEMORY-IMPLEMENTATION" / "001" / "staged"
TEST = STAGING / "gift/tests/unit_tests/test_campaign_phase_memory.py"
spec = importlib.util.spec_from_file_location("phase_review_fakes", TEST)
fakes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fakes)

with pytest.MonkeyPatch.context() as patch:
    counters = fakes.stats(allocated_peak=900, reserved_peak=950, active_peak=90)
    del counters["allocated_bytes.all.peak"]
    native = fakes.fake_modules([counters])
    for name, module in native.modules.items():
        patch.setitem(sys.modules, name, module)
    for name, value in {
        "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
        "TORCHTITAN_PHASE_MEMORY_RUN_ID": "review",
        "TORCHTITAN_PHASE_MEMORY_SOURCE_ID": "007",
        "TORCHTITAN_PHASE_MEMORY_CONFIG_ID": "fake",
    }.items():
        patch.setenv(name, value)
    adapter, name = fakes.load_adapter()
    observer = adapter.install_phase_memory_observer(
        HERE / "unused-output", {"startup_end_step": 1, "expected_end_step": 2}
    )
    try:
        processor = native.processor_type()
        processor.log(1, 1.0, 1.0, 1.0)
        snapshot = next(r for r in observer._records if r["event_kind"] == "native_snapshot")
        envelope = observer._phase_envelopes()[0]
        result = {
            "snapshot_status": snapshot["status"],
            "snapshot_reasons": snapshot["reasons"],
            "envelope_status": envelope["status"],
            "envelope_partial_values_only": envelope["partial_values_only"],
            "envelope_allocated_peak": envelope["max_allocated_peak_bytes"],
            "document_status": observer._document(None)["status"],
            "memory_stats_calls": native.device.memory_stats_calls,
            "mem_get_info_calls": native.device.mem_get_info_calls,
            "reset_calls": native.device.reset_calls,
        }
        assert snapshot["reasons"]
        assert snapshot["status"] == "complete"
        assert envelope["status"] == "complete"
        assert envelope["max_allocated_peak_bytes"] is None
        assert result["document_status"] == "invalid"
        print(json.dumps(result, indent=2))
    finally:
        observer.restore()
        sys.modules.pop(name, None)
