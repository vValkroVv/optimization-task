"""Exercise the real allocator observer with accepted CPU fake native modules."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from unittest.mock import patch

PREP = Path(__file__).resolve().parent
CAMPAIGN = PREP.parent.parent
TASK = CAMPAIGN.parents[4]


def main() -> None:
    fixture_path = TASK / "gift/tests/unit_tests/test_campaign_phase_memory.py"
    spec = importlib.util.spec_from_file_location("accepted_phase_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    required = ["TORCHTITAN_PHASE_MEMORY_RUN_ID", "TORCHTITAN_PHASE_MEMORY_SOURCE_ID",
                "TORCHTITAN_PHASE_MEMORY_CONFIG_ID"]
    results = []
    for number in range(6, 11):
        launch = (CAMPAIGN / "attempts" / f"{number:03d}" / "launch.sh").read_text()
        values = dict(re.findall(r'^export (TORCHTITAN_PHASE_MEMORY_\w+)="([^"\n]+)"$',
                                 launch, flags=re.MULTILINE))
        assert set(values) == set(required)
        native = fixture.fake_modules([
            fixture.stats(allocated_peak=100, reserved_peak=200, active_peak=80)
            for _ in range(5)
        ])
        env = dict(os.environ)
        for name in required:
            env.pop(name, None)
        env.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="8",
                   CUDA_VISIBLE_DEVICES="GPU-CPU-FIXTURE")
        with tempfile.TemporaryDirectory(dir=PREP) as temporary:
            with patch.dict(sys.modules, native.modules), patch.dict(os.environ, env, clear=True):
                adapter, module_name = fixture.load_adapter()
                try:
                    if number == 6:
                        try:
                            adapter.install_phase_memory_observer(
                                Path(temporary) / "broken", {"startup_end_step": 20, "expected_end_step": 40})
                        except RuntimeError as error:
                            assert "TORCHTITAN_PHASE_MEMORY_RUN_ID" in str(error)
                        else:
                            raise AssertionError("missing identities did not reproduce original failure")
                    os.environ.update(values)
                    observer = adapter.install_phase_memory_observer(
                        Path(temporary) / "fixed", {"startup_end_step": 20, "expected_end_step": 40})
                    try:
                        assert (observer.run_id, observer.source_id, observer.config_id) == tuple(values[n] for n in required)
                        processor = native.processor_type()
                        for step in (1, 10, 20, 30, 40):
                            fixture.run_log(processor, step)
                        observer.finish_and_flush()
                        doc = json.loads(observer.output_path.read_text())
                        assert doc["status"] == "complete", doc["coverage"]
                        assert doc["coverage"]["last_training_metrics_step"] == 40
                        assert doc["phase_boundaries"] == {"startup_end_step": 20, "expected_end_step": 40}
                        assert native.device.memory_stats_calls == 5
                        assert native.device.reset_calls == 6
                    finally:
                        observer.restore()
                finally:
                    sys.modules.pop(module_name, None)
        results.append({"attempt": f"{number:03d}", "constructor_identities": values,
                        "status": "complete", "native_fake_metrics_steps": [1, 10, 20, 30, 40]})
    report = {"status": "PASS", "original_missing_identity_failure_reproduced": True,
              "actual_adapter_constructor_and_installer": True,
              "fixture": str(fixture_path.relative_to(TASK)),
              "real_torch_imported": False, "gpu_initialization_or_launch": False,
              "results": results}
    (PREP / "cpu-identity-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: original missing identity reproduced; all five corrected launch identities installed, completed fake native metrics40, and restored.")


if __name__ == "__main__":
    main()
