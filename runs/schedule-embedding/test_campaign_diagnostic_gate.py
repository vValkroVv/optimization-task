"""Stubbed executor control-flow checks for the update-21 diagnostic gate."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


CAMPAIGN_PATH = Path(__file__).with_name("campaign.py")
SPEC = importlib.util.spec_from_file_location(
    "_schedule_embedding_diagnostic_gate_campaign", CAMPAIGN_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import campaign from {CAMPAIGN_PATH}")
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def _stub_executor(tmp_path, monkeypatch, *, prior_timing=False, failed_run=None):
    cache = tmp_path / "saved-cache"
    cache.mkdir()
    raw = tmp_path / "raw"
    receipts = tmp_path / "campaign.jsonl"
    freeze = {"test": "stable-freeze"}
    freeze_sha256 = campaign._freeze_sha256(freeze)

    monkeypatch.setattr(campaign, "RAW", raw)
    monkeypatch.setattr(campaign, "RECEIPTS", receipts)
    monkeypatch.setattr(campaign, "SAVED_D_CACHE", cache)
    monkeypatch.setattr(campaign, "PYTHON", sys.executable)
    monkeypatch.setattr(campaign, "_requested_signal", None)
    monkeypatch.setattr(campaign, "_active", None)
    monkeypatch.delenv("ABCD_COMPONENT_FIXTURE_DIR", raising=False)

    base = SimpleNamespace(
        _validate_completed_compiler_cache_root=lambda _path: None
    )
    flash = SimpleNamespace(_load_base_runner=lambda: base)
    monkeypatch.setattr(
        campaign, "_load_runner_modules", lambda: (object(), flash)
    )
    monkeypatch.setattr(campaign, "_load_module", lambda *_args: object())
    monkeypatch.setattr(campaign, "_campaign_freeze", lambda *_args: freeze)
    monkeypatch.setattr(campaign, "_run_argv", lambda name: [name])
    monkeypatch.setattr(
        campaign,
        "_validate_finished",
        lambda name, *_args: {"name": name, "status": "PASS"},
    )
    monkeypatch.setattr(
        campaign, "_preflight_idle_h100s", lambda: {"stub": "idle"}
    )
    monkeypatch.setattr(campaign, "_wait_group_exit", lambda *_args: True)
    monkeypatch.setattr(campaign, "_stop_owned_group", lambda *_args: True)

    launched = []

    class FakeProcess:
        def __init__(self, argv, **_kwargs):
            self.name = argv[0]
            self.pid = 1000 + len(launched)
            self.code = int(self.name == failed_run)
            launched.append(self.name)

        def wait(self, timeout):
            assert timeout == 1
            return self.code

    monkeypatch.setattr(campaign.subprocess, "Popen", FakeProcess)

    if prior_timing:
        with receipts.open("w", encoding="utf-8") as handle:
            for name in campaign.TIMING_RUNS:
                handle.write(
                    json.dumps(
                        {
                            "event": "run_complete",
                            "name": name,
                            "run_id": campaign.RUN_IDS[name],
                            "source_freeze_sha256": freeze_sha256,
                        }
                    )
                    + "\n"
                )
    return receipts, launched


def _read_receipts(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_full_campaign_runs_all_four_diagnostics_after_all_timing_pass(
    tmp_path, monkeypatch
):
    receipts_path, launched = _stub_executor(tmp_path, monkeypatch)

    assert campaign._execute(list(campaign.RUN_NAMES)) == 0

    records = _read_receipts(receipts_path)
    completed = next(row for row in records if row.get("event") == "campaign_complete")
    assert completed["status"] == "PASS"
    assert completed["completed"] == sorted(campaign.RUN_NAMES)
    assert launched == list(campaign.RUN_NAMES)
    assert not any(row.get("event") == "diagnostics_skipped" for row in records)


def test_diagnostics_only_uses_prior_validated_timing_receipts(tmp_path, monkeypatch):
    receipts_path, launched = _stub_executor(
        tmp_path, monkeypatch, prior_timing=True
    )

    assert campaign._execute(list(campaign.DIAGNOSTIC_RUNS)) == 0

    records = _read_receipts(receipts_path)
    completed = next(row for row in records if row.get("event") == "campaign_complete")
    assert completed["status"] == "PASS"
    assert completed["completed"] == sorted(campaign.DIAGNOSTIC_RUNS)
    assert launched == list(campaign.DIAGNOSTIC_RUNS)
    assert not any(row.get("event") == "diagnostics_skipped" for row in records)


def test_failed_timing_run_blocks_all_diagnostics(tmp_path, monkeypatch):
    failed_run = campaign.TIMING_RUNS[2]
    receipts_path, launched = _stub_executor(
        tmp_path, monkeypatch, failed_run=failed_run
    )

    assert campaign._execute(list(campaign.RUN_NAMES)) == 1

    records = _read_receipts(receipts_path)
    skipped = next(row for row in records if row.get("event") == "diagnostics_skipped")
    completed = next(row for row in records if row.get("event") == "campaign_complete")
    assert failed_run in skipped["reason"]
    assert completed["status"] == "FAIL"
    assert launched == list(campaign.TIMING_RUNS)
    assert not any(name in campaign.DIAGNOSTIC_RUNS for name in launched)
