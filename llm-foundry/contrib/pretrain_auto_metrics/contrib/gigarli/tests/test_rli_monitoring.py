from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "rli_monitoring.py"


def _load_monitoring_module() -> ModuleType:
    assert SCRIPT_PATH.exists(), "expected scripts/rli_monitoring.py to exist"
    module_name = "gigarli_rli_monitoring_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def monitoring() -> ModuleType:
    return _load_monitoring_module()


def _snapshot() -> dict[str, Any]:
    return {
        "collected_at": "2026-06-04T10:00:00Z",
        "base_url": "https://rli.example.invalid",
        "elapsed_ms": 12,
        "responses": {
            "dashboard": {
                "ok": True,
                "url": "https://rli.example.invalid/v1/system/dashboard?window=5m",
                "status_code": 200,
                "payload": {"status": "ok", "result": {}},
            }
        },
        "summary": {
            "dashboard": {
                "active_sessions": 7,
                "sessions_by_status": {"running": 6, "failed": 1},
                "avg_step_latency_ms": 180,
                "error_rate_percent_1m": 0.05,
            },
            "recent_errors_count": 0,
            "recent_errors": [],
            "failed_requests": {},
        },
    }


def test_parse_args_defaults_to_stdout_only(monitoring: ModuleType) -> None:
    args = monitoring.parse_args([])

    assert args.save_json is False
    assert args.quiet is False
    assert args.iterations == 1
    assert args.interval == 60.0
    assert args.output_dir.name == "rli_monitoring_snapshots"


def test_quiet_requires_save_json(
    monitoring: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class UnexpectedClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("client must not be created for invalid arguments")

    monkeypatch.setattr(monitoring, "RLIHttpClient", UnexpectedClient)

    assert monitoring.main(["--quiet"]) == 2

    captured = capsys.readouterr()
    assert "--quiet requires --save-json" in captured.err


def test_collect_snapshot_uses_dashboard_and_recent_errors(
    monitoring: ModuleType,
) -> None:
    class FakeClient:
        base_url = "https://rli.example.invalid"

        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, str | int] | None]] = []

        def get(
            self,
            path: str,
            query: dict[str, str | int] | None = None,
        ) -> dict[str, Any]:
            self.calls.append((path, query))
            if path == "/v1/system/dashboard":
                return {
                    "ok": True,
                    "url": "https://rli.example.invalid/v1/system/dashboard?window=15m",
                    "status_code": 200,
                    "payload": {
                        "status": "ok",
                        "result": {
                            "active_sessions": 42,
                            "sessions_by_status": {
                                "running": 30,
                                "creating": 5,
                                "failed": 2,
                                "closed": 100,
                            },
                            "avg_step_latency_ms": 234.5,
                            "error_rate_percent_1m": 0.05,
                            "ignored_extra": "not printed in summary",
                        },
                    },
                }
            if path == "/v1/errors/recent":
                return {
                    "ok": True,
                    "url": "https://rli.example.invalid/v1/errors/recent?limit=2",
                    "status_code": 200,
                    "payload": {
                        "status": "ok",
                        "result": [
                            {
                                "timestamp": "2026-06-04T09:59:59Z",
                                "http_status": 503,
                                "code": "UNAVAILABLE",
                                "category": "infrastructure_error",
                                "endpoint": "/v1/sessions/step",
                                "request_id": "req-123",
                                "id": "err-123",
                                "session_id": "sess-123",
                                "message": "Backend unavailable",
                                "diagnostic": "not printed in summary",
                            },
                            "unexpected non-dict item",
                        ],
                    },
                }
            raise AssertionError(f"unexpected path {path!r}")

    client = FakeClient()

    snapshot = monitoring._collect_snapshot(
        client,
        window="15m",
        errors_recent_limit=2,
        include_recent_errors=True,
    )

    assert client.calls == [
        ("/v1/system/dashboard", {"window": "15m"}),
        ("/v1/errors/recent", {"limit": 2}),
    ]
    assert snapshot["base_url"] == "https://rli.example.invalid"
    assert snapshot["responses"]["dashboard"]["ok"] is True
    assert snapshot["responses"]["errors_recent"]["ok"] is True
    assert snapshot["summary"]["dashboard"] == {
        "active_sessions": 42,
        "sessions_by_status": {
            "running": 30,
            "creating": 5,
            "failed": 2,
            "closed": 100,
        },
        "avg_step_latency_ms": 234.5,
        "error_rate_percent_1m": 0.05,
    }
    assert snapshot["summary"]["recent_errors_count"] == 2
    assert snapshot["summary"]["recent_errors"] == [
        {
            "timestamp": "2026-06-04T09:59:59Z",
            "http_status": 503,
            "code": "UNAVAILABLE",
            "category": "infrastructure_error",
            "endpoint": "/v1/sessions/step",
            "request_id": "req-123",
            "id": "err-123",
            "session_id": "sess-123",
            "message": "Backend unavailable",
        }
    ]


def test_collect_snapshot_can_skip_recent_errors(monitoring: ModuleType) -> None:
    class FakeClient:
        base_url = "https://rli.example.invalid"

        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, str | int] | None]] = []

        def get(
            self,
            path: str,
            query: dict[str, str | int] | None = None,
        ) -> dict[str, Any]:
            self.calls.append((path, query))
            return {
                "ok": True,
                "url": "https://rli.example.invalid/v1/system/dashboard?window=5m",
                "status_code": 200,
                "payload": {"status": "ok", "result": {"active_sessions": 1}},
            }

    client = FakeClient()

    snapshot = monitoring._collect_snapshot(
        client,
        window="5m",
        errors_recent_limit=20,
        include_recent_errors=False,
    )

    assert client.calls == [("/v1/system/dashboard", {"window": "5m"})]
    assert "errors_recent" not in snapshot["responses"]
    assert snapshot["summary"]["recent_errors_count"] == 0


def test_build_summary_records_failed_requests(monitoring: ModuleType) -> None:
    summary = monitoring._build_summary(
        {
            "dashboard": {
                "ok": False,
                "url": "https://rli.example.invalid/v1/system/dashboard?window=5m",
                "status_code": 503,
                "error": "Service unavailable",
                "payload": {
                    "status": "error",
                    "error": {
                        "code": "UNAVAILABLE",
                        "message": "Service unavailable",
                    },
                },
            }
        }
    )

    assert summary["failed_requests"] == {
        "dashboard": {
            "status_code": 503,
            "error_code": "UNAVAILABLE",
            "error": "Service unavailable",
            "url": "https://rli.example.invalid/v1/system/dashboard?window=5m",
        }
    }


def test_main_prints_summary_without_writing_json_by_default(
    monitoring: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def collect_snapshot(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return _snapshot()

    def write_snapshot(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("JSON must not be written unless --save-json is set")

    monkeypatch.setattr(monitoring, "RLIHttpClient", FakeClient)
    monkeypatch.setattr(monitoring, "_collect_snapshot", collect_snapshot)
    monkeypatch.setattr(monitoring, "_write_snapshot", write_snapshot)

    assert monitoring.main(["--iterations", "1", "--output-dir", str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert "RLI monitoring snapshot" in captured.out
    assert "active_sessions: 7" in captured.out
    assert "written:" not in captured.out
    assert list(tmp_path.iterdir()) == []


def test_main_writes_json_only_when_save_json_is_set(
    monitoring: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def collect_snapshot(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return _snapshot()

    monkeypatch.setattr(monitoring, "RLIHttpClient", FakeClient)
    monkeypatch.setattr(monitoring, "_collect_snapshot", collect_snapshot)

    assert (
        monitoring.main(
            [
                "--iterations",
                "1",
                "--save-json",
                "--quiet",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert capsys.readouterr().out == ""
    latest = tmp_path / "latest.json"
    assert latest.exists()
    assert json.loads(latest.read_text(encoding="utf-8")) == _snapshot()
    snapshot_paths = list(tmp_path.glob("rli_monitoring_*.json"))
    assert len(snapshot_paths) == 1
    assert json.loads(snapshot_paths[0].read_text(encoding="utf-8")) == _snapshot()
