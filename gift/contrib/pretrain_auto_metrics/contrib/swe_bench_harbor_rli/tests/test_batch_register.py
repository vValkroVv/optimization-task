from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from swe_bench_harbor_rli.swe_uv_image_poc.scripts import batch_register


def test_registers_only_passing_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "summary.json"
    output = tmp_path / "report.json"
    summary.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "django__django-13343",
                        "passed": True,
                        "image": "image:1.0.1",
                        "digest": "sha256:abc",
                    },
                    {
                        "task_id": "sphinx-doc__sphinx-9258",
                        "passed": False,
                        "image": None,
                        "digest": None,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, str, str | None]] = []

    def fake_register(
        name: str,
        image: str,
        *,
        digest: str | None = None,
    ) -> None:
        calls.append((name, image, digest))

    monkeypatch.setattr(batch_register, "register", fake_register)
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_register.py", str(summary), "--output", str(output), "--apply"],
    )

    assert batch_register.main() == 0
    assert calls == [
        ("swe-bench-django-13343", "image:1.0.1", "sha256:abc"),
    ]
    assert json.loads(output.read_text())["verified"] == 1


def test_rejects_summary_without_passing_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "summary.json"
    output = tmp_path / "report.json"
    summary.write_text(
        json.dumps({"tasks": [{"task_id": "task", "passed": False}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_register.py", str(summary), "--output", str(output)],
    )

    with pytest.raises(SystemExit, match="no passing tasks"):
        batch_register.main()


def test_dry_run_does_not_call_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "manifest.json"
    output = tmp_path / "report.json"
    summary.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": "django__django-13343", "image": "image:1.0.1"}
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        batch_register,
        "register",
        lambda *args, **kwargs: pytest.fail("dry-run called register"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_register.py", str(summary), "--output", str(output)],
    )

    assert batch_register.main() == 0
    assert json.loads(output.read_text())["tasks"][0]["status"] == "planned"


def test_task_filter_registers_only_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "manifest.json"
    output = tmp_path / "report.json"
    summary.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": "django__django-13343", "image": "django:1.0.1"},
                    {"task_id": "sphinx-doc__sphinx-9258", "image": "sphinx:1.0.1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        batch_register,
        "register",
        lambda name, image, digest=None: calls.append(name),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_register.py",
            str(summary),
            "--output",
            str(output),
            "--task-id",
            "django__django-13343",
            "--apply",
        ],
    )

    assert batch_register.main() == 0
    assert calls == ["swe-bench-django-13343"]


def test_failures_are_reported_and_other_tasks_continue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "manifest.json"
    output = tmp_path / "report.json"
    summary.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": "django__django-13343", "image": "django:1.0.1"},
                    {"task_id": "sphinx-doc__sphinx-9258", "image": "sphinx:1.0.1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_register(name, image, digest=None):
        calls.append(name)
        if "django" in name:
            raise TimeoutError("registry unavailable")

    monkeypatch.setattr(batch_register, "register", fake_register)
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_register.py", str(summary), "--output", str(output), "--apply"],
    )

    with pytest.raises(SystemExit, match="registration failed for 1 tasks"):
        batch_register.main()

    report = json.loads(output.read_text())
    assert calls == ["swe-bench-django-13343", "swe-bench-sphinx-9258"]
    assert report["verified"] == 1
    assert report["failed"] == 1
