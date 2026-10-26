from __future__ import annotations

import json
from pathlib import Path

from swe_bench_harbor_rli.swe_uv_image_poc.scripts.batch_summary import summarize


def test_summary_contains_pass_fail_image_digest_and_artifacts(tmp_path: Path) -> None:
    task_root = tmp_path / "django__django-13343"
    task_root.mkdir()
    (task_root / "offline-report.json").write_text(
        json.dumps({"reward": 1, "resolved": True, "image": "image:1.0.1"}),
        encoding="utf-8",
    )
    (task_root / "image-digests.json").write_text(
        json.dumps(["image@sha256:abc"]),
        encoding="utf-8",
    )

    summary = summarize(["django__django-13343"], tmp_path)

    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["tasks"][0]["digest"] == "sha256:abc"


def test_summary_rejects_passing_result_without_digest(tmp_path: Path) -> None:
    task_root = tmp_path / "django__django-13343"
    task_root.mkdir()
    (task_root / "offline-report.json").write_text(
        json.dumps({"reward": 1, "resolved": True, "image": "image:1.0.1"}),
        encoding="utf-8",
    )

    summary = summarize(["django__django-13343"], tmp_path)

    assert summary["passed"] == 0
    assert summary["failed"] == 1


def test_summary_aggregates_failure_stages(tmp_path: Path) -> None:
    task_root = tmp_path / "django__django-13343"
    task_root.mkdir()
    (task_root / "failure-report.json").write_text(
        json.dumps({"stage": "offline_validation", "exit_code": 1}),
        encoding="utf-8",
    )

    summary = summarize(["django__django-13343"], tmp_path)

    assert summary["failure_stages"] == {"offline_validation": 1}
    assert summary["tasks"][0]["failure_stage"] == "offline_validation"
    assert summary["tasks"][0]["exit_code"] == 1
