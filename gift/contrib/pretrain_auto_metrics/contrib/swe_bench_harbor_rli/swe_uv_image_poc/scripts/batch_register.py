from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .registry import (
        ENV_VERSION,
        HARNESS_ENV_VERSION,
        SUPPORTED_ENV_VERSIONS,
        environment_name,
        is_sha256_digest,
        register,
    )
except ImportError:
    from registry import (
        ENV_VERSION,
        HARNESS_ENV_VERSION,
        SUPPORTED_ENV_VERSIONS,
        environment_name,
        is_sha256_digest,
        register,
    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--version", choices=SUPPORTED_ENV_VERSIONS)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    tasks = [
        task
        for task in payload["tasks"]
        if isinstance(task, dict) and task.get("passed", True) is True
    ]
    task_ids = [task.get("task_id") for task in tasks]
    if not all(isinstance(task_id, str) for task_id in task_ids):
        raise SystemExit("registration input contains an invalid task_id")
    if len(set(task_ids)) != len(task_ids):
        raise SystemExit("registration input contains duplicate task_id values")
    missing = sorted(set(args.task_id) - set(task_ids))
    if missing:
        raise SystemExit(f"registration tasks not found: {', '.join(missing)}")
    if args.task_id:
        selected = set(args.task_id)
        tasks = [task for task in tasks if task["task_id"] in selected]
    if not tasks:
        raise SystemExit("register:batch has no passing tasks")

    version = args.version or ENV_VERSION
    registration_options: dict[str, str] = {}
    if args.version is not None:
        registration_options["version"] = version
    if args.apply and version == HARNESS_ENV_VERSION:
        invalid_task = next(
            (
                task["task_id"]
                for task in tasks
                if not is_sha256_digest(task.get("digest"))
            ),
            None,
        )
        if invalid_task is not None:
            raise SystemExit(
                f"sha256 image digest missing or invalid for {invalid_task}"
            )

    report: dict[str, Any] = {
        "apply": args.apply,
        "version": version,
        "total": len(tasks),
        "verified": 0,
        "failed": 0,
        "tasks": [],
    }
    for task in tasks:
        task_id = task["task_id"]
        image = task.get("image")
        if not isinstance(image, str) or not image:
            raise SystemExit(f"registration image missing for {task_id}")
        result: dict[str, object] = {"task_id": task_id, "image": image}
        if not args.apply:
            result["status"] = "planned"
        else:
            digest = task.get("digest")
            try:
                register(
                    environment_name(task_id),
                    image,
                    digest=digest if isinstance(digest, str) else None,
                    **registration_options,
                )
                result["status"] = "verified"
                report["verified"] += 1
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = f"{type(exc).__name__}: {exc}"
                report["failed"] += 1
        report["tasks"].append(result)
        _write_report(args.output, report)

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if report["failed"]:
        raise SystemExit(f"registration failed for {report['failed']} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
