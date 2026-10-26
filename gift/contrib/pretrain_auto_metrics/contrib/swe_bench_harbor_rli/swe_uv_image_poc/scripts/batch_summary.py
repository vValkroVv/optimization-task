from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        return None
    value = str(values[0])
    return value.rsplit("@", 1)[1] if "@" in value else value


def summarize(task_ids: list[str], output_root: Path) -> dict[str, object]:
    tasks: list[dict[str, object]] = []
    for task_id in task_ids:
        task_root = output_root / task_id
        report = _load_json(task_root / "offline-report.json")
        failure = _load_json(task_root / "failure-report.json")
        image = report.get("image")
        digest = _digest(task_root / "image-digests.json")
        passed = (
            report.get("reward") == 1
            and report.get("resolved") is True
            and isinstance(image, str)
            and bool(image)
            and digest is not None
        )
        tasks.append(
            {
                "task_id": task_id,
                "passed": passed,
                "reward": report.get("reward"),
                "resolved": report.get("resolved"),
                "image": image,
                "digest": digest,
                "failure_stage": None
                if passed
                else failure.get("stage", "unknown"),
                "exit_code": None if passed else failure.get("exit_code"),
                "artifact_dir": str(task_root),
            }
        )
    failure_stages = Counter(
        task["failure_stage"] for task in tasks if not task["passed"]
    )
    return {
        "total": len(tasks),
        "passed": sum(1 for task in tasks if task["passed"]),
        "failed": sum(1 for task in tasks if not task["passed"]),
        "failure_stages": dict(sorted(failure_stages.items())),
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("task_ids", nargs="+")
    args = parser.parse_args()
    summary = summarize(args.task_ids, args.output_root)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
