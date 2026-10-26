from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

try:
    from .image_builder import find_task_assets, source_image_from_dockerfile, target_image
except ImportError:
    from image_builder import find_task_assets, source_image_from_dockerfile, target_image


EXPECTED_TASK_COUNT = 500
BATCH_SIZE = 5
PILOT_PATH = Path(__file__).with_name("pilot_tasks.txt")
BUILT_PREFIX = "# built:"


def read_pilot_tasks(path: Path = PILOT_PATH) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def read_built_tasks(path: Path = PILOT_PATH) -> list[str]:
    return [
        line.strip()[len(BUILT_PREFIX) :].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(BUILT_PREFIX)
    ]


def build_manifest(dataset_root: Path, *, require_full: bool = True) -> list[dict[str, str]]:
    task_ids = sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and "__" in path.name
    )
    if require_full and len(task_ids) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TASK_COUNT} tasks, found {len(task_ids)} in {dataset_root}"
        )
    records: list[dict[str, str]] = []
    for task_id in task_ids:
        assets = find_task_assets(task_id, dataset_root)
        source_image = source_image_from_dockerfile(assets.environment_dockerfile)
        records.append(
            {
                "task_id": task_id,
                "source_image": source_image,
                "image": target_image(source_image),
                "assets": str(assets.root),
            }
        )
    return records


def select_tasks(
    records: Iterable[dict[str, str]],
    mode: str,
    batch_index: int | None = None,
    pilot_tasks: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    records_by_id = {record["task_id"]: record for record in records}
    pilot = list(pilot_tasks if pilot_tasks is not None else read_pilot_tasks())
    built = set() if pilot_tasks is not None else set(read_built_tasks())
    if mode == "pilot":
        missing = [task_id for task_id in pilot if task_id not in records_by_id]
        if missing:
            raise ValueError(f"pilot tasks missing from manifest: {', '.join(missing)}")
        return [records_by_id[task_id] for task_id in pilot]
    if mode not in ("all", "batch"):
        raise ValueError(f"unsupported image mode: {mode}")
    pilot_set = set(pilot) | built
    candidates = [record for record in records if record["task_id"] not in pilot_set]
    if mode == "all":
        return candidates
    if batch_index is None or batch_index < 0:
        raise ValueError("SWE_BATCH_INDEX must be a non-negative integer")
    start = batch_index * BATCH_SIZE
    selected = candidates[start : start + BATCH_SIZE]
    if not selected:
        raise ValueError(f"batch index {batch_index} is outside the manifest")
    return selected


def write_manifest(path: Path, dataset_ref: str, records: list[dict[str, str]]) -> None:
    path.write_text(
        json.dumps(
            {"dataset_ref": dataset_ref, "task_count": len(records), "tasks": records},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-ref", required=True)
    args = parser.parse_args()
    records = build_manifest(args.dataset_root)
    write_manifest(args.output, args.dataset_ref, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
