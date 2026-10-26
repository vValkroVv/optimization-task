from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from .manifest import BATCH_SIZE, read_pilot_tasks, select_tasks
except ImportError:
    from manifest import BATCH_SIZE, read_pilot_tasks, select_tasks


IMAGE_VERSION = "1.0.1"
HARNESS_IMAGE_VERSION = "1.0.2"
HARNESS_PILOT_PATH = Path(__file__).with_name("pilot_tasks_1_0_2.txt")


def _job_name(task_id: str) -> str:
    return "build-" + re.sub(r"[^a-zA-Z0-9]+", "-", task_id).strip("-")


def render_child_pipeline(
    records: list[dict[str, str]],
    mode: str,
    batch_index: int | None = None,
    pilot_tasks: list[str] | None = None,
    image_version: str = IMAGE_VERSION,
    max_parallel_jobs: int = BATCH_SIZE,
) -> str:
    if max_parallel_jobs < 1:
        raise ValueError("max_parallel_jobs must be a positive integer")
    if image_version == HARNESS_IMAGE_VERSION:
        build_script = "build_push_image_1_0_2.sh"
        validation_root = "swe_uv_image_poc/outputs/validation-1.0.2"
        if pilot_tasks is None:
            pilot_tasks = read_pilot_tasks(HARNESS_PILOT_PATH)
    elif image_version == IMAGE_VERSION:
        build_script = "build_push_uv_image.sh"
        validation_root = "swe_uv_image_poc/outputs/validation"
    else:
        raise ValueError(f"unsupported image version: {image_version}")
    selected = select_tasks(records, mode, batch_index, pilot_tasks)
    jobs = [_job_name(record["task_id"]) for record in selected]
    lines = [
        "stages:",
        "  - build",
        "  - summary",
        "",
        ".swe-image-build:",
        "  stage: build",
        "  tags: [advanced]",
        "  allow_failure: true",
        "  variables:",
        '    DOCKER_BUILDKIT: "1"',
        (
            "    SWE_BENCH_DATASET_ROOT: "
            '"$CI_PROJECT_DIR/swe_uv_image_poc/build/harbor-datasets/'
            'datasets/swebench-verified"'
        ),
        "  needs:",
        '    - pipeline: "$PARENT_PIPELINE_ID"',
        '      job: "prepare:swe-dataset"',
        "      artifacts: true",
    ]
    if image_version == HARNESS_IMAGE_VERSION:
        lines.extend(
            [
                '    - pipeline: "$PARENT_PIPELINE_ID"',
                '      job: "build:harness-runtime-1.0.2"',
                "      artifacts: true",
            ]
        )
    lines.extend(
        [
            "  before_script:",
            "    - docker --version",
            '    - test -n "$RLI_USER"',
            '    - test -n "$RLI_PASSWORD"',
            (
                '    - echo "$RLI_PASSWORD" | docker login docker.sberdevices.ru '
                '-u "$RLI_USER" --password-stdin'
            ),
            "    - |",
            '      if [[ -n "${DOCKERHUB_USER:-}" && -n "${DOCKERHUB_TOKEN:-}" ]]; then',
            (
                '        echo "$DOCKERHUB_TOKEN" | docker login docker.io '
                '-u "$DOCKERHUB_USER" --password-stdin'
            ),
            "      else",
            '        echo "Docker Hub creds not set; anonymous pulls may hit rate limits."',
            "      fi",
        ]
    )
    if image_version == HARNESS_IMAGE_VERSION:
        lines.extend(
            [
                (
                    "    - export SWE_HARNESS_RUNTIME_IMAGE_REF="
                    '"$(cat swe_uv_image_poc/outputs/'
                    'harness-runtime-1.0.2/image-ref.txt)"'
                ),
            ]
        )
    lines.extend(
        [
            "  after_script:",
            "    - docker logout docker.sberdevices.ru || true",
            "    - docker logout docker.io || true",
            "",
        ]
    )
    for index, (record, job) in enumerate(zip(selected, jobs)):
        task_id = record["task_id"]
        lines.extend(
            [
                f'"{job}":',
                "  extends: .swe-image-build",
                f"  resource_group: swe-image-build-{index % max_parallel_jobs}",
                "  variables:",
                f"    TASK_ID: {json.dumps(task_id)}",
                "  script:",
                (f'    - ./swe_uv_image_poc/scripts/{build_script} "$TASK_ID"'),
                "  artifacts:",
                "    when: always",
                "    paths:",
                f"      - {validation_root}/{task_id}/",
                "  timeout: 3h",
                "",
            ]
        )
    task_ids = [record["task_id"] for record in selected]
    task_args = " ".join(json.dumps(task_id) for task_id in task_ids)
    lines.extend(
        [
            '"summary:batch":',
            "  stage: summary",
            "  tags: [advanced]",
            "  when: always",
            "  script:",
            (
                "    - python3 ./swe_uv_image_poc/scripts/batch_summary.py "
                f"{validation_root} batch-summary.json {task_args}"
            ),
            "  artifacts:",
            "    when: always",
            "    paths:",
            "      - batch-summary.json",
            f"      - {validation_root}/",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("pilot", "batch", "all"), required=True)
    parser.add_argument("--batch-index", type=int)
    parser.add_argument(
        "--image-version",
        choices=(IMAGE_VERSION, HARNESS_IMAGE_VERSION),
        default=IMAGE_VERSION,
    )
    parser.add_argument("--pilot-tasks-path", type=Path)
    parser.add_argument("--max-parallel-jobs", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    pilot_tasks = (
        read_pilot_tasks(args.pilot_tasks_path)
        if args.pilot_tasks_path is not None
        else None
    )
    args.output.write_text(
        render_child_pipeline(
            payload["tasks"],
            args.mode,
            args.batch_index,
            pilot_tasks=pilot_tasks,
            image_version=args.image_version,
            max_parallel_jobs=args.max_parallel_jobs,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
