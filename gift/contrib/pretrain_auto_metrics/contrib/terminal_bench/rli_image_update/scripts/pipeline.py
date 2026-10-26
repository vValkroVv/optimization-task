#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MAX_PARALLEL_BUILDS = 5
PILOT_TASKS = (
    "configure-git-webserver",
    "git-multibranch",
    "path-tracing",
    "path-tracing-reverse",
)
BUILT_PILOT_TASKS = (
    "make-doom-for-mips",
    "compile-compcert",
    "constraints-scheduling",
    "distribution-search",
    "chess-best-move",
    "configure-git-webserver",
    "pytorch-model-cli",
    "git-multibranch",
    "rstan-to-pystan",
    "portfolio-optimization",
    "protein-assembly",
    "build-cython-ext",
    "bn-fit-modify",
    "install-windows-3.11",
    "reshard-c4-data",
    "train-fasttext",
    "headless-terminal",
    "fix-ocaml-gc",
)


def select_tasks(
    records: list[dict[str, object]], mode: str
) -> list[dict[str, object]]:
    if mode == "all":
        excluded_tasks = {*PILOT_TASKS, *BUILT_PILOT_TASKS}
        return [
            record for record in records if str(record["task_id"]) not in excluded_tasks
        ]
    if mode != "pilot":
        raise ValueError(f"unsupported image mode: {mode}")

    by_task_id = {str(record["task_id"]): record for record in records}
    missing = [task_id for task_id in PILOT_TASKS if task_id not in by_task_id]
    if missing:
        raise ValueError(f"pilot tasks missing from manifest: {', '.join(missing)}")
    return [by_task_id[task_id] for task_id in PILOT_TASKS]


def _job_name(task_id: str) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9]+", "-", task_id).strip("-")
    return f"build-{suffix}"


def render_child_pipeline(records: list[dict[str, object]], mode: str) -> str:
    selected = select_tasks(records, mode)
    lines = [
        "stages:",
        "  - build",
        "",
        ".terminal-image-build:",
        "  stage: build",
        "  tags: [advanced]",
        "  variables:",
        '    DOCKER_BUILDKIT: "1"',
        (
            "    TERMINAL_BENCH_DATASET_ROOT: "
            '"$CI_PROJECT_DIR/rli_image_update/build/terminal-bench-2"'
        ),
        "  needs:",
        '    - pipeline: "$PARENT_PIPELINE_ID"',
        '      job: "prepare:terminal-dataset"',
        "      artifacts: true",
        "  before_script:",
        "    - docker --version",
        '    - test -n "$RLI_USER"',
        '    - test -n "$RLI_PASSWORD"',
        "    - |",
        "      for attempt in 1 2 3 4 5; do",
        (
            '        if echo "$RLI_PASSWORD" | docker login '
            'docker.sberdevices.ru -u "$RLI_USER" --password-stdin; then'
        ),
        "          break",
        "        fi",
        '        [ "$attempt" -lt 5 ] || exit 1',
        "        sleep $((attempt * 15))",
        "      done",
        "  after_script:",
        "    - docker logout docker.sberdevices.ru || true",
        "",
    ]
    for index, record in enumerate(selected):
        task_id = str(record["task_id"])
        lines.extend(
            [
                f'"{_job_name(task_id)}":',
                "  extends: .terminal-image-build",
                (
                    "  resource_group: terminal-image-build-"
                    f"{index % MAX_PARALLEL_BUILDS}"
                ),
                "  variables:",
                f"    TASK_ID: {json.dumps(task_id)}",
                "  script:",
                (
                    "    - ./rli_image_update/scripts/build_push_image.sh "
                    '"$TASK_ID" "$TERMINAL_BENCH_DATASET_ROOT"'
                ),
                "  artifacts:",
                "    when: always",
                "    paths:",
                (f"      - rli_image_update/outputs/validation/{task_id}/"),
                "  timeout: 4h",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("pilot", "all"), required=True)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.write_text(
        render_child_pipeline(payload["tasks"], args.mode), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
