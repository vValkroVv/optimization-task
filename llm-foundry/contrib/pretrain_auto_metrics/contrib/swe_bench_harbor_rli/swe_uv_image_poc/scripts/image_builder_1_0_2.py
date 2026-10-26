#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from .image_builder import (
        DATASET_ROOT,
        _copy_task_context,
        find_task_assets,
        source_image_from_dockerfile,
        target_image,
    )
except ImportError:
    from image_builder import (
        DATASET_ROOT,
        _copy_task_context,
        find_task_assets,
        source_image_from_dockerfile,
        target_image,
    )

BASE_IMAGE_VERSION = "1.0.1"
IMAGE_VERSION = "1.0.2"
RUNTIME_IMAGE_ENV = "SWE_HARNESS_RUNTIME_IMAGE_REF"


def render_dockerfile(base_image: str, runtime_image: str) -> str:
    return f"""# syntax=docker/dockerfile:1
FROM {runtime_image} AS harness-runtime
FROM {base_image}

RUN set -eux; \\
    command -v apt-get; \\
    apt-get update; \\
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux; \\
    rm -rf /var/lib/apt/lists/*

COPY --from=harness-runtime /root/.local /root/.local
COPY --from=harness-runtime /opt/openhands-sdk-venv /opt/openhands-sdk-venv
COPY --from=harness-runtime /opt/sweagent-venv /opt/sweagent-venv
COPY --from=harness-runtime /opt/sweagent-repo /opt/sweagent-repo
COPY --from=harness-runtime /opt/sweagent-configs /opt/sweagent-configs
COPY --from=harness-runtime /usr/local/bin/sweagent /usr/local/bin/sweagent
COPY --from=harness-runtime /etc/profile.d/testbed-conda.sh /etc/profile.d/testbed-conda.sh

ENV PATH="/root/.local/bin:$PATH"
"""


def prepare_build(
    task_id: str,
    dataset_root: Path,
    output_root: Path,
    runtime_image: str,
) -> dict[str, object]:
    if "@sha256:" not in runtime_image:
        raise ValueError("harness runtime image must be pinned by sha256 digest")
    assets = find_task_assets(task_id, dataset_root)
    original_source_image = source_image_from_dockerfile(assets.environment_dockerfile)
    base_image = target_image(original_source_image, BASE_IMAGE_VERSION)
    image = target_image(original_source_image, IMAGE_VERSION)
    build_dir = output_root / task_id
    build_dir.mkdir(parents=True, exist_ok=True)
    _copy_task_context(assets, build_dir)
    (build_dir / "Dockerfile").write_text(
        render_dockerfile(base_image, runtime_image),
        encoding="utf-8",
    )
    info = {
        "task_id": task_id,
        "original_source_image": original_source_image,
        "source_image": base_image,
        "harness_runtime_image": runtime_image,
        "image": image,
        "build_dir": str(build_dir),
        "dataset_root": str(dataset_root),
    }
    (build_dir / "build-info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build" / IMAGE_VERSION,
    )
    parser.add_argument(
        "--harness-runtime-image",
        default=os.environ.get(RUNTIME_IMAGE_ENV),
    )
    args = parser.parse_args()
    if not args.harness_runtime_image:
        parser.error(f"--harness-runtime-image or {RUNTIME_IMAGE_ENV} is required")
    info = prepare_build(
        args.task_id,
        args.dataset_root,
        args.output_root,
        args.harness_runtime_image,
    )
    print(json.dumps(info, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
