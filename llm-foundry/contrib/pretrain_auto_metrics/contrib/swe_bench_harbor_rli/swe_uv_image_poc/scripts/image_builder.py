#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

DATASET_ROOT = Path(
    os.environ.get(
        "SWE_BENCH_DATASET_ROOT",
        "/home/user/swe-bench-verified/datasets/swebench-verified",
    )
)
IMAGE_VERSION = "1.0.1"
UV_VERSION = "0.7.13"
PARSER_PACKAGES = ("swebench==4.0.3", "datasets==2.16.1", "fastcore<1.11")
LOCAL_HTTPBIN_PACKAGES = ("httpbin==0.10.2", "gunicorn==23.0.0")
LEGACY_REQUESTS_VERSIONS = ("2.0", "2.3", "2.4")
_FROM_RE = re.compile(r"^\s*FROM(?:\s+--platform=\S+)?\s+(\S+)")


@dataclass(frozen=True)
class TaskAssets:
    task_id: str
    root: Path
    environment_dockerfile: Path
    tests_dir: Path
    solution_dir: Path
    config_path: Path


def find_task_assets(task_id: str, dataset_root: Path = DATASET_ROOT) -> TaskAssets:
    root = dataset_root / task_id
    if not root.is_dir():
        raise FileNotFoundError(f"unknown task_id or dataset root: {task_id}")

    assets = TaskAssets(
        task_id=task_id,
        root=root,
        environment_dockerfile=root / "environment" / "Dockerfile",
        tests_dir=root / "tests",
        solution_dir=root / "solution",
        config_path=root / "tests" / "config.json",
    )
    required = (
        assets.environment_dockerfile,
        assets.tests_dir / "test.sh",
        assets.config_path,
        assets.solution_dir / "solve.sh",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"task {task_id} has incomplete assets: {', '.join(missing)}"
        )
    return assets


def source_image_from_dockerfile(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = _FROM_RE.match(line)
        if match:
            image = match.group(1)
            if "${" in image:
                raise ValueError(f"unresolved source image in {path}: {image}")
            return image
    raise ValueError(f"no FROM image found in {path}")


def source_name(image: str) -> str:
    name = image.rsplit("/", 1)[-1].split("@", 1)[0]
    if ":" in name:
        name = name.rsplit(":", 1)[0]
    for prefix in ("sweb.eval.x86_64.", "sweb.eval."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if not name:
        raise ValueError(f"cannot derive source name from image {image!r}")
    return name


def target_image(source_image: str, version: str = IMAGE_VERSION) -> str:
    return (
        "docker.sberdevices.ru/gigarli/environments/"
        f"swe-bench_{source_name(source_image)}:{version}"
    )


def _load_config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"task config must be an object: {path}")
    return value


def compatibility_constraints(config: dict[str, object]) -> tuple[str, ...]:
    constraints = []
    if config.get("repo") == "astropy/astropy" and config.get("version") == "3.1":
        constraints.append("setuptools==59.8.0")
    if config.get("instance_id") == "astropy__astropy-8707":
        constraints.append("pytest==7.1.3")
    if config.get("instance_id") == "psf__requests-5414":
        constraints.extend(
            (
                "pytest==6.2.5",
                "pytest-httpbin==1.0.0",
                "pytest-mock==2.0.0",
                "httpbin==0.7.0",
                "Flask>=1.0,<2.0",
                "MarkupSafe<2.1",
                "trustme",
            )
        )
    return tuple(constraints)


def _copy_task_context(assets: TaskAssets, build_dir: Path) -> dict[str, object]:
    task_dir = build_dir / "task"
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "solution").mkdir(parents=True)
    test_script = assets.tests_dir / "test.sh"
    config = _load_config(assets.config_path)

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from swe_bench_harbor_rli.src.local_rli_test_setup import (
        prepare_test_config,
        prepare_test_script,
    )

    config = prepare_test_config(config)
    (task_dir / "tests" / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (task_dir / "tests" / "test.sh").write_text(
        prepare_test_script(
            test_script.read_text(encoding="utf-8"),
            repo=str(config.get("repo", "")),
            version=str(config.get("version", "")),
        ),
        encoding="utf-8",
    )
    shutil.copy2(assets.solution_dir / "solve.sh", task_dir / "solution" / "solve.sh")
    shutil.copy2(
        Path(__file__).with_name("cache_build_requirements.py"),
        task_dir / "cache_build_requirements.py",
    )
    return config


def render_dockerfile(
    source_image: str,
    task_id: str,
    config: dict[str, object],
) -> str:
    base_commit = str(config.get("base_commit", ""))
    if not base_commit:
        raise ValueError(f"task {task_id} config has no base_commit")
    environment_setup_commit = str(
        config.get("environment_setup_commit") or base_commit
    )
    runtime_packages = list(PARSER_PACKAGES)
    runtime_imports = "datasets, fastcore, swebench"
    runtime_setup = ""
    if (
        config.get("repo") == "psf/requests"
        and config.get("version") in LEGACY_REQUESTS_VERSIONS
    ):
        runtime_packages.extend(LOCAL_HTTPBIN_PACKAGES)
        runtime_imports += ", gunicorn, httpbin"
        runtime_setup = """    openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 3650 \\
      -subj "/CN=127.0.0.1" \\
      -addext "subjectAltName=IP:127.0.0.1" \\
      -addext "basicConstraints=critical,CA:TRUE" \\
      -keyout /tmp/test_runtime/httpbin-key.pem \\
      -out /tmp/test_runtime/httpbin-cert.pem; \\
"""
    packages = " ".join(f'"{package}"' for package in runtime_packages)
    constraints = compatibility_constraints(config)
    constraint_setup = ""
    sqlite_setup = ""
    if task_id == "django__django-10097":
        sqlite_setup = """RUN set -eux; \\
    curl -fsSL --http1.1 --retry 5 \\
      -o /tmp/sqlite-3.25.3.tar.bz2 \\
      https://conda.anaconda.org/conda-forge/linux-64/sqlite-3.25.3-h67949de_1000.tar.bz2; \\
    echo "d140c8e08258e55e1774d8be4167650c5609d39f5ce2876447672cf3bc1c7729  /tmp/sqlite-3.25.3.tar.bz2" | sha256sum -c -; \\
    tar -xjf /tmp/sqlite-3.25.3.tar.bz2 \\
      -C /opt/miniconda3/envs/testbed \\
      lib/libsqlite3.so lib/libsqlite3.so.0 lib/libsqlite3.so.0.8.6; \\
    rm /tmp/sqlite-3.25.3.tar.bz2; \\
    test "$(/opt/miniconda3/envs/testbed/bin/python -c 'import sqlite3; print(sqlite3.sqlite_version)')" = "3.25.3"

"""
    if constraints:
        constraint_packages = " ".join(f'"{package}"' for package in constraints)
        constraint_setup = f"""ENV PIP_CONSTRAINT=/usr/local/share/.dev_logs/testbed-constraints.txt

RUN set -eux; \\
    mkdir -p /usr/local/share/.dev_logs/build_wheels; \\
    printf '%s\\n' {constraint_packages} > "$PIP_CONSTRAINT"; \\
    /opt/miniconda3/envs/testbed/bin/python -m pip download \\
      --dest /usr/local/share/.dev_logs/build_wheels {constraint_packages}; \\
    /opt/miniconda3/envs/testbed/bin/python -m pip install \\
      --no-cache-dir {constraint_packages}

"""
    return f"""# syntax=docker/dockerfile:1
FROM {source_image}

ENV TASK_ID={task_id}
ENV UV_VERSION={UV_VERSION}
ENV BASE_COMMIT={base_commit}
ENV ENVIRONMENT_SETUP_COMMIT={environment_setup_commit}
ENV PATH="/root/.local/bin:$PATH"
ENV NO_COLOR=1
ENV PY_COLORS=0
ENV TERM=dumb

RUN set -eux; \\
    if command -v apt-get >/dev/null 2>&1; then \\
      apt-get update; \\
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl zstd; \\
      update-ca-certificates; \\
      rm -rf /var/lib/apt/lists/*; \\
    fi; \\
    printf '%s\\n' '--retry 5' '--retry-delay 2' '--connect-timeout 20' '--max-time 300' '--location' '--http1.1' > /root/.curlrc; \\
    /opt/miniconda3/bin/python -m pip install --no-cache-dir "uv=={UV_VERSION}"; \\
    ln -sf /opt/miniconda3/bin/uv /usr/local/bin/uv; \\
    uv --version

{constraint_setup}{sqlite_setup}RUN set -eux; \\
    mkdir -p /usr/local/share/.dev_logs; \\
    uv venv --python /opt/miniconda3/bin/python /tmp/test_runtime; \\
    uv pip install --python /tmp/test_runtime/bin/python {packages}; \\
    (cd /tmp && /tmp/test_runtime/bin/python -c 'import {runtime_imports}; print("test runtime dependencies ok")'); \\
{runtime_setup}    tar --zstd -cf /usr/local/share/.dev_logs/test_runtime.tar.zst -C /tmp test_runtime; \\
    tar --zstd -tf /usr/local/share/.dev_logs/test_runtime.tar.zst >/dev/null; \\
    chmod 0444 /usr/local/share/.dev_logs/test_runtime.tar.zst; \\
    rm -rf /tmp/test_runtime

ENV PIP_FIND_LINKS=/usr/local/share/.dev_logs/build_wheels

RUN --mount=type=bind,source=task,target=/opt/swebench-task \\
    set -eux; \\
    initial_commit="$(git -C /testbed rev-parse HEAD)"; \\
    git -C /testbed diff --binary HEAD > /tmp/testbed-initial.patch; \\
    git -C /testbed ls-files --others --ignored --exclude-standard -z > /tmp/testbed-ignored.files; \\
    if [ -s /tmp/testbed-ignored.files ]; then \\
      tar -C /testbed --null -T /tmp/testbed-ignored.files -cf /tmp/testbed-ignored.tar; \\
    fi; \\
    /opt/miniconda3/bin/python \\
      /opt/swebench-task/cache_build_requirements.py \\
      /testbed /usr/local/share/.dev_logs/build_wheels \\
      --python /opt/miniconda3/envs/testbed/bin/python; \\
    mkdir -p /tests /logs/verifier /tmp/t; \\
    cp /opt/swebench-task/tests/config.json /tests/config.json; \\
    bash /opt/swebench-task/solution/solve.sh; \\
    TMPDIR=/tmp/t bash /opt/swebench-task/tests/test.sh; \\
    test "$(cat /logs/verifier/reward.txt)" = 1; \\
    test "$(python3 -c 'import json; print(str(json.load(open("/logs/verifier/report.json"))["{task_id}"]["resolved"]).lower())')" = true; \\
    git -C /testbed reset --hard "$initial_commit"; \\
    git -C /testbed clean -fdx; \\
    if [ -s /tmp/testbed-initial.patch ]; then \\
      git -C /testbed apply --binary /tmp/testbed-initial.patch; \\
    fi; \\
    if [ -f /tmp/testbed-ignored.tar ]; then \\
      tar -C /testbed -xf /tmp/testbed-ignored.tar; \\
    fi; \\
    git -C /testbed diff --binary HEAD > /tmp/testbed-restored.patch; \\
    cmp /tmp/testbed-initial.patch /tmp/testbed-restored.patch; \\
    rm -rf /tests /logs /parser.py /testbed/solution_patch.diff \\
      /tmp/test_patch.diff /tmp/t \\
      /tmp/testbed-initial.patch /tmp/testbed-restored.patch \\
      /tmp/testbed-ignored.files /tmp/testbed-ignored.tar
"""


def prepare_build(
    task_id: str, dataset_root: Path, output_root: Path
) -> dict[str, object]:
    assets = find_task_assets(task_id, dataset_root)
    source_image = source_image_from_dockerfile(assets.environment_dockerfile)
    image = target_image(source_image)
    build_dir = output_root / task_id
    build_dir.mkdir(parents=True, exist_ok=True)
    config = _copy_task_context(assets, build_dir)
    (build_dir / "Dockerfile").write_text(
        render_dockerfile(source_image, task_id, config),
        encoding="utf-8",
    )
    info = {
        "task_id": task_id,
        "source_image": source_image,
        "image": image,
        "build_dir": str(build_dir),
        "dataset_root": str(dataset_root),
        "base_commit": config.get("base_commit"),
        "environment_setup_commit": config.get("environment_setup_commit"),
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
        default=Path(__file__).resolve().parents[1] / "build",
    )
    args = parser.parse_args()
    info = prepare_build(args.task_id, args.dataset_root, args.output_root)
    print(json.dumps(info, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
