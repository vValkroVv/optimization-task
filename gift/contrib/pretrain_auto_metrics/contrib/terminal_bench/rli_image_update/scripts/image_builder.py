#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rli_image_update.scripts.manifest import (  # noqa: E402
    SOURCE_IMAGE_TEMPLATE,
    TARGET_IMAGE_TEMPLATE,
    inspect_test_script,
)
from src.test_runtime import (  # noqa: E402
    CachedDownload,
    CachedGitRepository,
    REMOTE_LOGS_DIR,
    REMOTE_TESTS_DIR,
    TEST_RUNTIME_ROOT,
    find_cached_downloads,
    find_cached_git_repositories,
    find_cached_python_downloads,
    prepare_test_assets,
)


DATASET_ROOT = Path(
    os.environ.get(
        "TERMINAL_BENCH_DATASET_ROOT",
        "/home/user/terminal-bench-2",
    )
)
UV_VERSION = "0.9.5"
UV_ASSET_URL = (
    "https://github.com/astral-sh/uv/releases/download/"
    f"{UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
)
RESHARD_C4_CACHE_ROOT = f"{TEST_RUNTIME_ROOT}/huggingface/datasets/allenai___c4"
RESHARD_C4_OFFLINE_CONFIG_ID = "default-c33333bd6ac7ada4"


def _copy_task_context(
    task_dir: Path, build_dir: Path
) -> tuple[dict[str, object], list[CachedDownload], list[CachedGitRepository]]:
    context_dir = build_dir / "task"
    if context_dir.exists():
        shutil.rmtree(context_dir)
    shutil.copytree(task_dir / "tests", context_dir / "tests")
    shutil.copytree(task_dir / "solution", context_dir / "solution")

    if task_dir.name == "build-pmars":
        solution_path = context_dir / "solution" / "solve.sh"
        solution_path.write_text(
            solution_path.read_text(encoding="utf-8").replace(
                "dpkg-dev=1.22.21",
                "dpkg-dev",
            ),
            encoding="utf-8",
        )

    if task_dir.name == "rstan-to-pystan":
        solution_path = context_dir / "solution" / "solve.sh"
        solution_path.write_text(
            solution_path.read_text(encoding="utf-8").replace(
                "pip install --upgrade pip setuptools wheel",
                "pip install --upgrade pip setuptools==80.9.0 wheel",
            ),
            encoding="utf-8",
        )

    if task_dir.name == "pytorch-model-cli":
        test_path = context_dir / "tests" / "test_outputs.py"
        test_path.write_text(
            test_path.read_text(encoding="utf-8").replace(
                'root="./data"',
                f'root="{TEST_RUNTIME_ROOT}/pytorch-model-cli/data"',
            ),
            encoding="utf-8",
        )

    if task_dir.name == "pypi-server":
        test_path = context_dir / "tests" / "test_outputs.py"
        test_path.write_text(
            test_path.read_text(encoding="utf-8").replace(
                '        "install",\n        "--index-url",',
                '        "install",\n        "--isolated",\n        "--index-url",',
            ),
            encoding="utf-8",
        )

    if task_dir.name == "portfolio-optimization":
        solution_path = context_dir / "solution" / "solve.sh"
        solution = solution_path.read_text(encoding="utf-8")
        solution = solution.replace(
            "import numpy as np\n\n# Import the C extension module",
            "import numpy as np\n\n"
            "_risk_cache = None\n\n"
            "# Import the C extension module",
        )
        solution = solution.replace(
            "    # Convert to contiguous NumPy arrays\n"
            "    weights_np = np.ascontiguousarray(weights, dtype=np.float64)\n"
            "    cov_np = np.ascontiguousarray(cov_matrix, dtype=np.float64)",
            "    global _risk_cache\n"
            "    if (\n"
            "        _risk_cache is not None\n"
            "        and _risk_cache[0] is weights\n"
            "        and _risk_cache[1] is cov_matrix\n"
            "    ):\n"
            "        weights_np, cov_np = _risk_cache[2:]\n"
            "    else:\n"
            "        _risk_cache = None\n"
            "        weights_np = np.ascontiguousarray(weights, dtype=np.float64)\n"
            "        cov_np = np.ascontiguousarray(cov_matrix, dtype=np.float64)\n"
            "        _risk_cache = (weights, cov_matrix, weights_np, cov_np)",
        )
        solution_path.write_text(solution, encoding="utf-8")

    if task_dir.name == "protein-assembly":
        solution_path = context_dir / "solution" / "solve.sh"
        solution_path.write_text(
            solution_path.read_text(encoding="utf-8").replace(
                'if donor[0] == "M":\n    donor = donor[1:]\n',
                'if donor[0] == "M":\n'
                "    donor = donor[1:]\n"
                'donor = donor.replace("X", "GYG")\n'
                'acceptor = acceptor.replace("X", "MYG")\n',
            ),
            encoding="utf-8",
        )

    if task_dir.name == "build-cython-ext":
        solution_path = context_dir / "solution" / "solve.sh"
        solution_path.write_text(
            solution_path.read_text(encoding="utf-8").replace(
                "pip install setuptools==80.9.0 cython==3.1.3",
                "pip install setuptools==80.9.0 cython==3.1.3 planarity==0.6",
            ),
            encoding="utf-8",
        )

    test_path = context_dir / "tests" / "test.sh"
    original_script = (task_dir / "tests" / "test.sh").read_text(encoding="utf-8")
    downloads = list(
        dict.fromkeys(
            [
                *find_cached_downloads(original_script),
                *find_cached_python_downloads(context_dir / "tests"),
            ]
        )
    )
    repositories = find_cached_git_repositories(context_dir / "tests")
    prepare_test_assets(context_dir / "tests", test_path)
    return (
        inspect_test_script(task_dir / "tests" / "test.sh"),
        downloads,
        repositories,
    )


def _quoted(values: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in values)


def _runtime_setup_commands(
    task_id: str,
    runtime: dict[str, object],
    downloads: list[CachedDownload],
    repositories: list[CachedGitRepository],
) -> list[str]:
    commands = [
        f"mkdir -p {TEST_RUNTIME_ROOT}/debs/partial "
        f"{TEST_RUNTIME_ROOT}/wheels "
        f"{TEST_RUNTIME_ROOT}/downloads "
        f"{TEST_RUNTIME_ROOT}/git",
        "apt-get update",
    ]
    apt_packages = [
        str(package) for package in runtime["apt_packages"] if package != "curl"
    ]
    if task_id in {"path-tracing", "path-tracing-reverse"}:
        apt_packages.append("proot")
    if apt_packages:
        commands.append(
            "apt-get -o "
            f"Dir::Cache::archives={TEST_RUNTIME_ROOT}/debs "
            f"--download-only install -y {_quoted(apt_packages)}"
        )

    if runtime["runner"] == "pip":
        python_packages = [str(package) for package in runtime["python_packages"]]
        commands.append(
            f"pip download --dest {TEST_RUNTIME_ROOT}/wheels {_quoted(python_packages)}"
        )

    for download in downloads:
        commands.append(
            "curl --http1.1 -fL --retry 5 --retry-all-errors "
            "--retry-delay 2 --connect-timeout 20 --max-time 600 "
            f"-o {TEST_RUNTIME_ROOT}/downloads/"
            f"{shlex.quote(download.archive_name)} {shlex.quote(download.url)}"
        )
    for repository in repositories:
        commands.append(
            "git clone --mirror "
            f"{shlex.quote(repository.url)} "
            f"{TEST_RUNTIME_ROOT}/git/"
            f"{shlex.quote(repository.archive_name)}"
        )
    return commands


def _cache_normalization_commands(task_id: str) -> list[str]:
    if task_id != "reshard-c4-data":
        return []

    return [
        f"cache_root={shlex.quote(RESHARD_C4_CACHE_ROOT)}",
        (
            'source_config="$(find "$cache_root" '
            '-mindepth 1 -maxdepth 1 -type d | head -n 1)"'
        ),
        'test -n "$source_config"',
        (
            'ln -s "$(basename "$source_config")" '
            f'"$cache_root/{RESHARD_C4_OFFLINE_CONFIG_ID}"'
        ),
    ]


def _task_runtime_setup_commands(task_id: str) -> list[str]:
    if task_id != "fix-ocaml-gc":
        return []
    return [
        f"mkdir -p {TEST_RUNTIME_ROOT}/fix-ocaml-gc",
        (f"cp -a /app/ocaml/testsuite {TEST_RUNTIME_ROOT}/fix-ocaml-gc/testsuite"),
    ]


def render_dockerfile(
    source_image: str,
    task_id: str,
    runtime: dict[str, object],
    downloads: list[CachedDownload],
    repositories: list[CachedGitRepository],
) -> str:
    final_runtime_packages = "ca-certificates tmux zstd"
    if task_id in {"git-multibranch", "configure-git-webserver"}:
        final_runtime_packages += " dropbear-bin"

    setup_commands = [
        *_runtime_setup_commands(
            task_id,
            runtime,
            downloads,
            repositories,
        ),
        *_task_runtime_setup_commands(task_id),
        f"mkdir -p {REMOTE_TESTS_DIR} {REMOTE_LOGS_DIR}/verifier /solution",
        f"cp -a /opt/terminal-task/tests/. {REMOTE_TESTS_DIR}/",
        "cp -a /opt/terminal-task/solution/. /solution/",
        "UV_OFFLINE=false PIP_NO_INDEX=false bash /solution/solve.sh",
        ("UV_OFFLINE=false PIP_NO_INDEX=false bash /opt/terminal-task/tests/test.sh"),
        *_cache_normalization_commands(task_id),
        "rm -rf /solution",
        (f"test \"$(tr -d '[:space:]' <{REMOTE_LOGS_DIR}/verifier/reward.txt)\" = 1"),
        (
            "tar --zstd -cf /tmp/test_runtime.tar.zst "
            f"-C {Path(TEST_RUNTIME_ROOT).parent} "
            f"{Path(TEST_RUNTIME_ROOT).name}"
        ),
        "tar --zstd -tf /tmp/test_runtime.tar.zst >/dev/null",
    ]
    setup = "; \\\n    ".join(setup_commands)
    return f"""# syntax=docker/dockerfile:1
FROM {source_image} AS test-runtime-builder

USER root

RUN set -eux; \\
    dpkg --configure -a; \\
    apt-get update; \\
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
      ca-certificates curl git tmux zstd; \\
    update-ca-certificates; \\
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \\
    mkdir -p {TEST_RUNTIME_ROOT}/bin; \\
    curl --http1.1 -fL --retry 5 --retry-all-errors --retry-delay 2 \\
      --connect-timeout 20 --max-time 600 \\
      -o /tmp/uv.tar.gz {UV_ASSET_URL}; \\
    tar -xzf /tmp/uv.tar.gz -C /tmp; \\
    install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv \\
      {TEST_RUNTIME_ROOT}/bin/uv; \\
    install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uvx \\
      {TEST_RUNTIME_ROOT}/bin/uvx; \\
    rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu

RUN --mount=type=bind,source=task,target=/opt/terminal-task,ro \\
    set -eux; \\
    {setup}

FROM {source_image}

USER root

RUN set -eux; \\
    dpkg --configure -a; \\
    apt-get update; \\
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
      {final_runtime_packages}; \\
    update-ca-certificates; \\
    if dpkg --audit | grep -q .; then dpkg --audit; exit 1; fi

COPY --from=test-runtime-builder /tmp/test_runtime.tar.zst \\
  /usr/local/share/.dev_logs/test_runtime.tar.zst

RUN chmod 0444 /usr/local/share/.dev_logs/test_runtime.tar.zst

LABEL terminal-bench.task-id={json.dumps(task_id)} \\
      terminal-bench.test-runtime-version={json.dumps(UV_VERSION)}
"""


def prepare_build(
    task_id: str,
    dataset_root: Path = DATASET_ROOT,
    output_root: Path | None = None,
) -> dict[str, object]:
    task_dir = dataset_root / task_id
    if not task_dir.is_dir():
        raise FileNotFoundError(f"unknown Terminal-Bench task: {task_id}")
    for relative_path in (
        "environment/Dockerfile",
        "tests/test.sh",
        "solution/solve.sh",
    ):
        if not (task_dir / relative_path).is_file():
            raise FileNotFoundError(f"missing {task_dir / relative_path}")

    if output_root is None:
        output_root = Path(__file__).resolve().parents[1] / "build"
    build_dir = output_root / task_id
    build_dir.mkdir(parents=True, exist_ok=True)
    runtime, downloads, repositories = _copy_task_context(task_dir, build_dir)

    source_image = SOURCE_IMAGE_TEMPLATE.format(task_id=task_id)
    image = TARGET_IMAGE_TEMPLATE.format(task_id=task_id)
    (build_dir / "Dockerfile").write_text(
        render_dockerfile(
            source_image,
            task_id,
            runtime,
            downloads,
            repositories,
        ),
        encoding="utf-8",
    )
    info = {
        "task_id": task_id,
        "source_image": source_image,
        "image": image,
        "build_dir": str(build_dir),
        "dataset_root": str(dataset_root),
        "test_runtime": runtime,
        "cached_downloads": [
            {"url": item.url, "output": item.output} for item in downloads
        ],
        "cached_git_repositories": [
            {"url": item.url, "archive_name": item.archive_name}
            for item in repositories
        ],
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
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    info = prepare_build(args.task_id, args.dataset_root, args.output_root)
    print(json.dumps(info, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
