from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from terminal_bench.rli_image_update.scripts.image_builder import prepare_build


def _write_task(root: Path, task_id: str) -> Path:
    task = root / task_id
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests" / "test.sh").write_text(
        """#!/bin/bash
apt-get update
apt-get install -y curl imagemagick
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env
curl -L -o model.bin https://example.com/model.bin
uvx -p 3.13 -w pytest==8.4.1 pytest /tests/test_outputs.py
echo 1 > /logs/verifier/reward.txt
"""
    )
    (task / "tests" / "test_outputs.py").write_text(
        'open("/tests/private.bin")\n'
        "url = 'https://example.com/master.zip'\n"
        "urllib.request.urlretrieve(url, '/tmp/master.zip')\n"
        "git_cmd = ['git', 'clone', "
        "'https://example.com/project.git', '/tmp/project']\n"
    )
    (task / "tests" / "private.bin").write_bytes(b"private")
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\n")
    return task


def test_prepare_build_generates_isolated_multistage_image(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    build_root = tmp_path / "build"
    _write_task(dataset_root, "example-task")

    info = prepare_build("example-task", dataset_root, build_root)
    dockerfile = (build_root / "example-task" / "Dockerfile").read_text()
    final_stage = dockerfile.rsplit("FROM ", maxsplit=1)[1]

    assert info["image"].endswith("terminal-bench_example-task:1.0.1")
    assert "AS test-runtime-builder" in dockerfile
    assert "curl --http1.1 -fL" in dockerfile
    assert "uv-x86_64-unknown-linux-gnu.tar.gz" in dockerfile
    assert "--download-only install -y imagemagick" in dockerfile
    assert "downloads/model.bin https://example.com/model.bin" in dockerfile
    assert "downloads/master.zip https://example.com/master.zip" in dockerfile
    assert (
        "git clone --mirror https://example.com/project.git "
        "/var/lib/terminal-bench/test_runtime/git/project.git" in dockerfile
    )
    assert "ca-certificates tmux zstd" in final_stage
    assert "test_runtime.tar.zst" in final_stage
    assert "cp -a /opt/terminal-task/solution/. /solution/" in dockerfile
    assert "bash /solution/solve.sh" in dockerfile
    assert "mkdir -p /tmp/terminal-bench/tests" in dockerfile
    assert "/tmp/terminal-bench/logs/verifier/reward.txt" in dockerfile
    assert "\n+" not in dockerfile
    assert "/tests" not in final_stage
    assert "private.bin" not in final_stage


def test_prepare_build_writes_offline_test_script(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    build_root = tmp_path / "build"
    _write_task(dataset_root, "example-task")

    prepare_build("example-task", dataset_root, build_root)
    test_script = (
        build_root / "example-task" / "task" / "tests" / "test.sh"
    ).read_text()

    assert 'UV_OFFLINE="${UV_OFFLINE:-1}"' in test_script
    assert "apt-get" not in test_script
    assert "astral.sh" not in test_script
    assert 'cp "$TEST_RUNTIME_ROOT/downloads/model.bin" model.bin' in test_script
    python_test = (
        build_root / "example-task" / "task" / "tests" / "test_outputs.py"
    ).read_text()
    assert 'open("/tmp/terminal-bench/tests/private.bin")' in python_test
    assert "https://example.com" not in python_test


def test_reshard_c4_build_normalizes_offline_cache(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    build_root = tmp_path / "build"
    _write_task(dataset_root, "reshard-c4-data")

    prepare_build("reshard-c4-data", dataset_root, build_root)
    dockerfile = (build_root / "reshard-c4-data" / "Dockerfile").read_text()

    assert "huggingface/datasets/allenai___c4" in dockerfile
    assert "default-c33333bd6ac7ada4" in dockerfile
    assert 'ln -s "$(basename "$source_config")"' in dockerfile


def test_fix_ocaml_build_archives_pristine_testsuite(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    build_root = tmp_path / "build"
    _write_task(dataset_root, "fix-ocaml-gc")

    prepare_build("fix-ocaml-gc", dataset_root, build_root)
    dockerfile = (build_root / "fix-ocaml-gc" / "Dockerfile").read_text()

    assert (
        "cp -a /app/ocaml/testsuite "
        "/var/lib/terminal-bench/test_runtime/fix-ocaml-gc/testsuite"
    ) in dockerfile


def test_path_tracing_build_caches_proot(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    build_root = tmp_path / "build"
    _write_task(dataset_root, "path-tracing")

    prepare_build("path-tracing", dataset_root, build_root)
    dockerfile = (build_root / "path-tracing" / "Dockerfile").read_text()

    assert "--download-only install -y imagemagick proot" in dockerfile


def test_image_builder_runs_from_hyphenated_checkout(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "terminal-bench"
    scripts = checkout / "rli_image_update" / "scripts"
    source = checkout / "src"
    scripts.mkdir(parents=True)
    source.mkdir()
    for name in ("__init__.py", "image_builder.py", "manifest.py"):
        shutil.copy2(source_root / "rli_image_update" / "scripts" / name, scripts)
    for name in ("__init__.py", "test_runtime.py"):
        shutil.copy2(source_root / "src" / name, source)

    result = subprocess.run(
        [sys.executable, str(scripts / "image_builder.py"), "--help"],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
