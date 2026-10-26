from __future__ import annotations

from pathlib import Path

import pytest

from terminal_bench.rli_image_update.scripts.manifest import (
    build_manifest,
    inspect_test_script,
)


def _write_task(root: Path, task_id: str, test_script: str) -> Path:
    task = root / task_id
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests" / "test.sh").write_text(test_script)
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\n")
    return task


def test_inspect_test_script_extracts_uvx_runtime(tmp_path: Path) -> None:
    script = tmp_path / "test.sh"
    script.write_text(
        """#!/bin/bash
apt-get update
apt-get install -y curl imagemagick
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env
uvx \\
  -p 3.13 \\
  -w pytest==8.4.1 \\
  -w pillow==11.1.0 \\
  pytest /tests/test_outputs.py
"""
    )

    runtime = inspect_test_script(script)

    assert runtime == {
        "runner": "uvx",
        "python": "3.13",
        "apt_packages": ["curl", "imagemagick"],
        "python_packages": ["pytest==8.4.1", "pillow==11.1.0"],
        "standard_bootstrap": True,
    }


def test_inspect_test_script_extracts_direct_pip_runtime(
    tmp_path: Path,
) -> None:
    script = tmp_path / "test.sh"
    script.write_text(
        """#!/bin/bash
pip install pytest==8.4.1 requests==2.32.5 --break-system-packages
pytest /tests/test_outputs.py
"""
    )

    runtime = inspect_test_script(script)

    assert runtime == {
        "runner": "pip",
        "python": None,
        "apt_packages": [],
        "python_packages": ["pytest==8.4.1", "requests==2.32.5"],
        "standard_bootstrap": False,
    }


def test_build_manifest_uses_existing_rli_image_as_source(
    tmp_path: Path,
) -> None:
    _write_task(
        tmp_path,
        "example-task",
        "pip install pytest==8.4.1\npytest /tests/test_outputs.py\n",
    )

    manifest = build_manifest(tmp_path, expected_task_count=1)

    assert manifest == [
        {
            "task_id": "example-task",
            "source_image": (
                "docker.sberdevices.ru/gigarli/environments/"
                "terminal-bench_example-task:latest"
            ),
            "image": (
                "docker.sberdevices.ru/gigarli/environments/"
                "terminal-bench_example-task:1.0.1"
            ),
            "assets": str(tmp_path / "example-task"),
            "test_runtime": {
                "runner": "pip",
                "python": None,
                "apt_packages": [],
                "python_packages": ["pytest==8.4.1"],
                "standard_bootstrap": False,
            },
        }
    ]


def test_build_manifest_rejects_incomplete_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 89 tasks, found 0"):
        build_manifest(tmp_path)


def test_build_manifest_rejects_hidden_test_network(tmp_path: Path) -> None:
    task = _write_task(
        tmp_path,
        "example-task",
        "pip install pytest==8.4.1\npytest /tests/test_outputs.py\n",
    )
    (task / "tests" / "test_outputs.py").write_text(
        "import requests\nrequests.get('https://example.com/data.json')\n"
    )

    with pytest.raises(
        ValueError,
        match="example-task/test_outputs.py: external Python HTTP request",
    ):
        build_manifest(tmp_path, expected_task_count=1)
