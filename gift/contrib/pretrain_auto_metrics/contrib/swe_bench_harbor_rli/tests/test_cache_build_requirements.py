from __future__ import annotations

from pathlib import Path

from swe_bench_harbor_rli.swe_uv_image_poc.scripts.cache_build_requirements import (
    read_build_requirements,
)


def test_read_build_requirements_includes_defaults_and_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = ["flit_core<4,>=3.2", "setuptools_scm>=6.2"]
build-backend = "flit_core.buildapi"
""",
        encoding="utf-8",
    )

    assert read_build_requirements(tmp_path) == [
        "setuptools>=40.8.0",
        "wheel",
        "flit_core<4,>=3.2",
        "setuptools_scm>=6.2",
    ]


def test_read_build_requirements_uses_defaults_without_pyproject(
    tmp_path: Path,
) -> None:
    assert read_build_requirements(tmp_path) == [
        "setuptools>=40.8.0",
        "wheel",
    ]
