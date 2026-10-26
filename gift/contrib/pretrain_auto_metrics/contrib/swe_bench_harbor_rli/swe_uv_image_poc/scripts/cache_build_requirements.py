#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_BUILD_REQUIREMENTS = ("setuptools>=40.8.0", "wheel")


def read_build_requirements(project_root: Path) -> list[str]:
    requirements = list(DEFAULT_BUILD_REQUIREMENTS)
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.is_file():
        return requirements

    try:
        import tomllib
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib

    with pyproject_path.open("rb") as file:
        pyproject = tomllib.load(file)
    requirements.extend(pyproject.get("build-system", {}).get("requires", []))
    return list(dict.fromkeys(requirements))


def cache_build_requirements(
    project_root: Path,
    wheelhouse: Path,
    python: str = sys.executable,
) -> None:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "download",
            "--dest",
            str(wheelhouse),
            *read_build_requirements(project_root),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    cache_build_requirements(args.project_root, args.wheelhouse, args.python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
