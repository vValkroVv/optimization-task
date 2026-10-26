#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path


EXPECTED_TASK_COUNT = 89
SOURCE_IMAGE_TEMPLATE = (
    "docker.sberdevices.ru/gigarli/environments/terminal-bench_{task_id}:latest"
)
TARGET_IMAGE_TEMPLATE = (
    "docker.sberdevices.ru/gigarli/environments/terminal-bench_{task_id}:1.0.1"
)
STANDARD_BOOTSTRAP_MARKERS = (
    "apt-get update",
    "apt-get install -y curl",
    "https://astral.sh/uv/0.9.5/install.sh",
    "source $HOME/.local/bin/env",
)
SHELL_CONTROL_TOKENS = {"&&", "||", ";", "|"}
EXTERNAL_URL_RE = re.compile(r"https?://(?!localhost|0\.0\.0\.0|127\.0\.0\.1)")


def _logical_lines(script: str) -> list[str]:
    joined = re.sub(r"\\\r?\n[ \t]*", " ", script)
    return joined.splitlines()


def _tokens(line: str) -> list[str]:
    try:
        return shlex.split(line, comments=True)
    except ValueError as exc:
        raise ValueError(f"cannot parse test.sh line: {line!r}") from exc


def _package_arguments(tokens: list[str], start: int) -> list[str]:
    packages: list[str] = []
    for token in tokens[start:]:
        if token in SHELL_CONTROL_TOKENS:
            break
        if token.startswith("-"):
            continue
        packages.append(token)
    return packages


def _apt_packages(lines: list[str]) -> list[str]:
    packages: list[str] = []
    for line in lines:
        if "apt-get" not in line:
            continue
        tokens = _tokens(line)
        for index, token in enumerate(tokens):
            if token != "apt-get":
                continue
            command_index = index + 1
            while command_index < len(tokens) and tokens[command_index].startswith("-"):
                command_index += 1
            if command_index >= len(tokens) or tokens[command_index] != "install":
                continue
            packages.extend(_package_arguments(tokens, command_index + 1))
    return list(dict.fromkeys(packages))


def _uvx_runtime(lines: list[str]) -> tuple[str | None, list[str]] | None:
    for line in lines:
        tokens = _tokens(line)
        if not tokens or tokens[0] != "uvx":
            continue
        python = None
        packages: list[str] = []
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in {"-p", "--python"}:
                python = tokens[index + 1]
                index += 2
                continue
            if token in {"-w", "--with"}:
                packages.append(tokens[index + 1])
                index += 2
                continue
            index += 1
        return python, packages
    return None


def _pip_packages(lines: list[str]) -> list[str]:
    for line in lines:
        tokens = _tokens(line)
        if tokens[:2] == ["pip", "install"]:
            return _package_arguments(tokens, 2)
        if tokens[:3] == ["uv", "pip", "install"]:
            return _package_arguments(tokens, 3)
    return []


def inspect_test_script(path: Path) -> dict[str, object]:
    script = path.read_text(encoding="utf-8")
    lines = _logical_lines(script)
    apt_packages = []
    for shell_path in sorted(path.parent.rglob("*.sh")):
        apt_packages.extend(
            _apt_packages(_logical_lines(shell_path.read_text(encoding="utf-8")))
        )
    uvx_runtime = _uvx_runtime(lines)

    if uvx_runtime is not None:
        runner = "uvx"
        python, python_packages = uvx_runtime
    elif any(_tokens(line)[:2] == ["uv", "venv"] for line in lines):
        runner = "uv"
        python = next(
            (
                tokens[index + 1]
                for line in lines
                if (tokens := _tokens(line))[:2] == ["uv", "venv"]
                for index, token in enumerate(tokens)
                if token in {"-p", "--python"} and index + 1 < len(tokens)
            ),
            None,
        )
        python_packages = _pip_packages(lines)
    else:
        runner = "pip"
        python = None
        python_packages = _pip_packages(lines)

    if not python_packages:
        raise ValueError(f"unsupported verifier bootstrap in {path}")

    return {
        "runner": runner,
        "python": python,
        "apt_packages": list(dict.fromkeys(apt_packages)),
        "python_packages": python_packages,
        "standard_bootstrap": all(
            marker in script for marker in STANDARD_BOOTSTRAP_MARKERS
        ),
    }


def find_unsupported_test_network(tests_dir: Path) -> list[str]:
    findings = []
    for path in sorted(tests_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(tests_dir)
        if EXTERNAL_URL_RE.search(text) and ("urlopen(" in text or "requests." in text):
            findings.append(f"{relative_path}: external Python HTTP request")
        if EXTERNAL_URL_RE.search(text) and re.search(r"\b(?:curl|wget)\b", text):
            findings.append(f"{relative_path}: external subprocess download")
    return findings


def build_manifest(
    dataset_root: Path,
    *,
    expected_task_count: int = EXPECTED_TASK_COUNT,
) -> list[dict[str, object]]:
    task_dirs = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if len(task_dirs) != expected_task_count:
        raise ValueError(
            f"expected {expected_task_count} tasks, found "
            f"{len(task_dirs)} in {dataset_root}"
        )

    records: list[dict[str, object]] = []
    unsupported_network: list[str] = []
    for task_dir in task_dirs:
        test_script = task_dir / "tests" / "test.sh"
        dockerfile = task_dir / "environment" / "Dockerfile"
        solution = task_dir / "solution" / "solve.sh"
        missing = [
            str(path)
            for path in (test_script, dockerfile, solution)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"incomplete task {task_dir.name}: {', '.join(missing)}"
            )

        unsupported_network.extend(
            f"{task_dir.name}/{finding}"
            for finding in find_unsupported_test_network(task_dir / "tests")
        )

        records.append(
            {
                "task_id": task_dir.name,
                "source_image": SOURCE_IMAGE_TEMPLATE.format(task_id=task_dir.name),
                "image": TARGET_IMAGE_TEMPLATE.format(task_id=task_dir.name),
                "assets": str(task_dir),
                "test_runtime": inspect_test_script(test_script),
            }
        )
    if unsupported_network:
        raise ValueError(
            "unsupported test network operations:\n" + "\n".join(unsupported_network)
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-ref", required=True)
    args = parser.parse_args()

    records = build_manifest(args.dataset_root)
    args.output.write_text(
        json.dumps(
            {
                "dataset_ref": args.dataset_ref,
                "task_count": len(records),
                "tasks": records,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
