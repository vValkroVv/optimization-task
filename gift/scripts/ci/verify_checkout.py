#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _git_head(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Failed to resolve commit for {repo_root}") from exc
    return result.stdout.strip()


def _verify_checkout(repo_root: Path, expected_commit: str) -> None:
    actual_commit = _git_head(repo_root)
    if actual_commit != expected_commit:
        raise RuntimeError(
            "Cloud integration checkout mismatch: "
            f"expected {expected_commit}, got {actual_commit} at {repo_root}"
        )
    print(
        f"Verified TorchTitan checkout: {actual_commit} at {repo_root}",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a TorchTitan checkout before running an integration test"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("target", type=Path)
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = args.repo_root.resolve()
    target = args.target.resolve()

    if not target.is_relative_to(repo_root):
        raise ValueError(f"Target must be inside {repo_root}: {target}")

    _verify_checkout(repo_root, args.expected_commit)
    os.chdir(repo_root)
    os.execv(
        sys.executable,
        [sys.executable, str(target), *args.target_args],
    )


if __name__ == "__main__":
    main()
