#!/usr/bin/env python3
"""Capture non-secret Python/build and task-import identities for preflight."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import torch
import triton
import torchtitan


GIFT = Path(__file__).resolve().parents[7] / "gift"


def module_path(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"required task module has no import origin: {name}")
    path = Path(spec.origin).resolve(strict=True)
    if GIFT not in path.parents:
        raise RuntimeError(f"task module imports outside gift/: {name} -> {path}")
    return path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: preflight_software.py OUTPUT.json")
    cwd = Path.cwd().resolve()
    if cwd != GIFT.resolve():
        raise RuntimeError(f"preflight cwd is not the task gift checkout: {cwd}")
    modules = {
        "torchtitan": Path(torchtitan.__file__).resolve(strict=True),
        "config_pre_submit": module_path("torchtitan.config.pre_submit"),
        "campaign_train_scan_fix": module_path(
            "torchtitan.experiments.cuda_graph_campaign.train_scan_fix"
        ),
        "campaign_numerical_probe": module_path(
            "torchtitan.experiments.cuda_graph_campaign.numerical_probe"
        ),
    }
    if any(GIFT not in path.parents for path in modules.values()):
        raise RuntimeError(f"a task import resolved outside gift/: {modules}")
    inventory = sorted(
        (item.metadata.get("Name", "").lower().replace("_", "-"), item.version)
        for item in importlib.metadata.distributions()
    )
    inventory_bytes = json.dumps(inventory, separators=(",", ":")).encode("utf-8")
    result = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "task_cwd": str(cwd),
        "torch": {
            "version": torch.__version__,
            "git_version": torch.version.git_version,
            "cuda_build": torch.version.cuda,
            "import_path": str(Path(torch.__file__).resolve()),
        },
        "triton": {
            "version": triton.__version__,
            "import_path": str(Path(triton.__file__).resolve()),
        },
        "task_imports": {name: str(path) for name, path in modules.items()},
        "task_source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in modules.items()
        },
        "distribution_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "distribution_count": len(inventory),
        "python_path_gift_first": os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
        == str(GIFT),
    }
    if not result["python_path_gift_first"]:
        raise RuntimeError("PYTHONPATH does not put the task gift checkout first")
    output = Path(sys.argv[1])
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"preflight_software_sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
