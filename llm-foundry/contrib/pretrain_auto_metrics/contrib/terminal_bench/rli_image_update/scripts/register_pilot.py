#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import ssl
import tomllib
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from rli_image_update.scripts.pipeline import PILOT_TASKS


GATEWAY_URL = "https://rli.sberdevices.ru"
ENV_VERSION = "1.0.1"
MAX_CONCURRENCY = 5
IMAGE_TEMPLATE = (
    "docker.sberdevices.ru/gigarli/environments/terminal-bench_{task_id}:1.0.1"
)
# Sized volumes for workloads that exceed RLI's default writable mounts.
# Sizes are in MiB.
TASK_VOLUMES: dict[str, list[dict[str, object]]] = {
    "install-windows-3.11": [{"path": "/tmp", "size": 4096}],
    "mteb-leaderboard": [{"path": "/app", "size": 4096}],
    "caffe-cifar-10": [{"path": "/app", "size": 4096}],
    "count-dataset-tokens": [{"path": "/tmp", "size": 4096}],
    "reshard-c4-data": [{"path": "/tmp", "size": 4096}],
    "mcmc-sampling-stan": [{"path": "/tmp", "size": 4096}],
    "pytorch-model-recovery": [{"path": "/tmp", "size": 4096}],
    "headless-terminal": [{"path": "/tmp", "size": 4096}],
    "extract-moves-from-video": [
        {"path": "/app", "size": 4096},
        {"path": "/root/.cache", "size": 4096},
    ],
    "rstan-to-pystan": [
        {"path": "/opt/py310", "size": 4096},
        {"path": "/tmp", "size": 2048},
    ],
    "compile-compcert": [{"path": "/tmp", "size": 4096}],
    "pytorch-model-cli": [{"path": "/tmp", "size": 4096}],
    "sam-cell-seg": [{"path": "/tmp", "size": 4096}],
    "torch-pipeline-parallelism": [
        {"path": "/tmp", "size": 2048},
        {"path": "/var/lib/terminal-bench", "size": 6144},
    ],
    "torch-tensor-parallelism": [
        {"path": "/tmp", "size": 2048},
        {"path": "/var/lib/terminal-bench", "size": 6144},
    ],
}
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "registration"
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[5] / "terminal-bench-2"
DATASET_ROOT = Path(
    os.environ.get("TERMINAL_BENCH_DATASET_ROOT", str(DEFAULT_DATASET_ROOT))
)


def _resource_profile(task_id: str) -> dict[str, object]:
    task_config = tomllib.loads(
        (DATASET_ROOT / task_id / "task.toml").read_text(encoding="utf-8")
    )
    environment = task_config["environment"]
    cpu_limit = int(environment["cpus"]) * 1000
    memory_limit = int(environment["memory_mb"])
    storage_limit = int(environment["storage_mb"])
    return {
        "cpu": {
            "requests": max(cpu_limit // 2, 1000),
            "limits": cpu_limit,
        },
        "memory": {
            "requests": max(memory_limit // 2, 1024),
            "limits": memory_limit,
        },
        "ephemeral": {
            "requests": storage_limit,
            "limits": storage_limit,
        },
    }


def _task_ids(mode: str) -> tuple[str, ...]:
    if mode == "pilot":
        return PILOT_TASKS
    return tuple(path.parent.name for path in sorted(DATASET_ROOT.glob("*/task.toml")))


def _request(
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> dict[str, Any] | None:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        GATEWAY_URL + path,
        data=data,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
    )
    with opener.open(request, timeout=120) as response:
        body = response.read().decode("utf-8", errors="replace")
        result = json.loads(body) if body.strip() else None
        if isinstance(result, dict) and isinstance(result.get("result"), dict):
            return result["result"]
        return result


def _registry_path(name: str) -> str:
    quoted = urllib.parse.quote(name, safe="")
    return f"/v1/registry/ENV/{quoted}/{ENV_VERSION}"


def _get(name: str) -> dict[str, Any] | None:
    try:
        return _request("GET", _registry_path(name))
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _payload(task_id: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": f"terminal-bench-{task_id}",
        "version": ENV_VERSION,
        "type": "ENV",
        "image": IMAGE_TEMPLATE.format(task_id=task_id),
        "backend_type": "SWE",
        "max_concurrency": MAX_CONCURRENCY,
        "api_version": "v2",
        "resource_profile": _resource_profile(task_id),
        "volumes": TASK_VOLUMES.get(task_id, []),
    }
    return payload


def _matches(
    actual: dict[str, Any] | None,
    expected: dict[str, object],
) -> bool:
    return (
        isinstance(actual, dict)
        and all(actual.get(key) == value for key, value in expected.items())
        and actual.get("deprecated") is False
    )


def register(task_id: str) -> dict[str, object]:
    expected = _payload(task_id)
    name = str(expected["name"])
    existing = _get(name)
    if _matches(existing, expected):
        return {"task_id": task_id, "method": "unchanged", "entry": existing}

    method = "PUT" if existing is not None else "POST"
    path = _registry_path(name) if method == "PUT" else "/v1/registry"
    _request(method, path, expected)
    registered = _get(name)
    if not _matches(registered, expected):
        raise RuntimeError(f"RLI registry mismatch for {task_id}: {registered!r}")
    return {"task_id": task_id, "method": method, "entry": registered}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "all"), default="pilot")
    args = parser.parse_args()

    results = [register(task_id) for task_id in _task_ids(args.mode)]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = OUTPUT_DIR / f"{args.mode}_{timestamp}.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(f"{result['task_id']}\t{result['method']}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
