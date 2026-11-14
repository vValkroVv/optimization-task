#!/usr/bin/env python3
"""Native measurement bridge that installs an explicit FlashQLA backend per rank."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

TASK_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = Path(__file__).resolve().parent
BASE_MEASURE_PATH = TASK_ROOT / "runs/overnight-streaming-perf/measure.py"
BACKENDS = {"fla", "external_gate", "fused_gate"}

ENV_BACKEND = "TASK_FLASHQLA_BACKEND"
ENV_CAPTURE_MODE = "TASK_FLASHQLA_CAPTURE_MODE"
ENV_CAPTURE_DIR = "TASK_FLASHQLA_CAPTURE_DIR"
ENV_TILELANG_ROOT = "TASK_FLASHQLA_TILELANG_ROOT"
ENV_SOURCE_IDENTITY = "TASK_FLASHQLA_SOURCE_IDENTITY"


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native measurement adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_base_measure() -> Any:
    return _load_module(BASE_MEASURE_PATH, "_flash_qla_overnight_measure")


def _rank_environment(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} must be set for the FlashQLA native runner")
    try:
        rank = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer in [0, 7], got {value!r}") from exc
    if not 0 <= rank < 8:
        raise RuntimeError(f"{name} must be an integer in [0, 7], got {rank}")
    return rank


def _bridge_configuration() -> tuple[str, str, Path | None, dict[str, Any]]:
    backend = os.environ.get(ENV_BACKEND, "")
    if backend not in BACKENDS:
        raise RuntimeError(f"{ENV_BACKEND} must be one of {sorted(BACKENDS)}, got {backend!r}")
    capture_mode = os.environ.get(ENV_CAPTURE_MODE, "")
    if capture_mode not in {"disabled", "fixtures"}:
        raise RuntimeError(
            f"{ENV_CAPTURE_MODE} must be 'disabled' or 'fixtures', got {capture_mode!r}"
        )
    capture_text = os.environ.get(ENV_CAPTURE_DIR)
    capture_dir = Path(capture_text) if capture_text else None
    if capture_mode == "fixtures":
        if backend != "fla":
            raise RuntimeError("fixture capture requires the native FLA backend")
        if capture_dir is None:
            raise RuntimeError(f"{ENV_CAPTURE_DIR} is required in fixture capture mode")
    elif capture_dir is not None:
        raise RuntimeError(f"{ENV_CAPTURE_DIR} must be unset when fixture capture is disabled")
    try:
        source_identity = json.loads(os.environ[ENV_SOURCE_IDENTITY])
    except KeyError as exc:
        raise RuntimeError(f"{ENV_SOURCE_IDENTITY} must be set for every run") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{ENV_SOURCE_IDENTITY} must contain JSON") from exc
    if not isinstance(source_identity, dict):
        raise RuntimeError(f"{ENV_SOURCE_IDENTITY} must contain a JSON object")
    return backend, capture_mode, capture_dir, source_identity


def _measurement_dir(adapter: Any) -> Path:
    options = getattr(adapter, "options", None)
    if not isinstance(options, dict) or "output_dir" not in options:
        raise RuntimeError("native adapter does not expose its measurement output directory")
    measurement = Path(options["output_dir"]).resolve()
    try:
        measurement.relative_to(TASK_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"native measurement directory is outside the task root: {measurement}") from exc
    if not measurement.is_dir():
        raise RuntimeError(f"native measurement directory is missing: {measurement}")
    return measurement


def _configure_tilelang_cache(adapter: Any, global_rank: int) -> tuple[Path, Path]:
    measurement = _measurement_dir(adapter)
    expected_root = measurement.parent / "cache" / "tilelang"
    configured_root_text = os.environ.get(ENV_TILELANG_ROOT)
    if not configured_root_text:
        raise RuntimeError(f"{ENV_TILELANG_ROOT} must route TileLang caches into this run")
    configured_root = Path(configured_root_text).resolve()
    if configured_root != expected_root.resolve():
        raise RuntimeError(
            f"{ENV_TILELANG_ROOT} must name this attempt's TileLang cache: {configured_root}"
        )
    if configured_root.is_symlink() or not configured_root.is_dir():
        raise RuntimeError(f"task-owned TileLang cache root is missing or unsafe: {configured_root}")

    rank_root = configured_root / f"rank_{global_rank}"
    tmp_dir = rank_root / "tmp"
    if rank_root.is_symlink() or tmp_dir.is_symlink() or not rank_root.is_dir() or not tmp_dir.is_dir():
        raise RuntimeError(f"task-owned TileLang rank cache is missing or unsafe: {rank_root}")
    cache_dir = rank_root
    os.environ["TILELANG_CACHE_DIR"] = str(cache_dir)
    os.environ["TILELANG_TMP_DIR"] = str(tmp_dir)
    return cache_dir, tmp_dir


def _load_backend_installer() -> Callable[[str], Any]:
    module = importlib.import_module("torchtitan.experiments.flash_qla.backend")
    installer = getattr(module, "install_backend", None)
    if not callable(installer):
        raise RuntimeError("FlashQLA backend module does not expose install_backend")
    return installer


def _load_capture_installer() -> Callable[..., Any]:
    capture_path = BRIDGE_DIR / "capture.py"
    spec = importlib.util.spec_from_file_location("_flash_qla_fixture_capture", capture_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FlashQLA fixture capture adapter from {capture_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    installer = getattr(module, "install_capture", None)
    if not callable(installer):
        raise RuntimeError("FlashQLA capture module does not expose install_capture")
    return installer


def _validate_capture_directory(capture_dir: Path, measurement: Path) -> Path:
    attempt = measurement.parent.resolve()
    if not capture_dir.is_absolute() or ".." in capture_dir.parts:
        raise RuntimeError(f"{ENV_CAPTURE_DIR} must be an absolute task-owned path")
    try:
        relative = capture_dir.relative_to(attempt)
    except ValueError as exc:
        raise RuntimeError(f"{ENV_CAPTURE_DIR} must remain inside this run attempt") from exc
    if not relative.parts:
        raise RuntimeError(f"{ENV_CAPTURE_DIR} must name a fresh directory inside the attempt")
    current = attempt
    if current.is_symlink():
        raise RuntimeError(f"{ENV_CAPTURE_DIR} must not contain symlink components")
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise RuntimeError(f"{ENV_CAPTURE_DIR} must not contain symlink components")
    resolved = capture_dir.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"{ENV_CAPTURE_DIR} must be a run-owned directory: {resolved}")
    return resolved


def _write_bootstrap_receipt(
    *,
    measurement: Path,
    rank: int,
    local_rank: int,
    backend: str,
    capture_mode: str,
    capture_dir: Path | None,
    identity: Any,
    source_identity: dict[str, Any],
    cuda_device: int,
    tilelang_cache_dir: Path,
    tilelang_tmp_dir: Path,
) -> dict[str, Any]:
    try:
        json.dumps(identity)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("FlashQLA installer identity must be JSON-serializable") from exc
    receipt = {
        "schema_version": 1,
        "rank": rank,
        "local_rank": local_rank,
        "backend_variant": backend,
        "capture_mode": capture_mode,
        "capture_fixture_dir": None if capture_dir is None else str(capture_dir),
        "installer_identity": identity,
        "source_identity": source_identity,
        "cuda_device": cuda_device,
        "tilelang_cache_dir": str(tilelang_cache_dir),
        "tilelang_tmp_dir": str(tilelang_tmp_dir),
        "installation_stage": "before torchtitan.train.main import",
    }
    path = measurement / f"flash-qla-bootstrap-rank-{rank:02d}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print("flash_qla_bootstrap=" + json.dumps(receipt, sort_keys=True))
    return receipt


def _bootstrap_and_run(
    train_argv: list[str],
    adapter: Any,
    native_main: Callable[[list[str], Any], Any],
    *,
    torch_module: Any | None = None,
    backend_installer_loader: Callable[[], Callable[[str], Any]] | None = None,
    capture_installer_loader: Callable[[], Callable[..., Any]] | None = None,
) -> Any:
    backend, capture_mode, capture_dir, source_identity = _bridge_configuration()
    rank = _rank_environment("RANK")
    local_rank = _rank_environment("LOCAL_RANK")
    if rank != local_rank:
        raise RuntimeError(
            f"single-node FlashQLA run requires RANK == LOCAL_RANK, got {rank} and {local_rank}"
        )
    if os.environ.get("WORLD_SIZE") != "8":
        raise RuntimeError("WORLD_SIZE must be 8 for the FlashQLA native runner")
    triton_cache = Path(os.environ.get("TRITON_CACHE_DIR", ""))
    if triton_cache.name != f"rank_{rank}":
        raise RuntimeError(
            "base measurement adapter must route TRITON_CACHE_DIR before FlashQLA bootstrap"
        )

    measurement = _measurement_dir(adapter)
    if capture_dir is not None:
        capture_dir = _validate_capture_directory(capture_dir, measurement)
    original_cache = os.environ.get("TILELANG_CACHE_DIR")
    original_tmp = os.environ.get("TILELANG_TMP_DIR")
    cache_dir, tmp_dir = _configure_tilelang_cache(adapter, rank)
    try:
        if torch_module is None:
            torch_module = importlib.import_module("torch")
        torch_module.cuda.set_device(local_rank)
        cuda_device = int(torch_module.cuda.current_device())
        if cuda_device != local_rank:
            raise RuntimeError(
                f"CUDA current device {cuda_device} does not match LOCAL_RANK {local_rank}"
            )

        if capture_mode == "fixtures":
            loader = capture_installer_loader or _load_capture_installer
            installer = loader()
            identity = installer(capture_dir, max_fixtures=6)
        else:
            loader = backend_installer_loader or _load_backend_installer
            installer = loader()
            identity = installer(backend)

        _write_bootstrap_receipt(
            measurement=measurement,
            rank=rank,
            local_rank=local_rank,
            backend=backend,
            capture_mode=capture_mode,
            capture_dir=capture_dir,
            identity=identity,
            source_identity=source_identity,
            cuda_device=cuda_device,
            tilelang_cache_dir=cache_dir,
            tilelang_tmp_dir=tmp_dir,
        )
        return native_main(train_argv, adapter)
    finally:
        if original_cache is None:
            os.environ.pop("TILELANG_CACHE_DIR", None)
        else:
            os.environ["TILELANG_CACHE_DIR"] = original_cache
        if original_tmp is None:
            os.environ.pop("TILELANG_TMP_DIR", None)
        else:
            os.environ["TILELANG_TMP_DIR"] = original_tmp


def main(argv: list[str] | None = None) -> Any:
    base = _load_base_measure()
    original = base._run_native_main

    def startup_hook(train_argv: list[str], adapter: Any) -> Any:
        return _bootstrap_and_run(train_argv, adapter, original)

    base._run_native_main = startup_hook
    try:
        return base.main(argv)
    finally:
        base._run_native_main = original


if __name__ == "__main__":
    main()

