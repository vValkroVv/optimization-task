#!/usr/bin/env python3
"""Native measurement bridge that installs an explicit FlashQLA backend per rank."""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import inspect
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

TASK_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = Path(__file__).resolve().parent
BASE_MEASURE_PATH = TASK_ROOT / "runs/overnight-streaming-perf/measure.py"
BACKENDS = {"fla", "external_gate", "fused_gate"}
ROUTING_METADATA = {"original", "fused"}
CROSS_ENTROPY = {"original", "stable"}

ENV_BACKEND = "TASK_FLASHQLA_BACKEND"
ENV_CAPTURE_MODE = "TASK_FLASHQLA_CAPTURE_MODE"
ENV_CAPTURE_DIR = "TASK_FLASHQLA_CAPTURE_DIR"
ENV_TILELANG_ROOT = "TASK_FLASHQLA_TILELANG_ROOT"
ENV_SOURCE_IDENTITY = "TASK_FLASHQLA_SOURCE_IDENTITY"
ENV_ROUTING_METADATA = "TASK_FLASHQLA_ROUTING_METADATA"
ENV_CROSS_ENTROPY = "TASK_FLASHQLA_CROSS_ENTROPY"


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


def _bridge_configuration() -> tuple[str, str, Path | None, dict[str, Any], str, str]:
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
    routing_metadata = os.environ.get(ENV_ROUTING_METADATA, "")
    if routing_metadata not in ROUTING_METADATA:
        raise RuntimeError(
            f"{ENV_ROUTING_METADATA} must be one of {sorted(ROUTING_METADATA)}, "
            f"got {routing_metadata!r}"
        )
    cross_entropy = os.environ.get(ENV_CROSS_ENTROPY, "")
    if cross_entropy not in CROSS_ENTROPY:
        raise RuntimeError(
            f"{ENV_CROSS_ENTROPY} must be one of {sorted(CROSS_ENTROPY)}, "
            f"got {cross_entropy!r}"
        )
    return backend, capture_mode, capture_dir, source_identity, routing_metadata, cross_entropy


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


def _safe_task_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"{label} must be an absolute task-owned path: {path}")
    task_root = TASK_ROOT.resolve()
    try:
        relative = path.relative_to(task_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must remain inside the task root: {path}") from exc
    current = task_root
    if current.is_symlink():
        raise RuntimeError(f"{label} must not traverse symlinks: {path}")
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise RuntimeError(f"{label} must not traverse symlinks: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} is missing or unsafe: {path}") from exc
    if not resolved.is_dir():
        raise RuntimeError(f"{label} is not a directory: {resolved}")
    try:
        resolved.relative_to(task_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must remain inside the task root: {path}") from exc
    return resolved


def _configure_tilelang_cache(adapter: Any, global_rank: int) -> tuple[Path, Path]:
    _measurement_dir(adapter)
    compiler_cache_text = os.environ.get("TASK_COMPILER_CACHE_ROOT")
    if not compiler_cache_text:
        raise RuntimeError("TASK_COMPILER_CACHE_ROOT must select the validated task cache")
    compiler_cache_root = _safe_task_directory(
        Path(compiler_cache_text), "TASK_COMPILER_CACHE_ROOT"
    )
    expected_root = compiler_cache_root / "tilelang"
    configured_root_text = os.environ.get(ENV_TILELANG_ROOT)
    if not configured_root_text:
        raise RuntimeError(f"{ENV_TILELANG_ROOT} must route TileLang caches into this run")
    configured_root = _safe_task_directory(
        Path(configured_root_text), ENV_TILELANG_ROOT
    )
    if configured_root != expected_root.resolve():
        raise RuntimeError(
            f"{ENV_TILELANG_ROOT} must name the selected compiler cache's TileLang root: {configured_root}"
        )

    rank_root = configured_root / f"rank_{global_rank}"
    tmp_dir = rank_root / "tmp"
    cache_dir = _safe_task_directory(rank_root, "TileLang rank cache")
    tmp_dir = _safe_task_directory(tmp_dir, "TileLang rank temporary directory")
    os.environ["TILELANG_CACHE_DIR"] = str(cache_dir)
    os.environ["TILELANG_TMP_DIR"] = str(tmp_dir)
    return cache_dir, tmp_dir


def _load_backend_installer() -> Callable[[str], Any]:
    module = importlib.import_module("torchtitan.experiments.flash_qla.backend")
    installer = getattr(module, "install_backend", None)
    if not callable(installer):
        raise RuntimeError("FlashQLA backend module does not expose install_backend")
    return installer


def _load_native_modules() -> tuple[Any, Any, Any, Any, Any]:
    """Load the exact module globals used by HybridEP and CE constructors."""
    return (
        importlib.import_module("torchtitan.distributed.deepep.hybridep"),
        importlib.import_module("torchtitan.distributed.deepep.routing"),
        importlib.import_module("torchtitan.experiments.four_fusion.routing_metadata"),
        importlib.import_module("torchtitan.components.loss"),
        importlib.import_module("torchtitan.experiments.four_fusion.stable_ce"),
    )


def _unwrap_compiled_callable(function: Any) -> Any:
    current = function
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        original = getattr(current, "_torchdynamo_orig_callable", None)
        if not callable(original):
            original = getattr(current, "__wrapped__", None)
        if not callable(original):
            break
        current = original
    return current


def _callable_identity(function: Any) -> dict[str, Any]:
    if not callable(function):
        raise RuntimeError("selected native implementation is not callable")
    try:
        source_text = inspect.getsourcefile(function) or inspect.getfile(function)
    except (OSError, TypeError) as exc:
        raise RuntimeError("cannot identify selected native callable source") from exc
    source_path = Path(source_text).resolve()
    if not source_path.is_file():
        raise RuntimeError(f"selected native callable source is missing: {source_path}")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return {
        "object_id": int(id(function)),
        "module": getattr(function, "__module__", type(function).__module__),
        "qualname": getattr(
            function, "__qualname__", getattr(function, "__name__", type(function).__qualname__)
        ),
        "file": str(source_path),
        "sha256": digest,
    }


@contextmanager
def _bind_native_fusion_choices(
    routing_metadata: str, cross_entropy: str
) -> Iterator[dict[str, Any]]:
    if routing_metadata not in ROUTING_METADATA:
        raise RuntimeError(f"unsupported routing metadata selection: {routing_metadata!r}")
    if cross_entropy not in CROSS_ENTROPY:
        raise RuntimeError(f"unsupported cross-entropy selection: {cross_entropy!r}")

    hybridep, original_routing, fused_routing, loss_module, stable_ce = (
        _load_native_modules()
    )
    route_callable = getattr(
        original_routing if routing_metadata == "original" else fused_routing,
        "indices_to_map",
        None,
    )
    native_ce = getattr(loss_module, "cross_entropy_loss", None)
    stable_ce_callable = getattr(stable_ce, "cross_entropy_loss", None)
    ce_callable = native_ce if cross_entropy == "original" else stable_ce_callable
    if not callable(route_callable):
        raise RuntimeError("selected routing module has no callable indices_to_map")
    if not callable(native_ce) or not callable(stable_ce_callable) or not callable(ce_callable):
        raise RuntimeError("native and stable cross-entropy callables must be available")

    original_hybridep_route = getattr(hybridep, "indices_to_map", None)
    original_stable_ce = stable_ce_callable
    constructor_restore: list[tuple[type[Any], Any]] = []
    constructed_losses: list[dict[str, Any]] = []
    selected_ce_identity = _callable_identity(ce_callable)
    selection = {
        "routing_metadata": {
            "selection": routing_metadata,
            "callable": _callable_identity(route_callable),
        },
        "cross_entropy": {
            "selection": cross_entropy,
            "callable": selected_ce_identity,
        },
        "constructed_loss_callables": constructed_losses,
    }

    try:
        hybridep.indices_to_map = route_callable
        stable_ce.cross_entropy_loss = ce_callable
        for class_name in ("CrossEntropyLoss", "ChunkedCELoss"):
            loss_class = getattr(loss_module, class_name, None)
            if not isinstance(loss_class, type):
                raise RuntimeError(f"native loss module has no {class_name} class")
            original_init = loss_class.__init__
            constructor_restore.append((loss_class, original_init))

            def checked_init(
                self: Any,
                *args: Any,
                _original_init: Any = original_init,
                _class_name: str = class_name,
                **kwargs: Any,
            ) -> None:
                _original_init(self, *args, **kwargs)
                actual = getattr(self, "fn", None)
                unwrapped = _unwrap_compiled_callable(actual)
                if unwrapped is not ce_callable:
                    raise RuntimeError(
                        f"{_class_name} constructed with an unexpected cross-entropy callable"
                    )
                constructed_losses.append(
                    {
                        "class": f"{type(self).__module__}.{type(self).__qualname__}",
                        "callable": selected_ce_identity,
                        "runtime_callable_object_id": int(id(actual)),
                        "runtime_callable_type": (
                            f"{type(actual).__module__}.{type(actual).__qualname__}"
                        ),
                        "compiled_wrapper": actual is not unwrapped,
                    }
                )

            loss_class.__init__ = checked_init

        if hybridep.indices_to_map is not route_callable:
            raise RuntimeError("HybridEP routing binding did not take effect")
        if stable_ce.cross_entropy_loss is not ce_callable:
            raise RuntimeError("stable CE constructor binding did not take effect")
        yield selection
    finally:
        for loss_class, original_init in reversed(constructor_restore):
            loss_class.__init__ = original_init
        hybridep.indices_to_map = original_hybridep_route
        stable_ce.cross_entropy_loss = original_stable_ce


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
    fusion_selection: dict[str, Any],
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
        "routing_metadata": fusion_selection["routing_metadata"]["selection"],
        "routing_metadata_callable": fusion_selection["routing_metadata"]["callable"],
        "cross_entropy": fusion_selection["cross_entropy"]["selection"],
        "cross_entropy_callable": fusion_selection["cross_entropy"]["callable"],
        "constructed_loss_callables": [],
        "native_main_completed": None,
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


def _update_bootstrap_receipt(
    measurement: Path, rank: int, receipt: dict[str, Any]
) -> None:
    path = measurement / f"flash-qla-bootstrap-rank-{rank:02d}.json"
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)
    print("flash_qla_bootstrap_final=" + json.dumps(receipt, sort_keys=True))


def _bootstrap_and_run(
    train_argv: list[str],
    adapter: Any,
    native_main: Callable[[list[str], Any], Any],
    *,
    torch_module: Any | None = None,
    backend_installer_loader: Callable[[], Callable[[str], Any]] | None = None,
    capture_installer_loader: Callable[[], Callable[..., Any]] | None = None,
) -> Any:
    (
        backend,
        capture_mode,
        capture_dir,
        source_identity,
        routing_metadata,
        cross_entropy,
    ) = _bridge_configuration()
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

        with _bind_native_fusion_choices(routing_metadata, cross_entropy) as fusion_selection:
            receipt = _write_bootstrap_receipt(
                measurement=measurement,
                rank=rank,
                local_rank=local_rank,
                backend=backend,
                capture_mode=capture_mode,
                capture_dir=capture_dir,
                identity=identity,
                source_identity=source_identity,
                fusion_selection=fusion_selection,
                cuda_device=cuda_device,
                tilelang_cache_dir=cache_dir,
                tilelang_tmp_dir=tmp_dir,
            )
            try:
                result = native_main(train_argv, adapter)
            except BaseException as exc:
                receipt["native_main_completed"] = False
                receipt["native_main_error_type"] = type(exc).__name__
                receipt["constructed_loss_callables"] = list(
                    fusion_selection["constructed_loss_callables"]
                )
                _update_bootstrap_receipt(measurement, rank, receipt)
                raise
            receipt["native_main_completed"] = True
            receipt["constructed_loss_callables"] = list(
                fusion_selection["constructed_loss_callables"]
            )
            _update_bootstrap_receipt(measurement, rank, receipt)
            return result
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

