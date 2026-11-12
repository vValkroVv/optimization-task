"""Stage the pinned HybridEP fused-scan bound correction for this campaign."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from types import ModuleType


_RUNTIME_DIR_ENV = "GIFT_HYBRID_EP_SCAN_RUNTIME_DIR"
_RANK_ENV = "RANK"
_TARGET_MODULE = "torchtitan.distributed.deepep.hybridep"
_WRAPPER_SHA256 = "63e425916b3c6b0c881ad402af35a2b00c4c84203fcde611a6715744b89210cb"
_HEADER_SHA256 = "b507f65cfa4580f16342076fc2d00afe81e43f64ddefcc4a3cf43681aef11b4b"
_HEADER_RELATIVE_PATH = Path("backend/hybrid_ep_backend.cuh")
_OLD_BOUND = (
    b"int32_t thread_local_all_block_local_experts_sum[NUM_OF_EXPERTS_PER_RANK - 1];"
)
_NEW_BOUND = (
    b"int32_t thread_local_all_block_local_experts_sum[NUM_OF_EXPERTS_PER_RANK];"
)


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _tree_records(directory: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"HybridEP runtime source contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"HybridEP runtime source contains a non-file: {path}")
        relative_path = path.relative_to(directory).as_posix()
        records.append(
            {
                "path": relative_path,
                "sha256": _sha256(path.read_bytes()),
                "mode": stat.S_IMODE(path.stat().st_mode),
            }
        )
    return records


def _tree_sha256(records: list[dict[str, object]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(payload)


def _source_paths(
    wrapper_path: Path,
) -> tuple[Path, Path, bytes, list[dict[str, object]]]:
    wrapper_path = wrapper_path.resolve(strict=True)
    if (
        not wrapper_path.is_file()
        or _sha256(wrapper_path.read_bytes()) != _WRAPPER_SHA256
    ):
        raise ValueError(
            "Unsupported DeepEP HybridEP wrapper: expected the campaign-pinned SHA-256 "
            f"{_WRAPPER_SHA256} at {wrapper_path}"
        )

    backend_dir = wrapper_path.parent / "backend"
    header_path = backend_dir / _HEADER_RELATIVE_PATH.name
    if not backend_dir.is_dir() or not header_path.is_file():
        raise ValueError(f"Unsupported DeepEP HybridEP backend tree: {backend_dir}")
    if header_path.is_symlink():
        raise ValueError(
            f"DeepEP HybridEP backend header must be a regular file: {header_path}"
        )

    header_contents = header_path.read_bytes()
    if _sha256(header_contents) != _HEADER_SHA256:
        raise ValueError(
            "Unsupported DeepEP HybridEP backend header: expected the campaign-pinned "
            f"SHA-256 {_HEADER_SHA256} at {header_path}"
        )
    if header_contents.count(_OLD_BOUND) != 1:
        raise ValueError(
            "Pinned DeepEP HybridEP backend header does not contain exactly one expected bound"
        )

    return wrapper_path, backend_dir, header_contents, _tree_records(backend_dir)


def _write_exclusive(path: Path, contents: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())


def _stage_runtime_copy(
    wrapper_path: Path,
    runtime_root: Path,
    rank: int,
) -> tuple[Path, Path, Path]:
    """Publish a fresh rank-local wrapper/backend copy and an empty JIT cache root."""
    source_wrapper, source_backend, original_header, source_records = _source_paths(
        wrapper_path
    )
    runtime_root = runtime_root.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)

    rank_root = runtime_root / f"rank-{rank:05d}"
    try:
        rank_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RuntimeError(
            f"HybridEP scan-fix rank directory already exists; use a fresh attempt root: {rank_root}"
        ) from exc

    runtime_dir = rank_root / "runtime"
    cache_root = rank_root / "native-cache"
    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=rank_root))

    shutil.copy2(source_wrapper, staging_dir / "hybrid_ep_buffer.py")
    shutil.copytree(source_backend, staging_dir / "backend", copy_function=shutil.copy2)

    staged_header = staging_dir / _HEADER_RELATIVE_PATH
    patched_header = original_header.replace(_OLD_BOUND, _NEW_BOUND, 1)
    if patched_header == original_header:
        raise ValueError(
            "HybridEP scan-fix patch did not change the pinned backend header"
        )
    staged_header.write_bytes(patched_header)

    staged_records = _tree_records(staging_dir / "backend")
    wrapper_copy = staging_dir / "hybrid_ep_buffer.py"
    module_name = f"deep_ep.cuda_graph_campaign_scan_fix_rank_{rank}_pid_{os.getpid()}"
    cache_path = cache_root / ".deepep" / "hybrid_ep" / "jit" / f"proc-{os.getpid()}"
    manifest = {
        "rank": rank,
        "pid": os.getpid(),
        "entry_path": os.path.realpath(sys.argv[0]),
        "source_wrapper_path": str(source_wrapper),
        "source_wrapper_sha256": _sha256(source_wrapper.read_bytes()),
        "runtime_wrapper_path": str(runtime_dir / "hybrid_ep_buffer.py"),
        "runtime_wrapper_sha256": _sha256(wrapper_copy.read_bytes()),
        "source_backend_tree_sha256": _tree_sha256(source_records),
        "runtime_backend_tree_sha256": _tree_sha256(staged_records),
        "source_backend_files": source_records,
        "runtime_backend_files": staged_records,
        "patch": {
            "path": _HEADER_RELATIVE_PATH.as_posix(),
            "old_line": _OLD_BOUND.decode(),
            "new_line": _NEW_BOUND.decode(),
            "source_sha256": _sha256(original_header),
            "runtime_sha256": _sha256(patched_header),
            "mode": stat.S_IMODE(staged_header.stat().st_mode),
        },
        "wrapper_module": module_name,
        "native_cache_root": str(cache_root),
        "native_cache_path": str(cache_path),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _write_exclusive(staging_dir / "runtime_manifest.json", manifest_bytes)

    cache_root.mkdir(mode=0o700)
    if runtime_dir.exists():
        raise RuntimeError(f"HybridEP runtime directory already exists: {runtime_dir}")
    os.rename(staging_dir, runtime_dir)
    return (
        runtime_dir / "hybrid_ep_buffer.py",
        cache_root,
        runtime_dir / "runtime_manifest.json",
    )


def _load_relocated_class(
    wrapper_path: Path, module_name: str
) -> tuple[ModuleType, type]:
    if module_name in sys.modules:
        raise RuntimeError(
            f"HybridEP relocated wrapper module already exists: {module_name}"
        )
    spec = importlib.util.spec_from_file_location(module_name, wrapper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load staged HybridEP wrapper at {wrapper_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    relocated_class = getattr(module, "HybridEPBuffer", None)
    if (
        not isinstance(relocated_class, type)
        or relocated_class.__module__ != module_name
    ):
        sys.modules.pop(module_name, None)
        raise TypeError(f"Staged wrapper did not define HybridEPBuffer: {wrapper_path}")
    return module, relocated_class


def _replace_factory_class(
    target_module: ModuleType,
    original_class: type,
    relocated_class: type,
) -> None:
    if getattr(target_module, "_buffer", object()) is not None:
        raise RuntimeError(
            "HybridEP scan fix must be installed before buffer construction"
        )
    if getattr(target_module, "HybridEPBuffer", None) is not original_class:
        raise RuntimeError("GIFT HybridEP factory changed before scan-fix installation")
    if original_class.__module__ != "deep_ep.hybrid_ep_buffer":
        raise RuntimeError(
            "Unsupported GIFT HybridEP factory: expected the installed DeepEP native wrapper"
        )
    target_module.HybridEPBuffer = relocated_class


def install_hybrid_ep_scan_fix() -> Path:
    """Install the pinned header correction before GIFT constructs its buffer."""
    runtime_dir_value = os.environ.get(_RUNTIME_DIR_ENV)
    if not runtime_dir_value:
        raise ValueError(
            f"Set {_RUNTIME_DIR_ENV} to a fresh absolute campaign attempt path"
        )
    runtime_root = Path(runtime_dir_value)
    if not runtime_root.is_absolute():
        raise ValueError(
            f"{_RUNTIME_DIR_ENV} must be an absolute path: {runtime_dir_value}"
        )

    rank_value = os.environ.get(_RANK_ENV)
    if rank_value is None:
        raise ValueError(f"{_RANK_ENV} must be set by the distributed launcher")
    try:
        rank = int(rank_value)
    except ValueError as exc:
        raise ValueError(
            f"{_RANK_ENV} must be a non-negative integer: {rank_value}"
        ) from exc
    if rank < 0:
        raise ValueError(f"{_RANK_ENV} must be a non-negative integer: {rank_value}")

    target_module = importlib.import_module(_TARGET_MODULE)
    original_class = getattr(target_module, "HybridEPBuffer", None)
    if not isinstance(original_class, type):
        raise TypeError(
            f"{_TARGET_MODULE}.HybridEPBuffer is not the expected native wrapper"
        )
    if original_class.__module__ != "deep_ep.hybrid_ep_buffer":
        raise RuntimeError(
            "Unsupported GIFT HybridEP factory: expected the installed DeepEP native wrapper"
        )
    if getattr(target_module, "_buffer", object()) is not None:
        raise RuntimeError(
            "HybridEP scan fix must be installed before buffer construction"
        )

    source_wrapper = Path(inspect.getsourcefile(original_class) or "").resolve(
        strict=True
    )
    relocated_wrapper, cache_root, manifest_path = _stage_runtime_copy(
        source_wrapper, runtime_root, rank
    )
    manifest = json.loads(manifest_path.read_text())
    module_name = manifest["wrapper_module"]
    module_was_loaded = module_name in sys.modules
    cache_was_set = "HYBRID_EP_CACHE_DIR" in os.environ
    previous_cache = os.environ.get("HYBRID_EP_CACHE_DIR", "")
    try:
        os.environ["HYBRID_EP_CACHE_DIR"] = str(cache_root)
        _, relocated_class = _load_relocated_class(relocated_wrapper, module_name)
        _replace_factory_class(target_module, original_class, relocated_class)
        target_module.logger.info(
            "Campaign HybridEP fused-scan bound fix active: rank=%d wrapper=%s header_sha256=%s native_cache=%s",
            rank,
            relocated_wrapper,
            manifest["patch"]["runtime_sha256"],
            cache_root,
        )
    except BaseException:
        target_module.HybridEPBuffer = original_class
        if cache_was_set:
            os.environ["HYBRID_EP_CACHE_DIR"] = previous_cache
        else:
            os.environ.pop("HYBRID_EP_CACHE_DIR", None)
        if not module_was_loaded:
            sys.modules.pop(module_name, None)
        raise
    return manifest_path
