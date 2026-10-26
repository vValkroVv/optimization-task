from __future__ import annotations

import functools
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
from types import ModuleType

log = logging.getLogger(__name__)

_CSRC_DIR = Path(__file__).parents[1] / "csrc"
_SOURCES = (
    "activation_stats_kernels.cu",
    "actlog_observer.cpp",
    "attention_calls.cpp",
    "bindings.cpp",
    "expert_activation_calls.cpp",
    "linear_state_kernels.cu",
    "materialized_weight_observer.cpp",
    "monitor_observations.cpp",
    "quantile_kernels.cu",
    "recorder_copy_kernels.cu",
    "reduction_kernels.cu",
    "router_accumulate_kernels.cu",
    "router_balance_calls.cpp",
    "router_margin_calls.cpp",
    "router_margin_kernels.cu",
    "routing_calls.cpp",
    "row_norm_kernels.cu",
    "triton_launch.cpp",
    "weight_norm_kernels.cu",
)
_HEADER_SUFFIXES = (".h", ".cuh")
_MODULE_NAME = "metric_sampler_activation_ext"
_BUILD_ROOT_ENV = "METRIC_SAMPLER_ACTIVATION_EXT_DIR"
_BUILD_DIRECTORY_NAME_LENGTH = 12
# The JIT build gives C++ sources no optimization level of its own: without one, the bindings that every call of
# the extension goes through, and its gradient hooks, would be compiled unoptimized.
_CXX_FLAGS = ("-O3",)
_CUDA_FLAGS = ("-O3", "--expt-extended-lambda")


def _build_root() -> Path:
    configured = os.environ.get(_BUILD_ROOT_ENV)
    return Path(configured) if configured else Path(__file__).resolve().parents[3] / "build" / "activation_ext"


def _csrc_files() -> list[Path]:
    """The sources to compile, then the headers they include."""
    headers = sorted(path for path in _CSRC_DIR.iterdir() if path.suffix in _HEADER_SUFFIXES)
    return [_CSRC_DIR / source for source in _SOURCES] + headers


def build_directory(build_root: Path | None = None) -> Path:
    digest = hashlib.sha1()
    for path in _csrc_files():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    # Built with other flags, the same sources are another build.
    digest.update(" ".join(_CXX_FLAGS + _CUDA_FLAGS).encode())
    root = build_root if build_root is not None else _build_root()
    return root / digest.hexdigest()[:_BUILD_DIRECTORY_NAME_LENGTH]


def build(build_root: Path | None = None) -> ModuleType:
    """Compiles the extension into a directory named after the hash of its sources, headers and flags."""
    # Imported here: importing it queries the CUDA runtime, which ranks that only load the prebuilt module do not need.
    from torch.utils.cpp_extension import load as compile_extension

    directory = build_directory(build_root)
    directory.mkdir(parents=True, exist_ok=True)
    return compile_extension(
        name=_MODULE_NAME,
        sources=[str(_CSRC_DIR / source) for source in _SOURCES],
        build_directory=str(directory),
        extra_cflags=list(_CXX_FLAGS),
        extra_cuda_cflags=list(_CUDA_FLAGS),
    )


def load(build_root: Path | None = None) -> ModuleType | None:
    """The prebuilt module for the current sources, or None (with one warning) if there is none.

    A training rank never compiles: a missing or stale build falls back to the Triton kernels instead of
    stalling every rank on a compile."""
    try:
        candidates = sorted(build_directory(build_root).glob(f"{_MODULE_NAME}*.so"))
        if not candidates:
            log.warning("activation_ext: no prebuilt extension for these sources, using the Triton kernel")
            return None
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, candidates[0])
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        log.warning("activation_ext: loading the prebuilt extension failed, using the Triton kernel", exc_info=True)
        return None


@functools.cache
def prebuilt() -> ModuleType | None:
    """`load()` from the default build root, once per process."""
    return load()
