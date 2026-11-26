"""Prepare and execute pinned-wheel, BF16/FP32 cuBLASLt GEMMs."""
import ctypes as ct
from functools import lru_cache
import json
from pathlib import Path
import subprocess

import torch

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "lt_bridge.so"
CONFIG_FIELDS = (
    "algorithm_id", "tile_id", "split_k", "reduction_scheme", "cta_swizzle",
    "custom_option", "stages_id", "inner_shape_id", "cluster_shape_id", "heuristic_rank",
)


def build():
    torch_root = Path(torch.__file__).resolve().parent
    wheel = torch_root.parent / "nvidia/cu13"
    cublas = wheel / "lib/libcublasLt.so.13"
    torch_cuda = torch_root / "lib/libtorch_cuda.so"
    subprocess.run(
        [
            "c++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-fPIC", "-shared",
            f"-I{wheel / 'include'}", "-I/usr/local/cuda/include",
            str(ROOT / "lt_bridge.cpp"), str(cublas), str(torch_cuda),
            f"-Wl,-rpath,{cublas.parent}", f"-Wl,-rpath,{torch_cuda.parent}", "-o", str(LIBRARY),
        ],
        check=True,
    )


@lru_cache(maxsize=1)
def _library():
    library = ct.CDLL(str(LIBRARY))
    int_ptr = ct.POINTER(ct.c_int)
    addr = ct.c_void_p
    i64 = ct.c_int64
    signatures = {
        "head_lt_version": (ct.c_size_t, []),
        "head_lt_inherited_workspace_size": (ct.c_size_t, []),
        "head_lt_inherited_workspace": (addr, []),
        "lt_matrix_create": (
            addr, [i64] * 6 + [ct.c_int] * 6 + [i64] * 3 + [ct.c_uint32, ct.c_int, ct.c_size_t, ct.c_bool],
        ),
        "head_lt_destroy": (None, [addr]),
        "head_lt_counts": (None, [addr, int_ptr]),
        "head_lt_info": (ct.c_float, [addr, ct.c_int, int_ptr]),
        "head_lt_workspace_size": (ct.c_size_t, [addr, ct.c_int]),
        "head_lt_run": (ct.c_int, [addr, ct.c_int] + [addr] * 4),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(library, name)
        function.restype = result
        function.argtypes = arguments
    return library


def _alignment(tensors, batch_strides=(0, 0, 0)):
    alignment = 256
    for value in [*(tensor.data_ptr() for tensor in tensors), *(2 * stride for stride in batch_strides)]:
        if value:
            alignment = min(alignment, value & -value)
    return alignment


def _matrix_layout(tensor, desired_row):
    if tensor.stride(-1) == 1:
        return tensor.stride(-2), bool(desired_row)
    if tensor.stride(-2) == 1:
        return tensor.stride(-1), not desired_row
    raise ValueError(f"Unsupported matrix strides {tensor.stride()}")


def _problem(lhs, rhs, out, row):
    if any(t.dtype != torch.bfloat16 or not t.is_cuda or t.device != lhs.device for t in (lhs, rhs, out)):
        raise ValueError("Expected same-device CUDA BF16 operands")
    if lhs.ndim not in (2, 3) or rhs.ndim != 2 or out.ndim != lhs.ndim:
        raise ValueError("Expected two matrices or a batched left operand and broadcast right operand")
    if lhs.size(-1) != rhs.size(-2) or out.shape != (*lhs.shape[:-1], rhs.size(-1)):
        raise ValueError(f"Incompatible matrix dimensions: {lhs.shape}, {rhs.shape}, {out.shape}")
    if out.stride(-1) != 1 or out.stride(-2) < out.size(-1):
        raise ValueError("Expected nonoverlapping row-major output")
    batch = lhs.size(0) if lhs.ndim == 3 else 1
    left_stride, right_stride = ((lhs.stride(0), 0) if lhs.ndim == 3 else (0, 0))
    out_stride = out.stride(0) if lhs.ndim == 3 else 0
    if lhs.ndim == 3 and out_stride < out.stride(-2) * (out.size(-2) - 1) + out.size(-1):
        raise ValueError("Expected nonoverlapping output batches")

    if row:
        # A row-major output expresses the original product directly.
        m, n, k = lhs.size(-2), rhs.size(-1), lhs.size(-1)
        lda, trans_a = _matrix_layout(lhs, False)
        ldb, trans_b = _matrix_layout(rhs, False)
        strides = left_stride, right_stride, out_stride
    else:
        # Column-major C is the transpose of the original product.
        m, n, k = rhs.size(-1), lhs.size(-2), lhs.size(-1)
        lda, trans_a = _matrix_layout(rhs, False)
        ldb, trans_b = _matrix_layout(lhs, False)
        strides = right_stride, left_stride, out_stride
    alignment = _alignment((lhs, rhs, out), strides)
    return (m, n, k, lda, ldb, out.stride(-2), trans_a, trans_b, row, row, row, batch, *strides, alignment)


class Plan:
    def __init__(self, lhs, rhs, out, *, row=False, exhaustive=False, workspace=None, heuristics=96):
        self.library = _library()
        self.problem = _problem(lhs, rhs, out, row)
        self.row = row
        self.device = lhs.device
        if workspace is None:
            workspace = self.library.head_lt_inherited_workspace_size()
        self.workspace_limit = workspace
        with torch.cuda.device(self.device):
            self.handle = self.library.lt_matrix_create(*self.problem, heuristics, workspace, exhaustive)
        if not self.handle:
            raise RuntimeError("cuBLASLt could not prepare the descriptor")
        counts = (ct.c_int * 3)()
        self.library.head_lt_counts(self.handle, counts)
        self.counts = dict(zip(("exposed_ids", "checked_configs", "valid_configs"), counts, strict=True))
        self.configs = []
        for index in range(counts[2]):
            values = (ct.c_int * len(CONFIG_FIELDS))()
            waves = self.library.head_lt_info(self.handle, index, values)
            self.configs.append({
                "index": index,
                **dict(zip(CONFIG_FIELDS, values, strict=True)),
                "waves": waves,
                "workspace_bytes": self.library.head_lt_workspace_size(self.handle, index),
            })

    def __del__(self):
        handle = getattr(self, "handle", None)
        if handle:
            self.library.head_lt_destroy(handle)

    def launch(self, index, lhs, rhs, out):
        a, b = (lhs, rhs) if self.row else (rhs, lhs)
        return self.library.head_lt_run(
            self.handle, index, a.data_ptr(), b.data_ptr(), out.data_ptr(),
            torch.cuda.current_stream(self.device).cuda_stream,
        )

    def run(self, index, lhs, rhs, out):
        if status := self.launch(index, lhs, rhs, out):
            raise RuntimeError(f"cuBLASLt candidate {index} returned status {status}")
        return out


@lru_cache(maxsize=8)
def _load_choices(path):
    return json.loads(Path(path).read_text())


_PLANS = {}
_activations = {"candidate": 0, "fallback": 0, "candidate_direct_dhidden": 0}


def activation_counts():
    return dict(_activations)


def prepare(lhs, rhs, out, *, choices=None):
    """Cache a measured algorithm by actual descriptor, device and library version."""
    version = _library().head_lt_version()
    capability = torch.cuda.get_device_capability(lhs.device)
    for row in (False, True):
        problem = _problem(lhs, rhs, out, row)
        key = json.dumps([version, list(capability), list(problem)])
        if choices is not None:
            config = _load_choices(Path(choices).resolve()).get(key)
            if config is None:
                continue
        else:
            config = None
        cache_key = (lhs.device, key, str(choices))
        if cache_key not in _PLANS:
            plan = Plan(lhs, rhs, out, row=row, exhaustive=config is not None)
            index = next((item["index"] for item in plan.configs if config is not None and all(
                item[field] == config[field] for field in CONFIG_FIELDS[:-1]
            )), None)
            if index is None:
                continue
            _PLANS[cache_key] = plan, index
        return _PLANS[cache_key]
    return None


def lt_mm(lhs, rhs, *, out=None, choices=None):
    """Execute a measured descriptor or the equivalent native operation."""
    if choices is None:
        _activations["fallback"] += 1
        return torch.mm(lhs, rhs, out=out)
    if out is None:
        out = lhs.new_empty((lhs.size(0), rhs.size(1)))
    prepared = prepare(lhs, rhs, out, choices=choices)
    if prepared is None:
        _activations["fallback"] += 1
        return torch.mm(lhs, rhs, out=out)
    plan, index = prepared
    _activations["candidate"] += 1
    return plan.run(index, lhs, rhs, out)


def lt_dhidden(dlogits, weight, out, *, choices):
    """Write a sequence slice of the BF16 destination with one strided batched call."""
    prepared = prepare(dlogits, weight, out, choices=choices)
    if prepared is None:
        _activations["fallback"] += 1
        flat = torch.mm(dlogits.flatten(0, 1), weight)
        return out.copy_(flat.view_as(out))
    plan, index = prepared
    _activations["candidate_direct_dhidden"] += 1
    return plan.run(index, dlogits, weight, out)


if __name__ == "__main__":
    build()
