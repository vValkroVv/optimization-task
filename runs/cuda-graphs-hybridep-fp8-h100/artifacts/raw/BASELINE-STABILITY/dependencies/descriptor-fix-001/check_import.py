"""Verify the isolated dependency and its real binary include root without CUDA."""
import ctypes
import hashlib
import json
from pathlib import Path

import torch
import deep_gemm
from torchtitan.ops.float8.deep_gemm_handle import _DeepGemmHandle


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "deep_gemm"


class StringPrefix(ctypes.Structure):
    # This pinned Linux binary uses the libstdc++ C++11 string ABI. A filesystem
    # path starts with its string; this is diagnostic inspection, not an API.
    _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_size_t)]


def binary_string(binary, symbol):
    value = StringPrefix.in_dll(binary, symbol)
    assert value.size < 4096
    return ctypes.string_at(value.data, value.size).decode()


assert Path(deep_gemm.__file__).resolve() == PACKAGE / "__init__.py"
assert Path(deep_gemm._C.__file__).resolve().parent == PACKAGE
assert deep_gemm.__build_version__ == "2.3.0+f093a4c"
binary = ctypes.CDLL(deep_gemm._C.__file__)
library_root = binary_string(binary, "_ZN9deep_gemm8Compiler17library_root_pathB5cxx11E")
include_root = binary_string(binary, "_ZN9deep_gemm8Compiler20library_include_pathB5cxx11E")
library_version = binary_string(binary, "_ZN9deep_gemm8Compiler15library_versionB5cxx11E")
assert Path(library_root) == PACKAGE
assert Path(include_root) == PACKAGE / "include"
assert not torch.cuda.is_initialized()
original = json.loads((HERE / "original-manifest.json").read_text())
candidate = json.loads((HERE / "candidate-manifest.json").read_text())
for name, record in candidate["files"].items():
    assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == record["sha256"], name
for name, record in original["files"].items():
    assert hashlib.sha256((Path(original["package"]) / name).read_bytes()).hexdigest() == record["sha256"], name
assert candidate["files"][Path(deep_gemm._C.__file__).name] == original["files"][Path(deep_gemm._C.__file__).name]
print(json.dumps({
    "result": "PASS", "package": deep_gemm.__file__, "extension": deep_gemm._C.__file__,
    "build_version": deep_gemm.__build_version__, "binary_library_root": library_root,
    "binary_include_root": include_root, "binary_library_version": library_version,
    "original_files_unchanged": len(original["files"]),
    "candidate_files_verified": len(candidate["files"]),
    "gift_handle": _DeepGemmHandle.__module__,
    "custom_op_schema": str(torch.ops.torchtitan.k_grouped_fp8_gemm_contiguous.default._schema),
    "cuda_initialized": torch.cuda.is_initialized(),
}, indent=2))
