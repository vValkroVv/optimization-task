"""CUDA writers for the experimental FP32 LM-head reduce-scatter input."""

import ctypes as c
import weakref
from contextlib import ExitStack

import torch


class _CublasGemmEx:
    name = "cublasGemmEx"

    def __init__(self) -> None:
        lib = c.CDLL("libcublas.so.13")
        self.gemm = lib.cublasGemmEx
        self.gemm.argtypes = [
            _Ptr,
            _Int,
            _Int,
            _Int,
            _Int,
            _Int,
            _Ptr,
            _Ptr,
            _Int,
            _Int,
            _Ptr,
            _Int,
            _Int,
            _Ptr,
            _Ptr,
            _Int,
            _Int,
            _Int,
            _Int,
        ]
        self.gemm.restype = _Int
        self.pointer_mode = lib.cublasGetPointerMode_v2
        self.pointer_mode.argtypes = [_Ptr, c.POINTER(_Int)]
        self.pointer_mode.restype = _Int
        self.checked_handles: set[int] = set()

    def __call__(self, grad, hidden, slot, add, stream) -> None:
        # The row-major [V,H] destination is [H,V] to column-major cuBLAS.
        rows, vocab = grad.shape
        width = hidden.shape[1]
        handle = torch.cuda.current_blas_handle()
        if handle not in self.checked_handles:
            mode = _Int()
            _check(self.pointer_mode, handle, c.byref(mode))
            if mode.value != 0:
                raise RuntimeError(
                    "Direct LM-head dW requires host cuBLAS pointer mode"
                )
            self.checked_handles.add(handle)
        _check(
            self.gemm,
            handle,
            _OP_N,
            _OP_T,
            width,
            vocab,
            rows,
            c.byref(_ONE),
            hidden.data_ptr(),
            _BF16,
            width,
            grad.data_ptr(),
            _BF16,
            vocab,
            c.byref(_BETA[add]),
            slot.data_ptr(),
            _FP32,
            width,
            _COMPUTE_32F,
            _GEMM_DEFAULT,
        )


_Ptr, _Int, _U64, _Size = c.c_void_p, c.c_int, c.c_uint64, c.c_size_t
_OP_N, _OP_T = 0, 1
_BF16, _FP32, _COMPUTE_32F, _GEMM_DEFAULT = 14, 0, 68, -1
_TRANS_A, _TRANS_B, _LAYOUT_ORDER = 3, 4, 1
_ROW_MAJOR = 1
_ONE = c.c_float(1)
_BETA = (c.c_float(0), _ONE)
_Algo = _U64 * 8
_CUBLAS_LT_VERSION = 130101

# Selected beta=0/beta=1 entries in frontier-dw/tune-{2,8}.json.
# These are the serialized cuBLASLt 13.1.1 H100 algorithms, with 4 B workspace.
_LT_ALGOS = {
    2: (
        "9.14",
        True,
        (
            "4200000018000000230000000100000000000000000000000700000000000500"
            "0100000035fc01000e0000000e00000000000000440000000000000000000000",
            "42000000ef010000230000000100000000000000000000000700000000000500"
            "0100000035fc01000e0000000e00000000000000440000000000000000000000",
        ),
    ),
    8: (
        "5.5",
        False,
        (
            "420000001d000000230000000100000000000000000000000100000000000300"
            "0100000035fc01000e0000000e00000000000000440000000000000000000000",
        )
        * 2,
    ),
}


def _check(fn, *args) -> None:
    if status := fn(*args):
        raise RuntimeError(f"{fn.__name__} returned CUDA BLAS status {status}")


class _CublasLtMatmul:
    name = "cublasLtMatmul"

    def __init__(
        self, chunks: int, rows: int, vocab: int, width: int, stream: torch.Stream
    ) -> None:
        lib = c.CDLL("libcublasLt.so.13")

        def bind(name, *args, result=_Int):
            fn = getattr(lib, "cublasLt" + name)
            fn.argtypes, fn.restype = list(args), result
            return fn

        self.version = bind("GetVersion", result=_Size)()
        if self.version != _CUBLAS_LT_VERSION:
            raise RuntimeError(
                "Direct LM-head dW requires the tuned cuBLASLt 13.1.1 build "
                f"({_CUBLAS_LT_VERSION}), got {self.version}"
            )

        self.pair, self.row_major, serialized = _LT_ALGOS[chunks]
        self.algorithms = tuple(
            _Algo.from_buffer_copy(bytes.fromhex(x)) for x in serialized
        )
        self.stream = stream
        create = bind("Create", _Ptr)
        destroy = bind("Destroy", _Ptr)
        desc_create = bind("MatmulDescCreate", _Ptr, _Int, _Int)
        desc_destroy = bind("MatmulDescDestroy", _Ptr)
        desc_attr = bind("MatmulDescSetAttribute", _Ptr, _Int, _Ptr, _Size)
        layout_create = bind("MatrixLayoutCreate", _Ptr, _Int, _U64, _U64, c.c_int64)
        layout_destroy = bind("MatrixLayoutDestroy", _Ptr)
        layout_attr = bind("MatrixLayoutSetAttribute", _Ptr, _Int, _Ptr, _Size)
        self.matmul = bind(
            "Matmul",
            _Ptr,
            _Ptr,
            _Ptr,
            _Ptr,
            _Ptr,
            _Ptr,
            _Ptr,
            _Ptr,
            _Ptr,
            _Ptr,
            _Ptr,
            _Ptr,
            _Ptr,
            _Ptr,
            _Size,
            _Ptr,
        )
        with torch.cuda.stream(stream), ExitStack() as resources:
            self.handle = _Ptr()
            _check(create, c.byref(self.handle))
            resources.callback(_check, destroy, self.handle)
            self.operation = _Ptr()
            _check(desc_create, c.byref(self.operation), _COMPUTE_32F, _FP32)
            resources.callback(_check, desc_destroy, self.operation)
            transpose = _Int(_OP_T)
            _check(
                desc_attr,
                self.operation,
                _TRANS_A if self.row_major else _TRANS_B,
                c.byref(transpose),
                c.sizeof(transpose),
            )

            def matrix(dtype, m, n, ld):
                layout = _Ptr()
                _check(layout_create, c.byref(layout), dtype, m, n, ld)
                resources.callback(_check, layout_destroy, layout)
                if self.row_major:
                    order = _Int(_ROW_MAJOR)
                    _check(
                        layout_attr,
                        layout,
                        _LAYOUT_ORDER,
                        c.byref(order),
                        c.sizeof(order),
                    )
                return layout

            if self.row_major:
                self.a = matrix(_BF16, rows, vocab, vocab)
                self.b = matrix(_BF16, rows, width, width)
                self.cd = matrix(_FP32, vocab, width, width)
            else:
                self.a = matrix(_BF16, width, rows, width)
                self.b = matrix(_BF16, vocab, rows, vocab)
                self.cd = matrix(_FP32, width, vocab, width)
            self.workspace = torch.empty(4, dtype=torch.uint8, device=stream.device)
            cleanup = resources.pop_all()

        def close():
            with torch.cuda.device(stream.device):
                cleanup.close()

        weakref.finalize(self, close)

    def __call__(self, grad, hidden, slot, add, stream) -> None:
        if stream != self.stream:
            raise RuntimeError(
                "Direct cuBLASLt LM-head dW changed its workspace stream"
            )
        a, b = (grad, hidden) if self.row_major else (hidden, grad)
        a_ptr, b_ptr, out_ptr = a.data_ptr(), b.data_ptr(), slot.data_ptr()
        if (a_ptr | b_ptr | out_ptr) % 256:
            raise RuntimeError(
                "Tuned cuBLASLt LM-head dW requires 256-byte aligned operands"
            )
        _check(
            self.matmul,
            self.handle,
            self.operation,
            c.byref(_ONE),
            a_ptr,
            self.a,
            b_ptr,
            self.b,
            c.byref(_BETA[add]),
            out_ptr,
            self.cd,
            out_ptr,
            self.cd,
            c.byref(self.algorithms[add]),
            self.workspace.data_ptr(),
            self.workspace.numel(),
            stream.cuda_stream,
        )


HeadDwWriter = _CublasGemmEx | _CublasLtMatmul


def make_head_dw_writer(
    mode: str, chunks: int, rows: int, vocab: int, width: int, stream: torch.Stream
) -> HeadDwWriter:
    if mode == "previous":
        return _CublasGemmEx()
    return _CublasLtMatmul(chunks, rows, vocab, width, stream)
