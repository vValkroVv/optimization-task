"""Benchmark-only cuBLASLt writer using installed CUDA 13 API."""

import ctypes as c
from dataclasses import dataclass

import torch


Ptr, Int, U64, Size = c.c_void_p, c.c_int, c.c_uint64, c.c_size_t
BF16, FP32, COMPUTE_FP32 = 14, 0, 68
TRANS_A, TRANS_B, ROW_ORDER = 3, 4, 1
MAX_WORKSPACE, REDUCTION_MASK, MIN_ALIGN_A = 1, 3, 5
FP32_SPLIT_REDUCTIONS = 1 | 2
Algo = U64 * 8


class Heuristic(c.Structure):
    _fields_ = [
        ("algo", Algo),
        ("workspace", Size),
        ("state", Int),
        ("waves", c.c_float),
        ("reserved", Int * 4),
    ]


@dataclass
class Layout:
    operation: Ptr
    a: Ptr
    b: Ptr
    cd: Ptr
    row_major: bool


class LtWriter:
    def __init__(self, rows, vocab, width):
        lib = c.CDLL("libcublasLt.so.13")

        def bind(name, *args, result=Int):
            fn = getattr(lib, "cublasLt" + name)
            fn.argtypes, fn.restype = list(args), result
            return fn

        self.destroy = bind("Destroy", Ptr)
        create = bind("Create", Ptr)
        self.version = bind("GetVersion", result=Size)()
        self.layout_create = bind("MatrixLayoutCreate", Ptr, Int, U64, U64, c.c_int64)
        self.layout_attr = bind("MatrixLayoutSetAttribute", Ptr, Int, Ptr, Size)
        self.layout_destroy = bind("MatrixLayoutDestroy", Ptr)
        self.desc_create = bind("MatmulDescCreate", Ptr, Int, Int)
        self.desc_attr = bind("MatmulDescSetAttribute", Ptr, Int, Ptr, Size)
        self.desc_destroy = bind("MatmulDescDestroy", Ptr)
        self.pref_create = bind("MatmulPreferenceCreate", Ptr)
        self.pref_attr = bind("MatmulPreferenceSetAttribute", Ptr, Int, Ptr, Size)
        self.pref_destroy = bind("MatmulPreferenceDestroy", Ptr)
        self.heuristic = bind("MatmulAlgoGetHeuristic", Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Int, Ptr, Ptr)
        self.config_attr = bind("MatmulAlgoConfigGetAttribute", Ptr, Int, Ptr, Size, Ptr)
        self.matmul = bind("Matmul", Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Size, Ptr)
        self.handle = Ptr()
        self.check(create, c.byref(self.handle))
        self.layouts = {
            "col": self.new_layout(False, rows, vocab, width),
            "row": self.new_layout(True, rows, vocab, width),
        }
        self.alpha = c.c_float(1)
        self.beta = (c.c_float(0), c.c_float(1))

    @staticmethod
    def check(fn, *args):
        status = fn(*args)
        if status:
            raise RuntimeError(f"{fn.__name__} returned {status}")

    def attribute(self, fn, desc, key, value):
        self.check(fn, desc, key, c.byref(value), c.sizeof(value))

    def matrix(self, row_major, dtype, rows, cols, ld):
        matrix = Ptr()
        self.check(self.layout_create, c.byref(matrix), dtype, rows, cols, ld)
        if row_major:
            self.attribute(self.layout_attr, matrix, ROW_ORDER, Int(1))
        return matrix

    def new_layout(self, row_major, rows, vocab, width):
        operation = Ptr()
        self.check(self.desc_create, c.byref(operation), COMPUTE_FP32, FP32)
        self.attribute(self.desc_attr, operation, TRANS_A if row_major else TRANS_B, Int(1))
        if row_major:
            a = self.matrix(True, BF16, rows, vocab, vocab)
            b = self.matrix(True, BF16, rows, width, width)
            cd = self.matrix(True, FP32, vocab, width, width)
        else:
            a = self.matrix(False, BF16, width, rows, width)
            b = self.matrix(False, BF16, vocab, rows, vocab)
            cd = self.matrix(False, FP32, width, vocab, width)
        return Layout(operation, a, b, cd, row_major)

    def candidates(self, order, cap, count=16):
        layout = self.layouts[order]
        preference = Ptr()
        self.check(self.pref_create, c.byref(preference))
        try:
            self.attribute(self.pref_attr, preference, MAX_WORKSPACE, U64(cap))
            self.attribute(self.pref_attr, preference, REDUCTION_MASK, c.c_uint(FP32_SPLIT_REDUCTIONS))
            # All fixture operands and the active GFSDP slot must be aligned accordingly.
            for attr in range(MIN_ALIGN_A, MIN_ALIGN_A + 4):
                self.attribute(self.pref_attr, preference, attr, c.c_uint(256))
            results, returned = (Heuristic * count)(), Int()
            self.check(
                self.heuristic, self.handle, layout.operation, layout.a, layout.b, layout.cd,
                layout.cd, preference, count, results, c.byref(returned),
            )
            return [self.describe(order, entry) for entry in results[:returned.value] if entry.state == 0]
        finally:
            self.check(self.pref_destroy, preference)

    def describe(self, order, entry):
        config = {}
        names = ("id", "tile", "split_k", "reduction", "swizzle", "custom", "stages", "inner", "cluster")
        for attr, name in enumerate(names):
            value = c.c_uint16() if attr >= 7 else c.c_uint32()
            self.check(self.config_attr, c.byref(entry.algo), attr, c.byref(value), c.sizeof(value), None)
            config[name] = value.value
        return {
            "order": order, "algo": bytes(entry.algo).hex(), "workspace": entry.workspace,
            "waves": entry.waves, "config": config,
        }

    def prepare(self, config, scratch):
        algo = Algo.from_buffer_copy(bytes.fromhex(config["algo"]))
        return self.layouts[config["order"]], algo, scratch, config["workspace"]

    def apply(self, plan, grad, hidden, output, add):
        layout, algo, scratch, scratch_bytes = plan
        a, b = (grad, hidden) if layout.row_major else (hidden, grad)
        a_ptr, b_ptr, out_ptr = a.data_ptr(), b.data_ptr(), output.data_ptr()
        assert (a_ptr | b_ptr | out_ptr) % 256 == 0
        scratch_ptr = scratch.data_ptr() if scratch_bytes else None
        self.check(
            self.matmul, self.handle, layout.operation, c.byref(self.alpha),
            a_ptr, layout.a, b_ptr, layout.b, c.byref(self.beta[add]),
            out_ptr, layout.cd, out_ptr, layout.cd, c.byref(algo), scratch_ptr,
            scratch_bytes, torch.cuda.current_stream().cuda_stream,
        )

    def close(self):
        for layout in self.layouts.values():
            for matrix in (layout.a, layout.b, layout.cd):
                self.check(self.layout_destroy, matrix)
            self.check(self.desc_destroy, layout.operation)
        self.check(self.destroy, self.handle)


def scratch_for(*configs):
    nbytes = max(config["workspace"] for config in configs)
    return torch.empty(nbytes, dtype=torch.uint8, device="cuda") if nbytes else None


class PlannedWriter:
    """Reuse a selected algorithm pair with one requested workspace allocation."""

    def __init__(self, tuning, pair):
        self.configs = [tuning["configurations"][int(index)] for index in pair.split(".")]
        self.lt = LtWriter(tuning["rows_per_chunk"], tuning["vocab"], tuning["width"])
        if self.lt.version != tuning["cublasLt_version"]:
            raise RuntimeError("Tuned cuBLASLt algorithm requires the original library version")
        scratch = scratch_for(*self.configs)
        self.plans = [self.lt.prepare(config, scratch) for config in self.configs]

    def __call__(self, grad, hidden, output, add):
        self.lt.apply(self.plans[add], grad, hidden, output, add)

    def close(self):
        self.lt.close()
