"""Experimental FP32 writers for the native GFSDP LM-head gradient slot."""

import ctypes

import torch

from torchtitan.distributed.gfsdp import _staged_head_dw as staging
from torchtitan.distributed.gfsdp._common import TrainingState


class CublasWriter:
    """Use the PyTorch stream's cuBLAS handle and its existing workspace."""

    def __init__(self):
        lib = ctypes.CDLL("libcublas.so.13")
        self.gemm = lib.cublasGemmEx
        int32 = ctypes.c_int
        pointer = ctypes.c_void_p
        self.gemm.argtypes = [
            pointer, int32, int32, int32, int32, int32,
            pointer, pointer, int32, int32, pointer, int32, int32,
            pointer, pointer, int32, int32, int32, int32,
        ]
        self.gemm.restype = int32
        self.pointer_mode = lib.cublasGetPointerMode_v2
        self.pointer_mode.argtypes = [pointer, ctypes.POINTER(int32)]
        self.pointer_mode.restype = int32
        self.one = ctypes.c_float(1)
        self.zero = ctypes.c_float(0)
        mode = int32()
        if self.pointer_mode(torch.cuda.current_blas_handle(), ctypes.byref(mode)) or mode.value != 0:
            raise RuntimeError("Expected a cuBLAS handle with host pointer mode")

    def __call__(self, grad, hidden, output, add):
        # C is [V,H] row-major, i.e. [H,V] column-major. Compute hidden.T @ grad.
        cublas_op_n, cublas_op_t = 0, 1
        cuda_r_16bf, cuda_r_32f = 14, 0
        cublas_compute_32f, cublas_gemm_default = 68, -1
        rows, vocab = grad.shape
        width = hidden.shape[1]
        status = self.gemm(
            torch.cuda.current_blas_handle(), cublas_op_n, cublas_op_t, width, vocab, rows,
            ctypes.byref(self.one), hidden.data_ptr(), cuda_r_16bf, width,
            grad.data_ptr(), cuda_r_16bf, vocab,
            ctypes.byref(self.one if add else self.zero), output.data_ptr(), cuda_r_32f,
            width, cublas_compute_32f, cublas_gemm_default,
        )
        if status:
            raise RuntimeError(f"cublasGemmEx returned status {status}")


class TmaWriter:
    def __init__(self):
        from quack.epilogue.library import identity_epi

        self.identity = identity_epi

    def __call__(self, grad, hidden, output, add):
        self.identity.gemm(
            grad.t(), hidden.t(), output, epi_args={}, tile_M=128, tile_N=128,
            tile_K=64, cluster_M=1, cluster_N=2, add_to_output=add,
        )


class DirectHeadDw(staging.StagedHeadDw):
    """Keep the native callback and release protocol with one FP32 output per chunk."""

    def __init__(self, lease, num_chunks, writer):
        super().__init__(lease.group, lease.param, lease.context, lease.slot)
        self.num_chunks = num_chunks
        self.writer = writer

    def check_reduced(self):
        if self.completed_callbacks != self.num_chunks or self.context.owner is self.group:
            raise RuntimeError("GFSDP did not reduce the complete direct LM-head dW")

    def write(self, grad, hidden):
        group = self.group
        index = self.writes
        if (
            group._staged_head_dw is not self
            or group._training_state != TrainingState.PRE_BACKWARD
            or index >= self.num_chunks
            or index != self.completed_callbacks
            or group.reduce_grads != (index == self.num_chunks - 1)
            or self.param.unsharded_param.grad is not None
            or self.param.unsharded_accumulated_grad is not None
        ):
            raise RuntimeError("Unexpected GFSDP callback state for direct LM-head dW")
        if (
            grad.dtype != torch.bfloat16 or hidden.dtype != torch.bfloat16
            or not grad.is_contiguous() or not hidden.is_contiguous()
            or grad.size(0) != hidden.size(0)
            or (grad.size(1), hidden.size(1)) != self.slot.shape
        ):
            raise ValueError("Direct LM-head dW requires contiguous BF16 operands matching its FP32 slot")
        stream = group.device_handle.current_stream()
        if index == 0:
            if self.context.owner is not None:
                raise RuntimeError("The direct LM-head slot already has an owner")
            self.context.wait_for_release(stream)
            self.context.owner = group
            self.stream = stream
        elif self.context.owner is not group or self.stream != stream:
            raise RuntimeError("The direct LM-head lease changed between chunks")
        self.writer(grad, hidden, self.slot, index != 0)
        self.writes += 1

    def post_backward(self, was_in_backward):
        group = self.group
        if (
            not was_in_backward
            or self.writes != self.completed_callbacks + 1
            or self.context.owner is not group
            or self.stream != group.device_handle.current_stream()
            or group.reshard_after_backward
        ):
            raise RuntimeError("GFSDP did not complete the direct LM-head dW write")
        self.completed_callbacks += 1
        if not group.reduce_grads:
            return True
        if self.writes != self.num_chunks:
            raise RuntimeError("Every direct LM-head write must precede reduction")
        self.param.unsharded_accumulated_grad = self.slot
        group._staged_head_dw = None
        return False


def make_selector(writer):
    native = staging.select_staged_head_dw

    def select(head, chunks):
        # Reuse the existing eligibility checks without its exactly-two restriction.
        # The replay only provides nonempty, same-shape consecutive sequence chunks.
        assert chunks and all(
            x.shape == chunks[0].shape and x.dtype == chunks[0].dtype
            and x.device == chunks[0].device and x.is_contiguous() for x in chunks
        )
        lease = native(head, [chunks[0], chunks[-1]])
        if lease is None:
            raise RuntimeError("Direct LM-head dW replay requires an eligible GFSDP head")
        return DirectHeadDw(lease, len(chunks), writer)

    return select
