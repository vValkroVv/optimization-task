"""Launchers for centered CE that replaces each input logit with its BF16 gradient."""

import ctypes
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def fused_cluster(logits, target, loss, scales,
                  ROWS_PER_HEAD: tl.constexpr, VOCAB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    row_base = row.to(tl.int64) * VOCAB
    col = tl.arange(0, BLOCK)
    label = tl.load(target + row)
    if label == -100:
        tl.store(loss + row, 0.0)
        tl.store(logits + row_base + col, 0.0, col < VOCAB)
    else:
        tl.device_assert((label >= 0) & (label < VOCAB), "CE label outside vocabulary")
        values = tl.load(logits + row_base + col, col < VOCAB, other=float("-inf")).to(tl.float32)
        maximum = tl.max(values, 0)
        centered = values - maximum
        log_denominator = tl.log(tl.sum(tl.exp(centered), 0))
        chosen = tl.sum(tl.where(col == label, centered, 0.0), 0)
        tl.store(loss + row, -(chosen - log_denominator))
        scale = tl.load(scales + (row >= ROWS_PER_HEAD))
        grad = (tl.exp(centered - log_denominator) - (col == label).to(tl.float32)) * scale
        tl.store(logits + row_base + col, grad, col < VOCAB)


class Launcher:
    def __init__(self, name):
        self.name = name
        self.calls = 0
        if name.startswith("cuda-"):
            binary = os.environ.get("FUSED_CE_BINARY", "centered.so")
            lib = ctypes.CDLL(str(Path(__file__).with_name(binary)))
            fun = lib.centered_ce_launch
            pointer = ctypes.c_void_p
            fun.argtypes = [pointer, pointer, pointer, pointer, ctypes.c_int, pointer, ctypes.c_int]
            fun.restype = ctypes.c_int
            self.fun = fun

    def __call__(self, logits, target, losses, scales):
        rows, vocab = logits.shape
        assert vocab == 128256 and rows % 2 == 0
        assert logits.dtype == torch.bfloat16 and target.dtype == torch.int64
        assert scales.dtype == losses.dtype == torch.float32 and scales.shape == (2,)
        assert all(x.is_contiguous() for x in (logits, target, losses, scales))
        if self.name.startswith("cuda-"):
            status = self.fun(
                logits.data_ptr(), target.data_ptr(), losses.data_ptr(), scales.data_ptr(),
                rows, torch.cuda.current_stream().cuda_stream, int(self.name.split("-")[1]),
            )
            if status:
                raise RuntimeError(f"CUDA kernel launch failed with {status}")
        else:
            _, ctas, warps = self.name.split("-")
            fused_cluster[(rows,)](
                logits, target, losses, scales, rows // 2, vocab, triton.next_power_of_2(vocab),
                num_ctas=int(ctas), num_warps=int(warps), enable_fp_fusion=False, debug=True,
            )
        self.calls += 1


def incoming_scales(global_valid_tokens, weights, reference):
    """Reproduce the raw-loss VJP of (loss / tokens) * weight."""
    scale = reference.new_tensor(weights, dtype=torch.float32)
    if global_valid_tokens is not None:
        scale = scale / global_valid_tokens
    return scale
