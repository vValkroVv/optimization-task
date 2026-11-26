"""Run the pinned Liger Triton row kernel with GIFT's scaled BF16 VJP."""

import hashlib
import inspect
import linecache
import math
from dataclasses import dataclass
from functools import cache

import torch

from vendor_ops import IGNORE_INDEX, LIGER_REVISION, VendorUnavailable, _import_vendor, _source, _version


_SOURCE_SHA256 = "7149e6d6ae0f32498fb68b4974f5da79f410853c966bc9f83f805eee85d5d3f5"


@cache
def _load_kernel():
    module = _import_vendor("liger_kernel.ops.cross_entropy")
    original = _source(module)
    if original["sha256"] != _SOURCE_SHA256:
        raise VendorUnavailable(f"Expected Liger CE from {LIGER_REVISION}, got {original}")

    # Derive the benchmark kernel from the installed BSD-2-Clause source.
    # Copyright 2024 LinkedIn Corporation. The upstream LICENSE remains in its checkout.
    source = inspect.getsource(module.liger_cross_entropy_kernel.fn)
    start = source.index("            if HAS_SOFTCAPPING:\n                intermediate = tanh(")
    end = source.index("    # 5. Calculate the loss", start)
    source = source[:start] + (
        "            probability = tl.exp((X_block - m) - log_denominator)\n"
        "            gradient = probability - tl.where(X_offsets == y, 1.0, 0.0)\n"
        "            tl.store(X_ptr + X_offsets, gradient * scale, X_offsets < n_cols)\n\n"
    ) + source[end:]
    edits = (
        ("def liger_cross_entropy_kernel(\n", "def liger_early_ce_kernel(\n"),
        ("    X_ptr,\n", "    X_ptr,\n    scales_ptr,\n    ROWS_PER_HEAD: tl.constexpr,\n"),
        ("        return\n\n    loss_ptr +=", "        tl.store(loss_ptr + program_id * loss_stride, 0.0)\n"
         "        return\n\n    loss_ptr +="),
        ("    lse = m + tl.log(d)\n", "    log_denominator = tl.log(d)\n    lse = m + log_denominator\n"),
        ("    if HAS_GRADIENTS:\n        for i", "    if HAS_GRADIENTS:\n"
         "        scale = tl.load(scales_ptr + (program_id >= ROWS_PER_HEAD).to(tl.int32))\n        for i"),
        ("    loss = lse - ori_X_y\n", "    loss = log_denominator - (ori_X_y - m)\n"),
    )
    for before, after in edits:
        if source.count(before) != 1:
            raise VendorUnavailable(f"Unexpected Liger CE source near {before.strip()!r}")
        source = source.replace(before, after, 1)

    digest = hashlib.sha256(source.encode()).hexdigest()
    # Triton obtains the source with inspect.getsourcelines at decoration time.
    filename = f"{original['path']}.liger_early_{digest}"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(vars(module), __name__=__name__)
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    return namespace["liger_early_ce_kernel"], original, digest


@dataclass
class _PreparedScales:
    values: torch.Tensor
    checked: bool = False


class _LigerEarly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, scales, kernel, block, warps):
        vocab = logits.size(-1)
        rows = labels.numel()
        losses = logits.new_empty(rows, dtype=torch.float32)
        with torch.cuda.device(logits.device):
            kernel[(rows,)](
                X_ptr=logits, scales_ptr=scales.values, ROWS_PER_HEAD=rows // 2, X_stride=vocab,
                Y_ptr=labels, Y_stride=1, weight_ptr=None, loss_ptr=losses, z_loss_ptr=None,
                loss_stride=1, token_accuracy_ptr=None, token_accuracy_stride=0,
                predicted_tokens_ptr=None, predicted_tokens_stride=0, n_cols=vocab,
                n_non_ignore=0, sum_non_ignore_weight=0.0, weight_sum=0.0,
                ignore_index=IGNORE_INDEX, lse_square_scale=0.0, label_smoothing=0.0,
                reduction="sum", softcap=None, RETURN_Z_LOSS=False, RETURN_TOKEN_ACCURACY=False,
                RETURN_PREDICTED_TOKENS=False, BLOCK_SIZE=block, HAS_WEIGHT=False,
                HAS_SOFTCAPPING=False, HAS_GRADIENTS=True, num_warps=warps, enable_fp_fusion=False,
            )
        torch.autograd.graph.increment_version(logits)
        ctx.save_for_backward(logits)
        ctx.scales = scales
        return tuple(head.sum(dtype=torch.float32) for head in losses.chunk(2))

    @staticmethod
    def backward(ctx, main_scale, mtp_scale):
        if torch.is_grad_enabled():
            raise RuntimeError("Liger early-scale CE supports first-order backward only")
        (logits,) = ctx.saved_tensors
        scales = ctx.scales
        if not scales.checked:
            if not torch.equal(torch.stack((main_scale, mtp_scale)), scales.values):
                raise RuntimeError("Liger early-scale CE received a different weighted loss VJP")
            scales.checked = True
        return logits, None, None, None, None, None


def make_liger_early_ce():
    """Return FP32 head sums and consume exclusive BF16 logits for the fixed VJP.

    The caller must use `(main / denominator) * weights[0] + (mtp / denominator)
    * weights[1]` and run immediate first-order backward. Targets are validated
    by the fixture loader. As in GIFT's CUDA-13 route, the first backward for
    each cached scale checks that the caller uses the specified normalization.
    """
    kernel, original, digest = _load_kernel()
    triton = _import_vendor("triton")
    scale_key = None
    prepared = None

    def consume_if_supported(logits, labels, weights, denominator):
        nonlocal scale_key, prepared
        if (
            len(weights) != 2
            or any(not isinstance(weight, (int, float)) or not math.isfinite(weight) for weight in weights)
            or not isinstance(denominator, float)
            or not math.isfinite(denominator)
            or denominator <= 0.0
            or logits.ndim not in (2, 3)
            or not logits.numel()
            or logits.size(0) % 2
            or logits.shape[:-1] != labels.shape
            or logits.dtype != torch.bfloat16
            or labels.dtype != torch.int64
            or not logits.is_cuda
            or labels.device != logits.device
            or not logits.is_contiguous()
            or not labels.is_contiguous()
            or not torch.is_grad_enabled()
            or not logits.requires_grad
        ):
            return None
        stream = torch.cuda.current_stream(logits.device)
        key = (logits.device, stream.cuda_stream, tuple(weights), denominator)
        if scale_key != key:
            prepared = _PreparedScales(logits.new_tensor(weights, dtype=torch.float32) / denominator)
            scale_key = key
        block = min(32768, triton.next_power_of_2(logits.size(-1)))
        warps = 16 if torch.version.hip else 32
        return _LigerEarly.apply(logits, labels, prepared, kernel, block, warps)

    def ce(logits, labels, weights, denominator):
        result = consume_if_supported(logits, labels, weights, denominator)
        if result is None:
            raise ValueError("Liger early-scale CE requires two folded BF16 CUDA heads and a positive float denominator")
        return result

    ce.consume_if_supported = consume_if_supported
    ce.backend_receipt = {
        "backend": "Liger Triton CE with centered FP32 loss and early-scaled dZ",
        "api_reference_revision": LIGER_REVISION,
        "distribution_version": _version("liger-kernel"),
        "sources": [original],
        "transformed_kernel_sha256": digest,
        "dz_dtype": "bfloat16",
        "scale_contract": "FP32 per-head GIFT VJP before the in-place BF16 store",
    }
    return ce
