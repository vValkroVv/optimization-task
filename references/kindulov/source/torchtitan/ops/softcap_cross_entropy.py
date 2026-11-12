"""Sum-reduced softcap/CE without materializing FP32 logits or log-softmax."""

import math

import torch
import torch.nn.functional as F


class _SoftcapCrossEntropy(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, logits: torch.Tensor, labels: torch.Tensor, cap: float):
        from torchtitan.ops._softcap_cross_entropy import forward_kernel

        rows, vocab = logits.shape
        losses = torch.empty(rows, device=logits.device, dtype=torch.float32)
        lse = torch.empty_like(losses)
        forward_kernel[(rows,)](
            logits,
            labels,
            losses,
            lse,
            vocab,
            cap,
            4096,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
        ctx.save_for_backward(logits, labels, lse)
        ctx.cap = cap
        return losses.sum()

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, scale: torch.Tensor):
        from torchtitan.ops._softcap_cross_entropy import backward_kernel

        logits, labels, lse = ctx.saved_tensors
        if torch.is_grad_enabled():
            raise ValueError("Fused softcap CE supports first-order gradients only")
        grad = torch.empty_like(logits)
        backward_kernel[(logits.shape[0],)](
            logits,
            labels,
            lse,
            scale,
            grad,
            logits.shape[1],
            ctx.cap,
            4096,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return grad, None, None


def softcap_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, cap: float
) -> torch.Tensor:
    """Apply the existing softcap rounding contract and sum CE, ignoring -100."""
    if not math.isfinite(cap) or cap < 0.0:
        raise ValueError("Softcap must be finite and >= 0")
    if labels.dtype != torch.int64:
        raise ValueError("Cross-entropy targets must have dtype int64")
    if tuple(logits.shape[:-1]) != tuple(labels.shape):
        raise ValueError("Cross-entropy logits and targets have incompatible shapes")
    if logits.device != labels.device:
        raise ValueError("Cross-entropy logits and targets must share a device")
    flat = logits.reshape(-1, logits.shape[-1])
    targets = labels.reshape(-1)
    if (
        logits.device.type == "cuda"
        and flat.numel()
        and logits.dtype
        in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        )
    ):
        return _SoftcapCrossEntropy.apply(flat.contiguous(), targets.contiguous(), cap)
    capped = (
        logits
        if cap == 0.0
        else (cap * torch.tanh(logits.float() / cap)).to(logits.dtype)
    )
    return F.cross_entropy(
        capped.reshape(-1, logits.shape[-1]).float(),
        targets,
        reduction="sum",
        ignore_index=-100,
    )
