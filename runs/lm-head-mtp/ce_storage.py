"""Measure native centered CE with separate versus consumed logits storage."""

import torch

from torchtitan.experiments.four_fusion import stable_ce


class StorageControl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, num_heads, forward, reuse_logits):
        parts = [
            forward(logit_part, label_part)
            for logit_part, label_part in zip(logits.chunk(num_heads), labels.chunk(num_heads), strict=True)
        ]
        losses, maxima, log_denominators = zip(*parts, strict=True)
        ctx.save_for_backward(logits, labels, *maxima, *log_denominators)
        ctx.reuse_logits = reuse_logits
        return losses

    @staticmethod
    def backward(ctx, *scales):
        if torch.is_grad_enabled():
            raise RuntimeError("The isolated storage experiment supports first-order gradients only")
        logits, labels, *stats = ctx.saved_tensors
        num_heads = len(scales)
        # The owning loss immediately runs backward. Linear saved hidden and
        # weight only, and no acceptance monitor or other consumer reads logits.
        output = logits if ctx.reuse_logits else torch.empty_like(logits)
        for parts in zip(
            logits.chunk(num_heads), labels.chunk(num_heads), stats[:num_heads],
            stats[num_heads:], scales, output.chunk(num_heads), strict=True,
        ):
            stable_ce._launch_backward(*parts)
        return output, None, None, None, None


def make(backend, reuse_logits):
    forward = torch.compile(stable_ce._forward, backend=backend, fullgraph=True)

    def storage(logits, labels, num_heads):
        assert stable_ce.supports_folded_cross_entropy(logits, labels)
        flat = logits.view(-1, logits.shape[-1])
        return StorageControl.apply(flat, labels.view(-1), num_heads, forward, reuse_logits)

    return storage
