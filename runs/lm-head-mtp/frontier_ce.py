"""Benchmark-only tiled Stable CE backward and native replay integration."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def tiled_backward(
    X, Y, MAX0, MAX1, LOG0, LOG1, SCALE0, SCALE1, DX,
    VOCAB: tl.constexpr, HEAD_ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    target = tl.load(Y + row)
    head1 = row >= HEAD_ROWS
    head_row = row % HEAD_ROWS
    row_base = row.to(tl.int64) * VOCAB
    if target == -100:
        tl.store(DX + row_base + cols, 0.0, cols < VOCAB)
    else:
        maximum = tl.load(tl.where(head1, MAX1, MAX0) + head_row)
        log_denominator = tl.load(tl.where(head1, LOG1, LOG0) + head_row)
        scale = tl.load(tl.where(head1, SCALE1, SCALE0)).to(tl.float32)
        values = tl.load(X + row_base + cols, cols < VOCAB, other=0.0).to(tl.float32)
        grad = (tl.exp((values - maximum) - log_denominator)
                - tl.where(cols == target, 1.0, 0.0)) * scale
        tl.store(DX + row_base + cols, grad, cols < VOCAB)


def launch(logits, labels, maxima, logs, scales, grad, block, warps):
    rows, vocab = logits.shape
    tiled_backward[(rows, triton.cdiv(vocab, block))](
        logits, labels, *maxima, *logs, *scales, grad,
        vocab, rows // 2, block, num_warps=warps, enable_fp_fusion=False,
    )


def install(block, warps):
    from torchtitan.experiments.four_fusion import stable_ce

    native_backward = stable_ce._folded_backward

    @torch.library.custom_op("frontier_ce::tiled_backward", mutates_args=())
    def backward(
        logits: torch.Tensor,
        labels: torch.Tensor,
        maxima: list[torch.Tensor],
        log_denominators: list[torch.Tensor],
        scales: list[torch.Tensor],
    ) -> torch.Tensor:
        output = torch.empty_like(logits)
        launch(logits, labels, maxima, log_denominators, scales, output, block, warps)
        return output

    @backward.register_fake
    def fake(logits, labels, maxima, log_denominators, scales):
        return torch.empty_like(logits)

    def select(logits, labels, maxima, log_denominators, scales):
        if logits.dtype != torch.bfloat16 or logits.shape[1] != 128256 or len(scales) != 2:
            return native_backward(logits, labels, maxima, log_denominators, scales)
        return backward(logits, labels, list(maxima), list(log_denominators), list(scales))

    stable_ce._folded_backward = select


def install_forward():
    from torchtitan.experiments.four_fusion import stable_ce

    native_forward = stable_ce._FoldedCrossEntropy.forward

    @torch.library.custom_op("frontier_ce::folded_forward", mutates_args=())
    def forward(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, vocab = logits.shape
        losses = logits.new_empty((rows,), dtype=torch.float32)
        maximum = torch.empty_like(losses)
        log_denominator = torch.empty_like(losses)
        stable_ce.forward_kernel[(rows,)](
            logits, labels, losses, maximum, log_denominator, vocab,
            4096, num_warps=4, enable_fp_fusion=False, debug=True,
        )
        loss0, loss1 = losses.chunk(2)
        return loss0.sum(), loss1.sum(), maximum, log_denominator

    @forward.register_fake
    def fake(logits, labels):
        return (
            logits.new_empty((), dtype=torch.float32),
            logits.new_empty((), dtype=torch.float32),
            logits.new_empty((logits.shape[0],), dtype=torch.float32),
            logits.new_empty((logits.shape[0],), dtype=torch.float32),
        )

    def select(ctx, logits, labels, num_heads):
        if logits.dtype != torch.bfloat16 or logits.shape[1] != 128256 or num_heads != 2:
            return native_forward(ctx, logits, labels, num_heads)
        loss0, loss1, maximum, log_denom = forward(logits, labels)
        ctx.save_for_backward(logits, labels, *maximum.chunk(2), *log_denom.chunk(2))
        return loss0, loss1

    stable_ce._FoldedCrossEntropy.forward = staticmethod(select)


def save_digest(values, output):
    import hashlib
    import json
    from pathlib import Path

    receipt = {}
    for name, value in values.items():
        assert value.device.type == "cpu"
        receipt[name] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest(),
        }
    Path(output).write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    import sys
    import replay

    mode = os.environ.get("FRONTIER_CE_MODE", "native")
    if mode in ("tiled", "consolidated"):
        sys.path.insert(0, str(replay.CURRENT / "gift"))
        block = int(os.environ.get("FRONTIER_CE_BLOCK", 8192))
        warps = int(os.environ.get("FRONTIER_CE_WARPS", 4))
        install(block, warps)
        if mode == "consolidated":
            install_forward()
        print(f"frontier CE {mode} installed with {block=} {warps=}", flush=True)
    elif mode != "native":
        raise ValueError(mode)
    if "--dump" in sys.argv:
        torch.save = save_digest
    replay.main()


if __name__ == "__main__":
    main()
