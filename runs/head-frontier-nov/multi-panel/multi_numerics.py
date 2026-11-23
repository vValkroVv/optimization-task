"""Compare complete isolated-loop outputs and use a full-shape FP64 oracle."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

import multi_replay as panel_replay
import replay

sys.path.insert(0, str(replay.CURRENT / "gift"))

from torchtitan.components.loss import MTPLoss
from torchtitan.experiments.four_fusion import stable_ce


def compare(actual, ref):
    assert actual.shape == ref.shape
    error_sq = torch.zeros((), dtype=torch.float64, device="cuda")
    ref_sq = torch.zeros_like(error_sq)
    max_abs = torch.zeros_like(error_sq)
    unequal = torch.zeros((), dtype=torch.int64, device="cuda")
    nonfinite = torch.zeros_like(unequal)
    for a, b in zip(
        actual.reshape(-1).split(1 << 22), ref.reshape(-1).split(1 << 22), strict=True,
    ):
        a, b = a.cuda().double(), b.cuda().double()
        difference = a - b
        error_sq += difference.square().sum()
        ref_sq += b.square().sum()
        max_abs = torch.maximum(max_abs, difference.abs().amax())
        unequal += (a != b).sum()
        nonfinite += (~a.isfinite() | ~b.isfinite()).sum()
    relative = (error_sq / ref_sq).sqrt() if ref_sq.item() else error_sq.sqrt()
    return {
        "relative_l2": relative.item(), "max_abs": max_abs.item(),
        "different_elements": unequal.item(), "nonfinite": nonfinite.item(),
        "elements": actual.numel(),
    }


def digest(tensor):
    buf = tensor.reshape(-1).contiguous().view(torch.uint8).numpy()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(buf).hexdigest()}


def panel_operand_checks(hidden, weight, labels, logits, grad, split, scales):
    rows, vocab = logits.shape
    workspace = hidden.new_empty(rows * split)
    maximum, log_den, chosen, losses = [
        torch.empty(rows, dtype=torch.float32, device=hidden.device) for _ in range(4)
    ]
    projection, gradient = {}, {}
    for start, stop in panel_replay.ranges(split, vocab):
        tile = panel_replay.project(hidden, weight[start:stop], workspace)
        projection[str(start)] = compare(tile, logits[:, start:stop].contiguous())
        panel_replay.panel_stats[(rows,)](
            tile, labels, maximum, log_den, chosen, losses,
            start, stop - start, vocab, stop == vocab, 4096,
            num_warps=4, enable_fp_fusion=False, debug=True,
        )
    dx = torch.empty_like(hidden, dtype=torch.float32)
    for start, stop in reversed(panel_replay.ranges(split, vocab)):
        columns = stop - start
        tile = (
            workspace[:rows * columns].view(rows, columns)
            if stop == vocab else panel_replay.project(hidden, weight[start:stop], workspace)
        )
        panel_replay.panel_grad[(rows, 1)](
            tile, labels, maximum, log_den, *scales, start, columns, rows // 2,
            4096, (columns + 4095) // 4096, num_warps=4, enable_fp_fusion=False,
        )
        gradient[str(start)] = compare(tile, grad[:, start:stop].contiguous())
        if stop == vocab:
            torch.mm(tile, weight[start:stop], out_dtype=torch.float32, out=dx)
        else:
            torch.addmm(dx, tile, weight[start:stop], out_dtype=torch.float32, out=dx)
    return {"projection": projection, "bf16_dlogits": gradient}, dx


def fp64_loss(logits, labels, scale):
    values = logits.double()
    target = labels.clamp_min(0).view(-1, 1)
    chosen = values.gather(1, target).flatten()
    maximum = values.amax(dim=1)
    values.sub_(maximum[:, None]).exp_()
    loss = ((values.sum(dim=1).log() - (chosen - maximum)).masked_fill(labels == -100, 0))
    return loss.sum() * scale.double()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=int, choices=(2, 8))
    parser.add_argument("output", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidates", nargs="+", help="split:path pairs")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    canonical = torch.load(args.control, mmap=True, weights_only=True)
    candidates = {}
    for spec in args.candidates:
        split, path = spec.split(":", 1)
        candidates[int(split)] = torch.load(path, mmap=True, weights_only=True)

    capture = torch.load(replay.FIXTURE, mmap=True, weights_only=True)
    record = capture["ce"]
    hidden = record["hidden"].cuda()
    weight = record["weight"].cuda().bfloat16()
    labels = record["labels"].cuda()
    positions = record["positions"]
    if positions is not None:
        positions = positions.cuda()
    denominator = record["denominator"]
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.cuda()
    weights = record["head_weights"]
    targets = torch.cat((labels, MTPLoss._shift_labels(labels, 1, positions)))
    assert weights == [1.0, 0.3] and record["ignore_index"] == -100

    dw_ref = torch.zeros_like(weight, dtype=torch.float64)
    dx_ref = torch.empty_like(hidden.flatten(0, 1), dtype=torch.float64)
    dx_fp32 = {split: torch.empty_like(dx_ref, dtype=torch.float32) for split in candidates}
    weight64 = weight.double()
    loss_ref = torch.zeros((), dtype=torch.float64, device="cuda")
    per_chunk = []
    sequence = 0
    for h, y in zip(hidden.flatten(0, 1).chunk(args.chunks, dim=1), targets.chunk(args.chunks, dim=1), strict=True):
        h, y = h.contiguous(), y.contiguous()
        flat_h, flat_y = h.flatten(0, 1), y.flatten()
        logits = torch.mm(flat_h, weight.T).requires_grad_()
        main, mtp = stable_ce.folded_cross_entropy(logits.view(*y.shape, weight.size(0)), y, 2)
        loss = (main / denominator) * weights[0] + (mtp / denominator) * weights[1]
        scales = torch.autograd.grad(loss, (main, mtp), retain_graph=True)
        (grad,) = torch.autograd.grad(loss, logits)
        gd = grad.double()
        nxt = sequence + h.shape[1]
        with torch.no_grad():
            dx_ref[:, sequence:nxt] = (gd @ weight64).view_as(h)
            dw_ref.addmm_(gd.T, flat_h.double())
            for logit_half, y_half, scale in zip(logits.chunk(2), flat_y.chunk(2), scales, strict=True):
                loss_ref += fp64_loss(logit_half, y_half, scale)
            item = {
                "rows": flat_y.numel(),
                "valid_main": (flat_y.chunk(2)[0] != -100).sum().item(),
                "valid_mtp": (flat_y.chunk(2)[1] != -100).sum().item(),
                "head_scales": [scale.item() for scale in scales],
                "split_checks": {},
            }
            for split in candidates:
                values, dx = panel_operand_checks(flat_h, weight, flat_y, logits, grad, split, scales)
                dx_fp32[split][:, sequence:nxt] = dx.view_as(h)
                item["split_checks"][split] = values
            per_chunk.append(item)
        sequence = nxt
    assert sequence == hidden.shape[2]
    dx_ref = dx_ref.view_as(hidden)
    rounded_dx = dx_ref.bfloat16()

    def receipt(actual):
        return {
            "loss_fp64": compare(actual["loss"], loss_ref),
            "dx_fp64": compare(actual["dx"], dx_ref),
            "dx_rounded_fp64": compare(actual["dx"], rounded_dx),
            "dw_fp64": compare(actual["dw"], dw_ref),
            "digests": {name: digest(tensor) for name, tensor in actual.items()},
        }

    result = {
        "chunks": args.chunks,
        "fixture": str(replay.FIXTURE),
        "oracle": "full BF16 control projection; FP64 centered CE on these logits; FP64 GEMMs of canonical scale-weighted BF16 dLogits, BF16 gathered weight and BF16 hidden",
        "denominator": float(denominator),
        "weights": weights,
        "label_range": [labels[labels >= 0].amin().item(), labels.amax().item()],
        "unique_main_labels": labels[labels >= 0].unique().numel(),
        "fp64_loss": loss_ref.item(),
        "per_chunk": per_chunk,
        "control": receipt(canonical),
        "candidates": {},
    }
    for split, actual in candidates.items():
        row = receipt(actual)
        row["vs_control"] = {name: compare(actual[name], canonical[name]) for name in actual}
        row["standalone_fp32_dx_vs_fp64"] = compare(dx_fp32[split].view_as(hidden), dx_ref)
        row["standalone_bf16_dx_vs_integrated"] = compare(
            dx_fp32[split].view_as(hidden).bfloat16(), actual["dx"],
        )
        result["candidates"][split] = row
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
