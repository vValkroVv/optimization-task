"""Compare complete isolated gradients against a streamed FP32 reference."""

import argparse
import json
from pathlib import Path
import sys

import torch

from replay import CURRENT, FIXTURE

sys.path.insert(0, str(CURRENT / "gift"))

from dw_check import error
from torchtitan.components.loss import MTPLoss, cross_entropy_loss
from torchtitan.experiments.four_fusion import stable_ce
from ce_storage import make


def oracle(record):
    x = record["hidden"].cuda().float().requires_grad_()
    weight = record["weight"].cuda().float().requires_grad_()
    targets = record["labels"].cuda()
    positions = record["positions"].cuda() if record["positions"] is not None else None
    denominator = record["denominator"]
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.cuda()
    total = torch.zeros((), device="cuda")
    y_mtp = MTPLoss._shift_labels(targets, 1, positions)
    for head, labels, scale in zip(x, (targets, y_mtp), (1.0, 0.3), strict=True):
        for h, y in zip(head.flatten(0, 1).split(256), labels.flatten().split(256), strict=True):
            logits = torch.nn.functional.linear(h, weight)
            value = torch.nn.functional.cross_entropy(logits, y, ignore_index=-100, reduction="sum")
            value = value / denominator * scale
            value.backward(retain_graph=True)
            total += value.detach()
    return {"loss": total, "dx": x.grad, "dw": weight.grad}


def ce_metrics(record, chunks):
    hidden = record["hidden"].cuda().flatten(0, 1)
    weight = record["weight"].cuda()
    labels = record["labels"].cuda()
    positions = record["positions"].cuda() if record["positions"] is not None else None
    labels = torch.cat((labels, MTPLoss._shift_labels(labels, 1, positions)))
    denominator = record["denominator"]
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.cuda()
    hidden_tiles = hidden.chunk(chunks, dim=1)
    target_tiles = labels.chunk(chunks, dim=1)
    compiled_base = torch.compile(cross_entropy_loss, backend="inductor")
    compiled_stable = torch.compile(stable_ce.cross_entropy_loss, backend="inductor")
    routes = {
        "torch_ce": lambda x, y: tuple(compiled_base(a, b) for a, b in zip(x.chunk(2), y.chunk(2))),
        "stable_ce": lambda x, y: tuple(compiled_stable(a, b) for a, b in zip(x.chunk(2), y.chunk(2))),
        "folded_ce": lambda x, y: stable_ce.folded_cross_entropy(x, y, 2),
        "separate_storage": lambda x, y: make("inductor", False)(x, y, 2),
        "in_place": lambda x, y: make("inductor", True)(x, y, 2),
    }
    records = []
    for h, y in zip(hidden_tiles, target_tiles, strict=True):
        y = y.contiguous()
        logits = torch.nn.functional.linear(h.contiguous(), weight)
        v = logits.shape[-1]
        target = y.flatten()
        reference = torch.empty((target.numel(), v), dtype=torch.float64, device="cuda")
        for i in range(0, target.numel(), 128):
            stop = min(i + 128, target.numel())
            fp_logits = logits.view(-1, v)[i:stop].double().requires_grad_()
            value = torch.nn.functional.cross_entropy(fp_logits, target[i:stop], ignore_index=-100, reduction="sum")
            (a,) = torch.autograd.grad((value / denominator) * (1.0 if i < target.numel() // 2 else 0.3), fp_logits)
            reference[i:stop].copy_(a)

        ref_bf16 = None
        row = {}
        for name, call in routes.items():
            operand = logits.clone().requires_grad_()
            main, mtp = call(operand, y)
            value = main / denominator + (mtp / denominator) * 0.3
            (grad,) = torch.autograd.grad(value, operand)
            flat = grad.view(-1, v)
            if name == "stable_ce":
                ref_bf16 = flat
            row[name] = {
                **error(flat, reference),
                "loss": float(value.detach()),
                "equal_to_stable_bf16": torch.equal(flat, ref_bf16) if ref_bf16 is not None else None,
                "ignored_nonzero": int(torch.count_nonzero(flat[target == -100])),
            }
            del grad, flat, operand
        records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("gradients", nargs="*", type=Path)
    ce = parser.add_mutually_exclusive_group()
    ce.add_argument("--ce", action="store_true")
    ce.add_argument("--ce-chunks", type=int, nargs="+", choices=(2, 4, 8))
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._inductor.config.triton.cudagraphs = False
    record = torch.load(FIXTURE, map_location="cpu", weights_only=True, mmap=True)["ce"]
    output = {}
    if args.ce:
        output["dlogits_against_float64_native_logits"] = ce_metrics(record, 2)
    if args.ce_chunks:
        matrix = {}
        for count in args.ce_chunks:
            results = ce_metrics(record, count)
            assert len(results) == count
            for row in results:
                for name in ("folded_ce", "separate_storage", "in_place"):
                    result = row[name]
                    assert result["equal_to_stable_bf16"]
                    assert result["ignored_nonzero"] == result["nonfinite"] == 0
            matrix[count] = results
        output["ce_storage_by_loss_chunks"] = matrix
    if args.gradients:
        expected = oracle(record)
        rows = {}
        for path in args.gradients:
            actual = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
            rows[path.stem] = {
                "loss_abs_error": float(abs(actual["loss"].item() - expected["loss"].item())),
                "dx": error(actual["dx"].cuda().reshape(-1, 4096), expected["dx"].reshape(-1, 4096)),
                "dw": error(actual["dw"].cuda(), expected["dw"]),
            }
            del actual
        output["against_float32_operands"] = rows
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
