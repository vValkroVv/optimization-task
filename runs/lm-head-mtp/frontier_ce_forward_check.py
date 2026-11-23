"""Compare FP32 online forward reductions against Stable CE and FP64."""

import argparse
import json
from pathlib import Path

import torch
import triton

from frontier_ce_check import MTPLoss, capture, event_median, replay, stable_ce
from frontier_ce_forward import alloc_outputs, launch_cluster, launch_split
from frontier_ce import launch


def quality(logits, labels, results, scales, rows):
    row_logits = logits[rows].double()
    row_labels = labels[rows]
    oracle_ce = torch.logsumexp(row_logits, dim=-1)
    oracle_ce -= row_logits[torch.arange(len(rows), device=logits.device), row_labels.clamp_min(0)]
    oracle_ce.masked_fill_(row_labels == -100, 0)
    oracle_dx = row_logits.softmax(dim=1)
    for i, row in enumerate(rows):
        label = int(row_labels[i])
        if label == -100:
            oracle_dx[i].zero_()
        else:
            oracle_dx[i, label] -= 1
            oracle_dx[i] *= scales[int(row >= len(labels) // 2)]
    l2 = torch.linalg.vector_norm(oracle_dx)
    grads = torch.empty_like(logits)
    grad_by_mode = {}
    stats = {}
    for mode, (losses, maxima, logs) in results.items():
        launch(logits, labels, maxima.chunk(2), logs.chunk(2), scales, grads, 8192, 8)
        selected = grads[rows].double()
        grad_by_mode[mode] = selected
        stats[mode] = {
            "head_loss": [float(x.sum()) for x in losses.chunk(2)],
            "selected_fp64_max_abs_loss_error": float((losses[rows] - oracle_ce).abs().max()),
            "selected_fp64_grad_rel_l2": float(torch.linalg.vector_norm(selected - oracle_dx) / l2),
            "different_selected_dlogits": int((selected != grad_by_mode["native"]).sum()),
            "maximum_equal": torch.equal(maxima, results["native"][1]),
            "log_denom_max_delta": float((logs - results["native"][2]).abs().max()),
            "ignored_gradient_nonzero": int(torch.count_nonzero(grads[labels == -100])),
        }
    return stats


@torch.no_grad()
def main(output, try_cluster):
    torch.backends.cuda.matmul.allow_tf32 = False
    fixture = torch.load(replay.FIXTURE, map_location="cpu", weights_only=True, mmap=True)
    record = fixture["ce"]
    weight = record["weight"].cuda().to(torch.bfloat16)
    labels = record["labels"].cuda()
    positions = record["positions"]
    if positions is not None:
        positions = positions.cuda()
    all_labels = torch.cat((labels, MTPLoss._shift_labels(labels, 1, positions)))
    denominator = torch.as_tensor(record["denominator"], device="cuda", dtype=torch.float32)
    scales = tuple(weight / denominator for weight in (1.0, 0.3))
    receipts = []
    for chunks in (2, 16):
        end = labels.shape[-1] // chunks
        hidden = record["hidden"].flatten(0, 1)[:, :end].contiguous().cuda()
        logits = torch.nn.functional.linear(hidden, weight).view(-1, weight.shape[0])
        target = all_labels[:, :end].contiguous().view(-1)
        target[0] = -100
        target[1] = 0
        target[-1] = weight.shape[0] - 1
        peaks = {}
        partials_by_mode = {}
        outputs = {"native": alloc_outputs(logits)}

        def native():
            for x, y, l, m, d in zip(
                logits.chunk(2), target.chunk(2), *(output.chunk(2) for output in outputs["native"]),
                strict=True,
            ):
                stable_ce.forward_kernel[(x.shape[0],)](
                    x, y, l, m, d, x.shape[1], 4096, num_warps=4, enable_fp_fusion=False, debug=True,
                )

        outputs["native-combined"] = alloc_outputs(logits)

        def native_combined():
            stable_ce.forward_kernel[(logits.shape[0],)](
                logits, target, *outputs["native-combined"], logits.shape[1], 4096,
                num_warps=4, enable_fp_fusion=False, debug=True,
            )

        graphs = {"native": capture(native), "native-combined": capture(native_combined)}
        assert all(
            torch.equal(x, y) for x, y in zip(outputs["native"], outputs["native-combined"], strict=True)
        )
        failures = {}
        for block, warps in ((2048, 4), (4096, 4), (8192, 4), (8192, 8), (16384, 8)):
            mode = f"split-{block}-{warps}"
            outputs[mode] = alloc_outputs(logits)
            partials = tuple(
                torch.empty((logits.shape[0], triton.cdiv(logits.shape[1], block)),
                            device=logits.device, dtype=torch.float32)
                for _ in range(2)
            )
            partials_by_mode[mode] = partials
            peaks[mode] = sum(x.nbytes for x in partials)
            graphs[mode] = capture(
                lambda: launch_split(logits, target, partials, outputs[mode], block, warps)
            )
        if try_cluster:
            for ctas, warps in ((4, 8), (8, 8), (8, 16)):
                mode = f"cluster-{ctas}-{warps}"
                outputs[mode] = alloc_outputs(logits)
                try:
                    graphs[mode] = capture(
                        lambda: launch_cluster(logits, target, outputs[mode], ctas, warps)
                    )
                except Exception as error:
                    failures[mode] = str(error)
                    del outputs[mode]
        rows = list(range(0, len(target), max(len(target) // 64, 1))) + [1, len(target) - 1]
        receipt = {
            "chunks": chunks, "shape": list(logits.shape),
            "timings": event_median(graphs), "temporary_bytes": peaks,
            "quality": quality(logits, target, outputs, scales, rows), "failures": failures,
        }
        receipts.append(receipt)
        output.write_text(json.dumps(receipts, indent=2) + "\n")
        del graphs
        torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cluster", action="store_true")
    args = parser.parse_args()
    main(args.output, args.cluster)
