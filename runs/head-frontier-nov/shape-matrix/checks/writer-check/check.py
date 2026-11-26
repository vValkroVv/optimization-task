import json
import logging
import math
import os
from pathlib import Path

import torch

from torchtitan.distributed.gfsdp._staged_head_dw_writer import make_head_dw_writer


def error(slot, oracle):
    squares = torch.zeros((), dtype=torch.float64, device=slot.device)
    largest = torch.zeros_like(squares)
    for actual, expected in zip(slot.split(2048), oracle.split(2048), strict=True):
        diff = actual.double() - expected
        squares += diff.square().sum()
        largest = torch.maximum(largest, diff.abs().amax())
    return {
        "relative_l2": (squares / oracle.square().sum()).sqrt().item(),
        "max_abs": largest.item(),
    }


def check_case(name, row_counts, vocab, width, historical_chunks):
    stream = torch.cuda.Stream()
    modes = ["generic", "previous"]
    if historical_chunks is not None:
        modes.append("new")
    results = {}
    with torch.cuda.stream(stream):
        operands = [
            (
                torch.randn((rows, vocab), device=stream.device, dtype=torch.bfloat16),
                torch.randn((rows, width), device=stream.device, dtype=torch.bfloat16),
            )
            for rows in row_counts
        ]
        slots = {
            mode: torch.full((vocab, width), float("nan"), device=stream.device)
            for mode in modes
        }
        oracle = torch.empty((vocab, width), dtype=torch.float64, device=stream.device)
        writers = {
            mode: make_head_dw_writer(
                mode, historical_chunks or len(row_counts), row_counts[0], vocab, width, stream
            )
            for mode in modes
        }
        for step, (grad, hidden) in enumerate(operands):
            hidden_double = hidden.double()
            for target_row in range(0, vocab, 2048):
                reference = oracle[target_row : target_row + 2048]
                grad_double = grad[:, target_row : target_row + 2048].double()
                torch.addmm(
                    reference, grad_double.T, hidden_double,
                    beta=float(step != 0), out=reference,
                )
            for mode, writer in writers.items():
                writer(grad, hidden, slots[mode], step != 0, stream)
                stats = error(slots[mode], oracle)
                assert math.isfinite(stats["relative_l2"])
                assert stats["relative_l2"] < 5e-5
                results.setdefault(mode, []).append(stats)
        if historical_chunks is not None:
            results["generic_vs_historical"] = error(slots["generic"], slots["new"].double())
        results["generic_vs_previous"] = error(slots["generic"], slots["previous"].double())
        # Warm, preallocated diagnostic span of the synthetic writer chain.
        results["event_ms"] = {}
        for mode, writer in writers.items():
            for _ in range(2):
                for step, (grad, hidden) in enumerate(operands):
                    writer(grad, hidden, slots[mode], step != 0, stream)
            start = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            for _ in range(4):
                for step, (grad, hidden) in enumerate(operands):
                    writer(grad, hidden, slots[mode], step != 0, stream)
            finish.record(stream)
            finish.synchronize()
            results["event_ms"][mode] = start.elapsed_time(finish) / 4
    result = {"case": name, "row_counts": row_counts, "vocab": vocab, "width": width, **results}
    print(json.dumps(result), flush=True)
    return result


logging.basicConfig(level=logging.INFO)
torch.manual_seed(548742)
assert torch.cuda.device_count() == 1
print({
    "torch": torch.__version__, "device": torch.cuda.get_device_name(),
    "cuda": torch.version.cuda, "uuid": os.environ["CUDA_VISIBLE_DEVICES"],
    "writer_source": str(make_head_dw_writer.__code__.co_filename),
}, flush=True)
cases = [
    ("uneven", (128, 97, 128), 1024, 256, None),
    ("100b_outer2", (8192, 8192), 128256, 4096, 2),
    ("100b_outer8", (2048, 2048), 128256, 4096, 8),
    ("700b_outer8", (3072, 3072), 128256, 7168, None),
]
result = [check_case(*args) for args in cases]
Path(__file__).with_name("result.json").write_text(json.dumps(result, indent=2) + "\n")
