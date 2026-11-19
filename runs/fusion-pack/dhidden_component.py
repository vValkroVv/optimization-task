"""Compare disjoint gradient writes using gradients from the native folded head."""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runs/backward-components"))
from folded_ce import Replay, equal_bits, load_fixture
from torchtitan.components.loss import GradAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture, capture = load_fixture(args.fixture)
    replay = Replay(fixture, capture.get("compile_backend", "inductor"))
    loss = replay.make_loss("folded")
    chunks = []
    add = GradAccumulator.add

    def capture_chunk(self, tensor):
        chunks.append(tensor.detach().clone())
        return add(self, tensor)

    with patch.object(GradAccumulator, "add", capture_chunk):
        replay.run(loss)
    expected = replay.hidden.grad.flatten(0, 1).detach().clone()
    assert len(chunks) == 2 and all(c.dtype == torch.bfloat16 for c in chunks)
    del replay, loss, fixture
    gc.collect()

    def assemble(direct):
        acc = GradAccumulator(
            expected, num_chunks=len(chunks), dtype=torch.float32, direct_bf16=direct
        )
        for chunk in chunks:
            acc.add(chunk)
        return acc.result().to(expected.dtype)

    report = {}
    for name, direct in (("baseline", False), ("candidate", True)):
        ref = assemble(direct)
        # The full native callback may flatten or transpose the folded head axis.
        exact = equal_bits(ref, expected)
        if not exact:
            raise AssertionError(
                f"{name} disagrees with the native folded hidden gradient"
            )
        del ref
        for _ in range(10):
            assemble(direct)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        events, wall = [], []
        for _ in range(40):
            start = time.perf_counter()
            begin.record()
            assemble(direct)
            end.record()
            end.synchronize()
            events.append(begin.elapsed_time(end))
            wall.append((time.perf_counter() - start) * 1000)
        report[name] = dict(
            bitwise=exact,
            elements=expected.numel(),
            median_ms=statistics.median(events),
            median_wall_ms=statistics.median(wall),
            allocated=torch.cuda.max_memory_allocated(),
            reserved=torch.cuda.max_memory_reserved(),
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(3):
                assemble(direct)
            torch.cuda.synchronize()
        path = args.output / f"{name}.trace.json"
        prof.export_chrome_trace(str(path))
        physical = [
            e
            for e in json.loads(path.read_text())["traceEvents"]
            if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
        ]
        report[name].update(
            device_ms_three_calls=sum(e["dur"] for e in physical) / 1000,
            gpu_events=len(physical),
        )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
