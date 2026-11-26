"""Compare the three fastest low-memory CE routes on the same card per shape."""
import argparse
import json
from pathlib import Path
from statistics import median
import time

import torch

from ce_bench import evaluate, make_route, project_chunk
from torchtitan.tools.utils import device_memory_usage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--chunks", type=int, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--exact-so", type=Path, required=True)
    p.add_argument("--exact-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.part = args.chunks - 1
    args.extra_fixture = None
    torch.backends.cuda.matmul.allow_tf32 = False
    pristine, labels, weights, denom, fixture = project_chunk(args)
    working = torch.empty_like(pristine, requires_grad=True)
    names = ["cuda13", "liger-early", "fla"]
    routes = {name: make_route(args, name)[0] for name in names}
    samples = {name: [] for name in names}
    for cycle in range(42):
        order = names[cycle % len(names):] + names[:cycle % len(names)]
        if cycle == 10:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        for name in order:
            working.detach().copy_(pristine)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = evaluate(routes[name], working, labels, weights, denom)
            torch.cuda.synchronize()
            wall = 1000 * (time.perf_counter() - start)
            if cycle >= 10:
                total = device_memory_usage(torch.cuda.memory_stats())[1]
                samples[name].append({"ms": wall, "total_gib": total})
            del result
    summary = {name: {"wall_ms": median(x["ms"] for x in series),
                      "total_gib": max(x["total_gib"] for x in series)}
               for name, series in samples.items()}
    receipt = dict(fixture=fixture, samples=samples, summary=summary,
                   gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
    args.output.write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
