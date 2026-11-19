"""Compare Hopper CuTe norm plus residual against its compiled unfused path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from torchtitan.models.common.norms.cute_gated_norm import gated_norm


def sac_context():
    return create_selective_checkpoint_contexts(
        lambda _ctx, _op, *_args, **_kwargs: CheckpointPolicy.PREFER_RECOMPUTE
    )


def build(fused, use_checkpoint):
    def call(x, residual, w, gate_up, gate_down):
        y = gated_norm(
            x,
            1.0 + w.float(),
            gate_up,
            gate_down,
            eps=1e-5,
            gating_weight=2.0,
            residual=residual if fused else None,
        )
        return y if fused else residual + y

    def region(*args):
        if use_checkpoint:
            return checkpoint(call, *args, use_reentrant=False, context_fn=sac_context)
        return call(*args)

    return torch.compile(region, fullgraph=True, options={"triton.cudagraphs": False})


def train(function, tensors, dy):
    inputs = [value.detach().requires_grad_() for value in tensors]
    y = function(*inputs)
    return y.detach(), *torch.autograd.grad(y, inputs, dy)


def measure(run, repeats=20):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(2048)
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    shapes = ((2, 4096, 4096), (2, 4096, 4096), (4096,), (16, 4096), (4096, 16))
    inputs = [
        torch.randn(shape, dtype=torch.bfloat16, device="cuda") * scale
        for shape, scale in zip(shapes, (1.0, 0.5, 0.03, 0.02, 0.2), strict=True)
    ]
    dy = torch.randn_like(inputs[0])
    output = {}
    for use_checkpoint in (False, True):
        baseline = build(False, use_checkpoint)
        candidate = build(True, use_checkpoint)
        for expected, actual in zip(
            train(baseline, inputs, dy), train(candidate, inputs, dy), strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        version_results = {}
        for label, fn in (("baseline", baseline), ("candidate", candidate)):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            starting = torch.cuda.memory_allocated()
            forward_ms = measure(lambda: fn(*inputs))
            forward_peak = torch.cuda.max_memory_allocated() - starting
            torch.cuda.reset_peak_memory_stats()
            combined_ms = measure(lambda: train(fn, inputs, dy))
            version_results[label] = {
                "forward_event_ms": forward_ms,
                "forward_backward_event_ms": combined_ms,
                "forward_peak_delta_bytes": forward_peak,
                "forward_backward_peak_delta_bytes": torch.cuda.max_memory_allocated()
                - starting,
            }
        output["sac" if use_checkpoint else "plain"] = {
            "exact_output_and_all_gradients": True,
            "separate_forward_and_training_measurements": version_results,
        }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
