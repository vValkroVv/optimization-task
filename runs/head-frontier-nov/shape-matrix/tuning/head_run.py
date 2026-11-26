"""Apply only selected region implementations to the existing GFSDP head replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

import head_bench


def install_linear_ops(operations, choices):
    from torchtitan.distributed.gfsdp._staged_head_dw import _StagedHeadLinear
    from linear_policy import LinearPolicy

    policy = LinearPolicy(choices)

    counts = {"projection": 0, "dhidden": 0}
    if "projection" in operations or "linear" in operations:
        def forward(ctx, inputs, weight, lease):
            if weight is not lease.param.unsharded_param or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
                raise RuntimeError("Expected the lease's contiguous BF16 weight")
            ctx.save_for_backward(inputs, weight)
            ctx.lease = lease
            flat = inputs.flatten(0, -2)
            if not policy.supports_shape(flat, weight.T):
                return torch.nn.functional.linear(inputs, weight)
            counts["projection"] += 1
            return policy.run(flat, weight.T, (*inputs.shape[:-1], weight.shape[0]))

        _StagedHeadLinear.forward = staticmethod(forward)
    if "dhidden" in operations or "linear" in operations:
        def backward(ctx, grad):
            inputs, weight = ctx.saved_tensors
            dz = grad.flatten(0, -2)
            if policy.supports_shape(dz, weight):
                counts["dhidden"] += 1
                dx = policy.run(dz, weight).view_as(inputs)
            else:
                dx = torch.mm(dz, weight).view_as(inputs)
            ctx.lease.write(dz, inputs.flatten(0, -2))
            return dx, None, None

        _StagedHeadLinear.backward = staticmethod(backward)
    return counts, policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operations", nargs="*", default=[])
    parser.add_argument("--choices", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunks", type=int, required=True)
    for name in ("fixture", "out", "reference"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    args = parser.parse_args()
    allowed = {"projection", "dhidden", "dw", "ce", "linear"}
    if set(args.operations) - allowed:
        parser.error("Unknown operation")
    counts = {}
    writers = []
    if "dw" in args.operations:
        from torchtitan.distributed.gfsdp import _staged_head_dw as staged
        from tuning.lt.lt_writer import make_dw_writer

        def factory(*unused):
            writer = make_dw_writer(policy="replay", receipt=args.plan)
            writers.append(writer)
            return writer

        staged.make_head_dw_writer = factory
    if {"projection", "dhidden", "linear"}.intersection(args.operations):
        counts, policy = install_linear_ops(args.operations, args.choices)
    import head_oracle
    oracle_compare = head_oracle.compare
    digests = {}

    def compare(record, chunks, rank, world, dh, dw, components, total, refdir):
        digests["dweight_shard_sha256"] = hashlib.sha256(dw.detach().cpu().contiguous().numpy()).hexdigest()
        return oracle_compare(record, chunks, rank, world, dh, dw, components, total, refdir)

    head_oracle.compare = compare
    head_bench.run(args)
    for op in {"projection", "dhidden"}.intersection(args.operations):
        if counts[op] != (args.warmup + args.samples + 3) * args.chunks:
            raise RuntimeError(f"Unexpected {op} count {counts[op]}")
    if "dw" in args.operations and len(writers) != 1:
        raise RuntimeError(f"Expected one native dW writer, got {len(writers)}")
    if writers:
        steps = args.warmup + args.samples + 3
        expected = [steps, steps * (args.chunks - 1)]
        if writers[0].receipt()["tuned_calls_beta0_beta1"] != expected:
            raise RuntimeError("A requested dW plan fell back or missed a chunk")
    route = {}
    if {"projection", "dhidden", "linear"}.intersection(args.operations):
        route = policy.calls
        if route["fallback"] or route["candidate"] != sum(counts.values()):
            raise RuntimeError(f"The requested cuBLASLt path did not cover every call: {route}")
    activation = dict(linear_route=route, operations=args.operations, counts=counts, plans=[w.receipt() for w in writers],
                      ce=os.environ["GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256"], **digests)
    (args.out / f"activation-{os.environ['RANK']}.json").write_text(json.dumps(activation, default=str))


if __name__ == "__main__":
    main()
