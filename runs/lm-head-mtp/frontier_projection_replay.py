"""Evaluate a tuned projection through the compiled LM-head replay."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

import replay
from frontier_dw_lib import scratch_for
from frontier_projection_screen import Bf16OutputLt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "lt"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--storage", choices=("native", "reuse"), default="native")
    parser.add_argument("--numerics", action="store_true")
    args = parser.parse_args()

    receipt = json.loads(args.screen.read_text())
    run = next(run for run in receipt["runs"] if run["chunks"] == 8 and run["kind"] == "projection")
    config = run["finalists"][0]
    assert config["confirmation"]["numerics"]["bits_different"] == 0
    for key in ("GIFT_STABLE_CE_FOLDED_BACKWARD", "GIFT_DIRECT_BF16_DHIDDEN", "GIFT_STAGED_HEAD_DW", "GIFT_FUSED_HIDDEN_Z_LOSS"):
        os.environ[key] = "1"
    from torchtitan.distributed.gfsdp import _staged_head_dw as staging

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    original = staging._StagedHeadLinear.forward
    writer = None
    plan = None
    calls = 0
    tuned_calls = 0

    def forward(ctx, input, weight, lease):
        nonlocal writer, plan, calls, tuned_calls
        calls += 1
        if args.arm == "native":
            return original(ctx, input, weight, lease)
        assert weight is lease.param.unsharded_param
        assert input.is_contiguous() and weight.is_contiguous()
        assert input.dtype == weight.dtype == torch.bfloat16
        left = input.flatten(0, -2)
        assert left.shape == (2048, 4096) and weight.shape == (128256, 4096)
        if writer is None:
            writer = Bf16OutputLt("projection", left.size(0), *weight.shape)
            assert writer.version == run["cublas_lt_version"]
            plan = writer.prepare(config, scratch_for(config))
        ctx.save_for_backward(input, weight)
        ctx.lease = lease
        logits = input.new_empty((*input.shape[:-1], weight.size(0)))
        writer.execute(plan, left, weight, logits)
        tuned_calls += 1
        return logits

    staging._StagedHeadLinear.forward = staticmethod(forward)
    numerics = args.output.with_suffix(".numerics.json")
    save = torch.save

    def capture(dump, destination, *positional, **keywords):
        if destination != numerics:
            return save(dump, destination, *positional, **keywords)
        result = {}
        for key, value in dump.items():
            bytes_ = value.contiguous().reshape(-1).view(torch.uint8)
            result[key] = hashlib.sha256(memoryview(bytes_.numpy()).cast("B")).hexdigest()
        destination.write_text(json.dumps(result, indent=2) + "\n")

    sys.argv = [
        sys.argv[0], "current", str(args.output), "--chunks", "8", "--writer", "cublas",
        "--sum-reduction", "--warmup", "10", "--samples", "10", "--storage", args.storage,
    ]
    if args.numerics:
        torch.save = capture
        sys.argv += ["--dump", str(numerics)]
    replay.main()
    assert calls == (22 + int(args.numerics)) * 8, calls
    assert tuned_calls == (calls if args.arm == "lt" else 0)
    result = json.loads(args.output.read_text())
    result.update(projection=args.arm, projection_calls=calls, cublas_lt_calls=tuned_calls,
                  allow_bf16_reduced_precision_reduction=False)
    if writer is not None:
        result.update(cublas_lt_version=writer.version, cublas_lt_projection_config=config)
        writer.close()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
