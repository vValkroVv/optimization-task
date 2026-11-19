"""Check actual FP8Linear forward, dgrad, and wgrad at the dense tail shapes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from torchtitan.distributed.gfsdp import (
    MixedPrecisionPolicy,
    RaggedShardDTensor,
    fully_shard,
)
from torchtitan.models.common.ffn import FeedForward
from torchtitan.ops.float8.cuda_gemm import FP8Linear


class Stack(torch.nn.Module):
    def __init__(self, *, use_checkpoint: bool, gf: bool):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        layers = []
        for _ in range(3 if gf else 1):
            config = FeedForward.Config(
                dim=4096,
                hidden_dim=10944,
                w1=FP8Linear.Config(in_features=4096, out_features=10944),
                w2=FP8Linear.Config(in_features=10944, out_features=4096),
                w3=FP8Linear.Config(in_features=4096, out_features=10944),
            )
            config.use_shared_quantizer = True
            layer = config.build()
            layer = layer.cuda().to(torch.float32 if gf else torch.bfloat16)
            for weight in layer.parameters():
                weight.detach().normal_(std=0.0115)
            layers.append(layer)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint:
                x = checkpoint(layer, x, use_reentrant=False, context_fn=sac_context)
            else:
                x = layer(x)
        return x


def sac_context():
    return create_selective_checkpoint_contexts(
        lambda _ctx, _op, *_args, **_kwargs: CheckpointPolicy.PREFER_RECOMPUTE
    )


@torch.no_grad()
def local(tensor):
    return (
        tensor.to_local()
        if isinstance(tensor, (torch.distributed.tensor.DTensor, RaggedShardDTensor))
        else tensor
    )


def build(enabled, *, gf, mesh, use_checkpoint):
    os.environ["GIFT_FP8_QUANTIZE_WEIGHT_TAIL"] = str(int(enabled))
    torch.manual_seed(2048)
    model = Stack(use_checkpoint=use_checkpoint, gf=gf)
    if gf:
        mixed = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        # Match the training order: compile each inner region before GFSDP
        # places its pre/post forward and backward hooks outside that region.
        for layer in model.layers:
            layer.compile(fullgraph=True, options={"triton.cudagraphs": False})
            fully_shard(layer, mesh=mesh, mp_policy=mixed, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, mp_policy=mixed)
        model.set_shard_order(list(model.layers))
        return model
    return torch.compile(model, fullgraph=True, options={"triton.cudagraphs": False})


def run(model, x, grad):
    model.zero_grad(set_to_none=True)
    x = x.detach().requires_grad_()
    y = model(x)
    y.backward(grad)
    return y.detach(), x.grad, *(local(p.grad) for p in model.parameters())


def timing(model, x, grad, repetitions=12):
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run(model, x, grad)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gfsdp", action="store_true")
    args = parser.parse_args()
    mesh = None
    rank = 0
    if args.gfsdp:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    torch.manual_seed(400 + rank)
    rows = 128 if args.gfsdp else 4096
    x = torch.randn(2, rows, 4096, dtype=torch.bfloat16, device="cuda")
    grad = torch.randn_like(x)
    output = {}
    for use_checkpoint in (False, True):
        baseline = build(False, gf=args.gfsdp, mesh=mesh, use_checkpoint=use_checkpoint)
        candidate = build(True, gf=args.gfsdp, mesh=mesh, use_checkpoint=use_checkpoint)
        reference = run(baseline, x, grad)
        actual = run(candidate, x, grad)
        for left, right in zip(reference, actual, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        label = "sac" if use_checkpoint else "plain"
        output[label] = {"exact_output_input_gradient_and_weight_gradients": True}
        if args.gfsdp:
            groups = [
                layer._get_fsdp_state()._fsdp_param_groups[0]
                for layer in candidate.layers
            ]
            if groups[0].data_buffer_ctx is not groups[2].data_buffer_ctx:
                raise RuntimeError(
                    "The first and third GFSDP groups did not share an all-gather buffer"
                )
            output[label]["shared_full_weight_buffer"] = True
        else:
            repeats = [
                (timing(baseline, x, grad), timing(candidate, x, grad))
                for _ in range(3)
            ]
            output[label]["paired_forward_backward_ms"] = repeats
    path = (
        args.output if not args.gfsdp else args.output.with_suffix(f".rank{rank}.json")
    )
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)
    if args.gfsdp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
