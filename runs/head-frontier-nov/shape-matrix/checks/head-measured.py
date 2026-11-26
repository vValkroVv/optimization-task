"""One complete isolated LM-head replay on an eight-rank native GFSDP group."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from matrix_fixture import load_inputs


def make_head(weight, world):
    from torchtitan.distributed import ParallelDims
    from torchtitan.distributed.gfsdp import MixedPrecisionPolicy, fully_shard
    from torchtitan.models.common.decoder_blocks.sharding import set_decoder_sharding_config
    from torchtitan.models.common.nn_modules import Linear

    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",))
    dims = ParallelDims(dp_replicate=1, dp_shard=world, cp=1, tp=1, pp=1, ep=1, world_size=world)
    dims.build_mesh()
    v, d = weight.shape
    cfg = SimpleNamespace(tok_embeddings=SimpleNamespace(), norm=SimpleNamespace(),
                          lm_head=Linear.Config(in_features=d, out_features=v, bias=False))
    set_decoder_sharding_config(cfg, loss_parallel=True, enable_sp=True)
    with torch.device("meta"):
        head = cfg.lm_head.build()
    head.weight = torch.nn.Parameter(weight.to(device="cuda", dtype=torch.float32))
    head.parallelize(dims)
    head = fully_shard(head, mesh=mesh, reshard_after_forward=True,
                       mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                      reduce_dtype=torch.float32, cast_forward_inputs=False))
    head.set_own_grad_buffer()
    head.set_force_sum_reduction_for_comms(True)
    head.set_gradient_divide_factor(1.0)
    (group,) = head._get_fsdp_state()._fsdp_param_groups
    assert group.gradient_divide_factor == 1.0 and group.force_sum_reduction_for_comms
    return head


def fused_terms(variant, h, w, y):
    x, t = h.flatten(0, 1), y.flatten()
    if variant == "apple":
        from cut_cross_entropy import linear_cross_entropy
        nll = linear_cross_entropy(x, w, t, impl="cce_exact", reduction="none", ignore_index=-100)
    elif variant.startswith("liger-linear"):
        from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
        budget = 1 if variant.endswith("default") else (w.size(0) + w.size(1) - 1) // w.size(1)
        nll = LigerFusedLinearCrossEntropyFunction.apply(
            x, w, t, None, None, -100, 0.0, 0.0, "none", None, False,
            torch.float32, False, False, False, None, None, budget)[0]
    else:
        from fla.modules.fused_linear_cross_entropy.ops import fused_linear_cross_entropy_loss
        inner = 8 if variant.endswith("default") else 1
        return tuple(fused_linear_cross_entropy_loss(xh, yh, w, num_chunks=inner,
                      reduction="sum", accumulate_grad_in_fp32=True)
                     for xh, yh in zip(x.chunk(2), t.chunk(2), strict=True))
    return tuple(part.sum() for part in nll.chunk(2))


def install_fused(loss, head, variant):
    original_loop, original_forward = loss._chunked_loss_and_grad, head.forward
    state = {}

    def loop(this, hidden, targets, denominator, **kwargs):
        state["targets"] = iter(targets.chunk(this.num_chunks, dim=1))
        try:
            result = original_loop(hidden, targets, denominator, **kwargs)
            assert next(state["targets"], None) is None
            return result
        finally:
            state.clear()

    def forward(this, x, **kwargs):
        if "targets" not in state:
            return original_forward(x, **kwargs)
        return torch.stack(fused_terms(variant, x, this.weight, next(state["targets"]).contiguous()))

    def terms(this, raw, targets, denom, *, num_heads, **unused):
        assert raw.shape == (num_heads,) and num_heads == 2
        a, b = raw.unbind()
        a, b = a / denom, b / denom
        this.add_component(this.component_name, a.detach())
        this.add_component("loss_mtp", b.detach())
        return a + b * this.loss_weights[0]

    loss._chunked_loss_and_grad = MethodType(loop, loss)
    loss._chunk_loss = MethodType(terms, loss)
    head.forward = MethodType(forward, head)


def install_fla_lt(loss):
    # Same FLA fwd/bwd functions called by upstream FusedLinearCrossEntropyLoss.
    # The two-headed GIFT adapter supplies dloss before BF16 write and reuses
    # GIFT's two GEMMs including the direct FP32 GFSDP writer.
    from fla.modules.fused_linear_cross_entropy.ops import cross_entropy_fwd, cross_entropy_bwd

    class Inner(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, y):
            z, t = logits.flatten(0, 1), y.flatten()
            row, _, lse, classes, start = cross_entropy_fwd(
                logits=z, target=t, label_smoothing=0.0, logit_scale=1.0,
                logit_softcapping=None, ignore_index=-100, process_group=None)
            ctx.save_for_backward(logits, t, lse)
            ctx.classes, ctx.start = classes, start
            return tuple(chunk.sum() for chunk in row.chunk(2))

        @staticmethod
        def backward(ctx, a, b):
            z, t, lse = ctx.saved_tensors
            dz = cross_entropy_bwd(
                logits=z.flatten(0, 1), target=t, lse=lse,
                dloss=torch.stack([a, b]).repeat_interleave(t.numel() // 2),
                label_smoothing=0.0, logit_scale=1.0, logit_softcapping=None, ignore_index=-100,
                total_classes=ctx.classes, class_start_idx=ctx.start, inplace_backward=True)
            return dz.view_as(z), None

    loss._exact_folded_ce = SimpleNamespace(consume_if_supported=lambda logits, y, weights, den: Inner.apply(logits, y))


def run(args):
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert rank == int(os.environ["RANK"]) and world == 8
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    from torchtitan.components import loss as loss_module
    from torchtitan.config import CompileConfig
    from torchtitan.models.common.regularization import hidden_z_loss
    from torchtitan.tools.utils import device_memory_usage

    torch.backends.cuda.matmul.allow_tf32 = False
    torch._inductor.config.triton.cudagraphs = False
    if hasattr(torch._inductor.config.triton, "mix_order_reduction_non_strict_mode"):
        torch._inductor.config.triton.mix_order_reduction_non_strict_mode = True
    rec = load_inputs(args.model, args.fixture)
    h = rec["hidden"].cuda().requires_grad_()
    y, pos = rec["labels"].cuda(), rec["positions"].cuda()
    denom = float(rec["denominator"])
    z = rec["z_hidden"].cuda().requires_grad_() if args.model == "100b" else None
    head = make_head(rec["weight"], world)
    with torch.no_grad():
        head(h[0, :, :1].contiguous())
    dist.barrier()
    v, width = rec["weight"].shape
    loss = loss_module.MTPLoss(loss_module.MTPLoss.Config(num_chunks=args.chunks, loss_weights=rec["head_weights"][1:]),
                               compile_config=CompileConfig(enable=True, components=["loss"], backend="inductor"))
    loss.set_lm_head(head)
    if args.variant == "baseline":
        loss.fn = torch.compile(loss_module.cross_entropy_loss, backend="inductor")
    direct = args.variant in ("cuda13", "cuda13-tuned", "folded-lt", "fla-linear-lt", "apple-direct") or args.variant.startswith("standalone-")
    if args.variant == "apple-direct":
        from apple_head import install_apple_direct
        install_apple_direct(loss)
    elif args.variant == "standalone-liger-early":
        from liger_early import make_liger_early_ce
        loss._exact_folded_ce = SimpleNamespace(consume_if_supported=make_liger_early_ce())
    elif args.variant.startswith("standalone-"):
        from vendor_ops import make_vendor_ce
        loss._exact_folded_ce = SimpleNamespace(consume_if_supported=make_vendor_ce(args.variant.removeprefix("standalone-")))
    elif args.variant == "fla-linear-lt":
        install_fla_lt(loss)
    elif args.variant == "apple" or args.variant.startswith(("liger-linear", "fla-linear")):
        install_fused(loss, head, args.variant)

    def invoke(reset=False):
        h.grad = None
        if reset:
            head.weight.grad = None
        loss.reset_components()
        if z is not None:
            z.grad = None
            z_value = hidden_z_loss(z, coef=1.0e-7)
        result = loss(h, loss.pack_targets(y, {"positions": pos}), denom)
        if z is not None:
            torch.autograd.backward((result, z_value))
        else:
            result.backward()
        return result.detach()

    staged, calls = [], {"gather": 0, "scatter": 0}
    original_ag, original_rs = dist.all_gather_into_tensor, dist.reduce_scatter_tensor
    def ag(output_tensor, input_tensor, *a, **kw):
        out, inp = output_tensor, input_tensor
        if inp.dtype == torch.bfloat16 and out.numel() == v * width:
            assert inp.untyped_storage().data_ptr() != out.untyped_storage().data_ptr()
            calls["gather"] += 1
        return original_ag(out, inp, *a, **kw)
    def rs(out, inp, *a, **kw):
        if inp.numel() == v * width:
            assert inp.dtype == out.dtype == torch.float32 and out.numel() * world == inp.numel()
            assert kw.get("op", a[0] if a else dist.ReduceOp.SUM) == dist.ReduceOp.SUM
            calls["scatter"] += 1
        return original_rs(out, inp, *a, **kw)
    def hook(_head, inp, kwargs):
        staged.append(any("staged_head_dw" in k and val is not None for k, val in kwargs.items()))
    dist.all_gather_into_tensor, dist.reduce_scatter_tensor = ag, rs
    try:
        with head.register_forward_pre_hook(hook, with_kwargs=True):
            invoke()
    finally:
        dist.all_gather_into_tensor, dist.reduce_scatter_tensor = original_ag, original_rs
    assert len(staged) == args.chunks and calls == {"gather": 1, "scatter": 1}, (staged, calls)
    assert not direct or all(staged), (args.variant, staged)
    if args.variant.startswith("cuda13"):
        assert loss._exact_folded_ce.scales_checked and not loss._exact_folded_ce_fallback_logged
    for _ in range(args.warmup):
        invoke()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    invoke()
    torch.cuda.synchronize()
    metrics = []
    for _ in range(args.samples):
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        invoke()
        submitted = time.perf_counter()
        torch.cuda.synchronize()
        stop = time.perf_counter()
        metrics.append(dict(ms=1000*(stop-start), submit_ms=1000*(submitted-start),
                            wait_ms=1000*(stop-submitted),
                            total_gib=device_memory_usage(torch.cuda.memory_stats(device))[1]))
    value = invoke(reset=True)
    with torch.no_grad():
        weight_grad = head.weight.grad.to_local().view(v//world, width)
    assert weight_grad.dtype == torch.float32
    assert h.grad.dtype == torch.bfloat16
    ce_comp = {k: float(val) for k, val in loss.components().items()}
    output = dict(rank=rank, variant=args.variant, model=args.model, chunks=args.chunks,
                  rows=h.shape[0]*h.shape[1]*h.shape[2]//args.chunks, width=width,
                  vocab=v, smoke_collectives=calls, staged_chunks=staged,
                  gradient_divide_factor=1.0, force_sum_reduction_for_comms=True,
                  components=ce_comp, total_loss=float(value), measured=metrics,
                  initialization=rec["provenance"], source=loss_module.__file__,
                  source_hash=hashlib.sha256(Path(loss_module.__file__).read_bytes()).hexdigest(),
                  bf16_direct=getattr(loss, "_direct_bf16_dhidden", False))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"rank-{rank}.json"
    path.write_text(json.dumps(output, allow_nan=False))
    if args.reference:
        from head_oracle import compare
        output["numerics"] = compare(rec, args.chunks, rank, world, h.grad, weight_grad,
                                     ce_comp, value, args.reference)
        path.write_text(json.dumps(output, allow_nan=False))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("100b", "700b"), required=True)
    p.add_argument("--chunks", type=int, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=10)
    run(p.parse_args())
