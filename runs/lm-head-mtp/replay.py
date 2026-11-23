"""Replay native compiled main/MTP loss and real one-rank GFSDP head backward."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


TASK = Path(__file__).resolve().parents[4]
RAW = TASK / "runs/kernel-fusion/artifacts/raw"
FIXTURE = RAW / "backward-inputs-001/component-inputs/rank0.pt"
Z_FIXTURE = TASK / "runs/production-data-abcd/artifacts/raw/proddata-component-capture-001/native-inputs"
CURRENT = Path(__file__).resolve().parents[2]
OLD = CURRENT.parent / "lm-head-mtp-original"
SWITCHES = (
    "GIFT_STABLE_CE_FOLDED_BACKWARD",
    "GIFT_DIRECT_BF16_DHIDDEN",
    "GIFT_STAGED_HEAD_DW",
    "GIFT_FUSED_HIDDEN_Z_LOSS",
)
VARIANTS = {
    "original": (False, False, False, False),
    "all-off": (False, False, False, False),
    "vendor": (False, True, False, True),
    "torch-ce": (False, True, True, True),
    "unfolded": (False, True, True, True),
    "old-dh": (True, False, True, True),
    "ordinary-dw": (True, True, False, True),
    "old-z": (True, True, True, False),
    "current": (True, True, True, True),
}


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--writer", choices=("native", "cublas", "tma"), default="native")
    parser.add_argument("--vendor", choices=("fla", "q4096", "q8192", "liger", "cce"))
    parser.add_argument("--sum-reduction", action="store_true")
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--storage", choices=("native", "separate", "reuse"), default="native")
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    return parser.parse_args()


def make_head(weight):
    from torchtitan.distributed import ParallelDims
    from torchtitan.distributed.gfsdp import MixedPrecisionPolicy, fully_shard
    from torchtitan.models.common.decoder_blocks.sharding import set_decoder_sharding_config
    from torchtitan.models.common.nn_modules import Linear

    mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("fsdp",))
    dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1)
    dims.build_mesh()
    vocab, width = weight.shape
    decoder = SimpleNamespace(
        tok_embeddings=SimpleNamespace(),
        norm=SimpleNamespace(),
        lm_head=Linear.Config(in_features=width, out_features=vocab, bias=False),
    )
    set_decoder_sharding_config(decoder, loss_parallel=True, enable_sp=True)
    with torch.device("meta"):
        head = decoder.lm_head.build()
    head.weight = torch.nn.Parameter(weight.to(device="cuda", dtype=torch.float32))
    head.parallelize(dims)
    head = fully_shard(
        head,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False,
        ),
    )
    head.set_own_grad_buffer()
    head.set_gradient_divide_factor(1.0)
    return head


def main():
    args = parse()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = OLD if args.variant == "original" else CURRENT
    sys.path[:0] = [str(source / "gift"), str(TASK / "runs/production-data-abcd")]
    for name, enabled in zip(SWITCHES, VARIANTS[args.variant], strict=True):
        os.environ[name] = str(int(enabled))
    from component_fixtures import load_fixture
    from torchtitan.components import loss as loss_module
    from torchtitan.config import CompileConfig
    from torchtitan.models.common.regularization import hidden_z_loss
    from torchtitan.tools.utils import device_memory_usage

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._inductor.config.triton.cudagraphs = False
    if hasattr(torch._inductor.config.triton, "mix_order_reduction_non_strict_mode"):
        torch._inductor.config.triton.mix_order_reduction_non_strict_mode = True
    torch.manual_seed(2048)
    fixture = torch.load(FIXTURE, map_location="cpu", weights_only=True, mmap=True)
    z_fixture, z_receipt = load_fixture(Z_FIXTURE, 0)
    record = fixture["ce"]
    backend = fixture["metadata"]["compile_backend"]
    assert backend == "inductor" and record["n_chunks"] == 2
    assert record["hidden"].shape == (2, 2, 4096, 4096)
    assert record["weight"].shape == (128256, 4096)
    assert record["head_weights"] == [1.0, 0.3] and record["ignore_index"] == -100
    hidden = record["hidden"].cuda().requires_grad_()
    z_hidden = z_fixture["mtp"]["hidden"].cuda().requires_grad_()
    labels = record["labels"].cuda()
    positions = record["positions"]
    if positions is not None:
        positions = positions.cuda()
    denominator = record["denominator"]
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.cuda()
    del z_fixture
    rendezvous = args.output.with_suffix(".rendezvous").resolve().as_uri()
    dist.init_process_group("nccl", init_method=rendezvous, rank=0, world_size=1)
    try:
        head = make_head(record["weight"])
        if args.sum_reduction:
            head.set_force_sum_reduction_for_comms(True)
        del record, fixture
        with torch.no_grad():
            head(hidden[0, :, :1].contiguous())
        loss = loss_module.MTPLoss(
            loss_module.MTPLoss.Config(num_chunks=args.chunks, loss_weights=[0.3]),
            compile_config=CompileConfig(enable=True, components=["loss"], backend=backend),
        )
        if args.variant in ("torch-ce", "all-off"):
            loss.fn = torch.compile(loss_module.cross_entropy_loss, backend=backend)
        loss.set_lm_head(head)
        if args.writer != "native":
            if not VARIANTS[args.variant][2]:
                raise ValueError("Direct writers require a GFSDP lease-enabled variant")
            from direct_dw import CublasWriter, TmaWriter, make_selector
            from torchtitan.distributed.gfsdp import _staged_head_dw as staging

            writer = (CublasWriter if args.writer == "cublas" else TmaWriter)()
            staging.select_staged_head_dw = make_selector(writer)
        if args.storage != "native":
            if loss._folded_ce is None or args.writer == "native":
                raise ValueError("Storage experiments require folded CE with a direct FP32 writer")
            from ce_storage import make
            loss._folded_ce = make(backend, args.storage == "reuse")
        if args.variant == "vendor":
            from vendor_ce import install
            vendor_calls = install(args.vendor, backend, loss, head)
        else:
            vendor_calls = None
        assert hasattr(loss.fn, "_torchdynamo_orig_callable")
        ce_fn = loss.fn._torchdynamo_orig_callable
        runs = []

        def run(reset_weight=False):
            hidden.grad = None
            z_hidden.grad = None
            if reset_weight:
                head.weight.grad = None
            loss.reset_components()
            targets = loss.pack_targets(labels, {"positions": positions})
            z_value = hidden_z_loss(z_hidden, coef=1e-7)
            value = loss(hidden, targets, denominator)
            torch.autograd.backward((value, z_value))
            return value.detach(), z_value.detach()

        def observe_stage(_head, _args, kwargs):
            runs.append(kwargs.get("_staged_head_dw") is not None)

        routes = {"folded": 0, "ce": 0}
        eager_ce = loss.fn
        folded_fn = getattr(loss, "_folded_ce", None)

        def counted_ce(*args):
            routes["ce"] += 1
            return eager_ce(*args)

        def counted_folded(*args):
            routes["folded"] += 1
            return folded_fn(*args)

        loss.fn = counted_ce
        if folded_fn is not None:
            loss._folded_ce = counted_folded
        with head.register_forward_pre_hook(observe_stage, with_kwargs=True):
            result = run()
        loss.fn = eager_ce
        if folded_fn is not None:
            loss._folded_ce = folded_fn
        if args.variant == "vendor":
            assert routes == {"folded": 0, "ce": 0}
        elif folded_fn is None:
            assert routes == {"folded": 0, "ce": 2 * args.chunks}, routes
        else:
            assert routes == {"folded": args.chunks, "ce": 0}, routes
        if vendor_calls is not None:
            assert vendor_calls["calls"] == args.chunks
        expected_stage = VARIANTS[args.variant][2] and (args.writer != "native" or args.chunks == 2)
        assert runs == [expected_stage] * args.chunks, runs
        folded = getattr(loss, "_folded_ce", None)
        assert (folded is not None) == VARIANTS[args.variant][0]
        assert getattr(loss, "_direct_bf16_dhidden", False) == VARIANTS[args.variant][1]
        for _ in range(args.warmup):
            result = run()
        torch.cuda.synchronize()
        hidden.grad = None
        z_hidden.grad = None
        torch.cuda.empty_cache()
        result = run()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_stats()
        before = dict(torch._dynamo.utils.counters["stats"])
        elapsed, total = [], []
        for _ in range(args.samples):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            result = run()
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - t0) * 1000)
            total.append(device_memory_usage(torch.cuda.memory_stats())[1])
        mem_after = torch.cuda.memory_stats()
        allocator = {key: mem_after[key] - mem_before[key] for key in ("num_alloc_retries", "num_ooms")}
        assert not any(allocator.values()), allocator
        after = dict(torch._dynamo.utils.counters["stats"])
        assert before == after, (before, after)
        with torch.no_grad():
            assert head.weight.grad.to_local().dtype == torch.float32
        receipt = {
            "source": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
            "loss_source": loss_module.__file__,
            "variant": args.variant,
            "chunks": args.chunks,
            "writer": args.writer,
            "vendor": args.vendor,
            "vendor_chunk_calls": vendor_calls,
            "logit_storage": args.storage,
            "folded_rows_per_chunk": 16384 // args.chunks,
            "compile": {"backend": backend, "scope": "native loss", "counters": after, "cuda_graphs": False},
            "head": "bias-free BF16; FP32 rank-local GFSDP accumulation/reduction",
            "activation": {"ce": f"{ce_fn.__module__}.{ce_fn.__name__}", "folded": folded is not None,
                           "staged_chunk_calls": runs, **{s: os.environ[s] for s in SWITCHES}},
            "fixture": str(FIXTURE), "hidden_z_fixture_sha256": z_receipt["tensors_sha256"],
            "device": {"torch": torch.__version__, "cuda": torch.version.cuda,
                       "name": torch.cuda.get_device_name(0), "visible": os.environ["CUDA_VISIBLE_DEVICES"]},
            "scope": "host wall incl. device sync; native label packing, head CE and BF16 dHidden, FP32 GFSDP dW, independent z-loss retained across head backward",
            "singleton_reduction": "force-sum" if args.sum_reduction else "native-factor-1",
            "allocator_retries_and_ooms": allocator,
            "ce_route_calls": routes,
            "accumulation": "pre-existing sharded weight gradient retained across timed calls",
            "warmup": args.warmup, "elapsed_ms": elapsed, "median_ms": statistics.median(elapsed),
            "total_gpu_gib": total, "peak_total_gpu_gib": max(total),
            "loss": [float(value) for value in result],
        }
        if args.dump is not None:
            result = run(reset_weight=True)
            torch.cuda.synchronize()
            with torch.no_grad():
                torch.save({"loss": result[0].cpu(), "dx": hidden.grad.cpu(),
                            "dw": head.weight.grad.to_local().view(128256, 4096).cpu(),
                            "z_loss": result[1].cpu(), "z_dx": z_hidden.grad.cpu()}, args.dump)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
