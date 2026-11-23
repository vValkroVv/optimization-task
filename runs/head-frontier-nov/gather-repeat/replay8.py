"""Replay the isolated compiled main/MTP head and z-loss with eight NCCL ranks."""

import argparse
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parent.parent
TASK = ROOT.parents[1]
PRIOR = TASK / ".worktrees/lm-head-mtp-fix/runs/lm-head-mtp"
CURRENT = TASK.parent / "lm10-c1"
RANKS = 8
VOCAB, WIDTH = 128256, 4096
FIXTURE_SHA = "15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727"
BASELINE_SHA = "e3871b36c0be1cb5c027b811d1684558fb11b02b"
PAIRS = {2: "9.14", 8: "5.5"}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_digest(tensor):
    raw = tensor.detach().cpu().reshape(-1).contiguous().view(torch.uint8).numpy()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(raw).hexdigest()}


def make_distributed_head(replay, weight):
    """Apply the same mesh-only changes as the two-rank frontier replay."""
    body = inspect.getsource(replay.make_head)
    for before, after in (
        ('init_device_mesh("cuda", (1,),', f'init_device_mesh("cuda", ({RANKS},),'),
        ("dp_shard=1", f"dp_shard={RANKS}"),
        ("world_size=1", f"world_size={RANKS}"),
    ):
        assert body.count(before) == 1, before
        body = body.replace(before, after)
    methods = {}
    exec(compile(body, __file__, "exec"), vars(replay), methods)
    return methods["make_head"](weight)


def install_reduce_probe(rank):
    collective = dist.reduce_scatter_tensor
    receipt = {"calls": 0}

    def counted_reduce(output, input, op=dist.ReduceOp.SUM, group=None, **kwargs):
        assert (group.rank(), group.size(), dist.get_backend(group)) == (rank, RANKS, "nccl")
        assert op == dist.ReduceOp.SUM and output.dtype == input.dtype == torch.float32
        assert output.numel() == VOCAB * WIDTH // RANKS and input.numel() == VOCAB * WIDTH
        if receipt["calls"] == 0:
            assert dist.get_process_group_ranks(group) == list(range(RANKS))
        receipt["calls"] += 1
        return collective(output, input, op=op, group=group, **kwargs)

    dist.reduce_scatter_tensor = counted_reduce
    return receipt


def run(args, rank, device, visible):
    import replay
    from component_fixtures import load_fixture
    from gather8 import install_gather_probe
    from torchtitan.components import loss as loss_module
    from torchtitan.config import CompileConfig
    from torchtitan.distributed.gfsdp import _collectives, _param_group
    from torchtitan.models.common import regularization
    from torchtitan.tools import utils

    candidate = args.arm != "baseline"
    source = CURRENT if candidate else replay.OLD
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert candidate or revision == BASELINE_SHA, revision
    modules = [loss_module, _collectives, _param_group, regularization, utils]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._inductor.config.triton.cudagraphs = False
    if hasattr(torch._inductor.config.triton, "mix_order_reduction_non_strict_mode"):
        torch._inductor.config.triton.mix_order_reduction_non_strict_mode = True
    torch.manual_seed(2048)

    # All processes map the same file. Only rank 0 streams it for identity verification.
    if rank == 0:
        assert file_sha(replay.FIXTURE) == FIXTURE_SHA, replay.FIXTURE
    head_capture = torch.load(replay.FIXTURE, map_location="cpu", weights_only=True, mmap=True)
    record = head_capture["ce"]
    backend = head_capture["metadata"]["compile_backend"]
    assert backend == "inductor" and record["n_chunks"] == 2
    assert record["hidden"].shape == (2, 2, 4096, WIDTH)
    assert record["weight"].shape == (VOCAB, WIDTH)
    assert record["head_weights"] == [1.0, 0.3] and record["ignore_index"] == -100
    hidden = record["hidden"].to(device).requires_grad_()
    labels = record["labels"].to(device)
    positions = record["positions"]
    if positions is not None:
        positions = positions.to(device)
    denominator = record["denominator"]
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.to(device)
    z_capture, z_receipt = load_fixture(replay.Z_FIXTURE, 0)
    z_hidden = z_capture["mtp"]["hidden"].to(device).requires_grad_()
    del z_capture
    assert z_hidden.dtype == torch.bfloat16 and z_hidden.shape[-1] == WIDTH and z_hidden.is_contiguous()

    head = make_distributed_head(replay, record["weight"])
    del record, head_capture
    head.set_force_sum_reduction_for_comms(True)
    (group,) = head._get_fsdp_state()._fsdp_param_groups
    assert group.gradient_divide_factor == 1.0 and group.force_sum_reduction_for_comms
    gather = install_gather_probe(head, args.arm == "alias")
    reduction = install_reduce_probe(rank)

    launcher = None
    if candidate:
        import fused
        import native_ce
        from torchtitan.distributed.gfsdp import _staged_head_dw, _staged_head_dw_writer
        from torchtitan.experiments.four_fusion import stable_ce

        modules.extend((_staged_head_dw, _staged_head_dw_writer, stable_ce))
        launcher = fused.Launcher("cuda-13")
        scale_checks, prepared = native_ce.install(launcher)

    with torch.no_grad():
        head(hidden[0, :, :1].contiguous())
    loss = loss_module.MTPLoss(
        loss_module.MTPLoss.Config(num_chunks=args.chunks, loss_weights=[0.3]),
        compile_config=CompileConfig(enable=True, components=["loss"], backend=backend),
    )
    loss.set_lm_head(head)
    assert loss.num_chunks == args.chunks
    ce_original = loss.fn
    ce_fn = ce_original._torchdynamo_orig_callable
    folded = getattr(loss, "_folded_ce", None) is not None
    direct_dx = getattr(loss, "_direct_bf16_dhidden", False)
    release_logits = getattr(loss, "_release_chunk_logits", False)
    assert folded == direct_dx == release_logits == candidate
    if candidate:
        assert loss._folded_ce_compiled and loss._folded_ce_consume is not None
        assert prepared == [backend]

    def invoke(reset_weight=False):
        hidden.grad = None
        z_hidden.grad = None
        if reset_weight:
            head.weight.grad = None
        loss.reset_components()
        targets = loss.pack_targets(labels, {"positions": positions})
        z_value = regularization.hidden_z_loss(z_hidden, coef=1e-7)
        value = loss(hidden, targets, denominator)
        torch.autograd.backward((value, z_value))
        return value.detach(), z_value.detach()

    seen_staging = []
    native_ce_calls = 0
    z_fusion_calls = 0

    def observe_stage(_head, _args, kwargs):
        seen_staging.append(kwargs.get("staged_head_dw") is not None)

    def counted_ce(*inputs, **kwargs):
        nonlocal native_ce_calls
        native_ce_calls += 1
        return ce_original(*inputs, **kwargs)

    if candidate:
        from torchtitan.ops import hidden_z_loss as fused_z

        modules.append(fused_z)
        original_z = fused_z.magnitude_logsumexp

        def counted_z(*inputs, **kwargs):
            nonlocal z_fusion_calls
            z_fusion_calls += 1
            return original_z(*inputs, **kwargs)

        fused_z.magnitude_logsumexp = counted_z

    loss.fn = counted_ce
    with head.register_forward_pre_hook(observe_stage, with_kwargs=True):
        invoke()
    loss.fn = ce_original
    if candidate:
        fused_z.magnitude_logsumexp = original_z
    assert z_fusion_calls == int(candidate)
    assert seen_staging == [candidate] * args.chunks, seen_staging
    assert native_ce_calls == (0 if candidate else 2 * args.chunks), native_ce_calls
    writer_receipt = None
    if candidate:
        assert launcher.calls == args.chunks
        writer = group._staged_head_dw_writer
        assert writer.pair == PAIRS[args.chunks], writer.pair
        writer_receipt = {"name": writer.name, "pair": writer.pair, "version": writer.version}

    for _ in range(args.warmup):
        invoke()
    torch.cuda.synchronize(device)
    hidden.grad = None
    z_hidden.grad = None
    torch.cuda.empty_cache()
    invoke()
    torch.cuda.synchronize(device)
    before_stats = torch.cuda.memory_stats(device)
    compile_before = dict(torch._dynamo.utils.counters["stats"])
    assert compile_before.get("unique_graphs", 0) > 0, compile_before
    elapsed_ms, submit_ms, wait_ms, total_gib = [], [], [], []
    for _ in range(args.samples):
        dist.barrier(device_ids=[rank])
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        invoke()
        submitted = time.perf_counter()
        torch.cuda.synchronize(device)
        finished = time.perf_counter()
        elapsed_ms.append((finished - start) * 1000)
        submit_ms.append((submitted - start) * 1000)
        wait_ms.append((finished - submitted) * 1000)
        assert torch.cuda.current_device() == rank
        total_gib.append(utils.device_memory_usage(torch.cuda.memory_stats(device))[1])
    after_stats = torch.cuda.memory_stats(device)
    allocator = {key: after_stats[key] - before_stats[key] for key in ("num_alloc_retries", "num_ooms")}
    assert not any(allocator.values()), allocator
    compile_after = dict(torch._dynamo.utils.counters["stats"])
    assert compile_after == compile_before, (compile_before, compile_after)

    value, z_value = invoke(reset_weight=True)
    torch.cuda.synchronize(device)
    with torch.no_grad():
        dw = head.weight.grad.to_local().view(VOCAB // RANKS, WIDTH)
        assert dw.dtype == torch.float32
        digests = {name: tensor_digest(tensor) for name, tensor in (
            ("loss", value), ("dx", hidden.grad), ("dw", dw),
            ("z_loss", z_value), ("z_dx", z_hidden.grad),
        )}

    # First route check, warmup, post-cache settling call, samples, and a reset call.
    invocations = args.warmup + args.samples + 3
    assert reduction["calls"] == invocations, reduction
    assert gather["native_all_gather_into_tensor_calls"] == gather["gather_calls"]
    assert gather["group_rank"] == rank and gather["group_size"] == RANKS
    assert gather["gather_calls"] == invocations + 1, gather
    allocator_settings = torch.cuda.memory._snapshot()["allocator_settings"]
    assert allocator_settings["expandable_segments"] == candidate, allocator_settings
    exact = None
    if candidate:
        assert launcher.calls == invocations * args.chunks, launcher.calls
        (incoming, expected), = scale_checks
        actual = torch.stack(incoming).float()
        assert torch.equal(actual, expected), (actual, expected)
        binary = Path(fused.__file__).with_name("centered.so")
        exact = {"method": launcher.name, "calls": launcher.calls, "incoming_fp32_scales": actual.tolist(),
                 "binary_sha256": file_sha(binary), "hook_sha256": file_sha(native_ce.__file__),
                 "launcher_sha256": file_sha(fused.__file__)}

    source_hashes = {}
    for module in modules:
        path = Path(module.__file__).resolve()
        source_hashes[str(path.relative_to(source))] = file_sha(path)
    return {
        "arm": args.arm, "rank": rank, "world_size": RANKS, "group_ranks": list(range(RANKS)),
        "chunks": args.chunks, "warmup": args.warmup, "samples": args.samples, "process_id": os.getpid(),
        "torchelastic_run_id": os.environ["TORCHELASTIC_RUN_ID"],
        "sample_iterations": list(range(args.warmup + 1, args.warmup + args.samples + 1)),
        "folded_rows_per_chunk": 16384 // args.chunks, "elapsed_ms": elapsed_ms,
        "host_submit_ms": submit_ms, "host_wait_ms": wait_ms, "total_gpu_gib": total_gib,
        "scope": "Host wall including native target packing, isolated head and z-loss forward/backward, collective submission and completion, and CUDA synchronization. Excludes start barrier, memory reads, and reset-output digest.",
        "accumulation": "Primed sharded weight gradient retained across timed calls",
        "reduction": "Eight-rank NCCL FP32 SUM reduce-scatter with gradient divide factor 1",
        "source": revision, "source_hashes": source_hashes, "capture_rank": 0,
        "fixture": str(replay.FIXTURE), "fixture_sha256": FIXTURE_SHA,
        "hidden_z_fixture_sha256": z_receipt["tensors_sha256"],
        "compile": {"backend": backend, "scope": "native loss", "counters": compile_after, "cuda_graphs": False},
        "device": {"torch": torch.__version__, "torch_git": torch.version.git_version,
                   "cuda": torch.version.cuda, "name": torch.cuda.get_device_name(device),
                   "rank": rank, "current_device": torch.cuda.current_device(),
                   "visible": ",".join(visible), "visible_uuid": visible[rank]},
        "activation": {"ce": f"{ce_fn.__module__}.{ce_fn.__name__}", "folded": folded,
                       "staged_chunk_calls": seen_staging, "direct_bf16_dhidden": direct_dx,
                       "early_logits_release": release_logits, "fused_hidden_z_loss": bool(z_fusion_calls),
                       "exact_ce_calls": 0 if exact is None else exact["calls"], "dw_writer": writer_receipt,
                       "gift_flags": {key: os.environ[key] for key in (*replay.SWITCHES,
                                      "GIFT_EXPERIMENT_LM_HEAD_MATRIX", "GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS")}},
        "candidate": exact, "ce_route_calls": {"native": native_ce_calls, "exact_setup": args.chunks if candidate else 0},
        "expandable_segments": allocator_settings["expandable_segments"], "allocator_retries_and_ooms": allocator,
        "digests": digests, "gather_probe": gather, "native_fp32_sum_reduce_scatter_calls": reduction["calls"],
        "harness_sha256": file_sha(__file__), "historical_replay_sha256": file_sha(replay.__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "regular", "alias"), required=True)
    parser.add_argument("--chunks", type=int, choices=(2, 8), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if not args.out.is_dir() or args.warmup < 0 or args.samples < 1:
        parser.error("--out must exist, --warmup must be nonnegative, and --samples must be positive")
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == int(os.environ["LOCAL_WORLD_SIZE"]) == RANKS
    assert rank == int(os.environ["LOCAL_RANK"]) and 0 <= rank < RANKS
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert len(visible) == len(set(visible)) == RANKS and all(uuid.startswith("GPU-") for uuid in visible)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    logging.basicConfig(level=logging.INFO)

    sys.path[:0] = [str(PRIOR), str(TASK / "runs/production-data-abcd")]
    import replay

    candidate = args.arm != "baseline"
    source = CURRENT if candidate else replay.OLD
    sys.path[:0] = [str(source / "gift"), str(ROOT / "native-ce-finalist"),
                    str(ROOT / "layout-order"), str(ROOT / "fused-ce")]
    os.environ.update({name: str(int(candidate)) for name in replay.SWITCHES})
    os.environ["GIFT_EXPERIMENT_LM_HEAD_MATRIX"] = "new" if candidate else ""
    os.environ["GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS"] = str(args.chunks) if candidate else ""
    os.environ["FUSED_CE_BINARY"] = "centered.so"
    dist.init_process_group("nccl", device_id=device)
    try:
        assert (dist.get_rank(), dist.get_world_size()) == (rank, RANKS)
        receipt = run(args, rank, device, visible)
        path = args.out / f"rank-{rank}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.rename(path)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
