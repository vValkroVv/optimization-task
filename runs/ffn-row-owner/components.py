"""Actual GFSDP/row-owner embedding components; CUDA execution is explicit.

The plan/parser path imports no torch and never initializes a device. Each
variant runs in a fresh eight-rank process group; allocator caches are retained
within a run. Diagnostic stages are collected separately from integrated timing.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VOCAB, WIDTH, WORLD, BATCH, SEQUENCE = 128256, 4096, 8, 2, 4096
CASES = ("fixture", "duplicates", "skew", "empty-requester", "empty-all", "boundaries")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    if not values:
        raise ValueError("cannot summarize an empty measurement")
    return {"samples": values, "median": statistics.median(values),
            "min": min(values), "max": max(values), "spread": max(values) - min(values)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute", action="store_true", help="explicitly run authorized CUDA/NCCL work")
    result.add_argument("--variant", choices=("baseline", "owner"), default="owner")
    result.add_argument("--scope", choices=("integrated", "phases", "stages"), default="integrated")
    result.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--samples", type=int, default=3)
    result.add_argument("--seed", type=int, default=2048)
    result.add_argument("--input-fixtures", type=Path, help="native captured rank-local IDs; replaces historical fixed MDS rows")
    result.add_argument("--fixture", type=Path, default=ROOT / "runs/overnight-streaming-perf/artifacts/raw/mds4096/train")
    result.add_argument("--output", type=Path, default=ROOT / "runs/ffn-row-owner/artifacts/components")
    result.add_argument("--profile", action="store_true", help="separate one-sample/all-rank diagnostic trace; requires diagnostic scope")
    return result


def validate(args):
    if args.warmup < 1 or not 1 <= args.samples <= 20:
        raise ValueError("require warmup >= 1 and 1 <= samples <= 20")
    if args.scope == "stages" and args.variant != "owner":
        raise ValueError("stage observer exists only on owner; baseline integrated/profile kernels remain available separately")
    if args.profile and args.scope == "integrated":
        raise ValueError("profiling must be separate from integrated timing")
    if args.input_fixtures is not None and "fixture" not in args.cases:
        raise ValueError("--input-fixtures requires the fixture case")
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("duplicate cases would repeat the experiment")


def plan(args):
    validate(args)
    index = args.fixture / "index.json"
    return {"execution": bool(args.execute), "variant": args.variant, "scope": args.scope,
            "shape": {"vocabulary": VOCAB, "width": WIDTH, "batch": BATCH, "sequence": SEQUENCE},
            "topology": "eight ranks, one dense NCCL shard group; H100 only",
            "warmup": args.warmup, "samples": args.samples, "cases": args.cases,
            "seed": args.seed, "input_fixtures": str(args.input_fixtures) if args.input_fixtures else None,
            "fixture_index": str(index) if args.input_fixtures is None else None,
            "fixture_index_sha256": sha256(index) if args.input_fixtures is None and index.exists() else None,
            "cache": "fresh process per variant; no row cache; allocator retained between cases",
            "completion": "FP32 sharded masters -> BF16 lookup -> completed dense accumulated FP32 shard gradient",
            "timers": "CUDA event elapsed spans and host call/end-sync wall spans; not kernel-duration sums",
            "stage_semantics": "separate observer capture on actual stream, no per-stage synchronization; spans include queue gaps",
            "temporary_bytes": {"native_bf16_full_gradient": VOCAB * WIDTH * 2,
                                "potential_native_fp32_scratch": VOCAB * WIDTH * 4},
            "resident_bytes": {"private_lm_head_gradient_buffer": VOCAB * WIDTH * 4},
            "memory_scope": "both variants keep the actual untied FP32 lm_head GFSDP unit and its private gradient buffer resident; owner scratch aliases that storage",
            "ffn_gate": "separate native FP8 exact-SAC factory owner; no toy replacement in this runner"}


def fixture_rows(path, rank):
    """Read actual raw MDS rows without a streaming loader, locks or downloads."""
    import numpy as np
    import torch
    from streaming.base.local import LocalDataset

    dataset = LocalDataset(local=str(path))
    rows = []
    for index in (2 * rank, 2 * rank + 1):
        item = dataset[index]
        # The fixture producer records little-endian uint32 token bytes.
        tokens = np.frombuffer(item["tokens"], dtype="<u4").astype(np.int64)
        if tokens.shape != (SEQUENCE,):
            raise ValueError(f"MDS row {index} has unsupported shape {tokens.shape}")
        rows.append(torch.from_numpy(tokens.copy()))
    return torch.stack(rows)


def case_ids(case, rank, fixture, *, vocab=VOCAB, sequence=SEQUENCE, world=WORLD):
    import torch

    n = BATCH * sequence
    if case == "fixture":
        return fixture.clone()
    if case == "duplicates":
        return (torch.arange(n, dtype=torch.int64) % 16).view(BATCH, sequence)
    if case == "skew":
        # All requesters mostly hit owner zero, with a small rank-dependent tail.
        ids = torch.arange(n, dtype=torch.int64) % (vocab // world)
        ids[::32] = (ids[::32] + rank * (vocab // world)) % vocab
        return ids.view(BATCH, sequence)
    if case == "empty-requester":
        return torch.empty((0, sequence), dtype=torch.int64) if rank else torch.zeros((BATCH, sequence), dtype=torch.int64)
    if case == "empty-all":
        return torch.empty((0, sequence), dtype=torch.int64)
    if case == "boundaries":
        choices = torch.tensor([0, vocab - 1, vocab // world - 1, vocab // world], dtype=torch.int64)
        return choices[torch.arange(n) % 4].view(BATCH, sequence)
    raise ValueError(case)


def traffic(ids, world=WORLD, vocab=VOCAB, width=WIDTH):
    """CPU-only input census, outside timing. Bytes are sender payloads, not bus traffic."""
    values = ids.reshape(-1).tolist()
    unique = sorted(set(values))
    if any(i < 0 or i >= vocab for i in values):
        raise ValueError("out-of-range fixture ID")
    counts = [sum(i // (vocab // world) == owner for i in unique) for owner in range(world)]
    return {"tokens": len(values), "unique_requests": len(unique), "unique_requests_by_owner": counts,
            "request_ids_bytes": len(unique) * 8, "value_return_bytes": len(unique) * width * 2,
            "gradient_send_bytes": len(unique) * width * 2, "payload_padding_bytes": 0,
            "count_metadata_local_bytes": (world + 1) * 8,
            "count_allgather_output_bytes": world * (world + 1) * 8,
            "explicit_count_host_reads_per_forward": 1,
            "count_host_read_bytes": world * (world + 1) * 8,
            "host_read_scope": "matrix.cpu plus CPU split/validity read; measured within counts stage"}


class StageObserver:
    """Reusable diagnostic contexts; record on the stream running the stage.

    No context synchronizes. Call collect only after outer operation completion.
    The same context factory can mark additional native block stages.
    """
    def __init__(self):
        self.records = []

    @contextmanager
    def __call__(self, name):
        import torch

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        host = time.perf_counter()
        with torch.profiler.record_function(f"components::{name}"):
            start.record(torch.cuda.current_stream())
            try:
                yield
            finally:
                end.record(torch.cuda.current_stream())
                self.records.append((name, start, end, (time.perf_counter() - host) * 1000))

    def collect(self):
        result = [{"stage": name, "cuda_elapsed_span_ms": start.elapsed_time(end),
                   "host_context_call_ms": wall} for name, start, end, wall in self.records]
        self.records.clear()
        return result


def build_embedding(mesh, variant, *, vocab=VOCAB, width=WIDTH, seed=2048):
    import torch
    from torch import nn
    from torchtitan.distributed.gfsdp import fully_shard, MixedPrecisionPolicy
    from torchtitan.distributed.gfsdp._state import _get_module_fsdp_state
    from torchtitan.distributed.gfsdp.row_owner_embedding import RowOwnerEmbedding

    class Stem(nn.Module):
        def __init__(self, embedding, head):
            super().__init__()
            self.tok_embeddings = embedding
            self.lm_head = head

        def forward(self, ids):
            return self.tok_embeddings(ids)

    torch.manual_seed(seed)
    embedding = nn.Embedding(vocab, width, device=mesh.device_type, dtype=torch.float32)
    # Equivalent full initialization on every rank, then identical complete-row
    # shards. This setup allocation is excluded from operation memory peaks.
    nn.init.xavier_normal_(embedding.weight)
    # The native crop already keeps this separate FP32 reduction bucket for
    # chunked CE. Include the same actual head wrapper in both variants so
    # owner scratch borrows existing storage; never create an embedding pool.
    head = nn.Linear(width, vocab, bias=False, device=mesh.device_type, dtype=torch.float32)
    fully_shard(head, mesh=mesh, reshard_after_forward=True,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
    head.set_own_grad_buffer()
    if variant == "owner":
        owner = RowOwnerEmbedding.from_embedding(embedding, mesh)
        del embedding
        model = Stem(owner, head)
        fully_shard(model, mesh=mesh, ignored_params={owner.weight}, reshard_after_forward=True)
        owner.attach_comm_context(model)
    else:
        model = Stem(embedding, head)
        fully_shard(embedding, mesh=mesh, reshard_after_forward=True,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
        # The real stem is followed by trunk units and is not the final shared
        # buffer user. Root auto mode on a standalone stem would retain its
        # table and incorrectly remove the baseline's backward all-gather.
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        embedding.set_gradient_divide_factor(1.0)
    state = _get_module_fsdp_state(model)
    state._lazy_init()
    if variant == "baseline":
        (group,) = _get_module_fsdp_state(model.tok_embeddings)._fsdp_param_groups
        if group.post_forward_mesh_info is None:
            raise RuntimeError("baseline unexpectedly retained the full table after forward")
    return model, state._comm_ctx


def gradient_buffer_inventory(model):
    """Count physical GFSDP gradient allocations once, despite aliased regions."""
    allocations = {}
    key_to_id = {}
    for module_path, module in model.named_modules():
        get_state = getattr(module, "_get_fsdp_state", None)
        if get_state is None:
            continue
        state = get_state()
        for group_index, group in enumerate(state._fsdp_param_groups):
            context = group.grad_buffer_ctx
            if context is None:
                continue
            for dtype, view in context.buffers.items():
                storage = view.untyped_storage()
                key = (str(view.device), storage.data_ptr(), storage.nbytes())
                if key not in allocations:
                    storage_id = f"gradient-storage-{len(allocations)}"
                    key_to_id[key] = storage_id
                    allocations[key] = {"storage_id": storage_id,
                                        "device": str(view.device),
                                        "storage_bytes": storage.nbytes(),
                                        "aliases": []}
                allocation = allocations[key]
                alias = {"module": module_path or "<root>",
                         "group_index": group_index,
                         "dtype": str(dtype),
                         "view_bytes": view.numel() * view.element_size(),
                         "private_unit_buffer": bool(state._own_grad_buffer)}
                if alias not in allocation["aliases"]:
                    allocation["aliases"].append(alias)

    head_state = model.lm_head._get_fsdp_state()
    head_contexts = {
        group.grad_buffer_ctx._storage_ctx
        for group in head_state._fsdp_param_groups
        if group.grad_buffer_ctx is not None
    }
    head_candidates = [buffer for context in head_contexts
                       for buffer in context.buffers.values()]
    if not head_state._own_grad_buffer or not head_candidates:
        raise RuntimeError("lm_head must own its initialized private GFSDP gradient buffer")
    head_buffer = max(head_candidates, key=lambda buffer: buffer.numel())
    head_storage = head_buffer.untyped_storage()
    head_key = (str(head_buffer.device), head_storage.data_ptr(), head_storage.nbytes())
    head_storage_id = key_to_id[head_key]

    owner_storage_id = None
    owner = model.tok_embeddings
    if hasattr(owner, "_sync_groups"):
        contexts = {
            group.grad_buffer_ctx._storage_ctx
            for group in owner._sync_groups
            if group.grad_buffer_ctx is not None
            and hasattr(group.grad_buffer_ctx, "buffers")
        }
        candidates = [(buffer.numel(), context, buffer)
                      for context in contexts for buffer in context.buffers.values()]
        if not candidates:
            raise RuntimeError("row-owner scratch has no existing GFSDP gradient storage")
        _, _, selected = max(candidates, key=lambda item: item[0])
        selected_storage = selected.untyped_storage()
        selected_key = (str(selected.device), selected_storage.data_ptr(), selected_storage.nbytes())
        owner_storage_id = key_to_id[selected_key]

    return {"unique_storage_count": len(allocations),
            "unique_storage_bytes": sum(item["storage_bytes"] for item in allocations.values()),
            "allocations": list(allocations.values()),
            "private_lm_head_storage_id": head_storage_id,
            "private_lm_head_storage_bytes": head_storage.nbytes(),
            "owner_scratch_storage_id": owner_storage_id,
            "owner_scratch_aliases_private_lm_head": (
                owner_storage_id == head_storage_id if owner_storage_id is not None else None)}


def join_streams(ctx):
    """Join all participating GFSDP streams before recording a phase endpoint."""
    import torch

    current = torch.cuda.current_stream()
    streams = {ctx.all_gather_stream, ctx.reduce_scatter_stream}
    for stream in streams:
        current.wait_event(stream.record_event())


def timed_call(fn, ctx):
    import torch

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    host_start = time.perf_counter()
    start.record()
    # Explicitly include a stream's work after this operation's start.
    for stream in {ctx.all_gather_stream, ctx.reduce_scatter_stream}:
        stream.wait_event(start)
    output = fn()
    submitted = time.perf_counter()
    join_streams(ctx)
    end.record()
    sync_start = time.perf_counter()
    end.synchronize()
    finished = time.perf_counter()
    return output, {"cuda_elapsed_span_ms": start.elapsed_time(end),
                    "host_call_ms": (submitted - host_start) * 1000,
                    "host_end_sync_ms": (finished - sync_start) * 1000,
                    "synchronized_wall_ms": (finished - host_start) * 1000,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def execute(args):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from torch.distributed.device_mesh import init_device_mesh

    if int(os.environ.get("WORLD_SIZE", 0)) != WORLD:
        raise ValueError("production component execution requires torchrun with exactly eight ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if "H100" not in torch.cuda.get_device_name():
        raise ValueError("this measurement plan supports the agreed H100 crop only")
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    args.output.mkdir(parents=True, exist_ok=True)
    report = plan(args)
    report.update({"rank": rank, "command": sys.argv, "runtime": {
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "nccl": list(torch.cuda.nccl.version()), "gpu": torch.cuda.get_device_name(),
        "hostname": os.uname().nodename}, "source": {
        "revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "harness_sha256": sha256(__file__),
        "owner_sha256": sha256(ROOT / "gift/torchtitan/distributed/gfsdp/row_owner_embedding.py"),
        "gift_diff_sha256": hashlib.sha256(subprocess.check_output([
            "git", "-C", str(ROOT), "diff", "--binary", "--", "gift"])).hexdigest(),
        "gfsdp_file_sha256": {p.name: sha256(p) for p in sorted((ROOT / "gift/torchtitan/distributed/gfsdp").glob("*.py"))}},
        "attribution": {"kernel_compute": "unavailable without diagnostic trace analysis",
                        "copies_and_collectives": "unavailable without diagnostic trace analysis",
                        "host_work": "host call span includes dispatcher, submission and internal waits; no subtraction attribution",
                        "synchronization": "explicit endpoint synchronize span; internal count wait is inside call/stage",
        "memory": "phase absolute peaks include the resident private lm_head GFSDP gradient buffer; backing-storage inventory deduplicates aliases; allocator cache and initialization reservations retained; resets exclude gradient zeroing"},
        "results": []})
    try:
        fixture_receipt = None
        if "fixture" in args.cases and args.input_fixtures is not None:
            import importlib.util

            path = ROOT / "runs/production-data-abcd/component_fixtures.py"
            spec = importlib.util.spec_from_file_location("component_fixtures", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            captured, fixture_receipt = module.load_fixture(args.input_fixtures, rank)
            fixture = captured["input_ids"]
            del captured
        else:
            fixture = fixture_rows(args.fixture, rank) if "fixture" in args.cases else None
        report["input_fixture"] = fixture_receipt
        mesh = init_device_mesh("cuda", (WORLD,), mesh_dim_names=("fsdp",))
        model, ctx = build_embedding(mesh, args.variant, seed=args.seed)
        gradient_buffers = gradient_buffer_inventory(model)
        expected_head_buffer_bytes = VOCAB * WIDTH * 4
        if gradient_buffers["private_lm_head_storage_bytes"] != expected_head_buffer_bytes:
            raise RuntimeError(
                "private lm_head gradient storage has unexpected capacity: "
                f"{gradient_buffers['private_lm_head_storage_bytes']} != {expected_head_buffer_bytes}"
            )
        if args.variant == "owner" and not gradient_buffers["owner_scratch_aliases_private_lm_head"]:
            raise RuntimeError("owner scratch did not select the existing private lm_head gradient storage")
        head_weight = model.lm_head.weight
        if tuple(head_weight.shape) != (VOCAB, WIDTH) or head_weight.dtype != torch.float32:
            raise RuntimeError("component root must retain an untied FP32 [V,D] lm_head weight")
        report["gfsdp_gradient_buffer_inventory"] = gradient_buffers
        report["resident_lm_head"] = {"weight_global_shape": list(head_weight.shape),
                                       "weight_dtype": str(head_weight.dtype),
                                       "forward_computed": False,
                                       "private_gradient_buffer_resident_in_both_variants": True}
        report["mesh_ranks"] = mesh.mesh.tolist()
        report["activation"] = type(model.tok_embeddings).__name__
        for case in args.cases:
            cpu_ids = case_ids(case, rank, fixture)
            census = traffic(cpu_ids)
            census["input_ids_int64_sha256"] = hashlib.sha256(cpu_ids.numpy().tobytes()).hexdigest()
            ids = cpu_ids.cuda()
            generator = torch.Generator(device="cuda").manual_seed(args.seed + rank)
            gradient = torch.randn((*ids.shape, WIDTH), device="cuda", dtype=torch.bfloat16, generator=generator)
            observer = StageObserver() if args.scope == "stages" else None
            if args.variant == "owner":
                model.tok_embeddings.stage_observer = observer
            # Warm the actual forward/backward and establish an allocated FP32
            # accumulated gradient. Reset its values, not its existence.
            for _ in range(args.warmup):
                model(ids).backward(gradient)
            torch.cuda.synchronize()
            if observer:
                observer.records.clear()
            records = []
            for sample in range(args.samples):
                model.tok_embeddings.weight.grad.zero_()
                torch.cuda.synchronize()
                dist.barrier()  # between samples only, never inside a stage
                torch.cuda.reset_peak_memory_stats()
                starting = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}
                if args.scope == "phases":
                    output, forward = timed_call(lambda: model(ids), ctx)
                    torch.cuda.reset_peak_memory_stats()
                    _, backward = timed_call(lambda: output.backward(gradient), ctx)
                    measurements = {"forward": forward, "backward": backward}
                    del output
                else:
                    _, complete = timed_call(lambda: model(ids).backward(gradient), ctx)
                    measurements = {"complete": complete}
                parameter = model.tok_embeddings.weight
                if parameter.grad is None or parameter.grad.dtype != torch.float32:
                    raise RuntimeError("missing completed dense FP32 owner/GFSDP gradient")
                records.append({"sample": sample, "starting_bytes": starting,
                                **measurements,
                                "stages": observer.collect() if observer else []})
            if args.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    model(ids).backward(gradient)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.output / f"{args.variant}-{case}-rank{rank}.trace.json"))
                observer.records.clear()
            phases = ("forward", "backward") if args.scope == "phases" else ("complete",)
            distributions = {phase: {key: summary([sample[phase][key] for sample in records])
                                    for key in records[0][phase]} for phase in phases}
            report["results"].append({"case": case, "owner_route_census": census,
                                      "baseline_logical_collective_elements": {
                                          "forward_allgather_bf16": VOCAB * WIDTH,
                                          "backward_allgather_bf16": VOCAB * WIDTH,
                                          "backward_reduce_scatter_fp32": VOCAB * WIDTH},
                                      "distributions": distributions, "measurements": records})
        # Serialize only compact measurements, never full tables or gradients.
        all_reports = [None] * WORLD
        dist.all_gather_object(all_reports, report)
        (args.output / f"{args.variant}-{args.scope}-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        if rank == 0:
            aggregate = {"ranks": all_reports, "slowest_rank_by_sample": []}
            for case_index, case in enumerate(args.cases):
                entry = {"case": case}
                phases = ("forward", "backward") if args.scope == "phases" else ("complete",)
                for phase in phases:
                    for timer in ("cuda_elapsed_span_ms", "synchronized_wall_ms"):
                        values = [max(r["results"][case_index]["measurements"][s][phase][timer]
                                      for r in all_reports) for s in range(args.samples)]
                        entry[f"{phase}_{timer}"] = summary(values)
                aggregate["slowest_rank_by_sample"].append(entry)
            (args.output / f"{args.variant}-{args.scope}-all-ranks.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        (args.output / f"{args.variant}-{args.scope}-rank{rank}-failed.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        dist.destroy_process_group()


def main(argv=None):
    args = parser().parse_args(argv)
    validate(args)
    if args.execute:
        execute(args)
    else:
        print(json.dumps(plan(args), indent=2))


if __name__ == "__main__":
    main()
