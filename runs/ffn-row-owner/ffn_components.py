#!/usr/bin/env python3
"""Native crop FFN component checks. GPU execution requires eight authorized ranks."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import statistics
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
GIFT = ROOT / "gift"
COMPILER_CACHE_AREA = ROOT / "runs/ffn-row-owner/artifacts/raw/ffn-components"
COMPILER_CACHE_DIRS = (
    ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
    ("TRITON_CACHE_DIR", "triton"),
    ("TILELANG_CACHE_DIR", "tilelang"),
    ("TILELANG_TMP_DIR", "tilelang-tmp"),
)
sys.path.insert(0, str(GIFT))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--component", choices=("layer0", "mtp"), required=True)
    p.add_argument(
        "--mode", choices=("plan", "check", "benchmark", "diagnostic"), default="plan"
    )
    p.add_argument("--variant", choices=("control", "split"), default="control")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compiler-cache-root", type=Path)
    p.add_argument("--input-fixtures", type=Path, help="hashed native per-rank component captures")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--accumulation", type=int, choices=(1, 32), default=1)
    p.add_argument("--trace-microbatches", type=int, choices=(1, 2, 3), default=3)
    a = p.parse_args(argv)
    if a.warmup < 1 or a.samples < 1:
        p.error("warmup and samples must be positive")
    if a.mode != "check" and a.accumulation != 1:
        p.error("--accumulation 32 is supported by --mode check only")
    if a.input_fixtures is not None and a.mode == "check":
        p.error("captured fixtures refresh timings only; numeric check is outside this campaign")
    if a.mode == "diagnostic" and a.variant != "split":
        p.error("diagnostic requires --variant split")
    return a


def configure_compiler_caches(output, rank, compiler_cache_root=None):
    if compiler_cache_root is None:
        selected_root = output / "cache"
        rank_root = selected_root / f"rank{rank}"
        policy = "fresh_per_output"
        paths = {}
        for env, suffix in COMPILER_CACHE_DIRS:
            path = rank_root / suffix
            path.mkdir(parents=True, exist_ok=True)
            paths[env] = path.resolve()
    else:
        if not compiler_cache_root.is_absolute():
            raise ValueError("--compiler-cache-root must be an absolute path")
        allowed_root = COMPILER_CACHE_AREA.resolve(strict=True)
        try:
            selected_root = compiler_cache_root.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"--compiler-cache-root does not exist: {compiler_cache_root}"
            ) from exc
        if not selected_root.is_dir():
            raise NotADirectoryError(
                f"--compiler-cache-root is not a directory: {selected_root}"
            )
        if not selected_root.is_relative_to(allowed_root):
            raise ValueError(
                "--compiler-cache-root must remain inside this task's FFN artifact area"
            )
        rank_root = selected_root / f"rank{rank}"
        policy = "reuse_existing"
        paths = {}
        for env, suffix in COMPILER_CACHE_DIRS:
            path = rank_root / suffix
            if not path.is_dir():
                raise FileNotFoundError(f"compiler cache directory is missing: {path}")
            resolved_path = path.resolve(strict=True)
            if not resolved_path.is_relative_to(selected_root):
                raise ValueError(f"compiler cache directory escapes its root: {path}")
            paths[env] = resolved_path

    for env, path in paths.items():
        os.environ[env] = str(path)
    return {
        "policy": policy,
        "root": str(selected_root.resolve()),
        "rank": rank,
        "directories": {env: str(path) for env, path in paths.items()},
    }


def resolved_config(component, split=False):
    from torchtitan.config.cli import load_trainer_config

    previous_world = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = "8"
    try:
        loaded = load_trainer_config(
            [
                str(
                    GIFT
                    / "configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml"
                ),
                "--parallelism.gfsdp_reuse_bf16_expert_weights",
                "true",
                "--dataloader.num_prefetch_batches",
                "32",
                "--dataloader.dataset.epoch_size",
                "102400",
            ],
            logger=logging.getLogger("ffn-components"),
        )
    finally:
        if previous_world is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = previous_world
    cfg = loaded.trainer_config
    cfg.parallelism.gfsdp_split_layer0_ffn = split and component == "layer0"
    cfg.parallelism.gfsdp_split_mtp_ffn = split and component == "mtp"
    if getattr(cfg.parallelism, "gfsdp_row_owner_embeddings", False):
        raise ValueError("FFN components require row-owner embeddings disabled")
    model_cfg = cfg.model_spec.model
    model_cfg.owner().prepare_config(model_cfg, config=cfg)
    return cfg


def meta_component(cfg, component):
    """Construct native root/RoPE, selecting exact converted placed blocks before build.

    Unrelated tables are constructed on meta then removed before materialization.
    No MoE module/dispatcher is constructed. Original position-dependent init and
    source FQNs survive; this is a component allocation scope, not a model preset.
    """
    import torch
    from torchtitan.tools.utils import TORCH_DTYPE_MAP, set_default_dtype

    selected = copy.deepcopy(cfg.model_spec.model)
    selected.layers = [selected.placed_layers()[0]]
    if component == "layer0":
        selected.mtp = None
    with torch.device("meta"), set_default_dtype(TORCH_DTYPE_MAP[cfg.training.dtype]):
        model = selected.build()
    model.tok_embeddings = model.norm = model.lm_head = None
    if component == "mtp":
        model.layers.clear()

    def forward(
        self,
        tokens,
        input_embeds=None,
        attention_masks=None,
        positions=None,
        token_mask=None,
    ):
        if component == "layer0":
            return self.layers["0"](
                tokens, self.freqs_cis, attention_masks, positions, token_mask
            )
        # Native MTP performs the shared embedding shift and zero-fills its tail.
        return self.mtp(
            tokens, input_embeds, self.freqs_cis, attention_masks, positions, token_mask
        )[0]

    model.forward = MethodType(forward, model)
    return model


def contract(cfg, model, component):
    from torchtitan.components.quantization.utils import has_quantization

    unit = model.layers["0"] if component == "layer0" else model.mtp.layers["0"]
    block = getattr(unit, "block", unit)
    ffn = block.feed_forward
    shapes = {n: list(p.shape) for n, p in ffn.named_parameters()}
    if (
        cfg.model_spec.model.dim != 4096
        or cfg.training.seq_len != 4096
        or cfg.training.local_batch_size != 2
    ):
        raise ValueError("component contract requires B2 S4096 D4096")
    if (
        not cfg.compile.enable
        or cfg.compile.backend != "inductor"
        or "model" not in cfg.compile.components
    ):
        raise ValueError("native Inductor compilation must remain enabled")
    if not has_quantization(cfg.model_spec.model):
        raise ValueError("native FP8 conversion did not activate")
    if ffn.w1.weight.numel() != 4096 * 10944:
        raise ValueError(f"unexpected dense FFN shapes: {shapes}")
    return {
        "component": component,
        "shape": [2, 4096, 4096],
        "ffn_shapes": shapes,
        "ffn_types": {
            n: type(getattr(ffn, n)).__module__ + "." + type(getattr(ffn, n)).__name__
            for n in ("w1", "w2", "w3")
        },
        "parameters": sum(p.numel() for p in model.parameters()),
        "segments": [s.fqn for s in model.named_segments() if s.is_block],
        "sac": cfg.to_dict()["activation_checkpoint"],
        "precision": {
            "master": cfg.training.dtype,
            "compute": cfg.training.mixed_precision_param,
            "reduce": cfg.training.mixed_precision_reduce,
        },
        "compile": cfg.to_dict()["compile"],
        "scope": "selected native dense block plus parameter-free root/RoPE; no embedding, head, other trunk blocks or experts",
    }


def build(cfg, component, dims, device):
    import torch
    from torchtitan.distributed.backends.gfsdp2 import GFSDP2Backend

    from torchtitan.distributed.utils import set_determinism

    set_determinism(dims, device, cfg.debug, distinct_seed_mesh_dims=["pp"])
    torch.manual_seed(2048)
    model = meta_component(cfg, component)
    cfg.model_spec.parallelize_fn(
        model,
        parallel_dims=dims,
        training=cfg.training,
        parallelism=cfg.parallelism,
        compile_config=cfg.compile,
        ac_config=cfg.activation_checkpoint,
        dump_folder=str(cfg.dump_folder),
    )
    model.to_empty(device=device)
    backend = GFSDP2Backend(match_fsdp2_init=cfg.parallelism.gfsdp_match_fsdp2_init)
    with torch.no_grad(), backend.weight_init_context(model):
        model.init_weights(buffer_device=device)
    model.train()
    model.lazy_init()
    return model


def fixture_module():
    import importlib.util

    path = ROOT / "runs/production-data-abcd/component_fixtures.py"
    spec = importlib.util.spec_from_file_location("component_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def captured_inputs(path, device, rank, component):
    module = fixture_module()
    records, receipt = module.load_fixture(path, rank)
    row = records[component]
    hidden = row["hidden"].to(device).detach().requires_grad_()
    embedding = row["embedding"]
    if embedding is not None:
        embedding = embedding.to(device).detach().requires_grad_()
    return (hidden, embedding, row["upstream"].to(device),
            module.restore(row["positions"], device),
            module.restore(row["attention_masks"], device),
            module.restore(row["token_mask"], device)), receipt


def inputs(device, rank, microbatch, component):
    import torch

    gen = torch.Generator(device=device).manual_seed(2048 + rank * 1000 + microbatch)
    shape = (2, 4096, 4096)
    hidden = torch.randn(
        shape, generator=gen, device=device, dtype=torch.bfloat16
    ).requires_grad_()
    embedding = (
        torch.randn(
            shape, generator=gen, device=device, dtype=torch.bfloat16
        ).requires_grad_()
        if component == "mtp"
        else None
    )
    upstream = (
        torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16) / 32
    )
    positions = (
        torch.arange(4096, device=device, dtype=torch.int64).expand(2, -1).contiguous()
    )
    return hidden, embedding, upstream, positions


def microbatch(model, component, tensors):
    hidden, embedding, upstream, positions = tensors[:4]
    hidden.grad = None
    if embedding is not None:
        embedding.grad = None
    if len(tensors) == 6:
        masks, token_mask = tensors[4:]
    else:
        masks = token_mask = None
        from torchtitan.config.schema.models.common.attention.const import AttentionMaskKind

        if model.attention_config.attention_mask_kind != AttentionMaskKind.NONE:
            masks = model.get_attention_masks(positions)
    out = model(
        hidden,
        embedding if component == "mtp" else None,
        positions=positions,
        attention_masks=masks,
        token_mask=token_mask,
    )
    out.backward(upstream)
    return out


def canonical(name):
    return name.replace("_checkpoint_wrapped_module.", "")


def cpu_tensor(tensor):
    import torch

    with torch.no_grad():
        return (
            (tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor)
            .detach()
            .cpu()
            .clone()
        )


def snapshot(model, tensors=None, out=None):
    result = {}
    for name, param in model.named_parameters():
        key = canonical(name)
        result["weight/" + key] = cpu_tensor(param)
        if tensors is not None:
            if param.grad is None:
                raise RuntimeError(f"missing gradient: {key}")
            result["grad/" + key] = cpu_tensor(param.grad)
    if tensors is not None:
        result["output"] = cpu_tensor(out)
        result["input/hidden"] = cpu_tensor(tensors[0].grad)
        if tensors[1] is not None and tensors[1].grad is not None:
            result["input/embedding"] = cpu_tensor(tensors[1].grad)
    return result


def errors(reference, actual):
    import torch

    if reference.keys() != actual.keys():
        raise RuntimeError("different tensor/FQN coverage")
    result = {}
    for name, value in actual.items():
        ref = reference[name]
        if ref.shape != value.shape:
            raise RuntimeError(f"different shape for {name}")
        # Chunking bounds CPU temporary memory for native-width parameter tensors.
        max_abs = sum_sq = ref_sq = 0.0
        unequal = nonfinite = 0
        for a, b in zip(
            ref.reshape(-1).split(1048576), value.reshape(-1).split(1048576)
        ):
            a, b = a.double(), b.double()
            delta = a - b
            nonfinite += int((~torch.isfinite(a) | ~torch.isfinite(b)).sum())
            unequal += int((a != b).sum())
            max_abs = max(max_abs, float(delta.abs().max()))
            sum_sq += float(delta.square().sum())
            ref_sq += float(a.square().sum())
        result[name] = {
            "shape": list(ref.shape),
            "dtype": str(ref.dtype),
            "max_abs": max_abs,
            "rms": (sum_sq / ref.numel()) ** 0.5,
            "relative_l2": (sum_sq / ref_sq) ** 0.5 if ref_sq else None,
            "unequal": unequal,
            "nonfinite": nonfinite,
        }
    return result


def buffer_inventory(model):
    """Count unique storage bytes, retaining view offsets to audit split regions."""
    from torchtitan.distributed.gfsdp import GFSDPModule

    rows = []
    storages = {}
    for name, module in model.named_modules():
        if not isinstance(module, GFSDPModule):
            continue
        for group in module._get_fsdp_state()._fsdp_param_groups:
            for kind, context in (
                ("data", group.data_buffer_ctx),
                ("grad", group.grad_buffer_ctx),
            ):
                if context is None:
                    continue
                for dtype, tensor in context.buffers.items():
                    storage = tensor.untyped_storage()
                    key = str(storage.data_ptr())
                    storages[key] = storage.nbytes()
                    rows.append(
                        {
                            "fqn": canonical(name),
                            "kind": kind,
                            "dtype": str(dtype),
                            "storage": key,
                            "storage_bytes": storage.nbytes(),
                            "offset_bytes": tensor.storage_offset()
                            * tensor.element_size(),
                            "view_bytes": tensor.numel() * tensor.element_size(),
                        }
                    )
    return {
        "unique_storage_bytes": sum(storages.values()),
        "views": rows,
        "scope": "native GFSDP all-gather/reduce-scatter contexts; aliases counted once; compare control/split and inspect interval overlap",
    }


def optimizer_snapshot(optimizer):
    import torch

    result = {}

    def walk(value, path):
        if isinstance(value, torch.Tensor):
            result[canonical(path)] = cpu_tensor(value)
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}/{key}")
        elif isinstance(value, (tuple, list)):
            for key, child in enumerate(value):
                walk(child, f"{path}/{key}")

    walk(optimizer.state_dict(), "optimizer")
    return result


def optimizer_step(cfg, model, optimizer):
    from torchtitan.distributed.utils import clip_grad_norm_

    norm = clip_grad_norm_(
        model.parameters(), cfg.training.max_norm, foreach=True, ep_enabled=True
    )
    optimizer.step()
    return float(norm.item())


def numeric_check(args, dims, device, rank):
    import torch

    records = {}
    baseline = None
    for label, split in (("A", False), ("AA", False), ("B", True)):
        torch._dynamo.reset()
        cfg = resolved_config(args.component, split)
        cfg.dump_folder = str(args.output)
        model = build(cfg, args.component, dims, device)
        initial = snapshot(model)
        if baseline is not None:
            mismatch = errors(baseline["initial"], initial)
            if any(v["unequal"] for v in mismatch.values()):
                raise RuntimeError(
                    "initial FP32 logical parameters differ; matched check invalid"
                )
        optimizer = cfg.optimizer.build(model_parts=[model])
        updates = []
        comparisons = []
        for update in range(2):
            optimizer.zero_grad()
            last = out = None
            for mb in range(args.accumulation):
                last = inputs(device, rank, mb + update * 32, args.component)
                out = microbatch(model, args.component, last)
            current = snapshot(model, last, out)
            norm = optimizer_step(cfg, model, optimizer)
            updated = snapshot(model)
            updated.update(optimizer_snapshot(optimizer))
            if label == "A":
                updates.append({"current": current, "updated": updated})
            else:
                comparisons.append(
                    {
                        "before_update": errors(
                            baseline["updates"][update]["current"], current
                        ),
                        "after_clip_adam": errors(
                            baseline["updates"][update]["updated"], updated
                        ),
                        "norm": norm,
                    }
                )
        if label == "A":
            baseline = {"initial": initial, "updates": updates}
        else:
            records[label] = comparisons
        del model, optimizer, initial, current, updated, last, out, updates
        gc.collect()
        torch.cuda.empty_cache()
    records["verdict"] = (
        "Report A/A and A/B local errors; nonzero FP8 differences require precision-aware interpretation, no universal tolerance applied"
    )
    return records


def observe(model, component):
    """Diagnostic-only host markers; never installed in the timing path."""
    from torchtitan.distributed.gfsdp import GFSDPModule
    import torch

    records = []
    for name, module in model.named_modules():
        if not isinstance(module, GFSDPModule):
            continue
        for group in module._get_fsdp_state()._fsdp_param_groups:
            original = group.post_backward

            def observed(*a, group=group, original=original, name=name, **kw):
                ready = [p.unsharded_param.grad is not None for p in group.fsdp_params]
                if any(ready):
                    if not all(ready):
                        raise RuntimeError(f"incomplete wgrads at reduce: {name}")
                    records.append(
                        {
                            "kind": "post_backward",
                            "fqn": canonical(name),
                            "all_wgrads_ready": True,
                            "parameter_count": len(ready),
                        }
                    )
                with torch.profiler.record_function(
                    "component_ready/" + canonical(name)
                ):
                    return original(*a, **kw)

            group.post_backward = observed
            if canonical(name).endswith("feed_forward"):
                original_register = group._register_post_backward_hook

                def register_inputs(args, kwargs, original_register=original_register):
                    from torch.utils._pytree import tree_flatten

                    args, kwargs = original_register(args, kwargs)
                    flat, _ = tree_flatten((args, kwargs))

                    def input_ready(grad):
                        records.append({"kind": "ffn_input_grad_ready"})
                        with torch.profiler.record_function(
                            "component_ffn_input_grad_ready"
                        ):
                            return grad

                    for tensor in flat:
                        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                            tensor.register_hook(input_ready)
                    return args, kwargs

                # GFSDP's eager pre-forward boundary, outside compiled compute.
                group._register_post_backward_hook = register_inputs
    return records


def benchmark(args, cfg, model, device, rank):
    import torch
    import torch.distributed as dist

    if args.input_fixtures is not None:
        tensors, fixture_receipt = captured_inputs(args.input_fixtures, device, rank, args.component)
    else:
        tensors = inputs(device, rank, 0, args.component)
        fixture_receipt = None
    for _ in range(args.warmup):
        model.zero_grad(set_to_none=True)
        microbatch(model, args.component, tensors)
    torch.cuda.synchronize(device)
    inventory = buffer_inventory(model)
    if args.mode == "diagnostic":
        events = observe(model, args.component)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for mb in range(args.trace_microbatches):
                events.append({"kind": "microbatch_start", "microbatch": mb})
                model.zero_grad(set_to_none=True)
                with torch.profiler.record_function(f"component_microbatch/{mb}"):
                    microbatch(model, args.component, tensors)
                torch.cuda.synchronize(device)
        prof.export_chrome_trace(str(args.output / f"rank{rank}.trace.json"))
        ffn = [
            e
            for e in events
            if e["kind"] == "post_backward" and e["fqn"].endswith("feed_forward")
        ]
        input_ready = [e for e in events if e["kind"] == "ffn_input_grad_ready"]
        if len(input_ready) != args.trace_microbatches:
            raise RuntimeError(
                "missing or duplicate FFN input-gradient readiness markers"
            )
        for mb in range(args.trace_microbatches):
            starts = [
                i for i, e in enumerate(events) if e["kind"] == "microbatch_start"
            ]
            chunk = events[
                starts[mb] : starts[mb + 1] if mb + 1 < len(starts) else len(events)
            ]
            ready_at = next(
                i for i, e in enumerate(chunk) if e["kind"] == "ffn_input_grad_ready"
            )
            reduce_at = next(
                i
                for i, e in enumerate(chunk)
                if e["kind"] == "post_backward" and e["fqn"].endswith("feed_forward")
            )
            if ready_at >= reduce_at:
                raise RuntimeError("FFN reduction preceded its input-gradient boundary")
        if len(ffn) != args.trace_microbatches:
            raise RuntimeError(
                f"expected one FFN reduction per microbatch, observed {len(ffn)}"
            )
        return {
            "input_fixture": fixture_receipt,
            "component_state": "paired canonical harness initialization; no native weights exported",
            "buffers": inventory,
            "events": events,
            "overlap_verdict": "requires MCP Perfetto stream/event analysis on every rank; host marker order alone is insufficient",
        }
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(args.samples):
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        model.zero_grad(set_to_none=True)
        microbatch(model, args.component, tensors)
        # GFSDP root completion queues waits for reductions; device sync includes
        # all side streams, packing, RS, gathers, replay and input/weight backward.
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    all_samples = [None] * 8
    dist.all_gather_object(all_samples, samples)
    return {
        "buffers": inventory,
        "input_fixture": fixture_receipt,
        "component_state": "paired canonical harness initialization; no native weights exported",
        "milliseconds_by_rank": all_samples,
        "spread_by_rank_ms": [
            {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}
            for xs in all_samples
        ],
        "slowest_per_sample_ms": [
            max(xs[i] for xs in all_samples) for i in range(args.samples)
        ],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "timing_scope": "zero_grad + complete selected native root F/SAC replay/B + all GFSDP work; input loading/generation, H2D and barrier excluded; captured masks replayed verbatim, synthetic mask construction included only in synthetic mode; no profiler/hooks/optimizer",
    }


def main(argv=None):
    args = arguments(argv)
    args.output = args.output.resolve()
    os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
    os.environ["TORCHINDUCTOR_MIX_ORDER_REDUCTION"] = "0"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    compiler_cache = configure_compiler_caches(
        args.output,
        int(os.environ.get("RANK", "0")),
        args.compiler_cache_root,
    )
    cfg = resolved_config(args.component, args.variant == "split")
    model = meta_component(cfg, args.component)
    receipt = contract(cfg, model, args.component)
    receipt["resolved_config"] = cfg.to_dict()
    receipt["compiler_cache"] = compiler_cache
    receipt["source_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    receipt["source_files_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((GIFT / "torchtitan").rglob("*.py"))
    }
    receipt["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt["arguments"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    if args.mode == "plan":
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{args.component}-plan.json").write_text(
            json.dumps(receipt, indent=2, default=str) + "\n"
        )
        print(
            json.dumps(
                {
                    k: receipt[k]
                    for k in (
                        "component",
                        "shape",
                        "ffn_types",
                        "segments",
                        "parameters",
                    )
                },
                default=str,
            )
        )
        return
    del model
    import torch
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        raise RuntimeError(
            "requires torchrun WORLD_SIZE=8 on the authorized eight-H100 allocation"
        )
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.flash_qla.backend import install_backend

    receipt["external_gate"] = install_backend("external_gate")
    dims = ParallelDims.from_config(cfg.parallelism, world_size=8)
    dims.build_mesh()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg.dump_folder = str(args.output)
    receipt["software"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "gpu": torch.cuda.get_device_name(device),
    }
    try:
        if args.mode == "check":
            receipt["results"] = numeric_check(args, dims, device, rank)
        else:
            start = time.perf_counter()
            model = build(cfg, args.component, dims, device)
            torch.cuda.synchronize(device)
            receipt["startup_seconds_without_compile"] = time.perf_counter() - start
            receipt["results"] = benchmark(args, cfg, model, device, rank)
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(receipt, indent=2, default=str) + "\n"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
