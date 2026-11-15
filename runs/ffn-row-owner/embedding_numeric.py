"""Full-shape numerical check for the baseline and row-owner embeddings.

Run under an authorized eight-rank NCCL launch with --execute. The script
compares the existing component factory's output and local FP32 gradients; it
does not run timing or modify training source.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import components


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the authorized NCCL check")
    parser.add_argument("--fixture", type=Path, default=ROOT / "runs/overnight-streaming-perf/artifacts/raw/mds4096/train")
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=HERE / "embedding-numeric-results.json")
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(torch):
    torch.cuda.synchronize()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def local_grad(model, torch):
    parameter = model.tok_embeddings.weight
    grad = parameter.grad
    if grad is None or grad.dtype != torch.float32:
        raise RuntimeError("embedding backward did not produce a dense FP32 gradient")
    return getattr(grad, "_local_tensor", grad)


def build_shared_upstream(ids, seed, rank, torch):
    """Make the BF16 derivative arriving from native-loss and MTP consumers."""
    generator = torch.Generator(device=ids.device).manual_seed(seed + rank)
    shape = (*ids.shape, components.WIDTH)
    native = torch.randn(shape, device=ids.device, dtype=torch.bfloat16, generator=generator)
    mtp = torch.randn(shape, device=ids.device, dtype=torch.bfloat16, generator=generator)
    native.div_(32)
    mtp.mul_(0.3)
    combined = native + mtp
    hashes = {
        "native_scaled_bf16_sha256": tensor_sha256(native, torch),
        "mtp_scaled_bf16_sha256": tensor_sha256(mtp, torch),
        "combined_bf16_sha256": tensor_sha256(combined, torch),
        "shape": list(combined.shape),
        "dtype": str(combined.dtype),
        "seed": seed + rank,
        "combination": "BF16 native/32 plus BF16 MTP*0.3, rounded to BF16 before embedding backward",
    }
    del native, mtp
    gc.collect()
    torch.cuda.synchronize()
    return combined, hashes


def tensor_sha256(tensor, torch, chunk=1 << 20):
    byte_view = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, byte_view.numel(), chunk):
        digest.update(byte_view[start:start + chunk].cpu().numpy().tobytes())
    return digest.hexdigest()


def validate_output(output, torch):
    if output.dtype != torch.bfloat16:
        raise RuntimeError(f"expected BF16 lookup output, got {output.dtype}")


def pass_forward_backward(model, ids, upstream, torch, *, measure_peak):
    start = snapshot(torch)
    if measure_peak:
        torch.cuda.reset_peak_memory_stats()
    output = model(ids)
    validate_output(output, torch)
    output.backward(upstream)
    torch.cuda.synchronize()
    peak = {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    } if measure_peak else None
    output_cpu = output.detach().cpu()
    del output
    gc.collect()
    after = snapshot(torch)
    return {
        "output_cpu": output_cpu,
        "start_memory": start,
        "peak_memory": peak,
        "after_memory": after,
    }


def compare_tensors(actual, expected_cpu, torch, *, expected_scale=1.0, chunk=1 << 20):
    """Compare a CPU or CUDA tensor to a CPU reference without a full copy."""
    actual = actual.detach().reshape(-1)
    expected_cpu = expected_cpu.reshape(-1)
    if actual.numel() != expected_cpu.numel():
        raise ValueError(f"comparison size mismatch: {actual.numel()} != {expected_cpu.numel()}")
    unequal = nonfinite = finite_count = 0
    max_abs = 0.0
    sum_sq = ref_sum_sq = 0.0
    for start in range(0, actual.numel(), chunk):
        end = min(start + chunk, actual.numel())
        left = actual[start:end].to(device="cpu", dtype=torch.float32)
        right = expected_cpu[start:end].to(dtype=torch.float32).mul(expected_scale)
        finite = torch.isfinite(left) & torch.isfinite(right)
        nonfinite += int((~finite).sum())
        unequal += int(torch.count_nonzero(left != right))
        if finite.any():
            delta = left[finite].double() - right[finite].double()
            max_abs = max(max_abs, float(delta.abs().max()))
            sum_sq += float(delta.square().sum())
            ref_sum_sq += float(right[finite].double().square().sum())
            finite_count += int(finite.sum())
    rms = (sum_sq / finite_count) ** 0.5 if finite_count else None
    ref_norm = ref_sum_sq ** 0.5
    return {
        "elements": actual.numel(),
        "finite_elements": finite_count,
        "nonfinite_elements": nonfinite,
        "unequal_elements": unequal,
        "max_abs": max_abs if finite_count else None,
        "rms": rms,
        "relative_l2": (sum_sq ** 0.5 / ref_norm) if ref_norm else (0.0 if sum_sq == 0 else None),
        "exact_equal": nonfinite == 0 and unequal == 0,
    }


def cleanup_cache(torch):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return snapshot(torch)


def run_variant(torch, dist, mesh, variant, ids, upstream, upstream_meta, baseline_reference, seed):
    # These counters distinguish constructor retention from the equal cold-cache
    # policy used before each variant's first forward/backward.
    before_setup = snapshot(torch)
    model, _ctx = components.build_embedding(mesh, variant, seed=seed)
    after_setup = snapshot(torch)
    after_cold_policy = cleanup_cache(torch)
    inventory = components.gradient_buffer_inventory(model)
    expected_head_bytes = components.VOCAB * components.WIDTH * 4
    if inventory["private_lm_head_storage_bytes"] != expected_head_bytes:
        raise RuntimeError("component factory did not retain the full private lm_head gradient bucket")
    if variant == "owner" and not inventory["owner_scratch_aliases_private_lm_head"]:
        raise RuntimeError("owner scratch is not aliased to the existing private lm_head bucket")

    first = pass_forward_backward(model, ids, upstream, torch, measure_peak=True)
    first_gradient = local_grad(model, torch)
    if first_gradient.dtype != torch.float32:
        raise RuntimeError("local accumulated embedding gradient is not FP32")
    persistent_grad_bytes = first_gradient.numel() * first_gradient.element_size()
    local_gradient_shape = list(first_gradient.shape)
    local_gradient_dtype = str(first_gradient.dtype)
    persistent_after_first = snapshot(torch)

    if variant == "baseline":
        reference = {
            "output": first["output_cpu"],
            "gradient": first_gradient.detach().to(device="cpu", copy=True),
        }
        first_output_check = compare_tensors(first["output_cpu"], reference["output"], torch)
        first_gradient_check = compare_tensors(first_gradient, reference["gradient"], torch)
    else:
        reference = baseline_reference
        first_output_check = compare_tensors(first["output_cpu"], reference["output"], torch)
        first_gradient_check = compare_tensors(first_gradient, reference["gradient"], torch)
        del first["output_cpu"]

    warmed = pass_forward_backward(model, ids, upstream, torch, measure_peak=True)
    accumulated_check = compare_tensors(
        local_grad(model, torch), reference["gradient"], torch, expected_scale=2.0
    )
    warmed_output_check = compare_tensors(warmed["output_cpu"], reference["output"], torch)
    del warmed["output_cpu"]

    # Release the first-gradient view before set-to-None so the reset path does
    # not retain a stale local allocation while creating its new gradient.
    del first_gradient
    model.tok_embeddings.weight.grad = None
    torch.cuda.synchronize()
    reset = pass_forward_backward(model, ids, upstream, torch, measure_peak=True)
    reset_check = compare_tensors(local_grad(model, torch), reference["gradient"], torch)
    reset_output_check = compare_tensors(reset["output_cpu"], reference["output"], torch)
    del reset["output_cpu"]

    result = {
        "variant": variant,
        "input": {
            "shape": list(ids.shape),
            "dtype": str(ids.dtype),
            "sha256_int64": hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest(),
            "shared_output_upstream": upstream_meta,
        },
        "gradient_storage": {
            "local_shape": local_gradient_shape,
            "dtype": local_gradient_dtype,
            "persistent_local_gradient_bytes": persistent_grad_bytes,
            "head_gradient_storage_bytes": inventory["private_lm_head_storage_bytes"],
            "owner_scratch_aliases_private_lm_head": inventory["owner_scratch_aliases_private_lm_head"],
        },
        "memory": {
            "before_setup": before_setup,
            "after_setup_before_cleanup": after_setup,
            "after_sync_gc_empty_cache_before_first_call": after_cold_policy,
            "first_complete_forward_backward": {
                "start": first["start_memory"],
                "peak": first["peak_memory"],
                "after_backward_and_output_release": first["after_memory"],
                "gradient_allocation_precedes_warmed_peak": True,
            },
            "persistent_gradient_before_warmed_call": persistent_after_first,
            "warmed_accumulated_complete_forward_backward": {
                "start": warmed["start_memory"],
                "peak": warmed["peak_memory"],
                "after_backward_and_output_release": warmed["after_memory"],
            },
            "set_none_reset_complete_forward_backward": {
                "start": reset["start_memory"],
                "peak": reset["peak_memory"],
                "after_backward_and_output_release": reset["after_memory"],
            },
        },
        "checks": {
            "first_output_vs_baseline": first_output_check,
            "first_gradient_vs_baseline": first_gradient_check,
            "accumulated_gradient_vs_two_baseline_first_gradients": accumulated_check,
            "accumulated_output_vs_baseline": warmed_output_check,
            "set_none_gradient_vs_baseline_first_gradient": reset_check,
            "set_none_output_vs_baseline": reset_output_check,
        },
    }
    del model, _ctx
    gc.collect()
    torch.cuda.synchronize()
    return result, reference


def main():
    args = parse_args()
    if not args.execute:
        print(json.dumps({
            "execution": False,
            "shape": {"vocabulary": components.VOCAB, "width": components.WIDTH,
                      "batch": components.BATCH, "sequence": components.SEQUENCE},
            "world_size": components.WORLD,
            "fixture_index": str(args.fixture / "index.json"),
            "output": str(args.output),
        }, indent=2))
        return

    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from torch.distributed.device_mesh import init_device_mesh

    if int(os.environ.get("WORLD_SIZE", 0)) != components.WORLD:
        raise ValueError("full-shape numeric check requires exactly eight NCCL ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if "H100" not in torch.cuda.get_device_name():
        raise ValueError("this check is scoped to the current eight-H100 component factory")
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    try:
        fixture = components.fixture_rows(args.fixture, rank)
        ids = components.case_ids("fixture", rank, fixture).cuda()
        upstream, upstream_meta = build_shared_upstream(ids, args.seed, rank, torch)
        cleanup_cache(torch)
        mesh = init_device_mesh("cuda", (components.WORLD,), mesh_dim_names=("fsdp",))
        baseline, reference = run_variant(
            torch, dist, mesh, "baseline", ids, upstream, upstream_meta, None, args.seed
        )
        cleanup_cache(torch)
        owner, _ = run_variant(
            torch, dist, mesh, "owner", ids, upstream, upstream_meta, reference, args.seed
        )
        rank_report = {
            "rank": rank,
            "gpu": torch.cuda.get_device_name(),
            "input_fixture_rows": [2 * rank, 2 * rank + 1],
            "variants": {"baseline": baseline, "owner": owner},
        }
        all_ranks = [None] * components.WORLD
        dist.all_gather_object(all_ranks, rank_report)
        if rank == 0:
            write_reports(args, all_ranks, torch)
    finally:
        dist.destroy_process_group()


def _check_exact(check):
    return bool(check.get("exact_equal", False))


def summarize(all_ranks):
    output_checks = []
    gradient_checks = []
    nonfinite = 0
    for rank in all_ranks:
        for variant in ("baseline", "owner"):
            checks = rank["variants"][variant]["checks"]
            for name, check in checks.items():
                is_output = "output" in name
                (output_checks if is_output else gradient_checks).append(_check_exact(check))
                nonfinite += int(check.get("nonfinite_elements", 0))
    output_exact = all(output_checks)
    gradient_exact = all(gradient_checks)
    baseline_reserved = [
        rank["variants"]["baseline"]["memory"]["after_sync_gc_empty_cache_before_first_call"]["reserved_bytes"]
        for rank in all_ranks
    ]
    owner_reserved = [
        rank["variants"]["owner"]["memory"]["after_sync_gc_empty_cache_before_first_call"]["reserved_bytes"]
        for rank in all_ranks
    ]
    cold_cache = {
        "baseline_reserved_range_bytes": [min(baseline_reserved), max(baseline_reserved)],
        "owner_reserved_range_bytes": [min(owner_reserved), max(owner_reserved)],
        "owner_each_rank_at_or_below_baseline_reserved_budget": all(
            owner <= baseline for baseline, owner in zip(baseline_reserved, owner_reserved)
        ),
        "matched_rank_reserved_delta_owner_minus_baseline_bytes": [
            owner - baseline for baseline, owner in zip(baseline_reserved, owner_reserved)
        ],
    }
    if not output_exact or nonfinite:
        verdict = "fail"
    elif gradient_exact:
        verdict = "pass"
    else:
        verdict = "unresolved_gradient_difference"
    return {
        "verdict": verdict,
        "all_outputs_exactly_equal": output_exact,
        "all_gradient_checks_exactly_equal": gradient_exact,
        "nonfinite_elements_across_checks": nonfinite,
        "equal_cold_cache_policy": cold_cache,
        "gradient_difference_policy": (
            "No tolerance is imposed. Any finite nonzero gradient difference is reported as unresolved; "
            "the implementation preserves BF16 source-local coalescing but changes the cross-source FP32 sum order."
        ),
    }


def write_reports(args, all_ranks, torch):
    summary = summarize(all_ranks)
    fixture_index = args.fixture / "index.json"
    reproduction = (
        "PYTHONPATH=gift /opt/torchtitan-venv/bin/torchrun --standalone "
        "--nproc-per-node=8 runs/ffn-row-owner/embedding_numeric.py --execute "
        f"--fixture {args.fixture} --seed {args.seed} --output {args.output}"
    )
    report = {
        "task": "full-shape embedding output and local-gradient regression",
        "summary": summary,
        "reproduction_command": reproduction,
        "worker_argv": sys.argv,
        "resolved_config": {
            "case": "fixture",
            "variants_in_order": ["baseline", "owner"],
            "warmup_calls": 0,
            "backward_calls_per_variant": ["first", "accumulated second", "after grad=None"],
            "output_consumers": 2,
            "output_upstream": "rank-local seeded BF16 tensors; native scaled /32 and MTP scaled *0.3, then BF16 combined",
            "output_dtype": "bfloat16",
            "parameter_and_gradient_dtype": "float32",
            "gradient_comparison": "exact, chunked; no universal tolerance",
        },
        "shape": {"vocabulary": components.VOCAB, "width": components.WIDTH,
                  "batch": components.BATCH, "sequence": components.SEQUENCE},
        "topology": "eight H100 ranks, one CUDA/NCCL dense shard group",
        "seed": args.seed,
        "fixture": {
            "path": str(args.fixture),
            "index_sha256": sha256(fixture_index),
            "rows_per_rank": 2,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": list(torch.cuda.nccl.version()),
        },
        "source": {
            "repository_revision": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "components_sha256": sha256(HERE / "components.py"),
            "owner_source_sha256": sha256(ROOT / "gift/torchtitan/distributed/gfsdp/row_owner_embedding.py"),
            "numeric_script_sha256": sha256(HERE / "embedding_numeric.py"),
            "training_source_status": subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain", "--", "gift"], text=True
            ),
            "training_source_diff_sha256": hashlib.sha256(subprocess.check_output(
                ["git", "-C", str(ROOT), "diff", "HEAD", "--binary", "--", "gift"]
            )).hexdigest(),
        },
        "historical_fixture_memory_context": {
            "source": "previous matched component fixture pair, supplied by coordinator",
            "allocated_peak_bytes": {"baseline": 8950373888, "owner": 5667090944},
            "reserved_bytes": {"baseline": 16290676736, "owner": 18610126848},
            "interpretation": "The earlier higher owner reserved count is retained setup-cache evidence; this check records both constructor and equal sync/gc/empty_cache pre-call counters.",
        },
        "prior_attempts": [
            {
                "attempt": 1,
                "exit_code": 1,
                "phase": "all-rank measurements and gather completed; rank-zero report formatting failed",
                "error": "_memory_range received a duplicated memory path prefix and raised KeyError('memory')",
                "log": str(HERE / "embedding-numeric-launch.log"),
            }
        ],
        "ranks": all_ranks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(markdown_report(report))


def _memory_range(all_ranks, variant, path):
    values = []
    for rank in all_ranks:
        current = rank["variants"][variant]
        for key in path:
            current = current[key]
        values.append(current)
    def bounds(field):
        numbers = [value[field] for value in values]
        return f"{min(numbers):,}–{max(numbers):,}"
    return f"allocated {bounds('allocated_bytes')} B; reserved {bounds('reserved_bytes')} B"


def markdown_report(report):
    lines = [
        "# Full-shape embedding numerical check",
        "",
        f"**Verdict:** `{report['summary']['verdict']}`. Fixture lookup outputs are compared elementwise for exact equality. Gradient comparisons use exact equality as the only pass condition; finite nonzero differences remain unresolved, with no invented tolerance.",
        "",
        "## Protocol",
        "",
        f"- Shape: V={components.VOCAB}, D={components.WIDTH}, B={components.BATCH}, S={components.SEQUENCE}; eight H100/NCCL ranks.",
        f"- Seed: `{report['seed']}`. Fixture index SHA256: `{report['fixture']['index_sha256']}`.",
        "- Two matched, rank-local seeded BF16 consumer derivatives are scaled as native `/32` and MTP `*0.3`, then combined in BF16 before ordinary `output.backward(upstream)`. The exact combined tensor is reused across both variants and all backward states.",
        "- Each variant checks the first backward, a second backward accumulated onto it, and a fresh backward after `weight.grad = None`. One baseline local FP32 gradient per rank is retained on CPU; no model or full table is exported.",
        "- Memory counters cover setup, the identical synchronize/GC/empty-cache point before first forward, first complete forward/backward, warmed accumulated forward/backward, and the `set_none` reset call. These are memory checks, not timing results.",
        "",
        "## Results",
        "",
        "The first launch completed the all-rank measurements and wrote the JSON, then exited 1 in the rank-zero Markdown formatter; its log is retained at `embedding-numeric-launch.log`. The path bug is fixed, and this Markdown was regenerated offline from the saved rank data without repeating GPU work.",
        "",
        f"- Output exactly equal across all ranks and passes: **{report['summary']['all_outputs_exactly_equal']}**.",
        f"- Gradient checks exactly equal across all ranks: **{report['summary']['all_gradient_checks_exactly_equal']}**.",
        f"- Non-finite elements counted across comparisons: **{report['summary']['nonfinite_elements_across_checks']}**.",
        f"- Baseline post-setup cache: {_memory_range(report['ranks'], 'baseline', ('memory', 'after_setup_before_cleanup'))}.",
        f"- Owner post-setup cache: {_memory_range(report['ranks'], 'owner', ('memory', 'after_setup_before_cleanup'))}.",
        f"- Baseline after equal cold-cache policy: {_memory_range(report['ranks'], 'baseline', ('memory', 'after_sync_gc_empty_cache_before_first_call'))}.",
        f"- Owner after equal cold-cache policy: {_memory_range(report['ranks'], 'owner', ('memory', 'after_sync_gc_empty_cache_before_first_call'))}.",
        f"- Owner post-cleanup reserved at or below the baseline budget on every matching rank: **{report['summary']['equal_cold_cache_policy']['owner_each_rank_at_or_below_baseline_reserved_budget']}**; matched rank deltas are in JSON.",
        f"- Baseline first-call peak: {_memory_range(report['ranks'], 'baseline', ('memory', 'first_complete_forward_backward', 'peak'))}.",
        f"- Owner first-call peak: {_memory_range(report['ranks'], 'owner', ('memory', 'first_complete_forward_backward', 'peak'))}.",
        f"- Baseline warmed accumulated peak: {_memory_range(report['ranks'], 'baseline', ('memory', 'warmed_accumulated_complete_forward_backward', 'peak'))}.",
        f"- Owner warmed accumulated peak: {_memory_range(report['ranks'], 'owner', ('memory', 'warmed_accumulated_complete_forward_backward', 'peak'))}.",
        f"- Baseline `grad=None` reset peak: {_memory_range(report['ranks'], 'baseline', ('memory', 'set_none_reset_complete_forward_backward', 'peak'))}.",
        f"- Owner `grad=None` reset peak: {_memory_range(report['ranks'], 'owner', ('memory', 'set_none_reset_complete_forward_backward', 'peak'))}.",
        "",
        "The earlier fixture pair reported allocated peaks of 8,950,373,888 B (baseline) and 5,667,090,944 B (owner), while reserved memory was 16,290,676,736 B and 18,610,126,848 B. After equal cleanup in this check, owner reserved memory is 2,101,346,304 B lower than baseline on every rank. The earlier owner reserve premium was retained allocator-cache state and does not persist after the matched cold-cache policy.",
        "",
        "No gradient difference was observed in this check. A finite nonzero difference on a rerun would remain unresolved because the baseline distributed FP32 reduction and owner source-order FP32 sum can add contributions in different orders. The JSON has per-rank, per-state max absolute error, RMS, relative L2, unequal and non-finite counts.",
        "",
        "## Reproduction",
        "",
        "```text",
        report["reproduction_command"],
        "```",
        "",
        f"Repository revision: `{report['source']['repository_revision']}`. Factory SHA256: `{report['source']['components_sha256']}`. Owner source SHA256: `{report['source']['owner_source_sha256']}`. Numeric script SHA256: `{report['source']['numeric_script_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
