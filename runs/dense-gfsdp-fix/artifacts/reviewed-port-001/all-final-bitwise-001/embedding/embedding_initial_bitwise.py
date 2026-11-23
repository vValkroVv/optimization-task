#!/usr/bin/env python3
"""Compare GFSDP dense and shared row-owner embedding components on eight H100s."""

import argparse
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch


WORLD, VOCAB, WIDTH = 8, 128256, 4096
PASSES = ("initial", "accumulated", "reset")
OWNER_SHA256 = "f3a01c6cf99b981db7b4ef66866fe46259af84e9d2c21e2044e9ae18ff1696fd"
TORCH_GIT = "8b353065ce260336c5fdfa74d754e8bd9b91b062"


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def order_probe(torch, rank):
    # Each supplied row occurs once on each source, so this isolates the
    # order of FP32 additions after the two paths have the same BF16 operands.
    rows = torch.arange(WORLD, dtype=torch.int64) * (VOCAB // WORLD)
    derivative = torch.zeros((WORLD, WIDTH), dtype=torch.bfloat16)
    values = torch.tensor([2**24, 1, -(2**24), 1, 0, 0, 0, 0], dtype=torch.bfloat16)
    for row in range(WORLD):
        derivative[row, :WORLD] = values[(rank + row + torch.arange(WORLD)) % WORLD]
    return rows, derivative


def mismatch(actual, reference, harness, torch):
    actual = actual.detach().cpu().contiguous().reshape(-1)
    reference = reference.reshape(-1)
    word_type = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    mask = (1 << (actual.element_size() * 8)) - 1
    words_a, words_b = reference.view(word_type), actual.view(word_type)
    total, examples = 0, []
    for start in range(0, words_a.numel(), 1 << 20):
        a = words_a[start : start + (1 << 20)]
        b = words_b[start : start + (1 << 20)]
        different = a != b
        total += different.count_nonzero().item()
        if len(examples) < 8:
            for index in different.nonzero().flatten()[: 8 - len(examples)].tolist():
                examples.append({
                    "flat_index": start + index,
                    "dense_word": hex(int(a[index]) & mask),
                    "owner_word": hex(int(b[index]) & mask),
                })
    return {
        "different_words": total, "examples": examples,
        "values": harness["compare_tensors"](actual, reference, torch),
    }


def replay(variant, mesh, ids, upstream, seed, harness, torch, dist, baseline=None):
    tensor_hash = harness["tensor_sha256"]
    model, comm_context = harness["components"].build_embedding(mesh, variant, seed=seed)
    weight = model.tok_embeddings.weight
    master = tensor_hash(weight._local_tensor, torch)
    unique = ids.reshape(-1).unique(sorted=True)
    source = []
    reduce_scatter = dist.reduce_scatter_tensor
    all_to_all = dist.all_to_all_single

    def observe_dense(output, input_tensor, *args, **kwargs):
        # The isolated GFSDP factory reduces one full FP32 embedding table.
        if (input_tensor.dtype != torch.float32
                or input_tensor.numel() != VOCAB * WIDTH):
            raise RuntimeError("unexpected GFSDP embedding reduce-scatter input")
        source.append(input_tensor.view(VOCAB, WIDTH).index_select(0, unique))
        return reduce_scatter(output, input_tensor, *args, **kwargs)

    def observe_owner(output, input, *args, **kwargs):
        if input.dtype != torch.bfloat16 or tuple(input.shape) != (unique.numel(), WIDTH):
            raise RuntimeError("unexpected row-owner source BF16 gradient message")
        source.append(input.clone())
        return all_to_all(output, input, *args, **kwargs)

    @contextmanager
    def observe_stage(stage):
        if stage == "gradient_exchange":
            with patch.object(dist, "all_to_all_single", observe_owner):
                yield
        else:
            yield

    if variant == "owner":
        model.tok_embeddings.stage_observer = observe_stage

    results, references = {}, {}
    for name in PASSES:
        if name == "reset":
            weight.grad = None
        if variant == "baseline":
            with patch.object(dist, "reduce_scatter_tensor", observe_dense):
                call = harness["pass_forward_backward"](model, ids, upstream, torch, measure_peak=False)
        else:
            call = harness["pass_forward_backward"](model, ids, upstream, torch, measure_peak=False)
        (source_input,) = source
        source.clear()
        payload = source_input.to(torch.bfloat16)
        if variant == "baseline" and not torch.equal(
            source_input.view(torch.int32), payload.float().view(torch.int32)
        ):
            raise RuntimeError("dense source FP32 contributions did not round-trip through BF16")
        output = call["output_cpu"]
        gradient = harness["local_grad"](model, torch)
        if tuple(output.shape) != (*ids.shape, WIDTH) or gradient.numel() != VOCAB * WIDTH // WORLD:
            raise RuntimeError("unexpected owner output or FP32 gradient shape")
        receipt = {
            "output_bf16_sha256": tensor_hash(output, torch),
            "local_gradient_fp32_sha256": tensor_hash(gradient, torch),
            "source_gradient_bf16_sha256": tensor_hash(payload, torch),
        }
        if baseline is None:
            references[name] = (output, gradient.detach().to("cpu", copy=True).contiguous())
        else:
            dense, expected = baseline
            for field, value, reference in (
                ("output_bf16", output, expected[name][0]),
                ("local_gradient_fp32", gradient, expected[name][1]),
            ):
                key = field + "_sha256"
                if receipt[key] != dense["passes"][name][key]:
                    receipt[field + "_mismatch"] = mismatch(value, reference, harness, torch)
        results[name] = receipt
    del model, comm_context, weight, gradient, source_input, payload
    harness["cleanup_cache"](torch)
    return {"local_master_fp32_sha256": master, "passes": results}, references


def run(args):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    root = args.source_root.resolve(strict=True)
    helpers = root / "runs/dense-gfsdp-fix/artifacts/reviewed-port-001/embedding-combined-001"
    paths = {
        "driver": Path(__file__).resolve(),
        "owner": root / "gift/torchtitan/distributed/gfsdp/row_owner_embedding.py",
        "numeric": root / "runs/ffn-row-owner/embedding_numeric.py",
        "factory": root / "runs/ffn-row-owner/components.py",
        "capture": root / "runs/production-data-abcd/component_fixtures.py",
        "low_fixture": helpers / "compact_coalescing_replay.py",
        "fixture_reference": helpers / "embedding-replay-result.json",
        "route_numeric": helpers / "row_owner_route_numeric.py",
        "route_experiment": helpers / "row_owner_route_experiment.py",
    }
    hashes = {name: file_hash(path) for name, path in paths.items()}
    if hashes["owner"] != OWNER_SHA256 or args.output.exists():
        raise ValueError("unexpected shared owner source or existing output")
    os.environ["ROW_OWNER_REPLAY_SOURCE_ROOT"] = str(root)
    sys.path.insert(0, str(paths["low_fixture"].parent))
    from compact_coalescing_replay import low_fixture

    if (torch.__version__, torch.version.git_version, torch.version.cuda, torch.cuda.nccl.version()) != (
        "2.15.0.dev20260822+cu130", TORCH_GIT, "13.0", (2, 30, 7)
    ):
        raise ValueError("expected pinned torch, CUDA 13.0 and NCCL 2.30.7")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    gpu = torch.cuda.get_device_properties(rank)
    if (int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_WORLD_SIZE"])) != (WORLD, WORLD):
        raise ValueError("expected eight local and global ranks")
    if "H100" not in gpu.name or gpu.multi_processor_count != 132:
        raise ValueError("expected a 132-SM H100 per rank")
    harness = runpy.run_path(str(paths["numeric"]))
    components = harness["components"]
    load_fixture = runpy.run_path(str(paths["capture"]))["load_fixture"]
    from torchtitan.distributed.gfsdp import row_owner_embedding

    if Path(row_owner_embedding.__file__).resolve() != paths["owner"]:
        raise RuntimeError("the loaded embedding differs from the pinned shared source")
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    try:
        if rank != dist.get_rank():
            raise RuntimeError("local and global ranks differ")
        mesh = init_device_mesh("cuda", (WORLD,), mesh_dim_names=("fsdp",))
        captured, captured_meta = load_fixture(args.high_fixture, rank)
        high_ids = captured["input_ids"]
        del captured
        reference = json.loads(paths["fixture_reference"].read_text())
        if captured_meta["input_ids_int64_sha256"] != reference["input_ids_sha256_by_rank"][rank]:
            raise RuntimeError("real high-U IDs differ from the accepted capture")
        low_ids, low_meta = low_fixture(args.low_census, args.data_root, rank)
        empty_requester = components.case_ids("empty-requester", rank, None)
        if rank == 0:
            # An empty requester on rank one must still serve rank zero.
            empty_requester[:, 1::2] = VOCAB // WORLD
        probe_ids, probe_upstream = order_probe(torch, rank)
        cases = (
            ("real-high-u", high_ids, None, {"capture_sha256": captured_meta["tensors_sha256"]}),
            ("real-low-u", low_ids, None, low_meta),
            ("synthetic-empty-requester", empty_requester, None, {}),
            ("synthetic-fp32-order-probe", probe_ids, probe_upstream, {"source": "one token per row per rank"}),
        )
        rank_cases = {}
        for name, cpu_ids, derivative, provenance in cases:
            if cpu_ids.dtype != torch.int64 or not ((cpu_ids >= 0) & (cpu_ids < VOCAB)).all():
                raise ValueError(f"invalid signed int64 fixture IDs for {name}")
            ids = cpu_ids.cuda()
            if derivative is None:
                upstream, upstream_meta = harness["build_shared_upstream"](ids, args.seed, rank, torch)
            else:
                upstream, upstream_meta = derivative.cuda(), {"source": "finite BF16 cancellation operands"}
            dense, refs = replay("baseline", mesh, ids, upstream, args.seed, harness, torch, dist)
            owner, _ = replay("owner", mesh, ids, upstream, args.seed, harness, torch, dist, (dense, refs))
            del refs
            exact = {
                key: all(dense["passes"][name][key] == owner["passes"][name][key] for name in PASSES)
                for key in dense["passes"]["initial"]
            }
            exact["local_master_fp32_sha256"] = dense["local_master_fp32_sha256"] == owner["local_master_fp32_sha256"]
            exact["reset_reproduces_initial"] = all(
                arm["passes"]["initial"]["local_gradient_fp32_sha256"]
                == arm["passes"]["reset"]["local_gradient_fp32_sha256"]
                for arm in (dense, owner)
            )
            rank_cases[name] = {
                "ids_sha256": harness["tensor_sha256"](cpu_ids, torch),
                "ids_shape": list(cpu_ids.shape), "unique_ids": cpu_ids.unique().numel(),
                "upstream_sha256": harness["tensor_sha256"](upstream, torch),
                "upstream_shape": list(upstream.shape), "upstream": upstream_meta,
                "provenance": provenance, "exact": exact, "dense": dense, "owner": owner,
            }
            del upstream, ids
        rows = [None] * WORLD
        dist.all_gather_object(rows, {"rank": rank, "gpu_uuid": str(gpu.uuid), "cases": rank_cases})
        if [row["rank"] for row in rows] != list(range(WORLD)) or len({row["gpu_uuid"] for row in rows}) != WORLD:
            raise RuntimeError("missing rank receipts or duplicate CUDA devices")
        case_exact = {
            name: all(all(row["cases"][name]["exact"].values()) for row in rows)
            for name, *_ in cases
        }
        if rank == 0:
            receipt = {
                "schema_version": 1, "exact": all(case_exact.values()), "case_exact": case_exact,
                "scope": "component-only GFSDP dense versus shared row-owner, bitwise comparison",
                "seed": args.seed, "argv": sys.argv, "source_hashes": hashes,
                "source_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
                "ranks": rows,
            }
            with args.output.open("x") as output:
                json.dump(receipt, output, indent=2, allow_nan=False)
                output.write("\n")
            print(json.dumps(case_exact), flush=True)
        return int(not all(case_exact.values()))
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--high-fixture", type=Path, required=True)
    parser.add_argument("--low-census", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2048)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
