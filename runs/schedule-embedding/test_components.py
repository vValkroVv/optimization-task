"""CPU-hidden contracts for the staged component runner; no CUDA work."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PATH = Path(__file__).with_name("components.py")
SPEC = importlib.util.spec_from_file_location("schedule_embedding_components", PATH)
components = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(components)


def test_default_plan_uses_real_eight_rank_fixture_and_no_execute_flag():
    code = (
        "import runpy,sys; "
        f"sys.argv=[{str(PATH)!r}]; "
        f"runpy.run_path({str(PATH)!r},run_name='__main__'); "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    plan = json.loads(result.stdout)
    assert plan["execution"] is False
    assert plan["execute_flag_required_for_cuda"] is True
    assert plan["stage_id"] == "d0"
    assert plan["optimization_stages"] == []
    assert plan["warmup"] == plan["samples"] == 10
    assert plan["shape"] == {
        "vocabulary": 128256,
        "width": 4096,
        "batch": 2,
        "sequence": 4096,
        "tokens_per_rank": 8192,
        "dtype_contract": "FP32 owner master and returned local gradient; BF16 lookup and source coalescing",
    }
    assert [row["rank"] for row in plan["fixtures"]] == list(range(8))
    assert {row["world_size"] for row in plan["fixtures"]} == {8}
    assert {row["tokens"] for row in plan["fixtures"]} == {8192}
    assert all(row["input_ids_int64_sha256"] for row in plan["fixtures"])
    assert all(row["owner_distribution"] is None for row in plan["fixtures"])
    assert plan["compiler_cache"]["set_before_torch_or_triton_import"] is True


@pytest.mark.parametrize("index,stages,stage_id", [
    (0, [], "d0"),
    (1, ["compact"], "d1-compact"),
    (2, ["compact", "owner_add"], "d2-compact-owner_add"),
    (3, ["compact", "owner_add", "owner_lookup"], "d3-compact-owner_add-owner_lookup"),
    (4, ["compact", "owner_add", "owner_lookup", "routing"], "d4-compact-owner_add-owner_lookup-routing"),
])
def test_only_the_five_ordered_stage_prefixes_are_accepted(index, stages, stage_id):
    args = components.parser().parse_args(["--stages", *stages] if stages else [])
    assert tuple(components.validate(args)) == components.STAGE_LADDER[index]
    assert components._stage_id(tuple(stages)) == stage_id


@pytest.mark.parametrize("stages", [["owner_add"], ["compact", "owner_lookup"], ["routing", "compact"]])
def test_nonprefix_stage_combinations_are_rejected(stages):
    args = components.parser().parse_args(["--stages", *stages])
    with pytest.raises(ValueError, match="ordered ladder prefix"):
        components.validate(args)


def test_execution_requires_an_output_and_eight_ranks():
    args = components.parser().parse_args(["--execute"])
    with pytest.raises(ValueError, match="requires a fresh --output"):
        components.validate(args)
    args = components.parser().parse_args([
        "--execute",
        "--stages", "compact", "owner_add", "owner_lookup", "routing",
        "--output", str(components.RAW_ROOT / "cpu-parser-check"),
    ])
    assert components.validate(args) == components.STAGE_LADDER[-1]
    components.validate_world_size(8)
    with pytest.raises(ValueError, match="WORLD_SIZE=8"):
        components.validate_world_size(4)


def test_staged_factory_passes_only_the_stage_tuple_and_restores_original_classmethod():
    class DummyEmbedding:
        @classmethod
        def from_embedding(
            cls,
            embedding,
            mesh,
            *,
            shard_alignment=8,
            optimize=False,
            optimization_stages=None,
        ):
            return cls, embedding, mesh, shard_alignment, optimize, optimization_stages

    original = DummyEmbedding.__dict__["from_embedding"]
    stages = ("compact", "owner_add")
    with components.staged_factory(DummyEmbedding, stages):
        result = DummyEmbedding.from_embedding("embedding", "mesh")
        assert result == (DummyEmbedding, "embedding", "mesh", 8, False, stages)
        with pytest.raises(RuntimeError, match="must not override"):
            DummyEmbedding.from_embedding(
                "embedding", "mesh", optimization_stages=("routing",)
            )
    assert DummyEmbedding.__dict__["from_embedding"] is original


def test_fixture_manifest_validates_all_captured_rank_receipts():
    rows = components.fixture_manifest(components.FIXTURE_ROOT)
    assert len(rows) == 8
    assert [row["rank"] for row in rows] == list(range(8))
    assert len({row["input_ids_int64_sha256"] for row in rows}) == 8
    assert all(row["payload_hash_rechecked"] is False for row in rows)
    assert all(row["tensor_payload_sha256"] for row in rows)


def test_source_snapshot_pins_relevant_files_without_importing_torch():
    snapshot = components.source_snapshot()
    assert snapshot["revision"]
    assert "gift/torchtitan/distributed/gfsdp/row_owner_embedding.py" in snapshot["files_sha256"]
    assert "gift/torchtitan/distributed/gfsdp/_row_owner_embedding_kernels.py" in snapshot["files_sha256"]
    assert "runs/schedule-embedding/components.py" in snapshot["files_sha256"]


def test_cache_configuration_uses_existing_rank_paths_and_restores_environment(monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "previous-inductor")
    monkeypatch.setenv("TRITON_CACHE_DIR", "previous-triton")
    paths = components.configure_compiler_caches(3)
    assert paths["TORCHINDUCTOR_CACHE_DIR"] == str(components.D_CACHE_ROOT / "torchinductor")
    assert paths["TRITON_CACHE_DIR"] == str(components.D_CACHE_ROOT / "triton/rank_3")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == paths["TORCHINDUCTOR_CACHE_DIR"]
    assert os.environ["TRITON_CACHE_DIR"] == paths["TRITON_CACHE_DIR"]


def test_temporary_byte_accounting_keeps_route_scratch_and_unknown_native_partial():
    route = {"input_tokens": 8192, "unique_count": 80, "requested_count": 589}
    scratch = [
        {"scope": "first_warmup_integrated", "operation": "forward", "requested_bytes": 1000,
         "aliases_owner_gfsdp_gradient_storage": True},
        {"scope": "first_warmup_integrated", "operation": "backward", "requested_bytes": 2000,
         "aliases_owner_gfsdp_gradient_storage": True},
    ]
    coalescing = {
        "dispatch": "compact_native_output",
        "native_output_bytes": 80 * 4096 * 2,
        "native_full_bf16_table_bytes": 0,
        "native_fp32_partial_scratch_bytes": None,
    }
    result = components.temporary_bytes_accounting(
        route, scratch, coalescing, ("compact", "owner_add")
    )["rank_local"]
    assert result["saved_route_tensor_bytes"] == {
        "ids_or_compact_inverse": 8192 * 8,
        "unique_ids": 0,
        "requested_local_rows": 589 * 8,
        "total": (8192 + 589) * 8,
        "autograd_saved_tensors": [
            {
                "name": "inverse",
                "shape": [8192],
                "dtype": "int64",
                "bytes": 8192 * 8,
                "storage_note": "inverse tensor saved by value",
            },
            {"name": "local_requested_rows", "shape": [589], "dtype": "int64", "bytes": 589 * 8},
        ],
        "lifetime": "forward autograd context through backward; input-ID storage is retained by reference when compact is inactive",
    }
    assert result["forward_route_transient_bytes"]["known_tensor_total"] == 68104
    assert result["forward_route_transient_bytes"]["scope_note"].startswith("known int64")
    assert result["forward_output_bf16_bytes"] == 8192 * 4096 * 2
    assert result["required_owned_fp32_gradient_bytes"] == (128256 // 8) * 4096 * 4
    assert result["scratch_max_requested_bytes"] == 2000
    assert result["scratch_storage_is_borrowed"] is True
    assert result["native_source_coalescing"]["native_fp32_partial_scratch_bytes"] is None


def test_rank_max_aggregation_retains_per_rank_values_without_crossrank_sum():
    reports = []
    for rank in range(8):
        phases = {}
        for phase in ("forward", "backward", "integrated"):
            phases[phase] = [
                {
                    "cuda_elapsed_span_ms": float(rank + sample),
                    "host_call_ms": float(rank + sample + 1),
                    "host_end_sync_ms": float(rank + sample + 2),
                    "synchronized_wall_ms": float(rank + sample + 3),
                    "peak_allocated_bytes": rank * 10 + sample,
                    "peak_reserved_bytes": rank * 20 + sample,
                }
                for sample in range(10)
            ]
        reports.append({"rank": rank, "measurements": phases, "profile": {}})
    result = components._rank_max_aggregate(reports)
    forward = result["phases"]["forward"]
    assert forward["rank_max_by_sample"]["cuda_elapsed_span_ms"]["samples"] == [
        float(7 + sample) for sample in range(10)
    ]
    assert forward["peak_allocated_bytes"]["per_rank"] == [
        rank * 10 + 9 for rank in range(8)
    ]
    assert forward["peak_allocated_bytes"]["rank_max"] == 79
    assert len(forward["rank_local_medians"]["synchronized_wall_ms"]) == 8


def test_trace_summary_separates_kernels_nccl_copies_and_interval_union(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({
        "traceEvents": [
            {"ph": "X", "cat": "kernel", "name": "row_owner_lookup", "ts": 0, "dur": 10},
            {"ph": "X", "cat": "kernel", "name": "ncclDevKernel_SendRecv", "ts": 5, "dur": 10},
            {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoD", "ts": 20, "dur": 5},
            {"ph": "X", "cat": "gpu_memset", "name": "Memset", "ts": 25, "dur": 3},
            {"ph": "X", "cat": "cpu_op", "name": "not a device event", "ts": 0, "dur": 100},
        ]
    }))
    result = components._trace_summary(trace)
    assert result["kernel_count"] == 2
    assert result["local_kernel_count"] == 1
    assert result["nccl_kernel_count"] == 1
    assert result["copy_count"] == 1
    assert result["memset_count"] == 1
    assert result["device_event_interval_union_ms"] == pytest.approx(0.023)

