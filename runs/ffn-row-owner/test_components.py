"""GPU-hidden harness checks; no production-shape allocation or CUDA tests."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PATH = Path(__file__).with_name("components.py")
SPEC = importlib.util.spec_from_file_location("owner_components", PATH)
components = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(components)


def test_plan_is_torch_free():
    output = subprocess.check_output([
        sys.executable, "-c",
        "import runpy,sys; sys.argv=['components.py']; "
        f"runpy.run_path({str(PATH)!r},run_name='__main__'); "
        "assert 'torch' not in sys.modules",
    ], text=True)
    plan = json.loads(output)
    assert plan["execution"] is False
    assert plan["temporary_bytes"]["native_bf16_full_gradient"] == 1002 * 1024**2
    assert plan["resident_bytes"]["private_lm_head_gradient_buffer"] == 2004 * 1024**2
    assert plan["shape"]["width"] == 4096


@pytest.mark.parametrize("argv", [
    ["--warmup", "0"], ["--samples", "0"], ["--samples", "21"],
    ["--scope", "stages", "--variant", "baseline"], ["--profile"],
    ["--cases", "fixture", "fixture"],
])
def test_invalid_plan_refuses(argv):
    with pytest.raises(ValueError):
        components.plan(components.parser().parse_args(argv))


def test_sensitive_inputs_and_traffic():
    import torch

    fixture = torch.arange(16).reshape(2, 8)
    for case in components.CASES:
        for rank in (0, 1):
            ids = components.case_ids(case, rank, fixture, vocab=32, sequence=8, world=2)
            assert ids.dtype == torch.int64
            census = components.traffic(ids, world=2, vocab=32, width=8)
            assert sum(census["unique_requests_by_owner"]) == census["unique_requests"]
            assert census["payload_padding_bytes"] == 0
            assert census["tokens"] == ids.numel()
    assert components.case_ids("empty-requester", 1, fixture, vocab=32, sequence=8, world=2).numel() == 0
    assert components.traffic(torch.tensor([0, 0, 31]), world=2, vocab=32, width=8)["unique_requests_by_owner"] == [1, 1]
    with pytest.raises(ValueError):
        components.traffic(torch.tensor([-1]), world=2, vocab=32, width=8)


def test_real_mds_reader_preserves_bytes():
    import numpy as np
    import torch
    from streaming.base.local import LocalDataset

    args = components.parser().parse_args([])
    if not (args.fixture / "index.json").exists():
        pytest.skip("existing local MDS fixture is unavailable; no download")
    actual = components.fixture_rows(args.fixture, 0)
    raw = LocalDataset(local=str(args.fixture))[0]["tokens"]
    expected = torch.from_numpy(np.frombuffer(raw, dtype="<u4").astype(np.int64))
    assert torch.equal(actual[0], expected)
    assert actual.shape == (2, 4096)
    assert 0 <= actual.min() <= actual.max() < components.VOCAB


def _tiny_gfsdp_owner_rank(rank, world_size, rendezvous):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    assert not torch.cuda.is_initialized()
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            init_method=rendezvous)
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        baseline, _ = components.build_embedding(mesh, "baseline", vocab=16, width=8)
        candidate, _ = components.build_embedding(mesh, "owner", vocab=16, width=8)
        assert tuple(baseline.lm_head.weight.shape) == (16, 8)
        assert tuple(candidate.lm_head.weight.shape) == (16, 8)
        assert baseline.lm_head.weight.dtype == candidate.lm_head.weight.dtype == torch.float32
        assert baseline.lm_head.weight is not baseline.tok_embeddings.weight
        assert candidate.lm_head.weight is not candidate.tok_embeddings.weight
        baseline_head_state = baseline.lm_head._get_fsdp_state()
        candidate_head_state = candidate.lm_head._get_fsdp_state()
        assert baseline_head_state._own_grad_buffer
        assert candidate_head_state._own_grad_buffer
        baseline_head_group, = baseline_head_state._fsdp_param_groups
        candidate_head_group, = candidate_head_state._fsdp_param_groups
        baseline_head_buffer = next(iter(baseline_head_group.grad_buffer_ctx.buffers.values()))
        candidate_head_buffer = next(iter(candidate_head_group.grad_buffer_ctx.buffers.values()))
        baseline_embedding_group, = baseline.tok_embeddings._get_fsdp_state()._fsdp_param_groups
        baseline_embedding_buffer = next(iter(baseline_embedding_group.grad_buffer_ctx.buffers.values()))
        assert baseline_embedding_buffer.untyped_storage().data_ptr() != baseline_head_buffer.untyped_storage().data_ptr()
        baseline_buffers = components.gradient_buffer_inventory(baseline)
        candidate_buffers = components.gradient_buffer_inventory(candidate)
        assert baseline_buffers["private_lm_head_storage_bytes"] == 16 * 8 * 4
        assert candidate_buffers["private_lm_head_storage_bytes"] == 16 * 8 * 4
        assert baseline_buffers["unique_storage_count"] == 2
        assert candidate_buffers["unique_storage_count"] == 1
        assert candidate_buffers["owner_scratch_aliases_private_lm_head"] is True
        with candidate.tok_embeddings._scratch(8) as scratch:
            assert scratch.untyped_storage().data_ptr() == candidate_head_buffer.untyped_storage().data_ptr()
        assert candidate_head_group.grad_buffer_ctx.owner is None
        assert baseline.lm_head.weight._local_tensor.equal(candidate.lm_head.weight._local_tensor)
        ids = torch.tensor([[0, 0, 15, 3]], dtype=torch.int64)
        reference = baseline(ids)
        actual = candidate(ids)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        derivative = torch.arange(32).view(1, 4, 8).bfloat16() / 7
        reference.backward(derivative)
        actual.backward(derivative)
        left = baseline.tok_embeddings.weight.grad._local_tensor
        right = candidate.tok_embeddings.weight.grad._local_tensor
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        # Accumulated FP32 storage exists before the next measured operation.
        left.zero_()
        right.zero_()
        baseline(ids).backward(derivative)
        candidate(ids).backward(derivative)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_actual_tiny_gfsdp_and_owner_build(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import torch

    torch.multiprocessing.spawn(
        _tiny_gfsdp_owner_rank,
        args=(2, (tmp_path / "rendezvous").as_uri()),
        nprocs=2,
        join=True,
    )
    assert not torch.cuda.is_initialized()
