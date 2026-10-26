"""Sparse monitoring must rank selected keys before taking a smaller-k prefix."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability() != (9, 0):
    pytest.skip("DSA kernels require Hopper", allow_module_level=True)

from llmfoundry.models.layers import dsa_monitor, dsa_operator

pytestmark = [pytest.mark.unit, pytest.mark.requires_gpu]


def _inputs(k, packed):
    sk = 2 * k
    positions = (
        [0, 6, 36, 37, 38, 128, 299, 300, 301, sk - 1] if packed else [sk - 1] * 19
    )
    sq = len(positions)
    device = "cuda"
    q = torch.zeros(1, sq, 64, 128, device=device, dtype=torch.bfloat16)
    keys = torch.zeros(1, sk, 1, 128, device=device, dtype=torch.bfloat16)
    weights = torch.zeros(1, sq, 64, device=device, dtype=torch.bfloat16)
    q[0, :, 0, :3] = 1
    ids = torch.arange(sk, device=device)
    keys[0, :, 0, 0] = 1
    keys[0, :, 0, 1] = (ids // 128) * (8 / k)
    keys[0, :, 0, 2] = (ids % 128) / (16 * k)
    weights[0, :, 0] = torch.where(torch.arange(sq, device=device) % 2 == 0, 1, -1)
    positions = torch.tensor(positions, device=device)
    cu = (
        torch.tensor([0, 37, 300, sk], device=device, dtype=torch.int32)
        if packed
        else None
    )
    # Every score is exactly +/- (1 + key_id / (16*K)); there are no ties.
    # All positive scores share the selector's first coarse fp16 radix bucket.
    return q, keys, weights, positions, cu


def _reference(k, positions, rows, cu, values):
    positions, rows = positions.cpu().tolist(), rows.cpu().tolist()
    bounds = cu.cpu().tolist() if cu is not None else [0, 2 * k]
    stats = torch.zeros(
        len(values), dsa_monitor.NUM_INDEXER_TOPK_STATS, dtype=torch.float64
    )
    for row in rows:
        pos = positions[row]
        start = max(bound for bound in bounds if bound <= pos)
        ranked = list(range(start, pos + 1))
        if row % 2 == 0:
            ranked.reverse()
        for i, value in enumerate(values):
            selected = torch.tensor(ranked[: min(value, k)], dtype=torch.float64)
            stats[i, 0] += selected.max() - selected.min()
            stats[i, 1] += pos - selected.min()
            stats[i, 2] += selected.std(correction=0)
            stats[i, 3] += 1
    return stats


@pytest.mark.parametrize(
    "k,packed", [(128, False), (2048, False), (4096, False), (512, True)]
)
@pytest.mark.parametrize("autocast", [False, True])
def test_sparse_monitor_score_ranked_prefixes(k, packed, autocast):
    q, keys, weights, positions, cu = _inputs(k, packed)
    ids, lengths = dsa_operator.select_topk(q, keys, weights, 1.0, k, positions, cu)
    rows = torch.arange(positions.numel() - 1, -1, -1, device="cuda")
    values = (1, 32, k)
    before = ids.clone()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        actual = dsa_monitor.sparse_indexer_topk_stats(
            ids,
            lengths,
            positions,
            rows,
            values,
            index_q=q,
            index_k=keys,
            weights=weights,
            index_scale=1.0,
        )
    expected = _reference(k, positions, rows, cu, values)
    torch.testing.assert_close(actual.cpu().double(), expected, atol=1e-4, rtol=2e-6)
    assert torch.equal(ids, before), (
        "monitor must not reorder the attention/replay selection"
    )


def test_sparse_monitor_full_k_does_not_rescore(monkeypatch):
    q, keys, weights, positions, cu = _inputs(128, False)
    ids, lengths = dsa_operator.select_topk(q, keys, weights, 1.0, 128, positions, cu)
    rows = torch.arange(positions.numel(), device="cuda")

    def unexpected_matmul(*args, **kwargs):
        raise AssertionError("full-K monitoring must not rescore the selection")

    monkeypatch.setattr(torch, "bmm", unexpected_matmul)
    actual = dsa_monitor.sparse_indexer_topk_stats(
        ids,
        lengths,
        positions,
        rows,
        (128,),
        index_q=q,
        index_k=keys,
        weights=weights,
        index_scale=1.0,
    )
    expected = _reference(128, positions, rows, cu, (128,))
    torch.testing.assert_close(actual.cpu().double(), expected, atol=1e-4, rtol=2e-6)


@pytest.mark.parametrize("k", [128, 2048, 4096])
def test_sparse_monitor_ignores_selection_order(k):
    q, keys, weights, positions, _ = _inputs(k, False)
    ids, lengths = dsa_operator.select_topk(q, keys, weights, 1.0, k, positions)
    rows = torch.arange(positions.numel(), device="cuda")
    values = (1, 32, k)
    expected = _reference(k, positions, rows, None, values)
    for selection in (ids, ids.flip(-1), ids.roll(k // 3, dims=-1)):
        actual = dsa_monitor.sparse_indexer_topk_stats(
            selection,
            lengths,
            positions,
            rows,
            values,
            index_q=q,
            index_k=keys,
            weights=weights,
            index_scale=1.0,
        )
        torch.testing.assert_close(actual.cpu().double(), expected, atol=1e-4, rtol=2e-6)


@pytest.mark.parametrize("empty_rows", [False, True])
def test_sparse_monitor_empty_queries(empty_rows):
    q, keys, weights, positions, _ = _inputs(128, False)
    ids = torch.full((1, positions.numel(), 128), -1, device="cuda", dtype=torch.int32)
    lengths = torch.zeros(positions.numel(), device="cuda", dtype=torch.int32)
    rows = torch.arange(0 if empty_rows else positions.numel(), device="cuda")
    actual = dsa_monitor.sparse_indexer_topk_stats(
        ids,
        lengths,
        positions,
        rows,
        (1, 32, 128),
        index_q=q,
        index_k=keys,
        weights=weights,
        index_scale=1.0,
    )
    assert torch.equal(actual, torch.zeros_like(actual))
