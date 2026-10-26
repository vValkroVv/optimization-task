"""KV-pool accounting for the dense/DSA split, checked against the real
allocator (constructed on CPU, tiny size).

The split pool is exactly the sum of its DSA and dense sub-pools, and each dense
(indexer-removed) layer costs strictly less than a sparse DSA layer -- by the
indexer cache. This is the property _compute_cell_size mirrors by charging the
indexer only on the sparse layers.
"""

import pytest
import torch

from sglang.srt.mem_cache.dense_split_kv_pool import DenseSplitDSAKVPool
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool

SIZE = 256
PAGE = 64
DTYPE = torch.float8_e4m3fn
KV_LORA = 512
QK_ROPE = 64
IDX_HEAD = 128
KV_DIM = 656


def _dsa(layer_num):
    return DSATokenToKVPool(
        size=SIZE,
        page_size=PAGE,
        kv_lora_rank=KV_LORA,
        dtype=DTYPE,
        qk_rope_head_dim=QK_ROPE,
        layer_num=layer_num,
        device="cpu",
        index_head_dim=IDX_HEAD,
        enable_memory_saver=False,
        kv_cache_dim=KV_DIM,
    )


def _mla(layer_num):
    return MLATokenToKVPool(
        size=SIZE,
        page_size=PAGE,
        dtype=DTYPE,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=QK_ROPE,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
    )


def _split(dsa_ids, dense_ids):
    return DenseSplitDSAKVPool(
        size=SIZE,
        page_size=PAGE,
        dtype=DTYPE,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=QK_ROPE,
        dsa_layer_ids=dsa_ids,
        dense_layer_ids=dense_ids,
        device="cpu",
        index_head_dim=IDX_HEAD,
        enable_memory_saver=False,
        kv_cache_dim=KV_DIM,
        start_layer=0,
        end_layer=len(dsa_ids) + len(dense_ids),
    )


@pytest.mark.unit
def test_split_pool_is_sum_of_subpools():
    # The allocator: split == its DSA sub-pool + its dense sub-pool, built with
    # the same per-class layer counts. Catches estimator/allocator drift.
    dsa_ids, dense_ids = [0, 2, 3], [1, 4]
    split = _split(dsa_ids, dense_ids).get_kv_size_bytes()
    expected = _dsa(len(dsa_ids)).get_kv_size_bytes() + _mla(len(dense_ids)).get_kv_size_bytes()
    assert split == expected


@pytest.mark.unit
def test_dense_layer_marginal_cheaper_than_sparse():
    # Marginal (per-layer) cost, isolating the fixed per-pool overhead: a dense
    # plain-MLA layer drops the indexer k cache, so it is strictly cheaper.
    dsa_marginal = _dsa(3).get_kv_size_bytes() - _dsa(2).get_kv_size_bytes()
    mla_marginal = _mla(3).get_kv_size_bytes() - _mla(2).get_kv_size_bytes()
    assert mla_marginal < dsa_marginal


@pytest.mark.unit
def test_more_dense_layers_shrink_the_pool():
    # Moving a layer from sparse to dense lowers the total (the #7 premise: the
    # indexer budget is charged on fewer layers).
    fewer_dense = _split([0, 1, 2, 3], [4]).get_kv_size_bytes()
    more_dense = _split([0, 1], [2, 3, 4]).get_kv_size_bytes()
    assert more_dense < fewer_dense


def _stub_mr(dense_ids, n):
    import types

    hf = types.SimpleNamespace(
        architectures=["Gigachat35ForCausalLM"],
        index_topk=2048,
        index_head_dim=IDX_HEAD,
        index_n_heads=64,
        full_attention_layers=list(range(n)),
        dense_attention_layers=dense_ids,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=QK_ROPE,
        num_hidden_layers=n,
    )
    mc = types.SimpleNamespace(
        hf_config=hf, kv_lora_rank=KV_LORA, qk_rope_head_dim=QK_ROPE
    )
    return types.SimpleNamespace(
        model_config=mc,
        kv_cache_dtype=DTYPE,
        use_mla_backend=True,
        calculate_mla_kv_cache_dim=lambda: KV_DIM,
        start_layer=0,
        end_layer=n,
        num_effective_layers=n,
    )


@pytest.mark.unit
def test_compute_cell_size_charges_indexer_only_on_dense(monkeypatch):
    # Exercise the estimator itself (_compute_cell_size), not only the allocator:
    # each dense (indexer-removed) layer must drop exactly the indexer per-token
    # budget that a sparse DSA layer pays -- the #7 change.
    import sglang.srt.model_executor.pool_configurator as pc

    # The MLA path does not use tp_size, but it is read unconditionally and the
    # attn-TP group is not initialized in a CPU unit test.
    monkeypatch.setattr(pc, "get_attention_tp_size", lambda: 1)

    n = 5
    cell = lambda dense: pc.DefaultPoolConfigurator._compute_cell_size(
        None, _stub_mr(dense, n), n
    )
    indexer_per_token = (
        IDX_HEAD + IDX_HEAD // DSATokenToKVPool.quant_block_size * 4
    ) * torch._utils._element_size(DSATokenToKVPool.index_k_with_scale_buffer_dtype)

    all_sparse = cell([])
    two_dense = cell([1, 4])
    assert two_dense < all_sparse
    assert all_sparse - two_dense == 2 * indexer_per_token
