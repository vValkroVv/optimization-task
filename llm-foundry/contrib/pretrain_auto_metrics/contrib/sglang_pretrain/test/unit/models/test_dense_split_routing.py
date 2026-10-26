"""Layer-class routing / split decisions for dense_attention_layers.

Covers the predicates the dense/DSA split and the KV-budget accounting are gated
on: which full-attn layers run plain MLA, when the non-hybrid split fires, and
the fp8/bf16 + NPU gating of routes_dense_layers_to_plain_mla.
"""

from types import SimpleNamespace as NS

import pytest

from sglang.srt.configs.model_config import (
    dsa_has_active_dsa_layers,
    get_dsa_dense_full_layer_ids,
    get_dsa_sparse_attention_layers,
    get_plain_mla_full_layer_ids,
    is_deepseek_dsa,
    is_dsa_dense_split_model,
    routes_dense_layers_to_plain_mla,
)

FULL = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]


def _hybrid(dense):
    # GDN + DSA (40 layers, 10 full-attn); the rest are linear layers.
    return NS(
        architectures=["Gigachat35ForCausalLM"],
        index_topk=2048,
        num_hidden_layers=40,
        full_attention_layers=list(FULL),
        full_attention_layer_ids=list(FULL),
        linear_attention_type="Qwen3NextGatedDeltaNet",
        dense_attention_layers=list(dense),
    )


def _nonhybrid(dense, n=5):
    # No GDN: every layer is a full-attn DSA layer.
    return NS(
        architectures=["Gigachat35ForCausalLM"],
        index_topk=2048,
        num_hidden_layers=n,
        full_attention_layers=list(range(n)),
        full_attention_layer_ids=list(range(n)),
        linear_attention_type=None,
        dense_attention_layers=list(dense),
    )


def _true_dense(n=5):
    # Ordinary dense MLA+GDN, no indexer at all (index_topk absent -> not DSA).
    return NS(
        architectures=["Gigachat35ForCausalLM"],
        num_hidden_layers=n,
        full_attention_layers=[2],
        full_attention_layer_ids=[2],
        linear_attention_type="Qwen3NextGatedDeltaNet",
    )


@pytest.mark.unit
def test_mixed_hybrid():
    c = _hybrid(dense=[3, 15])
    assert is_deepseek_dsa(c)
    assert get_dsa_dense_full_layer_ids(c) == [3, 15]
    assert 3 not in get_dsa_sparse_attention_layers(c)
    assert dsa_has_active_dsa_layers(c)  # 8 sparse full-attn layers remain
    assert routes_dense_layers_to_plain_mla(c, "fp8_e4m3")
    assert not is_dsa_dense_split_model(c)  # hybrid split lives in HybridLinearAttnBackend


@pytest.mark.unit
def test_all_dense_hybrid():
    c = _hybrid(dense=list(FULL))
    assert get_dsa_dense_full_layer_ids(c) == FULL
    assert not dsa_has_active_dsa_layers(c)  # no sparse layer left
    assert routes_dense_layers_to_plain_mla(c, "fp8_e4m3")
    assert not is_dsa_dense_split_model(c)


@pytest.mark.unit
def test_nonhybrid_split():
    c = _nonhybrid(dense=[2], n=5)
    assert is_dsa_dense_split_model(c)  # all full-attn, >=1 dense and >=1 sparse
    assert get_dsa_dense_full_layer_ids(c) == [2]
    assert routes_dense_layers_to_plain_mla(c, "fp8_e4m3")


@pytest.mark.unit
def test_pure_dsa_no_dense():
    c = _hybrid(dense=[])
    assert get_dsa_dense_full_layer_ids(c) == []
    assert not get_plain_mla_full_layer_ids(c, FULL) == FULL  # no dense -> []
    assert not routes_dense_layers_to_plain_mla(c, "fp8_e4m3")
    assert not is_dsa_dense_split_model(c)


@pytest.mark.unit
def test_fp8_gate_non_dsa_true_dense():
    c = _true_dense()
    assert not is_deepseek_dsa(c)
    # ordinary dense Gigachat35: flashmla only under fp8, stays on fa3 under bf16
    assert routes_dense_layers_to_plain_mla(c, "fp8_e4m3")
    assert not routes_dense_layers_to_plain_mla(c, "bfloat16")


@pytest.mark.unit
def test_npu_gate():
    c = _hybrid(dense=[3, 15])
    assert routes_dense_layers_to_plain_mla(c, "fp8_e4m3", is_npu=False)
    assert not routes_dense_layers_to_plain_mla(c, "fp8_e4m3", is_npu=True)
