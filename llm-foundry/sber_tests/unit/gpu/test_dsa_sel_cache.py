"""Per-layer selection caches: GPU retention, CPU copies and checkpoint replay."""

import pytest

import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
from llmfoundry.models.layers import dsa_attention

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig  # noqa: E402

DEV = "cuda"


def make_layer(offload):
    cfg = GigaMixConfig(
        hidden_size=2048,
        num_attention_heads=64,
        num_key_value_heads=64,
        num_krope_heads=1,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        q_lora_rank=1536,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        norm_type="LlamaRMSNorm",
        attention_type="LlamaLatentSparseAttention",
        init_device=DEV,
        indexer_n_heads=64,
        indexer_head_dim=128,
        indexer_topk=128,
        aux_kl_loss_koeff=0.01,
        dsa_stage=2,
        dsa_sel_cache_offload=offload,
    )
    return dsa_attention.LlamaLatentSparseAttention(cfg, layer_idx=0).to(DEV).train()


def sel_pair(shape=(1, 4096, 2048), seed=0):
    g = torch.Generator(DEV).manual_seed(seed)
    ids = torch.randint(0, 1 << 20, shape, device=DEV, dtype=torch.int32, generator=g)
    ln = torch.randint(
        1, shape[-1], (shape[1],), device=DEV, dtype=torch.int32, generator=g
    )
    return ids, ln


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_default_keeps_selection_on_gpu():
    layer = make_layer(offload=False)
    assert layer.sel_cache_offload is False
    ids, ln = sel_pair()
    layer._sel_store(ids, ln)
    got_ids, got_ln = layer._sel_replay(DEV)
    assert got_ids.data_ptr() == ids.data_ptr() and got_ln.data_ptr() == ln.data_ptr()


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_offload_replay_matches_after_source_freed():
    layer = make_layer(offload=True)
    a = torch.randn(4096, 4096, device=DEV) / 128
    for it in range(5):
        ref_ids, ref_ln = sel_pair(seed=it)
        for _ in range(6):
            a = a @ a  # queue compute ahead of the copy so it starts late
        ids, ln = ref_ids.clone(), ref_ln.clone()
        layer._sel_store(ids, ln)
        del ids, ln
        junk = torch.empty(ref_ids.shape, device=DEV, dtype=torch.int32)
        junk.fill_(-777)  # tries to reuse the freed block before the copy ran
        got_ids, got_ln = layer._sel_replay(DEV)
        assert (
            got_ids.device.type == "cuda" and got_ids.data_ptr() != ref_ids.data_ptr()
        )
        assert torch.equal(got_ids, ref_ids), f"iter {it}: ids corrupted"
        assert torch.equal(got_ln, ref_ln), f"iter {it}: lengths corrupted"


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_offload_handles_shape_change():
    layer = make_layer(offload=True)
    for it, sq in enumerate([1024, 512, 1024, 768]):
        ref_ids, ref_ln = sel_pair(shape=(1, sq, 256), seed=10 + it)
        layer._sel_store(ref_ids.clone(), ref_ln.clone())
        got_ids, got_ln = layer._sel_replay(DEV)
        assert got_ids.shape == ref_ids.shape
        assert torch.equal(got_ids, ref_ids) and torch.equal(got_ln, ref_ln)


@pytest.mark.unit
@pytest.mark.requires_gpu
@pytest.mark.parametrize("offload", [False, True])
def test_replay_without_store_raises(offload):
    layer = make_layer(offload)
    with pytest.raises(AssertionError, match="no cached selection"):
        layer._sel_replay(DEV)


@pytest.mark.unit
@pytest.mark.requires_gpu
@pytest.mark.parametrize("offload", [False, True], ids=["gpu", "offload"])
def test_layers_keep_independent_selections(offload):
    layers = [make_layer(offload), make_layer(offload)]
    selections = [sel_pair(shape=(1, 128, 256), seed=seed) for seed in (31, 47)]
    for layer, (ids, lengths) in zip(layers, selections):
        layer._sel_store(ids.clone(), lengths.clone())

    assert layers[0]._sel_cache[0].data_ptr() != layers[1]._sel_cache[0].data_ptr()
    for layer, expected in reversed(list(zip(layers, selections))):
        ids, lengths = layer._sel_replay(DEV)
        assert torch.equal(ids, expected[0])
        assert torch.equal(lengths, expected[1])

    # Replacing one layer's cache must leave the other layer's selection intact.
    for sq in (128, 192):
        replacement = sel_pair(shape=(1, sq, 256), seed=sq)
        layers[0]._sel_store(*replacement)
        for layer, expected in zip(layers, [replacement, selections[1]]):
            ids, lengths = layer._sel_replay(DEV)
            assert torch.equal(ids, expected[0])
            assert torch.equal(lengths, expected[1])
