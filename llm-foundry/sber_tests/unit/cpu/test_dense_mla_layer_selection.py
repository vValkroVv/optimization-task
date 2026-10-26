from types import SimpleNamespace

import pytest

from llmfoundry.models.layers.blocks import _resolve_attention_type


def _config(**overrides):
    values = {
        "attention_type": "LlamaLatentSparseAttention",
        "linear_attention_type": None,
        "full_attention_layers": [],
        "dense_attention_layers": [],
        "num_hidden_layers": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
@pytest.mark.fast
def test_exact_dense_attention_layers_use_dense_mla():
    config = _config(dense_attention_layers=[1, 5])

    resolved = [
        _resolve_attention_type(config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)
    ]

    assert resolved[0] == "LlamaLatentSparseAttention"
    assert resolved[1] == "LlamaLatentAttention"
    assert resolved[5] == "LlamaLatentAttention"
    assert resolved[6] == "LlamaLatentSparseAttention"


@pytest.mark.unit
@pytest.mark.fast
def test_exact_dense_attention_layers_in_hybrid_model():
    config = _config(
        linear_attention_type="Qwen3NextGatedDeltaNet",
        full_attention_layers=[3, 7],
        dense_attention_layers=[7],
    )

    assert _resolve_attention_type(config, 0) == "Qwen3NextGatedDeltaNet"
    assert _resolve_attention_type(config, 3) == "LlamaLatentSparseAttention"
    assert _resolve_attention_type(config, 7) == "LlamaLatentAttention"


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize("value", [[-1], [1.5], [True], [None], "1", ()])
def test_dense_attention_layers_must_contain_valid_indices(value):
    config = _config(dense_attention_layers=value)

    with pytest.raises(ValueError, match="dense_attention_layers"):
        _resolve_attention_type(config, 0)


@pytest.mark.unit
@pytest.mark.fast
def test_dense_attention_layers_must_be_unique():
    config = _config(dense_attention_layers=[1, 1])

    with pytest.raises(ValueError, match="unique"):
        _resolve_attention_type(config, 0)


@pytest.mark.unit
@pytest.mark.fast
def test_dense_attention_layers_requires_dsa():
    config = _config(
        attention_type="LlamaLatentAttention",
        dense_attention_layers=[1],
    )

    with pytest.raises(ValueError, match="supported only"):
        _resolve_attention_type(config, 0)


@pytest.mark.unit
@pytest.mark.fast
def test_dense_attention_layers_must_be_full_attention_layers():
    config = _config(
        linear_attention_type="Qwen3NextGatedDeltaNet",
        full_attention_layers=[3, 7],
        dense_attention_layers=[5],
    )

    with pytest.raises(ValueError, match="subset of full_attention_layers"):
        _resolve_attention_type(config, 0)


@pytest.mark.unit
@pytest.mark.fast
def test_dense_attention_layers_must_be_in_range():
    config = _config(dense_attention_layers=[8])

    with pytest.raises(ValueError, match="out-of-range"):
        _resolve_attention_type(config, 0)
