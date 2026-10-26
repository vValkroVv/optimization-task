"""HF loads accept state the format lacks; HF exports refuse to drop it."""

import pytest
import torch

from tests.unit_tests.config_helpers import load_model_spec

_LORA = {"config": {"converters": [{"type": "lora", "rank": 2}]}}
_QK_NORM = {
    "config": {"blocks": {"dense": {"attention": {"qk_norm": {"type": "rmsnorm"}}}}}
}
_MTP_MOE = {"config": {"mtp": {"layers": [{"block": "moe", "repeat": 1}]}}}
_MTP_MLA_MOE = {"config": {"mtp": {"layers": [{"block": "mla_moe", "repeat": 1}]}}}
_HEAD_DIM = {
    "config": {
        "rope": {"dim": 32},
        "blocks": {"dense": {"attention": {"head_dim": 32, "n_kv_heads": 4}}},
    }
}


def _adapter_and_state(preset, overrides=None):
    spec = load_model_spec(preset, overrides)
    with torch.device("meta"):
        model = spec.model.build()
    # keep_vars: tied entries stay one object
    return spec.state_dict_adapter(spec.model, None), model.state_dict(keep_vars=True)


@pytest.mark.parametrize(
    "preset,overrides",
    [
        ("qwen3/debugmodel_moe.yaml", None),
        ("qwen3/30b_a3b.yaml", None),
        ("qwen3_vl/debugmodel_moe.yaml", None),
        ("llama4/debugmodel.yaml", None),
        ("gpt_oss/debugmodel.yaml", None),
        ("qwen3/debugmodel.yaml", _LORA),
        ("llama3/debugmodel.yaml", _LORA),
        ("deepseek_v3/debugmodel.yaml", _LORA),
        ("gift_moe/debugmodel.yaml", _LORA),
        ("llama3/debugmodel.yaml", _QK_NORM),
        ("qwen3/debugmodel_moe.yaml", _MTP_MOE),
        ("deepseek_v3/debugmodel.yaml", _MTP_MLA_MOE),
    ],
)
def test_hf_load_skips_unrepresentable_state_and_export_rejects_it(preset, overrides):
    adapter, state = _adapter_and_state(preset, overrides)
    # The load path builds its HF request with to_hf; it must not raise.
    loaded = adapter.from_hf(adapter.to_hf(dict(state)))
    skipped = set(state) - set(loaded)
    assert skipped
    with pytest.raises(ValueError, match="HF export cannot represent") as raised:
        adapter.check_hf_export(state)
    assert sorted(skipped)[0] in str(raised.value)
    if overrides is not None and "mtp" in overrides["config"]:
        assert "mtp.layers.0." in str(raised.value)


@pytest.mark.parametrize(
    "preset,overrides",
    [
        ("llama3/debugmodel.yaml", None),
        ("llama3/debugmodel.yaml", _HEAD_DIM),
        ("llama3/debugmodel_fused_qkv.yaml", _HEAD_DIM),
        ("qwen3/debugmodel.yaml", None),
        ("qwen3/debugmodel_fused_qkv.yaml", None),
        ("deepseek_v3/debugmodel.yaml", None),
    ],
)
def test_representable_models_export_and_round_trip_exactly(preset, overrides):
    adapter, meta_state = _adapter_and_state(preset, overrides)
    adapter.check_hf_export(meta_state)
    torch.manual_seed(0)
    by_param = {}
    state = {
        k: by_param.setdefault(id(v), torch.randn(v.shape))
        for k, v in meta_state.items()
    }
    back = adapter.from_hf(adapter.to_hf(dict(state)))
    assert back.keys() == state.keys()
    for key, value in state.items():
        assert torch.equal(back[key], value), key


def test_heterogeneous_attention_export_is_a_value_error():
    dense2 = {
        "attention": {"type": "gqa", "n_heads": 16, "n_kv_heads": 4},
        "feed_forward": {"type": "llama", "multiple_of": 256},
        "attention_norm": {"type": "rmsnorm"},
        "ffn_norm": {"type": "rmsnorm"},
    }
    layers = [{"block": "dense", "repeat": 3}, {"block": "dense2", "repeat": 3}]
    adapter, state = _adapter_and_state(
        "llama3/debugmodel.yaml",
        {"config": {"blocks": {"dense2": dense2}, "layers": layers}},
    )
    with pytest.raises(ValueError, match="cannot map this model's state"):
        adapter.check_hf_export(state)


def test_adapter_key_error_on_export_is_a_value_error(monkeypatch):
    adapter, state = _adapter_and_state("llama3/debugmodel.yaml")

    def to_hf(state_dict):
        raise KeyError("layers.{}.unknown")

    monkeypatch.setattr(adapter, "to_hf", to_hf)
    with pytest.raises(ValueError, match="KeyError") as raised:
        adapter.check_hf_export(state)
    assert isinstance(raised.value.__cause__, KeyError)
