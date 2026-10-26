"""Runtime checks for the older freeze, mask and HF review findings."""

from unittest.mock import patch

import pytest
import torch

from tests.unit_tests.config_helpers import load_experiment_config, load_model_spec


@pytest.mark.parametrize("selector", ["layers.*.attention", "layers.*.feed_forward"])
def test_freezing_preserves_forward_flop_accounting(selector):
    plain = load_model_spec("llama3/debugmodel.yaml").model
    frozen = load_model_spec(
        "llama3/debugmodel.yaml",
        {"config": {"converters": [{"type": "freeze", "selectors": [selector]}]}},
    ).model
    assert sum(b.fwd_flops_per_token(8) for b in plain.layers) == sum(
        b.fwd_flops_per_token(8) for b in frozen.layers
    )
    assert frozen.attention.attention_mask_kind == plain.attention.attention_mask_kind
    with torch.device("meta"):
        model = frozen.build()
    _, flops = frozen.owner().get_nparams_and_flops(frozen, model, 8)
    assert flops > 0


def test_llama4_masks_use_each_rope_blocks_chunk_size():
    config = load_experiment_config(
        "llama4_debug.yaml",
        "--model.config.blocks.dense.fixed_attn_block_size",
        "64",
        "--model.config.blocks.moe.fixed_attn_block_size",
        "128",
    )
    with torch.device("meta"):
        model = config.model_spec.model.build()
    with patch(
        "torchtitan.models.llama4.model.create_attention_mask",
        lambda mask, *args, **kwargs: mask,
    ):
        masks = model.get_attention_masks(torch.arange(128).reshape(1, -1))
    args = [torch.tensor(x) for x in (0, 0, 65, 63)]
    assert not masks["rope_64"](*args).item()
    assert masks["rope_128"](*args).item()
    assert masks["nope"](*args).item()


def test_gpt_oss_window_does_not_come_from_a_full_attention_block():
    config = load_experiment_config(
        "gpt_oss_debug.yaml",
        "--model.config.layers",
        "[{block: gpt_oss_sink_moe}, {block: sliding_gpt_oss_sink_moe}]",
        "--model.config.blocks.gpt_oss_sink_moe.attention.sliding_window_size",
        "128",
        "--model.config.blocks.sliding_gpt_oss_sink_moe.attention.sliding_window_size",
        "16",
    )
    with torch.device("meta"):
        model = config.model_spec.model.build()
    with patch(
        "torchtitan.models.gpt_oss.model.create_attention_mask",
        lambda mask, *args, **kwargs: mask,
    ):
        masks = model.get_attention_masks(torch.arange(64).reshape(1, -1))
    args = [torch.tensor(x) for x in (0, 0, 32, 0)]
    assert masks["basic_mask"](*args).item()
    assert not masks["sliding_window_mask"](*args).item()


def test_unknown_hf_state_is_rejected_instead_of_dropped():
    spec = load_model_spec("llama3/debugmodel.yaml")
    adapter = spec.state_dict_adapter(spec.model, None)
    with pytest.raises(ValueError, match="cannot represent"):
        adapter.check_hf_export({"layers.0.attention.extra.weight": torch.ones(2)})


def test_qwen_moe_export_rejects_unrepresentable_expert_bias():
    config = load_experiment_config("qwen3_moe_debug.yaml")
    spec = config.model_spec
    with torch.device("meta"):
        model = spec.model.build()
    adapter = spec.state_dict_adapter(spec.model, None)
    with pytest.raises(ValueError, match="expert_bias"):
        adapter.check_hf_export(model.state_dict())


def test_qwen_moe_without_correction_bias_round_trips_all_state_keys():
    config = load_experiment_config(
        "qwen3_moe_debug.yaml", "--model.config.blocks.moe.moe.load_balance_coeff", "0"
    )
    spec = config.model_spec
    with torch.device("meta"):
        model = spec.model.build()
    state = model.state_dict()
    adapter = spec.state_dict_adapter(spec.model, None)
    restored = adapter.from_hf(adapter.to_hf(dict(state)))
    assert state.keys() == restored.keys()
    assert all(value.shape == restored[key].shape for key, value in state.items())


@pytest.mark.parametrize("fused", [False, True])
def test_llama3_hf_conversion_uses_explicit_head_width(fused):
    attention = {"n_heads": 4, "head_dim": 32}
    if fused:
        attention["qkv_linear"] = {"type": "fused"}
    spec = load_model_spec(
        "llama3/debugmodel.yaml",
        {
            "config": {
                "rope": {"dim": 32},
                "blocks": {"dense": {"attention": attention}},
            }
        },
    )
    with torch.device("meta"):
        model = spec.model.build()
    native = {
        key: torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape)
        for key, value in model.state_dict().items()
        if key.startswith("layers.0.attention.qkv_linear.")
    }
    adapter = spec.state_dict_adapter(spec.model, None)
    restored = adapter.from_hf(adapter.to_hf(dict(native)))
    assert native.keys() == restored.keys()
    assert all(torch.equal(value, restored[key]) for key, value in native.items())


@pytest.mark.parametrize(
    "recipe,block_name,field,size,key,q,kv",
    [
        ("llama4_debug", "dense", "fixed_attn_block_size", 32, "rope_32", 32, 31),
        (
            "gpt_oss_debug",
            "sliding_gpt_oss_sink_moe",
            "attention.sliding_window_size",
            16,
            "sliding_window_16",
            32,
            0,
        ),
    ],
)
def test_mtp_blocks_get_their_own_mask_sizes(
    recipe, block_name, field, size, key, q, kv
):
    from copy import deepcopy
    from torchtitan.config import load_raw_document
    from torchtitan.config.decode import decode
    from torchtitan.config.sections import ModelSection

    raw = load_raw_document(f"configs/experiments/{recipe}.yaml").data["model"]
    block = deepcopy(raw["config"]["blocks"][block_name])
    target = block
    parts = field.split(".")
    for name in parts[:-1]:
        target = target[name]
    target[parts[-1]] = size
    raw["config"]["mtp"] = {"layers": [{"block": block}]}
    tree = decode(ModelSection, raw).config.materialize()
    with torch.device("meta"):
        model = tree.build()
    with patch(
        f"torchtitan.models.{tree.family}.model.create_attention_mask",
        lambda mask, *args, **kwargs: mask,
    ):
        masks = model.get_attention_masks(torch.arange(128).reshape(1, -1))
    assert key in masks
    assert not masks[key](*[torch.tensor(x) for x in (0, 0, q, kv)]).item()
