"""GIFT runtime plans and HF conversion follow modules rather than preset labels."""

from copy import deepcopy

import pytest
import torch

from torchtitan.config import load_raw_document
from torchtitan.config.build import prepare_document
from torchtitan.models.common.decoder_blocks.sharding import (
    gated_norm_config,
    norm_config,
)
from torchtitan.models.gift_moe._base.sharding import set_gift_moe_sharding_config
from torchtitan.models.gift_moe._base.state_dict_adapter import GiftMoEStateDictAdapter
from torchtitan.models.gift_moe.gift_3_5.sharding import set_gift_hybrid_sharding_config
from torchtitan.models.gift_moe.versions import BASE, HYBRID


def _prepared(variant):
    recipe = "gift_3_5_debug" if variant == "gdn" else "gift_moe_debug"
    raw = load_raw_document(f"configs/experiments/{recipe}.yaml")
    model = raw.data["model"]["config"]
    model["type"] = "gift_moe"
    if variant == "postnorm":
        for block in model["blocks"].values():
            block["post_attn_norm"] = deepcopy(model["norm"])
            block["post_ffn_norm"] = deepcopy(model["norm"])
    if variant == "gated":
        hybrid_init = load_raw_document("configs/presets/init/gift_hybrid.yaml")
        model["init"]["gated_norm"] = hybrid_init.data["model"]["config"]["init"][
            "gated_norm"
        ]
        model["norm"]["type"] = "gated_rmsnorm"
        for block in model["blocks"].values():
            for name in ("attention_norm", "ffn_norm"):
                block[name]["type"] = "gated_rmsnorm"
            block["attention"]["kv_norm"]["type"] = "gated_rmsnorm"
    return prepare_document(raw)


def test_compatibility_entrypoint_and_runtime_adapter_share_implementation():
    assert set_gift_hybrid_sharding_config is set_gift_moe_sharding_config
    assert BASE.state_dict_adapter is HYBRID.state_dict_adapter


@pytest.mark.parametrize("sp", [False, True])
def test_plain_norm_parameter_and_activation_placements_stay_identical(sp):
    ordinary, gated = norm_config(enable_sp=sp), gated_norm_config(enable_sp=sp)
    assert ordinary.state_shardings["weight"] == gated.state_shardings["weight"]
    assert ordinary.in_src_shardings == gated.in_src_shardings
    assert ordinary.in_dst_shardings == gated.in_dst_shardings
    assert ordinary.out_dst_shardings == gated.out_dst_shardings
    assert not set(ordinary.state_shardings) - set(gated.state_shardings)


@pytest.mark.parametrize("variant", ["plain", "postnorm", "gated", "gdn"])
def test_base_composition_prepare_build_and_hf_round_trip(variant):
    prepared = _prepared(variant)
    tree = prepared.model_config
    tree.owner().prepare_config(tree, config=prepared.trainer_config)
    assert tree.validate_hf_export() == []
    with torch.device("meta"):
        model = tree.build()
    # Distinct per-parameter values detect missing/swapped projection or norm keys.
    state = {
        name: torch.tensor(float(index + 1)).expand(value.shape)
        for index, (name, value) in enumerate(model.state_dict().items())
        if not name.endswith("freqs_cis")
    }
    adapter = BASE.state_dict_adapter(tree, hf_assets_path=None)
    exported = adapter.to_hf(state)
    restored = adapter.from_hf(exported)
    assert restored.keys() == state.keys()
    for name, value in state.items():
        assert torch.equal(restored[name], value), name
    if variant == "plain":
        legacy = GiftMoEStateDictAdapter(tree, hf_assets_path=None).to_hf(state)
        assert exported.keys() == legacy.keys()
        assert all(torch.equal(exported[key], value) for key, value in legacy.items())
    expected_extra = {
        "postnorm": "post_self_attn_layernorm.weight",
        "gated": "gate_up_projection.weight",
        "gdn": "in_proj_qkvz.weight",
    }.get(variant)
    if expected_extra is not None:
        assert any(expected_extra in name for name in exported)
