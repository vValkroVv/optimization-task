"""Initializer overrides use explicit child paths and persist in resolved YAML."""

import pytest

from tests.unit_tests.config_helpers import init_policy
from torchtitan.config.configurable import to_plain
from torchtitan.config.decode import decode
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.schema.models.common.ffn import FeedForwardConfig
from torchtitan.config.schema.models.common.attention.linear.gdn import (
    GatedDeltaNetConfig,
)
from torchtitan.config.schema.models.common.attention.linear.kda import (
    KimiDeltaAttentionConfig,
)
from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    GQAttentionConfig,
)
from torchtitan.config.schema.models.common.attention.softmax.mla import MLAConfig
from torchtitan.config.schema.models.common.moe.router import (
    TokenChoiceTopKRouterConfig,
)
from torchtitan.config.schema.models.common.init_policy import apply_init
from torchtitan.config.schema.protocols.module import Ones

GIFT_HYBRID_INIT = init_policy("gift_hybrid")


@pytest.mark.parametrize(
    "schema,alias",
    [
        (FeedForwardConfig, "w1_param_init"),
        (FeedForwardConfig, "w2w3_param_init"),
        (GQAttentionConfig, "qkv_param_init"),
        (GQAttentionConfig, "wo_param_init"),
        (GQAttentionConfig, "gate_param_init"),
        (MLAConfig, "linear_param_init"),
        (MLAConfig, "norm_param_init"),
        (GatedDeltaNetConfig, "in_proj_param_init"),
        (GatedDeltaNetConfig, "out_proj_param_init"),
        (GatedDeltaNetConfig, "conv_param_init"),
        (KimiDeltaAttentionConfig, "in_proj_param_init"),
        (TokenChoiceTopKRouterConfig, "gate_param_init"),
    ],
)
def test_flat_initializer_fields_are_rejected(schema, alias):
    with pytest.raises(
        ConfigSchemaError, match=f"{alias}.*unknown key|unknown key.*{alias}"
    ):
        decode(schema, {alias: {"weight": {"type": "zeros"}}})


@pytest.mark.parametrize("child", ["w1", "w2", "w3"])
def test_nested_initializer_survives_policy_and_resolved_reload(child):
    config = decode(
        FeedForwardConfig,
        {
            "dim": 8,
            "hidden_dim": 16,
            child: {"param_init": {"weight": {"type": "ones"}}},
        },
    )
    apply_init(config, GIFT_HYBRID_INIT, layer_id=2, dim=8)
    resolved = to_plain(config)
    assert resolved[child]["param_init"]["weight"]["type"] == "ones"
    reloaded = decode(FeedForwardConfig, resolved)
    apply_init(reloaded, GIFT_HYBRID_INIT, layer_id=2, dim=8)
    assert isinstance(getattr(reloaded, child).param_init["weight"], Ones)
    assert to_plain(reloaded) == resolved
