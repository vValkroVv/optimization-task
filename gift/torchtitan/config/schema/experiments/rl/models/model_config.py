"""Adapt a completed decoder tree to the runtime-only vLLM kernel."""

from copy import deepcopy
from typing import TypeVar

from torchtitan.config.schema.components.lora import unwrap_frozen
from torchtitan.config.schema.experiments.rl.models.attention import (
    VLLMAttentionWrapperConfig,
)
from torchtitan.config.schema.models.common.attention.const import MaskType
from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    GQAttentionConfig,
)
from torchtitan.config.schema.models.common.decoder_blocks.decoder import DecoderConfig

_Decoder = TypeVar("_Decoder", bound=DecoderConfig)


def with_vllm_attention(
    model_config: _Decoder, kernel_config_type: type[VLLMAttentionWrapperConfig]
) -> _Decoder:
    tree = deepcopy(model_config)
    for index, layer in enumerate(tree.placed_layers()):
        attention = unwrap_frozen(layer.attention)
        if not isinstance(attention, GQAttentionConfig):
            raise ValueError(f"layers.{index}.attention: vLLM requires GQA attention")
        if attention.dim is None:
            raise ValueError(
                f"layers.{index}.attention: vLLM needs completed dimensions"
            )
        backend = kernel_config_type(
            hidden_size=attention.dim,
            num_heads=attention.n_heads,
            num_kv_heads=attention.n_kv_heads or attention.n_heads,
            head_dim=attention.head_dim or attention.dim // attention.n_heads,
        )
        # vLLM owns causal masking and sequence boundaries through its KV metadata.
        layer.attention = layer.attention._copy(
            inner_attention=backend, mask_type=MaskType.CAUSAL
        )
    tree.validate_materialized()
    return tree
