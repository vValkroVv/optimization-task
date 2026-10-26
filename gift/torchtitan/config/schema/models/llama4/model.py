from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    GQAttentionConfig,
)
from torchtitan.config.schema.components.lora import unwrap_frozen
from torchtitan.config.schema.models.common.attention.softmax.kernels.flex import (
    FlexAttentionConfig,
)
from torchtitan.config.schema.models.common.decoder_blocks.decoder import DecoderConfig
from torchtitan.config.schema.models.common.decoder_blocks.transformer_block import (
    ResidualTransformerBlockConfig,
)


@config(type="llama4_chunked")
class Llama4TransformerBlockConfig(ResidualTransformerBlockConfig):
    """Llama 4's block: RoPE layers attend within fixed chunks (iRoPE)."""

    _owner_spec: ClassVar[str] = "torchtitan.models.llama4.model:Llama4TransformerBlock"
    # The chunked mask has no Context Parallel path.
    supports_context_parallel: ClassVar[bool] = False

    fixed_attn_block_size: int = 8192
    """Chunk size of the local attention mask on RoPE layers."""

    def __post_init__(self) -> None:
        if (
            type(self.fixed_attn_block_size) is not int
            or self.fixed_attn_block_size <= 0
        ):
            raise ConfigFieldError(
                "fixed_attn_block_size", "must be a positive integer"
            )
        # The chunked and NoPE masks are FlexAttention block masks.
        kernel = unwrap_frozen(getattr(self.attention, "inner_attention", None))
        if kernel is not None and not isinstance(kernel, FlexAttentionConfig):
            raise ValueError(
                "llama4_chunked blocks need attention.inner_attention: "
                f"{{type: flex}}, got {type(kernel).__name__}"
            )
        # slots=True breaks super(); call the parent explicitly.
        ResidualTransformerBlockConfig.__post_init__(self)


@config(type="llama4")
class Llama4ModelConfig(DecoderConfig):
    attention_types: ClassVar[tuple[type, ...]] = (GQAttentionConfig,)
    _owner_spec: ClassVar[str] = "torchtitan.models.llama4.model:Llama4Model"
    family: ClassVar[str] = "llama4"
    _runtime_spec: ClassVar[str] = "torchtitan.models.llama4.versions:BASE"
    block_types: ClassVar[tuple[type, ...]] = (Llama4TransformerBlockConfig,)
