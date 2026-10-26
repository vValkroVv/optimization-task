from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.softmax.kernels.base import (
    InnerAttentionConfig,
)


@config
class VLLMAttentionWrapperConfig(InnerAttentionConfig):
    """Unbound parameters; vLLM binds its kernel when the runtime is imported."""

    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float | None = None
