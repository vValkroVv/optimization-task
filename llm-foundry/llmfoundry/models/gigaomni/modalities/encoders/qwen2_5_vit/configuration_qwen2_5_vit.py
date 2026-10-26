from typing import Optional, Literal
from transformers.configuration_utils import PretrainedConfig


class Qwen2_5VisionConfig(PretrainedConfig):
    model_type = "qwen2_5_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 32,
        hidden_size: int = 1280,
        hidden_act: str = "silu",
        intermediate_size: int = 3456,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        tokens_per_second: int = 2,
        window_size: int = 112,
        out_hidden_size: int = 8192,
        fullatt_block_indexes: list[int] = [7, 15, 23, 31],
        initializer_range: float = 0.02,
        attn_impl: Optional[Literal["eager", "flash_attention_2"]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range
        self.image_size = spatial_merge_size * patch_size
        self.attn_impl = attn_impl
