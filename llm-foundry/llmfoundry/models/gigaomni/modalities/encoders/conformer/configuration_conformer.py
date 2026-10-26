import typing as tp

from transformers import PretrainedConfig


class ConformerConfig(PretrainedConfig):
    """Configuration for the GigaOmni Conformer audio encoder."""

    model_type: str = "giga_omni_conformer"
    ALLOWED_ATTN_TYPES: tuple[str] = ("chunk_rotary",)
    ALLOWED_SUBSAMPLING_TYPES: tuple[str] = (
        "striding_conv1d",
        "striding",
        "stacking",
    )

    def __init__(
        self,
        feat_in: int = 64,
        n_layers: int = 24,
        d_model: int = 1024,
        feat_out: int = -1,
        subsampling: str = "striding_conv1d",
        subsampling_factor: int = 4,
        subsampling_conv_channels: int = 1024,
        ff_expansion_factor: int = 4,
        self_attention_model: str = "chunk_rotary",
        n_heads: int = 16,
        pos_emb_max_len: int = 5000,
        conv_kernel_size: int = 5,
        conv_norm_type: str = "layer_norm",
        dropout: float = 0.1,
        dropout_emb: float = 0.1,
        dropout_att: float = 0.1,
        attn_chunk_size: int = 200,
        max_position_seq_len: int = 131072,
        use_causal_conv_chunking: bool = False,
        use_causal_conv: bool = False,
        window_size: tp.Tuple[int, int] = (-1, -1),
        attention_backend: str = "auto",
        use_per_layer_wrapping: bool = False,
        length_bucketing: bool = True,
        **kwargs: tp.Dict,
    ) -> None:
        super().__init__(**kwargs)

        self.feat_in = feat_in
        self.n_layers = n_layers
        self.d_model = d_model
        self.feat_out = feat_out

        self.subsampling = subsampling
        assert subsampling in self.ALLOWED_SUBSAMPLING_TYPES, (
            f"Only {self.ALLOWED_SUBSAMPLING_TYPES} allowed, but found {subsampling}!"
        )

        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.ff_expansion_factor = ff_expansion_factor


        self.self_attention_model = self_attention_model
        assert self_attention_model in self.ALLOWED_ATTN_TYPES, (
            f"Only {self.ALLOWED_ATTN_TYPES} allowed, but found "
            f"{self_attention_model}!"
        )

        self.attn_chunk_size = attn_chunk_size
        self.n_heads = n_heads
        self.conv_kernel_size = conv_kernel_size
        self.conv_norm_type = conv_norm_type
        self.dropout = dropout
        self.dropout_emb = dropout_emb
        self.dropout_att = dropout_att
        self.use_causal_conv_chunking = use_causal_conv_chunking
        self.use_causal_conv = use_causal_conv
        self.window_size = tuple(window_size)
        self.pos_emb_max_len = pos_emb_max_len
        self.max_position_seq_len = max_position_seq_len

        if attention_backend not in ("auto", "flash_attn", "sdpa"):
            raise ValueError(
                "attention_backend must be one of 'auto', 'flash_attn', or "
                f"'sdpa', got {attention_backend!r}"
            )
        self.attention_backend = attention_backend

        self.use_per_layer_wrapping = use_per_layer_wrapping
        self.length_bucketing = length_bucketing
