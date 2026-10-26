from transformers.configuration_utils import PretrainedConfig


class InternVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternVisionModel`]. It is used to
    instantiate a vision encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            Number of color channels in the input images (e.g., 3 for RGB).
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        image_size (`int`, *optional*, defaults to 448):
            The size (resolution) of each image.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries and values in the self-attention layers.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        qk_normalization (`bool`, *optional*, defaults to `False`):
            Whether to normalize the queries and keys in the self-attention layers.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        use_flash_attn (`bool`, *optional*, defaults to `False`):
            Whether to use flash attention mechanism.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate for stochastic depth.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for layer scale.
        get_tokens_from_layer (`int`, *optional*, defaults to -1):
            The layer to get the tokens for language modeling from.
        deterministic_attention ('bool'. *optional*, defaults to False)
            Whether to use determinism in flash attention. It may reduce performance
        use_per_layer_wrapping: bool = True,
            Whether to use per-layer wrapping for GigaFSDP.
    """

    model_type = "intern_vit"

    def __init__(
        self,
        num_channels: int = 3,
        patch_size: int = 14,
        image_size: int = 448,
        qkv_bias: bool = True,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        qk_normalization: bool = False,
        num_hidden_layers: int = 24,
        use_flash_attn: bool = False,
        hidden_act: str = "gelu",
        norm_type: str = "layer_norm",
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        initializer_factor: float = 1.0,
        get_tokens_from_layer: int = -1,
        deterministic_attention: bool = False,
        use_per_layer_wrapping: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.qkv_bias = qkv_bias
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.qk_normalization = qk_normalization
        self.num_hidden_layers = num_hidden_layers
        self.use_flash_attn = use_flash_attn
        self.hidden_act = hidden_act
        self.norm_type = norm_type
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.get_tokens_from_layer = get_tokens_from_layer
        self.deterministic_attention = deterministic_attention
        self.use_per_layer_wrapping = use_per_layer_wrapping
