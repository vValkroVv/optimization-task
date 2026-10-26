import math
from dataclasses import field, fields
from typing import ClassVar

from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.configurable import config, Derived, OwnerFilled
from torchtitan.config.schema.models.common.attention.softmax.kernels.flex import (
    FlexAttentionConfig,
)
from torchtitan.config.schema.models.common.nn_modules import GELUConfig, LinearConfig
from torchtitan.config.schema.models.common.norms.layernorm import LayerNormConfig
from torchtitan.config.schema.models.common.decoder_blocks.layer_group import (
    LayerGroupConfig,
    check_block_names,
    count_layers,
    place_blocks,
    placed_configs,
)
from torchtitan.config.schema.models.common.validation import (
    resolve_child,
    validate_values,
)
from torchtitan.config.schema.protocols.blocks import BlockConfig
from torchtitan.config.schema.protocols.module import (
    ModuleConfig,
    Ones,
    TruncNormal,
    Zeros,
)


_LINEAR_INIT = {"weight": TruncNormal(std=0.02), "bias": Zeros()}
_LAYER_NORM_INIT = {"weight": Ones(), "bias": Zeros()}
_POS_EMBED_INIT = {"pos_embed": TruncNormal(std=0.02, mean=0.0)}
_EPS = 1e-6


def _vl_linear(child, in_features: int, out_features: int, name: str) -> LinearConfig:
    return resolve_child(
        child,
        LinearConfig,
        name=name,
        dims=dict(in_features=in_features, out_features=out_features),
        require=dict(bias=True),
        init=_LINEAR_INIT,
    )


def _vl_layer_norm(norm: LayerNormConfig | None, normalized_shape: int, name: str):
    # Explicit init: the policy's norm table names ``weight`` only, a
    # LayerNorm has a bias too.
    if norm is None:
        norm = LayerNormConfig(eps=_EPS, param_init=dict(_LAYER_NORM_INIT))
    return resolve_child(
        norm, LayerNormConfig, name=name, dims={"normalized_shape": normalized_shape}
    )


@config
class PatchEmbedConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.qwen3_vl.vision_encoder:PatchEmbed"

    patch_size: int
    temporal_patch_size: int
    in_channels: int = 3
    # Filled by the encoder: its width, then the projection to it.
    dim: OwnerFilled[int] = None
    proj: Derived[LinearConfig] = None

    def __post_init__(self) -> None:
        if self.dim is not None:
            patch_dim = (
                self.in_channels
                * self.temporal_patch_size
                * self.patch_size
                * self.patch_size
            )
            self.proj = _vl_linear(self.proj, patch_dim, self.dim, "proj")


@config
class VisionRotaryEmbeddingConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.qwen3_vl.vision_encoder:VisionRotaryEmbedding"
    )

    # Filled by the encoder: half its head width.
    dim: OwnerFilled[int] = None
    theta: float = 10000.0


@config
class PatchMergerConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.qwen3_vl.vision_encoder:PatchMerger"

    use_postshuffle_norm: bool = False
    act_fn: GELUConfig = field(default_factory=lambda: GELUConfig(approximate="tanh"))
    # Filled by the encoder.
    hidden_size: OwnerFilled[int] = None
    spatial_merge_size: OwnerFilled[int] = None
    out_hidden_size: OwnerFilled[int] = None
    fc1: Derived[LinearConfig] = None
    fc2: Derived[LinearConfig] = None
    norm: Derived[LayerNormConfig] = None

    def __post_init__(self) -> None:
        hidden_size, spatial_merge_size, out_hidden_size = (
            self.hidden_size,
            self.spatial_merge_size,
            self.out_hidden_size,
        )
        if hidden_size is None or spatial_merge_size is None or out_hidden_size is None:
            return  # partial: the encoder completes it
        merged = hidden_size * spatial_merge_size**2
        self.fc1 = _vl_linear(self.fc1, merged, merged, "fc1")
        self.fc2 = _vl_linear(self.fc2, merged, out_hidden_size, "fc2")
        self.norm = _vl_layer_norm(
            self.norm,
            merged if self.use_postshuffle_norm else hidden_size,
            f"{type(self).__qualname__}.norm",
        )


@config
class VisionAttentionConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.qwen3_vl.vision_encoder:VisionAttention"
    )

    flex_attention: FlexAttentionConfig = field(
        default_factory=lambda: FlexAttentionConfig()
    )
    # Filled by the block.
    dim: OwnerFilled[int] = None
    n_heads: OwnerFilled[int] = None
    qkv: Derived[LinearConfig] = None
    proj: Derived[LinearConfig] = None

    def __post_init__(self) -> None:
        if self.dim is None or self.n_heads is None:
            return
        self.qkv = _vl_linear(self.qkv, self.dim, self.dim * 3, "qkv")
        self.proj = _vl_linear(self.proj, self.dim, self.dim, "proj")


@config
class VisionMLPConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.qwen3_vl.vision_encoder:VisionMLP"

    hidden_dim: int
    act_fn: GELUConfig = field(default_factory=lambda: GELUConfig(approximate="tanh"))
    # Filled by the block.
    dim: OwnerFilled[int] = None
    fc1: Derived[LinearConfig] = None
    fc2: Derived[LinearConfig] = None

    def __post_init__(self) -> None:
        if self.dim is None:
            return
        self.fc1 = _vl_linear(self.fc1, self.dim, self.hidden_dim, "fc1")
        self.fc2 = _vl_linear(self.fc2, self.hidden_dim, self.dim, "fc2")


@config
class VisionTransformerBlockConfig(BlockConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.qwen3_vl.vision_encoder:VisionTransformerBlock"
    )

    mlp: VisionMLPConfig
    attn: VisionAttentionConfig = field(default_factory=VisionAttentionConfig)
    norm1: Derived[LayerNormConfig] = None
    norm2: Derived[LayerNormConfig] = None
    # Filled by the encoder.
    dim: OwnerFilled[int] = None
    n_heads: OwnerFilled[int] = None

    def __post_init__(self) -> None:
        if self.dim is None or self.n_heads is None:
            return
        owner = type(self).__qualname__
        self.attn = resolve_child(
            self.attn,
            type(self.attn),
            name=f"{owner}.attn",
            dims={"dim": self.dim, "n_heads": self.n_heads},
        )
        self.mlp = resolve_child(
            self.mlp, type(self.mlp), name=f"{owner}.mlp", dims={"dim": self.dim}
        )
        self.norm1 = _vl_layer_norm(self.norm1, self.dim, f"{owner}.norm1")
        self.norm2 = _vl_layer_norm(self.norm2, self.dim, f"{owner}.norm2")


@config
class Qwen3VLVisionEncoderConfig(ModuleConfig):
    """Qwen3-VL vision encoder (ViT).

    ``pos_embed`` is a parameter of the encoder itself, so its shape fields
    (``num_position_embeddings``, ``dim``) live here. Layers are placed from
    named ``blocks`` as in the decoder.
    """

    _owner_spec: ClassVar[str] = (
        "torchtitan.models.qwen3_vl.vision_encoder:Qwen3VLVisionEncoder"
    )
    supports_context_parallel: ClassVar[bool] = False

    dim: int
    n_heads: int
    spatial_merge_size: int
    num_position_embeddings: int
    out_hidden_size: int
    """Width the merged visual tokens are projected to; the decoder's ``dim``."""

    # DeepStack: layer indices for extracting intermediate visual features
    deepstack_visual_indices: list[int]

    patch_embed: PatchEmbedConfig
    blocks: dict[str, VisionTransformerBlockConfig] = field(default_factory=dict)
    layers: list[LayerGroupConfig | VisionTransformerBlockConfig] = field(
        default_factory=list
    )
    rotary_pos_emb: Derived[VisionRotaryEmbeddingConfig] = None
    merger: Derived[PatchMergerConfig] = None
    deepstack_mergers: Derived[list[PatchMergerConfig]] = None

    def tree_fields(self) -> tuple:
        skip = {"blocks"} if self._get_metadata("materialized", False) else set()
        return tuple(f for f in fields(self) if f.name not in skip)

    def placed_layers(self) -> tuple[VisionTransformerBlockConfig, ...]:
        return placed_configs(
            self.layers, VisionTransformerBlockConfig, path="vision_encoder.layers"
        )

    def __post_init__(self) -> None:
        for name in (
            "dim",
            "n_heads",
            "spatial_merge_size",
            "num_position_embeddings",
            "out_hidden_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigFieldError(name, "must be a positive integer")
        if self.dim % self.n_heads:
            raise ConfigFieldError(
                "dim", f"must be divisible by n_heads ({self.n_heads}), got {self.dim}"
            )
        if (self.dim // self.n_heads) % 4:
            raise ConfigFieldError(
                "dim",
                "dim / n_heads must be divisible by 4 for two-axis rotary embeddings",
            )
        if (
            math.isqrt(self.num_position_embeddings) ** 2
            != self.num_position_embeddings
        ):
            raise ConfigFieldError(
                "num_position_embeddings",
                "must be a square for the position embedding grid",
            )
        for name in ("patch_size", "temporal_patch_size", "in_channels"):
            value = getattr(self.patch_embed, name)
            if type(value) is not int or value <= 0:
                raise ConfigFieldError(
                    f"patch_embed.{name}", "must be a positive integer"
                )

        owner = type(self).__qualname__
        check_block_names(self.layers, self.blocks)
        n_layers = count_layers(self.layers)
        if stray := [
            index
            for index in self.deepstack_visual_indices
            if not 0 <= index < n_layers
        ]:
            raise ValueError(
                f"{owner}.deepstack_visual_indices {stray} name no layer of "
                f"the {n_layers} in the tower."
            )
        # ``named_segments`` pairs merger i with layer indices[i]; an
        # unpaired one would fall out of the reported structure.
        if self.deepstack_mergers is not None:
            validate_values(
                owner,
                (
                    "deepstack_mergers",
                    len(self.deepstack_mergers),
                    len(self.deepstack_visual_indices),
                ),
            )

    def _completed_children(self) -> dict:
        """The children the encoder's widths size; written configs keep them
        partial, so a resolved config records only what was written."""
        owner = type(self).__qualname__
        merger_dims = {
            "hidden_size": self.dim,
            "spatial_merge_size": self.spatial_merge_size,
            "out_hidden_size": self.out_hidden_size,
        }
        deepstack = self.deepstack_mergers
        if deepstack is None:
            deepstack = [
                PatchMergerConfig(use_postshuffle_norm=True)
                for _ in self.deepstack_visual_indices
            ]
        return {
            "patch_embed": resolve_child(
                self.patch_embed,
                PatchEmbedConfig,
                name=f"{owner}.patch_embed",
                dims={"dim": self.dim},
            ),
            "rotary_pos_emb": resolve_child(
                self.rotary_pos_emb,
                VisionRotaryEmbeddingConfig,
                name=f"{owner}.rotary_pos_emb",
                dims={"dim": self.dim // self.n_heads // 2},
            ),
            "merger": resolve_child(
                self.merger, PatchMergerConfig, name=f"{owner}.merger", dims=merger_dims
            ),
            "deepstack_mergers": [
                resolve_child(
                    m,
                    PatchMergerConfig,
                    name=f"{owner}.deepstack_mergers",
                    dims=merger_dims,
                )
                for m in deepstack
            ],
        }

    def placed(self) -> "Qwen3VLVisionEncoderConfig":
        """A copy with every layer its own complete block."""
        layers = [
            resolve_child(
                block,
                type(block),
                name=f"vision_encoder.layers.{i}",
                dims=dict(dim=self.dim, n_heads=self.n_heads),
            )
            for i, block in enumerate(place_blocks(self.layers, self.blocks))
        ]
        placed = self._copy(layers=layers, **self._completed_children())
        placed._set_metadata("materialized", True)
        if placed.param_init is None:
            placed.param_init = dict(_POS_EMBED_INIT)
        return placed
