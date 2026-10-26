from dataclasses import field, fields
from typing import ClassVar

from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.validation import Issue
from torchtitan.config.configurable import config, Derived, independent_config_positions
from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.schema.models.common.decoder_blocks.layer_group import (
    LayerGroupConfig,
    check_block_names,
    count_layers,
    place_blocks,
    placed_configs,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.validation import resolve_child
from torchtitan.config.schema.models.flux.model.autoencoder import AutoEncoderConfig
from torchtitan.config.schema.models.flux.model.hf_embedder import FluxEmbedderConfig
from torchtitan.config.schema.models.flux.model.layers import (
    DoubleStreamBlockConfig,
    EmbedNDConfig,
    FluxBlockConfig,
    LastLayerConfig,
    MLPEmbedderConfig,
    SingleStreamBlockConfig,
    _linear,
)
from torchtitan.config.schema.models.flux.param_init import _XAVIER_LINEAR
from torchtitan.config.schema.protocols.model import BaseModelConfig

_T5_ENCODER_VERSION = "google/t5-v1_1-xxl"
_CLIP_ENCODER_VERSION = "openai/clip-vit-large-patch14"


@config(type="flux")
class FluxModelConfig(BaseModelConfig):
    """FLUX.1 rectified-flow transformer; runs under the ``flux`` trainer.

    ``hidden_size`` and ``num_heads`` size every block, as a decoder's
    ``dim`` does; the stream blocks are placed from named ``blocks``.
    """

    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.model:FluxModel"
    family: ClassVar[str] = "flux"
    trainer_type: ClassVar[str] = "flux"
    _runtime_spec: ClassVar[str] = "torchtitan.models.flux.versions:BASE"

    hidden_size: int
    num_heads: int
    in_channels: int = 64
    """Packed latent channels in (the autoencoder's 16 x a 2x2 patch)."""
    out_channels: int = 64
    vec_in_dim: int = 768
    """CLIP pooled-embedding width."""
    context_in_dim: int = 4096
    """T5 token-embedding width."""
    pe_config: EmbedNDConfig = field(
        default_factory=lambda: EmbedNDConfig(theta=10_000, axes_dim=(16, 56, 56))
    )
    """Rotary embedding; ``axes_dim`` sums to the head dim."""
    blocks: dict[str, FluxBlockConfig] = field(default_factory=dict)
    double_blocks: list[LayerGroupConfig | DoubleStreamBlockConfig] = field(
        default_factory=list
    )
    single_blocks: list[LayerGroupConfig | SingleStreamBlockConfig] = field(
        default_factory=list
    )
    img_in: Derived[LinearConfig] = None
    txt_in: Derived[LinearConfig] = None
    time_in_config: Derived[MLPEmbedderConfig] = None
    vector_in_config: Derived[MLPEmbedderConfig] = None
    final_layer_config: Derived[LastLayerConfig] = None

    autoencoder: AutoEncoderConfig = field(default_factory=AutoEncoderConfig)
    """Frozen VAE; the default is the FLUX.1 checkpoint's shape."""
    # The trainer can override version and random_init when it builds the
    # encoders (``encoder.clip_encoder`` / ``encoder.t5_encoder``).
    clip_encoder: FluxEmbedderConfig = field(
        default_factory=lambda: FluxEmbedderConfig(version=_CLIP_ENCODER_VERSION)
    )
    t5_encoder: FluxEmbedderConfig = field(
        default_factory=lambda: FluxEmbedderConfig(version=_T5_ENCODER_VERSION)
    )

    def tree_fields(self) -> tuple:
        skip = {"blocks"} if self.materialized else set()
        return tuple(f for f in fields(self) if f.name not in skip)

    def placed_double_blocks(self) -> tuple[DoubleStreamBlockConfig, ...]:
        return placed_configs(
            self.double_blocks, DoubleStreamBlockConfig, path="double_blocks"
        )

    def placed_single_blocks(self) -> tuple[SingleStreamBlockConfig, ...]:
        return placed_configs(
            self.single_blocks, SingleStreamBlockConfig, path="single_blocks"
        )

    @property
    def axes_dim(self) -> tuple[int, int, int]:
        return self.pe_config.axes_dim

    @property
    def depth(self) -> int:
        return count_layers(self.double_blocks)

    @property
    def depth_single_blocks(self) -> int:
        return count_layers(self.single_blocks)

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "num_heads",
            "in_channels",
            "out_channels",
            "vec_in_dim",
            "context_in_dim",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigFieldError(name, "must be a positive integer")
        if any(type(axis) is not int or axis < 0 or axis % 2 for axis in self.axes_dim):
            raise ConfigFieldError(
                "pe_config.axes_dim", "must contain nonnegative even integers"
            )

        if self.hidden_size % self.num_heads != 0:
            raise ConfigFieldError(
                "hidden_size",
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads "
                f"({self.num_heads})",
            )
        head_dim = self.hidden_size // self.num_heads
        if sum(self.axes_dim) != head_dim:
            raise ConfigFieldError(
                "pe_config.axes_dim",
                f"pe_config.axes_dim {self.axes_dim} sums to {sum(self.axes_dim)}, "
                f"but the head dim is {head_dim}",
            )
        check_block_names(self.double_blocks, self.blocks, path="double_blocks")
        check_block_names(self.single_blocks, self.blocks, path="single_blocks")

    def validate(self, ctx):
        # The Flux trainer packs each VAE latent into a fixed 2 x 2 patch.
        packed_width = 4 * self.autoencoder.z_channels
        return [
            Issue(
                path=name,
                message=f"{name} must match packed autoencoder.z_channels ({packed_width}), got {getattr(self, name)}",
            )
            for name in ("in_channels", "out_channels")
            if getattr(self, name) != packed_width
        ]

    def materialize(self) -> "FluxModelConfig":
        """The module tree: blocks placed and sized, projections derived."""
        if self.materialized:
            self.validate_materialized()
            return independent_config_positions(self)
        hs = self.hidden_size

        def placed(entries, kind):
            out = []
            for block in place_blocks(entries, self.blocks):
                if not isinstance(block, kind):
                    raise ValueError(
                        f"{type(block).__name__} placed where a {kind.__name__} goes"
                    )
                out.append(
                    resolve_child(
                        block,
                        kind,
                        name=kind.__name__,
                        dims=dict(hidden_size=hs, num_heads=self.num_heads),
                    )
                )
            return out

        tree = independent_config_positions(self)
        tree._set_metadata("materialized", True)
        tree.double_blocks = placed(self.double_blocks, DoubleStreamBlockConfig)
        tree.single_blocks = placed(self.single_blocks, SingleStreamBlockConfig)
        tree.pe_config = resolve_child(
            tree.pe_config,
            EmbedNDConfig,
            name="pe_config",
            dims={"dim": hs // self.num_heads},
        )
        tree.img_in = _linear(
            tree.img_in, self.in_channels, hs, _XAVIER_LINEAR, name="img_in"
        )
        tree.txt_in = _linear(
            tree.txt_in, self.context_in_dim, hs, _XAVIER_LINEAR, name="txt_in"
        )
        # Sinusoidal timestep embedding is 256 wide; the guidance vector is CLIP's.
        tree.time_in_config = resolve_child(
            tree.time_in_config,
            MLPEmbedderConfig,
            name="time_in_config",
            dims=dict(in_dim=256, hidden_dim=hs),
        )
        tree.vector_in_config = resolve_child(
            tree.vector_in_config,
            MLPEmbedderConfig,
            name="vector_in_config",
            dims=dict(in_dim=self.vec_in_dim, hidden_dim=hs),
        )
        tree.final_layer_config = resolve_child(
            tree.final_layer_config,
            LastLayerConfig,
            name="final_layer_config",
            dims=dict(hidden_size=hs, patch_size=1, out_channels=self.out_channels),
        )
        tree.validate_materialized()
        return tree

    def validate_materialized(self) -> None:
        BaseModelConfig.validate_materialized(self)
        self.__post_init__()
        for name, kind in (
            ("double_blocks", DoubleStreamBlockConfig),
            ("single_blocks", SingleStreamBlockConfig),
        ):
            for index, block in enumerate(getattr(self, name)):
                if isinstance(block, FrozenConfig):
                    block = block.inner
                if not isinstance(block, kind):
                    raise ValueError(f"{name}.{index}: expected {kind.__name__}")
                if (
                    block.hidden_size != self.hidden_size
                    or block.num_heads != self.num_heads
                ):
                    raise ValueError(
                        f"{name}.{index}: block dimensions disagree with its model"
                    )
