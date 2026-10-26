from typing import ClassVar

from torchtitan.config.configurable import config, Derived, OwnerFilled
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.norms.rmsnorm import RMSNormConfig
from torchtitan.config.schema.protocols.blocks import BlockConfig
from torchtitan.config.schema.protocols.module import ModuleConfig
from torchtitan.config.schema.models.common.validation import (
    require_completed,
    resolve_child,
    validate_values,
)


@config
class EmbedNDConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.layers:EmbedND"

    theta: int
    axes_dim: tuple[int, int, int]
    # Filled by the model: its head dim.
    dim: OwnerFilled[int] = None


@config
class MLPEmbedderConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.layers:MLPEmbedder"

    in_layer: Derived[LinearConfig] = None
    out_layer: Derived[LinearConfig] = None
    in_dim: OwnerFilled[int] = None
    hidden_dim: OwnerFilled[int] = None

    def __post_init__(self) -> None:
        from torchtitan.config.schema.models.flux.param_init import _NORMAL_02

        if self.in_dim is None or self.hidden_dim is None:
            return  # partial: the model completes it
        self.in_layer = _linear(
            self.in_layer,
            self.in_dim,
            self.hidden_dim,
            _NORMAL_02,
            name="in_layer",
        )
        self.out_layer = _linear(
            self.out_layer,
            self.hidden_dim,
            self.hidden_dim,
            _NORMAL_02,
            name="out_layer",
        )


@config
class QKNormConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.layers:QKNorm"

    query_norm: RMSNormConfig
    key_norm: RMSNormConfig


@config
class SelfAttentionConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.layers:SelfAttention"

    qkv: LinearConfig
    proj: LinearConfig
    norm: QKNormConfig
    num_heads: int = 8
    qkv_bias: bool = False


@config
class ModulationConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.layers:Modulation"

    lin: LinearConfig
    double: bool


def _linear(
    child, in_features: int, out_features: int, init, *, name: str, bias=True
) -> LinearConfig:
    """A projection with *init* under its own entries; a written one must have
    the owner's widths and ``bias``."""
    return resolve_child(
        child,
        LinearConfig,
        name=name,
        dims=dict(in_features=in_features, out_features=out_features),
        require=dict(bias=bias),
        init=init,
    )


def _qk_norm(child, head_dim: int, name: str) -> QKNormConfig:
    from torchtitan.config.schema.models.flux.param_init import _NORM_INIT

    if child is not None:
        return child._copy(
            query_norm=resolve_child(
                child.query_norm,
                RMSNormConfig,
                name=f"{name}.query_norm",
                dims=dict(normalized_shape=head_dim),
            ),
            key_norm=resolve_child(
                child.key_norm,
                RMSNormConfig,
                name=f"{name}.key_norm",
                dims=dict(normalized_shape=head_dim),
            ),
        )
    return QKNormConfig(
        query_norm=RMSNormConfig(
            normalized_shape=head_dim, param_init=dict(_NORM_INIT)
        ),
        key_norm=RMSNormConfig(normalized_shape=head_dim, param_init=dict(_NORM_INIT)),
    )


def _modulation(child, hidden_size: int, double: bool, name: str) -> ModulationConfig:
    from torchtitan.config.schema.models.flux.param_init import _ZERO_LINEAR

    width = (6 if double else 3) * hidden_size
    lin = _linear(
        child and child.lin, hidden_size, width, _ZERO_LINEAR, name=f"{name}.lin"
    )
    if child is None:
        return ModulationConfig(double=double, lin=lin)
    validate_values(name, ("double", child.double, double))
    return child._copy(lin=lin)


@config(family="flux_block")
class FluxBlockConfig(BlockConfig):
    """A FLUX transformer block; ``hidden_size`` and ``num_heads`` come from
    the model, the sublayers are sized from them."""

    hidden_size: OwnerFilled[int] = None
    num_heads: OwnerFilled[int] = None
    mlp_ratio: float = 4.0

    @property
    def mlp_hidden_dim(self) -> int:
        return int(
            require_completed(self.hidden_size, "FluxBlock.hidden_size")
            * self.mlp_ratio
        )


@config(type="double_stream")
class DoubleStreamBlockConfig(FluxBlockConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.layers:DoubleStreamBlock"

    qkv_bias: bool = False
    img_mod: Derived[ModulationConfig] = None
    txt_mod: Derived[ModulationConfig] = None
    img_attn: Derived[SelfAttentionConfig] = None
    txt_attn: Derived[SelfAttentionConfig] = None
    img_mlp_in: Derived[LinearConfig] = None
    img_mlp_out: Derived[LinearConfig] = None
    txt_mlp_in: Derived[LinearConfig] = None
    txt_mlp_out: Derived[LinearConfig] = None

    def _attention(
        self, child, hs: int, num_heads: int, name: str
    ) -> SelfAttentionConfig:
        from torchtitan.config.schema.models.flux.param_init import _XAVIER_LINEAR

        parts = dict(
            qkv=_linear(
                child and child.qkv,
                hs,
                hs * 3,
                _XAVIER_LINEAR,
                name=f"{name}.qkv",
                bias=self.qkv_bias,
            ),
            proj=_linear(
                child and child.proj, hs, hs, _XAVIER_LINEAR, name=f"{name}.proj"
            ),
            norm=_qk_norm(child and child.norm, hs // num_heads, f"{name}.norm"),
        )
        if child is None:
            return SelfAttentionConfig(
                num_heads=num_heads, qkv_bias=self.qkv_bias, **parts
            )
        validate_values(
            name,
            ("num_heads", child.num_heads, num_heads),
            ("qkv_bias", child.qkv_bias, self.qkv_bias),
        )
        return child._copy(**parts)

    def __post_init__(self) -> None:
        from torchtitan.config.schema.models.flux.param_init import _XAVIER_LINEAR

        if self.hidden_size is None or self.num_heads is None:
            return  # partial: the model completes it
        hs, mlp = self.hidden_size, self.mlp_hidden_dim
        self.img_mod = _modulation(self.img_mod, hs, double=True, name="img_mod")
        self.txt_mod = _modulation(self.txt_mod, hs, double=True, name="txt_mod")
        self.img_attn = self._attention(self.img_attn, hs, self.num_heads, "img_attn")
        self.txt_attn = self._attention(self.txt_attn, hs, self.num_heads, "txt_attn")
        self.img_mlp_in = _linear(
            self.img_mlp_in, hs, mlp, _XAVIER_LINEAR, name="img_mlp_in"
        )
        self.img_mlp_out = _linear(
            self.img_mlp_out, mlp, hs, _XAVIER_LINEAR, name="img_mlp_out"
        )
        self.txt_mlp_in = _linear(
            self.txt_mlp_in, hs, mlp, _XAVIER_LINEAR, name="txt_mlp_in"
        )
        self.txt_mlp_out = _linear(
            self.txt_mlp_out, mlp, hs, _XAVIER_LINEAR, name="txt_mlp_out"
        )


@config(type="single_stream")
class SingleStreamBlockConfig(FluxBlockConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.layers:SingleStreamBlock"

    qk_scale: float | None = None
    linear1: Derived[LinearConfig] = None
    linear2: Derived[LinearConfig] = None
    modulation: Derived[ModulationConfig] = None
    norm: Derived[QKNormConfig] = None

    def __post_init__(self) -> None:
        from torchtitan.config.schema.models.flux.param_init import _XAVIER_LINEAR

        if self.hidden_size is None or self.num_heads is None:
            return  # partial: the model completes it
        hs, mlp = self.hidden_size, self.mlp_hidden_dim
        self.linear1 = _linear(
            self.linear1, hs, hs * 3 + mlp, _XAVIER_LINEAR, name="linear1"
        )
        self.linear2 = _linear(
            self.linear2, hs + mlp, hs, _XAVIER_LINEAR, name="linear2"
        )
        self.modulation = _modulation(
            self.modulation, hs, double=False, name="modulation"
        )
        self.norm = _qk_norm(self.norm, hs // self.num_heads, "norm")


@config
class LastLayerConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.layers:LastLayer"

    linear: Derived[LinearConfig] = None
    adaln_linear: Derived[LinearConfig] = None
    hidden_size: OwnerFilled[int] = None
    patch_size: OwnerFilled[int] = None
    out_channels: OwnerFilled[int] = None

    def __post_init__(self) -> None:
        from torchtitan.config.schema.models.flux.param_init import _ZERO_LINEAR

        if None in (self.hidden_size, self.patch_size, self.out_channels):
            return  # partial: the model completes it
        self.linear = _linear(
            self.linear,
            self.hidden_size,
            self.patch_size**2 * self.out_channels,
            _ZERO_LINEAR,
            name="linear",
        )
        self.adaln_linear = _linear(
            self.adaln_linear,
            self.hidden_size,
            2 * self.hidden_size,
            _ZERO_LINEAR,
            name="adaln_linear",
        )
