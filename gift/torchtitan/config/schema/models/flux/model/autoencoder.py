from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.protocols.module import ModuleConfig


@config
class AttnBlockConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.autoencoder:AttnBlock"

    in_channels: int


@config
class ResnetBlockConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.autoencoder:ResnetBlock"

    in_channels: int
    out_channels: int


@config
class DownsampleConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.autoencoder:Downsample"

    in_channels: int


@config
class UpsampleConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.autoencoder:Upsample"

    in_channels: int


@config
class EncoderConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.autoencoder:Encoder"

    resolution: int
    in_channels: int
    ch: int
    ch_mult: tuple[int, ...]
    num_res_blocks: int
    z_channels: int


@config
class DecoderConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.autoencoder:Decoder"

    ch: int
    out_ch: int
    ch_mult: tuple[int, ...]
    num_res_blocks: int
    in_channels: int
    resolution: int
    z_channels: int


@config
class DiagonalGaussianConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.flux.model.autoencoder:DiagonalGaussian"
    )

    sample: bool = True
    chunk_dim: int = 1


@config
class AutoEncoderConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.autoencoder:AutoEncoder"

    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 16
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159
