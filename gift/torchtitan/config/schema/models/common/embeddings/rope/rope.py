import math
from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.embeddings.rope.const import DEFAULT_THETA
from torchtitan.config.schema.models.common.embeddings.rope.scaling import (
    LlamaScaling,
    NoScaling,
    ScalingConfig,
    validate_cos_sin_yarn_window,
    YarnScaling,
)
from torchtitan.config.schema.protocols.module import ModuleConfig


@config(family="rope")
class RoPEConfig(ModuleConfig):
    # Rotated width per head; ``DecoderConfig`` checks it against the layers.
    dim: int
    # The context the model declares. No scaling parameter depends on it.
    max_seq_len: int
    theta: float = DEFAULT_THETA
    # Positions to allocate; ``None`` means ``max_seq_len``. Allocation only --
    # entry ``i`` is the same number either way.
    cache_len: int | None = None
    scaling: ScalingConfig = field(default_factory=NoScaling)

    # Scalings this backend implements; empty on the abstract base.
    supported_scalings: ClassVar[tuple[type, ...]] = ()

    def __post_init__(self) -> None:
        if not self.supported_scalings:
            raise ValueError(
                f"{type(self).__qualname__} is abstract: a model names its "
                "backend by picking ComplexRoPEConfig (Llama3/4, DeepSeek V3) "
                "or CosSinRoPEConfig (Qwen3, GPT-OSS)."
            )
        if type(self.scaling) not in self.supported_scalings:
            supported = sorted(cls.__name__ for cls in self.supported_scalings)
            raise ValueError(
                f"{type(self).__qualname__} does not implement "
                f"{type(self.scaling).__name__}; it supports {supported}."
            )
        # An odd dim passes the width checks and dies in view_as_complex.
        if self.dim <= 0 or self.dim % 2:
            raise ValueError(
                "RoPE dim must be positive and even (rotation works on "
                f"pairs), got {self.dim}."
            )
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}.")
        if not math.isfinite(self.theta) or self.theta <= 1.0:
            # log(theta) divides the YaRN window; at 1.0 that is a zero divide.
            from torchtitan.config.errors import ConfigFieldError

            raise ConfigFieldError(
                "theta", f"theta must be finite and > 1.0, got {self.theta}."
            )
        if self.cache_len is not None and not 0 < self.cache_len <= self.max_seq_len:
            raise ValueError(
                f"cache_len must be in (0, max_seq_len={self.max_seq_len}], "
                f"got {self.cache_len}."
            )

    @property
    def cache_is_complex(self) -> bool:
        """Whether ``build().cache`` is complex; ``forward`` dispatches on it."""
        raise NotImplementedError

    @property
    def num_cached_positions(self) -> int:
        """Cache length actually allocated."""
        return self.max_seq_len if self.cache_len is None else self.cache_len


@config(type="complex")
class ComplexRoPEConfig(RoPEConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.embeddings.rope.rope:ComplexRoPE"
    )

    supported_scalings: ClassVar[tuple[type, ...]] = (
        NoScaling,
        LlamaScaling,
        YarnScaling,
    )

    @property
    def cache_is_complex(self) -> bool:
        return True


@config(type="cos_sin")
class CosSinRoPEConfig(RoPEConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.embeddings.rope.rope:CosSinRoPE"
    )

    supported_scalings: ClassVar[tuple[type, ...]] = (NoScaling, YarnScaling)

    def __post_init__(self) -> None:
        # slots=True breaks super(); call the parent explicitly.
        RoPEConfig.__post_init__(self)
        if isinstance(self.scaling, YarnScaling):
            # Needs this config's dim/theta, so not on the child.
            validate_cos_sin_yarn_window(self.scaling, dim=self.dim, theta=self.theta)

    @property
    def cache_is_complex(self) -> bool:
        return False
