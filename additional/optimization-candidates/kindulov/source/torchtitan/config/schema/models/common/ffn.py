from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.validation import resolve_child
from torchtitan.config.schema.protocols.module import ModuleConfig


@config(family="feed_forward", default="swiglu", type="swiglu")
class FeedForwardConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.ffn:FeedForward"

    # None: the owner fills it, then the projections are sized.
    dim: int | None = None
    hidden_dim: int
    # None: sized from the dims, with child-level initializers.
    w1: LinearConfig | None = None
    w2: LinearConfig | None = None
    w3: LinearConfig | None = None
    # Derived by FP8Config at runtime; not a second YAML settings surface.
    use_shared_quantizer: bool = field(
        default=False,
        init=False,
        metadata={"config_source": "fp8.dense_use_shared_quantizer"},
    )
    dense_fused_swiglu_quant: bool = field(
        default=True,
        init=False,
        metadata={"config_source": "fp8.dense_fused_swiglu_quant"},
    )
    swiglu_limit: float = 0.0
    recompute: bool = False
    """Recompute this FFN during backward, independently of block checkpointing."""

    def __post_init__(self) -> None:
        from torchtitan.config.errors import ConfigFieldError

        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ConfigFieldError(
                "hidden_dim", f"hidden_dim must be positive, got {self.hidden_dim}"
            )
        if type(self.recompute) is not bool:
            raise ValueError("recompute must be a boolean")
        owner = type(self).__qualname__
        if self.swiglu_limit < 0.0:
            raise ValueError(
                f"{owner}.swiglu_limit must be >= 0, got {self.swiglu_limit}."
            )
        if self.dim is None:
            return  # partial: the owner fills dim, then this runs again
        self.w1 = resolve_child(
            self.w1,
            LinearConfig,
            name=f"{owner}.w1",
            dims=dict(in_features=self.dim, out_features=self.hidden_dim),
        )
        self.w2 = resolve_child(
            self.w2,
            LinearConfig,
            name=f"{owner}.w2",
            dims=dict(in_features=self.hidden_dim, out_features=self.dim),
            extra=dict(init_role="linear_out"),
        )
        self.w3 = resolve_child(
            self.w3,
            LinearConfig,
            name=f"{owner}.w3",
            dims=dict(in_features=self.dim, out_features=self.hidden_dim),
            extra=dict(init_role="linear_out"),
        )

    def _projections(
        self,
    ) -> tuple[
        LinearConfig,
        LinearConfig,
        LinearConfig,
    ]:
        assert self.w1 is not None
        assert self.w2 is not None
        assert self.w3 is not None
        return self.w1, self.w2, self.w3

    def fwd_flops_per_token(self, seq_len: int) -> int:
        """Forward FLOPs per token of the SwiGLU matmuls (= 6·dim·hidden_dim).

        SiLU and the elementwise gate product are not counted.
        """
        return sum(w.fwd_flops_per_token(seq_len) for w in self._projections())


@config(type="llama")
class LlamaFeedForwardConfig(FeedForwardConfig):
    """SwiGLU feed-forward; ``hidden_dim`` defaults to
    :func:`compute_ffn_hidden_dim` of ``dim``."""

    _owner_spec: ClassVar[str] = FeedForwardConfig._owner_spec

    # pyrefly: ignore [bad-override]
    hidden_dim: int | None = None
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None

    def __post_init__(self) -> None:
        if self.multiple_of <= 0:
            raise ValueError("multiple_of must be positive")
        if self.dim is not None and self.hidden_dim is None:
            self.hidden_dim = compute_ffn_hidden_dim(
                self.dim,
                multiple_of=self.multiple_of,
                ffn_dim_multiplier=self.ffn_dim_multiplier,
            )
        # slots=True breaks super(); call the parent explicitly.
        FeedForwardConfig.__post_init__(self)


def compute_ffn_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 1,
    ffn_dim_multiplier: float | None = None,
) -> int:
    """Compute the SwiGLU hidden dimension for Llama3/4-style models.

    This applies the 2/3 scaling, optional multiplier, and rounds up to multiple_of.
    """
    hidden_dim = int(2 * 4 * dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
