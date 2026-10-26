from typing import ClassVar, cast

from torchtitan.config.configurable import config, Derived, OwnerFilled
from torchtitan.config.schema.components.lora import FrozenConfig, unwrap_frozen
from torchtitan.config.schema.models.common.nn_modules import Conv1dConfig, LinearConfig
from torchtitan.config.schema.models.common.validation import (
    resolve_child,
    validate_values,
)
from torchtitan.config.schema.protocols.module import ModuleConfig


@config(family="qkv", default="separate")
class BaseQKVLinearConfig(ModuleConfig):
    """How Q/K/V are projected (``separate`` or ``fused``); the owning attention
    fills the head geometry via :func:`resolve_qkv`."""

    head_dim: OwnerFilled[int] = None

    # The uniform view GQAttention checks, whatever the strategy.
    @property
    def in_features(self) -> int:
        raise NotImplementedError

    @property
    def n_heads(self) -> int:
        raise NotImplementedError

    @property
    def n_kv_heads(self) -> int:
        raise NotImplementedError

    def completed(
        self,
        *,
        name: str,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        extra,
        bias: bool | None = None,
    ) -> "BaseQKVLinearConfig":
        """This config with the owner's geometry filled and its projections built
        or checked. ``extra`` goes to built projections; a non-None ``bias`` is
        checked."""
        raise NotImplementedError

    def fwd_flops_per_token(self, seq_len: int) -> int:
        """Forward FLOPs per token of the Q/K/V projection matmuls."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fwd_flops_per_token."
        )


def resolve_qkv(
    qkv: BaseQKVLinearConfig | None,
    *,
    name: str,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    extra=None,
    bias: bool | None = None,
) -> BaseQKVLinearConfig:
    """The owner's Q/K/V projection: separate when absent, else the given
    strategy completed for the owner's geometry."""
    original = qkv
    inner = QKVLinearConfig() if qkv is None else unwrap_frozen(qkv)
    if not isinstance(inner, BaseQKVLinearConfig):
        raise ValueError(f"{name} must be a QKV projection config")
    completed = inner.completed(
        name=name,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        extra=extra,
        bias=bias,
    )
    if isinstance(original, FrozenConfig):
        # The checked inner schema supplies the fields, while the transparent
        # proxy must keep its own build method to freeze the completed module.
        return cast(BaseQKVLinearConfig, original._copy(inner=completed))
    return completed


@config(type="separate")
class QKVLinearConfig(BaseQKVLinearConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.attention.qkv:QKVLinear"

    wq: Derived[LinearConfig] = None
    wkv: Derived[LinearConfig] = None

    def _projections(self) -> tuple[LinearConfig, LinearConfig, int]:
        assert (
            self.wq is not None and self.wkv is not None and self.head_dim is not None
        ), "partial QKVLinear.Config: the owner has not completed it"
        return self.wq, self.wkv, self.head_dim

    @property
    def in_features(self) -> int:
        return self._projections()[0].require_features()[0]

    @property
    def n_heads(self) -> int:
        wq, _, head_dim = self._projections()
        return wq.require_features()[1] // head_dim

    @property
    def n_kv_heads(self) -> int:
        _, wkv, head_dim = self._projections()
        return wkv.require_features()[1] // head_dim

    def __post_init__(self) -> None:
        if (
            self.wq is None
            or self.wkv is None
            or self.head_dim is None
            or self.wq.in_features is None
            or self.wq.out_features is None
            or self.wkv.in_features is None
            or self.wkv.out_features is None
        ):
            return  # partial: the owner completes it
        validate_values(
            type(self).__qualname__,
            ("wkv.in_features", self.wkv.in_features, self.wq.in_features),
            ("wq.out_features % head_dim", self.wq.out_features % self.head_dim, 0),
            (
                "wkv.out_features % head_dim",
                self.wkv.out_features % self.head_dim,
                0,
            ),
        )

    def completed(
        self, *, name, dim, n_heads, n_kv_heads, head_dim, extra=None, bias=None
    ):
        checked = None if bias is None else dict(bias=bias)
        qkv = resolve_child(
            self, QKVLinearConfig, name=name, dims=dict(head_dim=head_dim)
        )
        return qkv._copy(
            wq=resolve_child(
                qkv.wq,
                LinearConfig,
                name=f"{name}.wq",
                dims=dict(in_features=dim, out_features=n_heads * head_dim),
                require=checked,
                extra=extra,
            ),
            wkv=resolve_child(
                qkv.wkv,
                LinearConfig,
                name=f"{name}.wkv",
                dims=dict(in_features=dim, out_features=n_kv_heads * head_dim),
                require=checked,
                extra=extra,
            ),
        )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        wq, wkv, _ = self._projections()
        # ``wkv`` builds both wk and wv, so count it twice.
        return wq.fwd_flops_per_token(seq_len) + 2 * wkv.fwd_flops_per_token(seq_len)


@config(type="fused")
class FusedQKVLinearConfig(BaseQKVLinearConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.attention.qkv:FusedQKVLinear"

    # pyrefly: ignore [bad-override]
    n_heads: OwnerFilled[int] = None
    # pyrefly: ignore [bad-override]
    n_kv_heads: OwnerFilled[int] = None
    wqkv: Derived[LinearConfig] = None

    @property
    def in_features(self) -> int:
        assert self.wqkv is not None, "partial FusedQKVLinear.Config"
        return self.wqkv.require_features()[0]

    def __post_init__(self) -> None:
        if None in (self.head_dim, self.n_heads, self.n_kv_heads, self.wqkv):
            return  # partial: the owner completes it
        assert (
            self.n_heads is not None
            and self.n_kv_heads is not None
            and self.wqkv is not None
            and self.head_dim is not None
        )
        if self.wqkv.in_features is None or self.wqkv.out_features is None:
            return  # partial projection: completed() fills its widths
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads}) for fused QKV"
            )
        validate_values(
            type(self).__qualname__,
            (
                "wqkv.out_features",
                self.wqkv.out_features,
                (self.n_heads + 2 * self.n_kv_heads) * self.head_dim,
            ),
        )

    def completed(
        self, *, name, dim, n_heads, n_kv_heads, head_dim, extra=None, bias=None
    ):
        checked = None if bias is None else dict(bias=bias)
        qkv = resolve_child(
            self,
            FusedQKVLinearConfig,
            name=name,
            dims=dict(head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads),
        )
        return qkv._copy(
            wqkv=resolve_child(
                qkv.wqkv,
                LinearConfig,
                name=f"{name}.wqkv",
                dims=dict(
                    in_features=dim, out_features=(n_heads + 2 * n_kv_heads) * head_dim
                ),
                require=checked,
                extra=extra,
            ),
        )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        assert self.wqkv is not None, "partial FusedQKVLinear.Config"
        # One matmul packing Q, K and V: equal to separate wq + wk + wv.
        return self.wqkv.fwd_flops_per_token(seq_len)


@config
class LinearAttnQKVConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.attention.qkv:LinearAttnQKV"

    head_k_dim: int
    head_v_dim: int
    num_k_heads: int
    num_v_heads: int
    qkv_proj: LinearConfig
    # Depthwise causal short conv; the GDN/KDA config sizes it.
    conv: Conv1dConfig
    # q/k L2-norm done inside the fused conv-unpack op (so the scan kernel uses
    # ``use_qk_l2norm_in_kernel=False``).
    normalize_qk: bool = True
    eps: float = 1e-6

    @property
    def in_features(self) -> int:
        return self.qkv_proj.require_features()[0]

    def __post_init__(self) -> None:
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by "
                f"num_k_heads ({self.num_k_heads}): each key head expands to "
                "num_v_heads // num_k_heads value heads"
            )
        conv = self.conv
        widths = (conv.in_channels, conv.out_channels, conv.groups, conv.padding)
        if None in (*widths, conv.kernel_size, self.qkv_proj.out_features):
            return  # partial: the owning attention completes and checks it
        # The packed width the fused op unpacks: q and k over key heads, v over
        # value heads. The parent knows only dim and conv_size, so check here.
        conv_dim = (
            2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim
        )
        validate_values(
            type(self).__qualname__,
            ("qkv_proj.out_features", self.qkv_proj.out_features, conv_dim),
            ("conv.in_channels", self.conv.in_channels, conv_dim),
            ("conv.out_channels", self.conv.out_channels, conv_dim),
            ("conv.groups", self.conv.groups, conv_dim),
            # Left-pad by kernel_size - 1, then slice: causal, no lookahead.
            ("conv.padding", self.conv.padding, self.conv.kernel_size - 1),
        )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        return self.qkv_proj.fwd_flops_per_token(
            seq_len
        ) + self.conv.fwd_flops_per_token(seq_len)
