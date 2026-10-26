import math
from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config, Derived, OwnerFilled
from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    SoftmaxAttentionConfig,
    causal_softmax_core_flops,
)
from torchtitan.config.schema.models.common.attention.softmax.kernels.base import (
    InnerAttentionConfig,
)
from torchtitan.config.schema.models.common.attention.softmax.kernels.sdpa import (
    ScaledDotProductAttentionConfig,
)
from torchtitan.config.schema.models.common.embeddings.rope.scaling import YarnScaling
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.norms.rmsnorm import RMSNormConfig
from torchtitan.config.schema.models.common.norms.types import NormConfig
from torchtitan.config.schema.models.common.validation import resolve_child


@config(type="mla")
class MLAConfig(SoftmaxAttentionConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.attention.softmax.mla:MLA"
    context_parallel_kernels: ClassVar[tuple[type, ...] | None] = (
        ScaledDotProductAttentionConfig,
    )

    n_heads: int
    # None: the owner fills it, then the projections are sized.
    dim: OwnerFilled[int] = None
    # None: sized from the dims, with the matching ``*_param_init``. A norm
    # passed in keeps its type (e.g. GatedRMSNormConfig).
    wkv_a: Derived[LinearConfig] = None
    wkv_b: Derived[LinearConfig] = None
    wo: Derived[LinearConfig] = None
    kv_norm: Derived[NormConfig] = None
    # q_lora_rank picks the path: 0 uses wq, > 0 uses wq_a/wq_b/q_norm. Setting
    # a child of the other path is an error.
    wq: LinearConfig | None = None
    wq_a: LinearConfig | None = None
    wq_b: LinearConfig | None = None
    q_norm: NormConfig | None = None
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    inner_attention: InnerAttentionConfig = field(
        default_factory=ScaledDotProductAttentionConfig
    )
    # YaRN amplitude coefficient; read only when the rope scales.
    mscale: float = 1.0
    # Scale the Q/KV latents by sqrt(dim / latent_dim) before their
    # up-projections (alpha_q applies only when q_lora_rank > 0).
    use_mla_scaling_factor: bool = False
    # Q is penalized after norm/scaling; KV before norm, excluding RoPE.
    q_latent_hidden_z_loss_coef: float = 0.0
    kv_latent_hidden_z_loss_coef: float = 0.0
    hidden_z_loss_projections: ClassVar[dict[str, str]] = {
        "q_latent": "q_latent_hidden_z_loss_coef",
        "kv_latent": "kv_latent_hidden_z_loss_coef",
    }
    # The YaRN factor to compensate for; None: no compensation. Stored so
    # ``build()`` needs nothing from a parent; ``accept_rope`` fills or checks it.
    yarn_factor: float | None = None

    def __post_init__(self) -> None:
        # slots=True breaks super(); call the parent explicitly.
        SoftmaxAttentionConfig.__post_init__(self)
        owner = type(self).__qualname__
        for name in self.hidden_z_loss_projections.values():
            coef = getattr(self, name)
            if not math.isfinite(coef) or coef < 0.0:
                from torchtitan.config.errors import ConfigFieldError

                raise ConfigFieldError(name, f"must be finite and >= 0, got {coef}.")
        lora = self.q_lora_rank > 0
        unwanted = [
            name
            for name in (("wq",) if lora else ("wq_a", "wq_b", "q_norm"))
            if getattr(self, name) is not None
        ]
        if unwanted:
            raise ValueError(
                f"{owner} has q_lora_rank={self.q_lora_rank}, so "
                f"{', '.join(unwanted)} belongs to the other q path; it takes "
                f"{'wq_a/wq_b/q_norm' if lora else 'wq'}."
            )
        if self.dim is None:
            return  # partial: the owner fills dim, then this runs again
        q_out = self.n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        if lora:
            self.wq_a = self._linear(
                self.wq_a, f"{owner}.wq_a", self.dim, self.q_lora_rank
            )
            self.wq_b = self._linear(
                self.wq_b, f"{owner}.wq_b", self.q_lora_rank, q_out
            )
            self.q_norm = self._norm(self.q_norm, f"{owner}.q_norm", self.q_lora_rank)
        else:
            self.wq = self._linear(self.wq, f"{owner}.wq", self.dim, q_out)
        self.wkv_a = self._linear(
            self.wkv_a,
            f"{owner}.wkv_a",
            self.dim,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )
        self.kv_norm = self._norm(self.kv_norm, f"{owner}.kv_norm", self.kv_lora_rank)
        self.wkv_b = self._linear(
            self.wkv_b,
            f"{owner}.wkv_b",
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        self.wo = resolve_child(
            self.wo,
            LinearConfig,
            name=f"{owner}.wo",
            dims=dict(
                in_features=self.n_heads * self.v_head_dim, out_features=self.dim
            ),
            extra=dict(init_role="linear_out"),
        )
        if self.gate is not None:
            # The gate multiplies the attention output before wo.
            self.gate = self.gate._copy(
                gate_proj=self._linear(
                    self.gate.gate_proj,
                    f"{owner}.gate.gate_proj",
                    self.dim,
                    self.wo.in_features,
                )
            )

    def _linear(self, child, name: str, in_features: int, out_features: int):
        return resolve_child(
            child,
            LinearConfig,
            name=name,
            dims=dict(in_features=in_features, out_features=out_features),
        )

    def _norm(self, child, name: str, normalized_shape: int):
        return resolve_child(
            child,
            RMSNormConfig,
            name=name,
            dims=dict(normalized_shape=normalized_shape),
        )

    def _projections(
        self,
    ) -> tuple[LinearConfig, LinearConfig, LinearConfig, NormConfig]:
        assert self.wkv_a is not None
        assert self.wkv_b is not None
        assert self.wo is not None
        assert self.kv_norm is not None
        return self.wkv_a, self.wkv_b, self.wo, self.kv_norm

    def accept_rope(self, rope) -> None:
        """Take the YaRN factor from the decoder's RoPE and check the width.

        Rotating only part of the head means it needs the complex cache:
        cos/sin is a different layout and a different place for ``mscale``.
        """
        if not rope.cache_is_complex:
            raise ValueError(
                f"{type(self).__qualname__} applies complex RoPE to its rope "
                f"half, but {type(rope).__qualname__} builds a real cache."
            )
        if rope.dim != self.qk_rope_head_dim:
            raise ValueError(
                f"{type(self).__qualname__}.qk_rope_head_dim is "
                f"{self.qk_rope_head_dim} but the decoder's RoPE was built "
                f"for {rope.dim} dims."
            )
        scaling = rope.scaling
        factor = scaling.rope_factor if isinstance(scaling, YarnScaling) else None
        # An explicit value is checked: once folded into ``softmax_scale`` a
        # mismatch is invisible.
        if self.yarn_factor is None:
            self.yarn_factor = factor
        if self.yarn_factor != factor:
            raise ValueError(
                f"{type(self).__qualname__}.yarn_factor is {self.yarn_factor} "
                f"but the decoder's RoPE scales by {factor}; set "
                f"yarn_factor={factor}."
            )
        if factor is None and self.mscale != 1.0:
            raise ValueError(
                f"{type(self).__qualname__}.mscale is {self.mscale} but the "
                "decoder's RoPE does not scale, so it weighs a compensation "
                "that never applies. Drop it, or give the rope a YarnScaling."
            )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        # q_norm/kv_norm FLOPs are not counted (0 for RMSNorm; a small gate MLP
        # for GatedRMSNorm).
        if self.wq is not None:
            q_linears: tuple[LinearConfig, ...] = (self.wq,)
        else:
            assert self.wq_a is not None and self.wq_b is not None
            q_linears = (self.wq_a, self.wq_b)
        wkv_a, wkv_b, wo, _ = self._projections()
        projections = sum(
            lin.fwd_flops_per_token(seq_len) for lin in (*q_linears, wkv_a, wkv_b, wo)
        )
        if self.gate is not None:
            projections += self.gate.fwd_flops_per_token(seq_len)
        # wkv_b materializes k and v per head before the inner kernel, so the
        # core is a standard causal softmax over n_heads with key dim
        # qk_nope+qk_rope and value dim v_head_dim.
        core = causal_softmax_core_flops(
            n_heads=self.n_heads,
            seq_len=seq_len,
            qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
        )
        return projections + core
