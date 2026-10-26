from typing import ClassVar, Protocol, TypeAlias

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

from torchtitan.config.schema.models.common.embeddings.rope.rope import (
    ComplexRoPEConfig,
    CosSinRoPEConfig,
    RoPEConfig,
)
from torchtitan.models.common.embeddings.rope.scaling import (
    complex_llama_scaling,
    complex_yarn_scaling,
    cos_sin_yarn_scaling,
    no_scaling,
)
from torchtitan.config.schema.models.common.embeddings.rope.scaling import (
    LlamaScaling,
    NoScaling,
    ScalingConfig,
    YarnScaling,
)
from torchtitan.protocols.module import Module

__all__ = [
    "ComplexRoPE",
    "CosSinRoPE",
    "RoPE",
    "apply_rotary_emb_complex",
    "apply_rotary_emb_cos_sin",
]


class ScalingFn(Protocol):
    """A scaling implementation: base frequencies -> (frequencies, mscale)."""

    def __call__(
        self, freqs: torch.Tensor, scaling: ScalingConfig, *, dim: int, theta: float
    ) -> tuple[torch.Tensor, float]: ...


def _maybe_check_max_pos(positions: torch.Tensor, *, max_valid_pos: int) -> None:
    """Async bounds check: verify all position values <= max_valid_pos.

    Uses ``torch._assert_async`` to avoid a device-host sync while still
    catching out-of-bounds positions (the assertion failure surfaces at a
    later kernel launch).  Skipped entirely under ``torch.compile``.
    """
    if torch.compiler.is_compiling():
        return
    pos_local = positions.to_local() if isinstance(positions, DTensor) else positions
    torch._assert_async(
        torch.all(pos_local <= max_valid_pos),
        f"position_ids exceed {max_valid_pos=}",
    )


class RoPE(Module):
    """Shared Rotary Position Embedding module.

    Abstract: the backend is the subclass config, the scaling is the type of the
    ``scaling`` child, and ``supported_scalings`` says which pairings a backend implements,
    so a bad one fails at config construction. One RoPE serves the whole decoder,
    which keeps ``cache`` in a non-persistent buffer.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = RoPEConfig
    _scaling_fns: ClassVar[dict[type, ScalingFn]] = {}

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.cache: torch.Tensor = self._precompute()

    def _precompute(self) -> torch.Tensor:
        # Template method: base frequencies -> scaling strategy -> backend cache.
        cfg = self.config
        freqs = self._base_freqs()
        scaling_fn = type(self)._scaling_fns[type(cfg.scaling)]
        freqs, mscale = scaling_fn(freqs, cfg.scaling, dim=cfg.dim, theta=cfg.theta)
        return self._build_cache(freqs, mscale)

    def _base_freqs(self) -> torch.Tensor:
        """Base inverse frequencies ``1 / theta**(arange(0, dim, 2)/dim)``.

        Shared by both backends. Kept free of an explicit ``device=`` so the
        cache is built on the ambient device (see ``_init_self_buffers``).
        """
        dim = self.config.dim
        return 1.0 / (
            self.config.theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )

    def _build_cache(self, freqs: torch.Tensor, mscale: float) -> torch.Tensor:
        """Build the backend cache from (scaled) frequencies and YaRN ``mscale``."""
        raise NotImplementedError

    def forward(
        self, seq_len: int, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the precomputed cache tensor (slicing is done by apply_rotary_emb)."""
        return self.cache

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is not None:
            with torch.device(buffer_device):
                self.cache = self._precompute()
        else:
            self.cache = self._precompute()


class ComplexRoPE(RoPE):
    """Complex-exponential RoPE (Llama3/4, DeepSeek V3).

    ``mscale`` is not applied here: MLA rotates only part of the head, so the
    compensation has to reach the whole logit. It folds it into ``softmax_scale``.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = ComplexRoPEConfig
    _scaling_fns: ClassVar[dict[type, ScalingFn]] = {
        NoScaling: no_scaling,
        LlamaScaling: complex_llama_scaling,
        YarnScaling: complex_yarn_scaling,
    }

    def _build_cache(self, freqs: torch.Tensor, mscale: float) -> torch.Tensor:
        del mscale  # see the class docstring
        t = torch.arange(self.config.num_cached_positions, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64


class CosSinRoPE(RoPE):
    """Cosine/sine RoPE (Qwen3, GPT-OSS). Llama scaling is unsupported.

    ``mscale`` multiplies cos and sin, landing ``mscale**2`` on the logits --
    valid because this backend rotates the whole head dim.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = CosSinRoPEConfig
    _scaling_fns: ClassVar[dict[type, ScalingFn]] = {
        NoScaling: no_scaling,
        YarnScaling: cos_sin_yarn_scaling,
    }

    def _build_cache(self, freqs: torch.Tensor, mscale: float) -> torch.Tensor:
        # Inverse frequencies here, despite the name from the base signature.
        t = torch.arange(
            self.config.num_cached_positions, dtype=freqs.dtype, device=freqs.device
        )
        freqs = torch.outer(t, freqs).float()
        theta = torch.cat([freqs, freqs], dim=-1)
        cos = theta.cos() * mscale
        sin = theta.sin() * mscale
        return torch.cat([cos, sin], dim=-1)


def _reshape_for_broadcast(
    cache: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor | None,
    *,
    last_dim: int,
) -> torch.Tensor:
    """Reshape a per-position RoPE cache slice for broadcasting against ``x``.

    Shared by both backends. ``last_dim`` is the expected final-dim size of the
    cache slice: ``x.shape[-1]`` for the complex view (``head_dim // 2``), or
    ``head_dim * 2`` for cos/sin. It cannot be derived from ``x.shape[-1]``
    because the cos/sin cache packs cos and sin into a doubled last dim.

    The broadcast view is ``[1, ..., seqlen, ..., last_dim]`` (seqlen on dim 1,
    ``last_dim`` on the last dim, 1 elsewhere). This is indexing/gather only —
    no float ops — so it is bit-wise identical across the refactor.

    cache: (max_seqlen, last_dim)
    x: (bsz, seqlen, n_heads, last_dim)
    """
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    if positions is None:
        cache = cache[0:seqlen]
        assert cache.shape == (seqlen, last_dim)
        shape = [1] * ndim
        shape[1], shape[-1] = seqlen, last_dim
        return cache.view(*shape)
    elif positions.size(0) == 1:
        assert positions.shape == (1, seqlen)
        cache = cache[positions.squeeze(0)]
        assert cache.shape == (seqlen, last_dim)
        shape = [1] * ndim
        shape[1], shape[-1] = seqlen, last_dim
        return cache.view(*shape)
    else:
        assert positions.shape == (x.shape[0], seqlen)
        cache_expanded = cache[None, :, None, :].expand(x.shape[0], -1, -1, -1)
        cache = torch.gather(
            cache_expanded,
            dim=1,
            index=positions.view(x.shape[0], seqlen, 1, 1).expand(
                x.shape[0], seqlen, 1, cache_expanded.shape[-1]
            ),
        )
        assert cache.shape == (x.shape[0], seqlen, 1, last_dim)
        return cache


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _maybe_wrap_positions(
    positions: torch.Tensor | None,
    x: torch.Tensor,
) -> torch.Tensor | None:
    """Wrap positions as a DTensor deriving mesh and placements from x (xq/xk).

    TODO: In a full DTensor rewrite, positions should be made a DTensor
    in/right after dataloading, together with inputs and labels.

    When TP uses use_local_output=False (DeepSeek V3, Qwen3, GPT-OSS),
    x is a DTensor but positions is a plain tensor. The downstream
    torch.gather requires both operands to be the same type.

    Positions (bsz, seqlen) has fewer dimensions than x (bsz, seqlen,
    n_heads, head_dim), so we only preserve Shard placements for shared
    dimensions. Shard dims beyond positions' rank (e.g. Shard(2) for TP
    on heads) become Replicate.
    """
    if (
        positions is not None
        and isinstance(x, DTensor)
        and not isinstance(positions, DTensor)
    ):
        ndim = positions.ndim
        placements = tuple(
            p if not isinstance(p, Shard) or p.dim < ndim else Replicate()
            for p in x.placements
        )
        positions = DTensor.from_local(
            positions,
            x.device_mesh,
            placements,
            run_check=False,
        )
    return positions


def apply_rotary_emb_complex(
    *xs: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Apply complex-format RoPE to one or more ``(bsz, seqlen, n_heads, head_dim)``
    tensors, returning one rotated tensor per input in its input dtype.

    The ``freqs_cis`` broadcast shape is derived from the first tensor and reused
    for the rest, so all inputs must share seqlen and head_dim.
    """
    positions = _maybe_wrap_positions(positions, xs[0])
    if positions is not None:
        _maybe_check_max_pos(positions, max_valid_pos=freqs_cis.shape[0] - 1)
    xs_ = [torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) for x in xs]
    freqs_cis = _reshape_for_broadcast(
        freqs_cis, xs_[0], positions, last_dim=xs_[0].shape[-1]
    )
    return tuple(
        torch.view_as_real(x_ * freqs_cis).flatten(3).type_as(x)
        for x, x_ in zip(xs, xs_)
    )


def apply_rotary_emb_cos_sin(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply cos/sin-format RoPE to query and key tensors (Qwen3/GPT-OSS style).

    Args:
        xq: (bsz, seqlen, n_heads, head_dim)
        xk: (bsz, seqlen, n_kv_heads, head_dim)
        rope_cache: (max_seqlen, head_dim * 2) with cos and sin concatenated,
            or (bsz, seqlen, 1, head_dim * 2) if already broadcast-shaped (MRoPE)
        positions: optional position indices
    """
    positions = _maybe_wrap_positions(positions, xq)
    head_dim = xq.shape[-1]
    if rope_cache.ndim != 4:
        # Qwen3-VL MRoPE validates position IDs before building its 4D cache.
        # Here shape[0] is max sequence length only for non-4D indexed caches.
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=rope_cache.shape[0] - 1)
        rope_cache = _reshape_for_broadcast(
            rope_cache, xq, positions, last_dim=xq.shape[-1] * 2
        )
    cos = rope_cache[..., :head_dim].to(device=xq.device)
    sin = rope_cache[..., head_dim:].to(device=xq.device)
    xq_f = xq.float()
    xk_f = xk.float()
    xq_out = (xq_f * cos) + (_rotate_half(xq_f) * sin)
    xk_out = (xk_f * cos) + (_rotate_half(xk_f) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)
