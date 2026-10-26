"""Public GatedNorm API: launch configuration, validation, autograd and module wrapper."""

from dataclasses import dataclass, replace

import torch

from .backward import gated_norm_bwd, precompile_bwd
from .common import BLK_H, RANK, resolve_num_sms
from .forward import gated_norm_fwd, pick_n_hc, precompile_fwd

_PARAM_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def check_params(H, weight, gate_up_weight, gate_down_weight):
    if H % (2 * BLK_H):
        raise ValueError(f"hidden size {H} must be a multiple of {2 * BLK_H}")
    if tuple(gate_up_weight.shape) != (RANK, H) or tuple(gate_down_weight.shape) != (H, RANK):
        raise ValueError(f"expected gate_up ({RANK}, {H}) and gate_down ({H}, {RANK})")
    if tuple(weight.shape) != (H,):
        raise ValueError(f"expected weight ({H},)")


def check_inputs(x, weight, gate_up_weight, gate_down_weight):
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise TypeError("x must be a bf16 CUDA tensor")
    check_params(x.shape[-1], weight, gate_up_weight, gate_down_weight)
    for name, tensor in (
        ("weight", weight),
        ("gate_up_weight", gate_up_weight),
        ("gate_down_weight", gate_down_weight),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on {x.device}, got {tensor.device}")
        if tensor.dtype not in _PARAM_DTYPES:
            raise TypeError(f"{name} must be bf16, fp16 or fp32, got {tensor.dtype}")


@dataclass(frozen=True)
class GnCfg:
    """Shape and launch configuration for one GatedNorm kernel family.

    num_sms=None uses the process-wide SM budget, which by default reserves 16 SMs for
    concurrent collectives. n_hc=None selects the F2 column split from the run-time M.
    """

    hidden_dim: int
    gate_dim: int = RANK
    num_sms: int | None = None
    n_hc: int | None = None
    eps: float = 1e-6
    lgw: float = 2.0

    def __post_init__(self):
        H = self.hidden_dim
        if not isinstance(H, int) or H <= 0:
            raise ValueError(f"hidden_dim must be a positive int, got {H!r}")
        if H % (2 * BLK_H):
            raise ValueError(f"hidden_dim={H} must be a multiple of {2 * BLK_H}")
        if self.gate_dim != RANK:
            raise ValueError(f"gate_dim must be {RANK}, got {self.gate_dim}")
        for name in ("num_sms", "n_hc"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive int or None, got {value!r}")
        if self.n_hc is not None and (
            H % self.n_hc or (H // self.n_hc) % (2 * BLK_H)
        ):
            raise ValueError(
                f"n_hc={self.n_hc} must split hidden_dim={H} into multiples of {2 * BLK_H}"
            )
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

    @staticmethod
    def of(x, weight, gate_up_weight, gate_down_weight, eps, lgw, cfg=None, num_sms=None):
        check_inputs(x, weight, gate_up_weight, gate_down_weight)
        H = x.shape[-1]
        if cfg is None:
            cfg = GnCfg(
                hidden_dim=H,
                gate_dim=gate_up_weight.shape[0],
                eps=float(eps),
                lgw=float(lgw),
            )
        elif cfg.hidden_dim != H or cfg.gate_dim != gate_up_weight.shape[0]:
            raise ValueError(
                f"cfg is for hidden_dim={cfg.hidden_dim}, gate_dim={cfg.gate_dim}; "
                f"input has H={H}, gate rank={gate_up_weight.shape[0]}"
            )
        if num_sms is not None:
            cfg = replace(cfg, num_sms=int(num_sms))
        return cfg

    def resolve(self, device, rows):
        sms = resolve_num_sms(device, self.num_sms)
        n_hc = self.n_hc
        if n_hc is None:
            n_hc = pick_n_hc(rows, self.hidden_dim, sms)
        return replace(self, num_sms=sms, n_hc=n_hc)


def _check_precompile_params(weight, gate_up_weight, gate_down_weight):
    H = weight.shape[0]
    check_params(H, weight, gate_up_weight, gate_down_weight)
    if not weight.is_cuda:
        raise TypeError("parameters must be on a CUDA device")
    for name, tensor in (
        ("weight", weight),
        ("gate_up_weight", gate_up_weight),
        ("gate_down_weight", gate_down_weight),
    ):
        if tensor.device != weight.device:
            raise ValueError(f"{name} must be on {weight.device}, got {tensor.device}")
        if tensor.dtype not in _PARAM_DTYPES:
            raise TypeError(f"{name} must be bf16, fp16 or fp32, got {tensor.dtype}")
    return H


def gated_norm_precompile(
    weight,
    gate_up_weight,
    gate_down_weight,
    eps=1e-6,
    lgw=2.0,
    *,
    cfg=None,
    num_sms=None,
):
    """Compile all variants reachable by this shape/configuration before the first batch."""
    H = _check_precompile_params(weight, gate_up_weight, gate_down_weight)
    if cfg is None:
        cfg = GnCfg(hidden_dim=H, eps=float(eps), lgw=float(lgw))
    elif cfg.hidden_dim != H or cfg.gate_dim != RANK:
        raise ValueError(
            f"cfg is for hidden_dim={cfg.hidden_dim}, gate_dim={cfg.gate_dim}; "
            f"parameters have H={H}, gate rank={RANK}"
        )
    if num_sms is not None:
        cfg = replace(cfg, num_sms=int(num_sms))
    sms = resolve_num_sms(weight.device, cfg.num_sms)
    precompile_fwd(
        H,
        RANK,
        (gate_up_weight.dtype, weight.dtype, gate_down_weight.dtype),
        cfg.eps,
        cfg.lgw,
        weight.device,
        num_sms=sms,
        n_hc=cfg.n_hc,
    )
    precompile_bwd(
        H,
        RANK,
        cfg.lgw,
        weight.device,
        (weight.dtype, gate_up_weight.dtype, gate_down_weight.dtype),
        num_sms=sms,
    )


class _GatedNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, gate_up_weight, gate_down_weight, cfg):
        input_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1])
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        y, rho, p, prep = gated_norm_fwd(
            x2d,
            weight,
            gate_up_weight,
            gate_down_weight,
            cfg.eps,
            cfg.lgw,
            num_sms=cfg.num_sms,
            n_hc=cfg.n_hc,
        )
        ctx.save_for_backward(x2d, rho, p, *prep)
        ctx.input_shape = input_shape
        ctx.cfg = cfg
        ctx.grad_dtypes = (weight.dtype, gate_up_weight.dtype, gate_down_weight.dtype)
        return y.view(input_shape)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x2d, rho, p, w_f, gd_bf, gd_t, gu_t = ctx.saved_tensors
        dy2d = dy.reshape(x2d.shape).contiguous()
        if dy2d.dtype != torch.bfloat16:
            dy2d = dy2d.to(torch.bfloat16)
        dx, dweight, dgate_up, dgate_down = gated_norm_bwd(
            dy2d,
            x2d,
            rho,
            p,
            w_f,
            gd_bf,
            gd_t,
            gu_t,
            ctx.cfg.lgw,
            ctx.grad_dtypes,
            num_sms=ctx.cfg.num_sms,
        )
        return dx.view(ctx.input_shape), dweight, dgate_up, dgate_down, None


def gated_norm(
    x,
    weight,
    gate_up_weight,
    gate_down_weight,
    eps=1e-6,
    lgw=2.0,
    *,
    cfg=None,
    num_sms=None,
):
    """Differentiable fused GatedNorm for bf16 CUDA inputs."""
    cfg = GnCfg.of(
        x,
        weight,
        gate_up_weight,
        gate_down_weight,
        eps,
        lgw,
        cfg,
        num_sms,
    )
    if x.numel() == 0:
        zero = (
            weight.sum().to(dtype=x.dtype)
            + gate_up_weight.sum().to(dtype=x.dtype)
            + gate_down_weight.sum().to(dtype=x.dtype)
        )
        return x + zero * 0
    rows = x.numel() // x.shape[-1]
    return _GatedNorm.apply(
        x,
        weight,
        gate_up_weight,
        gate_down_weight,
        cfg.resolve(x.device, rows),
    )


class GatedNorm(torch.nn.Module):
    """Drop-in fused GatedNorm module with an optional persistent-kernel SM budget."""

    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        layernorm_gating_weight=2.0,
        device=None,
        dtype=None,
        num_sms=None,
        n_hc=None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.gate_up_projection = torch.nn.Linear(
            hidden_size,
            RANK,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.gate_down_projection = torch.nn.Linear(
            RANK,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.variance_epsilon = eps
        self.layernorm_gating_weight = layernorm_gating_weight
        self.cfg = GnCfg(
            hidden_dim=hidden_size,
            num_sms=num_sms,
            n_hc=n_hc,
            eps=eps,
            lgw=layernorm_gating_weight,
        )

    def forward(self, x):
        return gated_norm(
            x,
            self.weight,
            self.gate_up_projection.weight,
            self.gate_down_projection.weight,
            cfg=self.cfg,
        )
