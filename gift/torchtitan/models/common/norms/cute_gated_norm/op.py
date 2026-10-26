"""Opaque forward/backward ops for model-level torch.compile and autograd.

Only the CUDA implementations import CuTe. FakeTensor propagation, config
construction and the default norm backend do not need the optional DSL package.
The extra forward outputs are private saved tensors, never shared scratch.
"""

import torch

from torchtitan.distributed.cute_gated_norm import register_cute_gated_norm_sharding

_PARAM_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _validate_inputs(x, weight, gate_up_weight, gate_down_weight, eps):
    if x.ndim < 1 or x.shape[-1] <= 0 or x.shape[-1] % 128:
        raise ValueError("CuTe gated norm requires a hidden size divisible by 128")
    if x.device.type not in ("cuda", "meta") or x.dtype != torch.bfloat16:
        raise ValueError("norm_backend='cute' requires BF16 CUDA inputs")
    hidden = x.shape[-1]
    # CuTe 4.4.1 unpacks dynamic shape leaves as int32. Keep row counts in
    # range; layouts/strides and row*hidden address arithmetic remain int64.
    if x.numel() // hidden > 2**31 - 64:
        raise ValueError("CuTe gated norm supports at most 2**31 - 64 rows")
    for name, tensor, shape in (
        ("weight", weight, (hidden,)),
        ("gate_up_weight", gate_up_weight, (16, hidden)),
        ("gate_down_weight", gate_down_weight, (hidden, 16)),
    ):
        if tensor.shape != shape:
            raise ValueError(f"CuTe gated norm expects {name}.shape={shape}")
        if tensor.device != x.device or tensor.dtype not in _PARAM_DTYPES:
            raise ValueError(f"{name} must be BF16, FP16 or FP32 on {x.device}")
    if not eps > 0:
        raise ValueError("CuTe gated norm requires eps > 0")


def _check_device(x):
    if torch.cuda.get_device_capability(x.device) != (9, 0):
        raise ValueError("norm_backend='cute' requires a Hopper (SM90) GPU")


def _aligned(tensor):
    tensor = tensor.contiguous()
    # A contiguous slice can still have an unaligned storage offset. TMA and
    # the fake CuTe tensor descriptors promise at least 16-byte alignment.
    return tensor.clone() if tensor.data_ptr() % 16 else tensor


def _load_kernels():
    try:
        import tvm_ffi  # noqa: F401

        from .backward import gated_norm_bwd
        from .forward import gated_norm_fwd
    except ImportError as exc:
        raise ImportError(
            "norm_backend='cute' requires nvidia-cutlass-dsl[cu13]==4.4.1 "
            "and apache-tvm-ffi==0.1.11; install torchtitan[cute]."
        ) from exc
    return gated_norm_fwd, gated_norm_bwd


@torch.library.custom_op(
    "torchtitan::cute_gated_norm_fwd", mutates_args=(), device_types="cuda"
)
def _forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_down_weight: torch.Tensor,
    eps: float,
    gating_weight: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _validate_inputs(x, weight, gate_up_weight, gate_down_weight, eps)
    _check_device(x)
    if x.numel() == 0:
        y, rho, p, *_ = _forward_fake(
            x, weight, gate_up_weight, gate_down_weight, eps, gating_weight
        )
        # Even an empty row shard must return the replicated auxiliary weights.
        return (
            y,
            rho,
            p,
            weight.to(dtype=torch.float32, copy=True),
            gate_down_weight.to(dtype=torch.bfloat16, copy=True),
            gate_down_weight.t().to(torch.bfloat16).contiguous(),
            gate_up_weight.t().to(torch.bfloat16).contiguous(),
        )
    forward, _ = _load_kernels()
    # Compilation and stream lookup must use the input's device, including
    # when this process's current device is a different GPU.
    with torch.cuda.device(x.device):
        y, rho, p, prep = forward(
            _aligned(x),
            _aligned(weight),
            _aligned(gate_up_weight),
            _aligned(gate_down_weight),
            eps,
            gating_weight,
        )
    # Preserve leading dimensions so DTensor can shard rows without flattening
    # a batch/sequence shard across the normalized hidden dimension.
    return y, rho.view(x.shape[:-1]), p.view(*x.shape[:-1], 16), *prep


@_forward.register_fake
def _forward_fake(x, weight, gate_up_weight, gate_down_weight, eps, gating_weight):
    _validate_inputs(x, weight, gate_up_weight, gate_down_weight, eps)
    hidden = x.shape[-1]
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        x.new_empty(x.shape[:-1], dtype=torch.float32),
        x.new_empty((*x.shape[:-1], 16), dtype=torch.float32),
        x.new_empty((hidden,), dtype=torch.float32),
        x.new_empty((hidden, 16)),
        x.new_empty((16, hidden)),
        x.new_empty((hidden, 16)),
    )


@torch.library.custom_op(
    "torchtitan::cute_gated_norm_bwd", mutates_args=(), device_types="cuda"
)
def _backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    rho: torch.Tensor,
    p: torch.Tensor,
    w_f: torch.Tensor,
    gd_bf: torch.Tensor,
    gd_t: torch.Tensor,
    gu_t: torch.Tensor,
    gating_weight: float,
    weight_dtype: torch.dtype,
    up_dtype: torch.dtype,
    down_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.numel() == 0:
        return (
            torch.empty_like(x, memory_format=torch.contiguous_format),
            torch.zeros_like(w_f, dtype=weight_dtype),
            torch.zeros_like(gd_t, dtype=up_dtype),
            torch.zeros_like(gd_bf, dtype=down_dtype),
        )
    _, backward = _load_kernels()
    with torch.cuda.device(x.device):
        return backward(
            _aligned(dy),
            _aligned(x),
            rho.reshape(-1),
            p.reshape(-1, 16),
            w_f,
            gd_bf,
            gd_t,
            gu_t,
            gating_weight,
            (weight_dtype, up_dtype, down_dtype),
        )


@_backward.register_fake
def _backward_fake(
    dy,
    x,
    rho,
    p,
    w_f,
    gd_bf,
    gd_t,
    gu_t,
    gating_weight,
    weight_dtype,
    up_dtype,
    down_dtype,
):
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(w_f, dtype=weight_dtype),
        torch.empty_like(gd_t, dtype=up_dtype),
        torch.empty_like(gd_bf, dtype=down_dtype),
    )


def _setup_context(ctx, inputs, output):
    x, weight, gate_up_weight, gate_down_weight, _, gating_weight = inputs
    ctx.save_for_backward(x, *output[1:])
    ctx.mark_non_differentiable(*output[1:])
    ctx.set_materialize_grads(False)
    ctx.gating_weight = gating_weight
    ctx.grad_dtypes = (weight.dtype, gate_up_weight.dtype, gate_down_weight.dtype)


def _autograd_backward(ctx, dy, *unused):
    if dy is None:
        return (None,) * 6
    grads = _backward(dy, *ctx.saved_tensors, ctx.gating_weight, *ctx.grad_dtypes)
    return *grads, None, None


_forward.register_autograd(_autograd_backward, setup_context=_setup_context)

register_cute_gated_norm_sharding(_forward._opoverload, _backward._opoverload)


def gated_norm(x, weight, gate_up_weight, gate_down_weight, *, eps, gating_weight):
    """Self-gated RMSNorm for BF16 Hopper inputs; supports first-order gradients."""
    _validate_inputs(x, weight, gate_up_weight, gate_down_weight, eps)
    return _forward(x, weight, gate_up_weight, gate_down_weight, eps, gating_weight)[0]
