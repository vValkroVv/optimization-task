"""``fla`` kernels wrapped as opaque ``torch.library`` custom ops.

Wraps the short conv (:func:`qkv_conv1d_unpack_op`), the per-family scans
(:func:`gdn_chunk_op`, :func:`kda_chunk_op`) and the output norm
(:func:`gated_rmsnorm_op`). Registering each as a custom op keeps
its int-heavy ``cu_seqlens`` index math out of the selective-AC op stream, which
otherwise desyncs the SAC recompute replay (``Autograd not support dtype: Int``);
the ops are also on the SAC MUST_SAVE list (see ``distributed/activation_checkpoint.py``).

A custom op is opaque to autograd, so each registers its own backward via
:func:`vjp_via_recompute`. ``fla`` is imported lazily inside the op bodies.
"""

from collections.abc import Callable

import torch

__all__ = [
    "fla_vjp",
    "gated_rmsnorm_op",
    "gdn_chunk_op",
    "kda_chunk_op",
    "qkv_conv1d_unpack_op",
    "vjp_via_recompute",
]


_RECOMPUTE: dict[str, "Callable"] = {}
"""kernel name -> ``(static_args) -> forward closure over the differentiable inputs``."""


@torch.library.custom_op("torchtitan::fla_vjp", mutates_args=())
def fla_vjp(
    kernel: str,
    leaves: list[torch.Tensor],
    grad_outputs: list[torch.Tensor],
    static: list[torch.Tensor | None],
    scalars: list[int | float | bool],
    text: str,
) -> list[torch.Tensor]:
    """One opaque backward for every fla op here.

    ``register_fake`` hides only a forward; a backward registered through
    ``register_autograd`` is traced into the backward graph, so recomputing an
    fla kernel there would hand it fake tensors. Routing all of them through one
    custom op keeps each a single node. The static args a kernel needs arrive
    split by type, because a custom-op schema cannot hold a mixed tuple.
    """
    forward_fn = _RECOMPUTE[kernel](static, scalars, text)
    return list(vjp_via_recompute(forward_fn, leaves, grad_outputs))


@fla_vjp.register_fake
def _(kernel, leaves, grad_outputs, static, scalars, text):
    return [torch.empty_like(t) for t in leaves]


def vjp_via_recompute(forward_fn, leaves, grad_outputs):
    """Backward for an opaque fla custom op: recompute its forward and differentiate.

    Re-runs ``forward_fn(*leaves)`` over freshly-leafed copies of the differentiable
    inputs under ``enable_grad`` (non-tensor args bound by the caller) and lets fla's
    own backward run. Returns grads aligned to ``leaves`` (``None`` where unused).
    """
    leaves = tuple(t.detach().requires_grad_(True) for t in leaves)
    with torch.enable_grad():
        outs = forward_fn(*leaves)
    outs = outs if isinstance(outs, (tuple, list)) else (outs,)
    grad_outputs = [
        torch.zeros_like(o) if g is None else g for o, g in zip(outs, grad_outputs)
    ]
    return torch.autograd.grad(outs, leaves, grad_outputs, allow_unused=True)


@torch.library.custom_op("torchtitan::qkv_conv1d_unpack", mutates_args=())
def qkv_conv1d_unpack_op(
    qkv: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    num_v_heads: int,
    head_dim: int,
    head_v_dim: int,
    cu_seqlens: torch.Tensor | None,
    normalize_qk: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Opaque custom-op wrapper over fla's ``fused_qkv_conv1d_unpack`` (short conv)."""
    from fla.ops.common.fused_conv_qkv import fused_qkv_conv1d_unpack

    with torch.no_grad():
        q, k, v = fused_qkv_conv1d_unpack(
            qkv,
            weight,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            head_dim=head_dim,
            head_v_dim=head_v_dim,
            cp_context=None,
            cu_seqlens=cu_seqlens,
            normalize_qk=normalize_qk,
            eps=eps,
        )
    return q.contiguous(), k.contiguous(), v.contiguous()


@qkv_conv1d_unpack_op.register_fake
def _(
    qkv,
    weight,
    num_heads,
    num_v_heads,
    head_dim,
    head_v_dim,
    cu_seqlens,
    normalize_qk,
    eps,
):
    bs, seqlen = qkv.shape[0], qkv.shape[1]
    return (
        qkv.new_empty(bs, seqlen, num_heads, head_dim),
        qkv.new_empty(bs, seqlen, num_heads, head_dim),
        qkv.new_empty(bs, seqlen, num_v_heads, head_v_dim),
    )


def _qkv_conv_setup(ctx, inputs, output):
    qkv, weight = inputs[0], inputs[1]
    ctx.save_for_backward(qkv, weight)
    # num_heads, num_v_heads, head_dim, head_v_dim, cu_seqlens, normalize_qk, eps
    ctx.fwd_args = inputs[2:]


def _qkv_conv_recompute(static, scalars, text):
    from fla.ops.common.fused_conv_qkv import fused_qkv_conv1d_unpack

    (cu_seqlens,) = static
    num_heads, num_v_heads, head_dim, head_v_dim, normalize_qk, eps = scalars

    def run(qkv, weight):
        return fused_qkv_conv1d_unpack(
            qkv,
            weight,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            head_dim=head_dim,
            head_v_dim=head_v_dim,
            cp_context=None,
            cu_seqlens=cu_seqlens,
            normalize_qk=normalize_qk,
            eps=eps,
        )

    return run


_RECOMPUTE["qkv_conv1d"] = _qkv_conv_recompute


def _qkv_conv_backward(ctx, grad_q, grad_k, grad_v):
    nh, nvh, hd, hvd, cu_seqlens, normalize_qk, eps = ctx.fwd_args
    d_qkv, d_weight = fla_vjp(
        "qkv_conv1d",
        list(ctx.saved_tensors),
        [grad_q, grad_k, grad_v],
        [cu_seqlens],
        [nh, nvh, hd, hvd, normalize_qk, eps],
        "",
    )
    return d_qkv, d_weight, None, None, None, None, None, None, None


qkv_conv1d_unpack_op.register_autograd(
    _qkv_conv_backward, setup_context=_qkv_conv_setup
)


@torch.library.custom_op("torchtitan::gdn_chunk", mutates_args=())
def gdn_chunk_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Opaque custom-op wrapper over fla's ``chunk_gated_delta_rule`` (output only).

    ``A_log`` / ``dt_bias`` are kernel inputs (``use_gate_in_kernel=True``): the raw
    decay input ``g`` is gated as ``-exp(A_log) * softplus(g + dt_bias)`` and cumsummed
    inside the kernel, so both receive grads via the recompute backward.
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    with torch.no_grad():
        o, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
    return o.contiguous()


@gdn_chunk_op.register_fake
def _(q, k, v, g, beta, A_log, dt_bias, cu_seqlens):
    return torch.empty_like(v)


def _gdn_chunk_setup(ctx, inputs, output):
    q, k, v, g, beta, A_log, dt_bias, cu_seqlens = inputs
    ctx.save_for_backward(q, k, v, g, beta, A_log, dt_bias)
    ctx.cu_seqlens = cu_seqlens


def _gdn_chunk_recompute(static, scalars, text):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    (cu_seqlens,) = static

    def run(q, k, v, g, beta, A_log, dt_bias):
        out = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=True,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        )
        return out[0]

    return run


_RECOMPUTE["gdn_chunk"] = _gdn_chunk_recompute


def _gdn_chunk_backward(ctx, grad_o):
    grads = fla_vjp(
        "gdn_chunk", list(ctx.saved_tensors), [grad_o], [ctx.cu_seqlens], [], ""
    )
    return (*grads, None)


gdn_chunk_op.register_autograd(_gdn_chunk_backward, setup_context=_gdn_chunk_setup)


@torch.library.custom_op("torchtitan::kda_chunk", mutates_args=())
def kda_chunk_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Opaque custom-op wrapper over fla's ``chunk_kda`` (output only).

    ``A_log`` / ``dt_bias`` are kernel inputs (``use_gate_in_kernel=True``), so they
    are op inputs and receive grads via the recompute backward.
    """
    from fla.ops.kda import chunk_kda

    with torch.no_grad():
        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
    return o.contiguous()


@kda_chunk_op.register_fake
def _(q, k, v, g, beta, A_log, dt_bias, cu_seqlens):
    return torch.empty_like(v)


def _kda_chunk_setup(ctx, inputs, output):
    q, k, v, g, beta, A_log, dt_bias, cu_seqlens = inputs
    ctx.save_for_backward(q, k, v, g, beta, A_log, dt_bias)
    ctx.cu_seqlens = cu_seqlens


def _kda_chunk_recompute(static, scalars, text):
    from fla.ops.kda import chunk_kda

    (cu_seqlens,) = static

    def run(q, k, v, g, beta, A_log, dt_bias):
        out = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        return out[0]

    return run


_RECOMPUTE["kda_chunk"] = _kda_chunk_recompute


def _kda_chunk_backward(ctx, grad_o):
    grads = fla_vjp(
        "kda_chunk", list(ctx.saved_tensors), [grad_o], [ctx.cu_seqlens], [], ""
    )
    return (*grads, None)


kda_chunk_op.register_autograd(_kda_chunk_backward, setup_context=_kda_chunk_setup)


@torch.library.custom_op("torchtitan::gated_rmsnorm", mutates_args=())
def gated_rmsnorm_op(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    activation: str,
    zero_centered: bool,
) -> torch.Tensor:
    """Opaque wrapper over fla's ``rms_norm_gated`` (the GDN/KDA output norm).

    fla marks its forward ``torch.compiler.disable``, which dynamo refuses to
    trace under ``fullgraph=True``; as a custom op it becomes one opaque node.
    ``bias`` / ``residual`` are always absent here, so they are not plumbed.
    """
    from fla.modules.fused_norm_gate import rms_norm_gated

    with torch.no_grad():
        return rms_norm_gated(
            x, g, weight, None, activation, eps=eps, zero_centered=zero_centered
        )


@gated_rmsnorm_op.register_fake
def _(x, g, weight, eps, activation, zero_centered):
    return torch.empty_like(x)


def _gated_rmsnorm_setup(ctx, inputs, output):
    x, g, weight, eps, activation, zero_centered = inputs
    ctx.save_for_backward(x, g, weight)
    ctx.norm_args = (eps, activation, zero_centered)


def _gated_rmsnorm_recompute(static, scalars, text):
    from fla.modules.fused_norm_gate import rms_norm_gated

    eps, zero_centered = scalars
    activation = text

    def run(x, g, weight):
        return rms_norm_gated(
            x, g, weight, None, activation, eps=eps, zero_centered=zero_centered
        )

    return run


_RECOMPUTE["gated_rmsnorm"] = _gated_rmsnorm_recompute


def _gated_rmsnorm_backward(ctx, grad_o):
    grads = fla_vjp(
        "gated_rmsnorm",
        list(ctx.saved_tensors),
        [grad_o],
        [],
        [ctx.norm_args[0], ctx.norm_args[2]],
        ctx.norm_args[1],
    )
    return (*grads, None, None, None)


gated_rmsnorm_op.register_autograd(
    _gated_rmsnorm_backward, setup_context=_gated_rmsnorm_setup
)
