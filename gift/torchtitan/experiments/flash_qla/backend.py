"""Opt-in raw-gate GDN backends for the native opaque GDN op."""
from __future__ import annotations

from functools import cache
import hashlib
import importlib
import inspect
import logging
import os
from pathlib import Path
import subprocess
import sys

import torch

_LOG = logging.getLogger(__name__)
_VARIANTS = {"fla", "external_gate", "fused_gate"}
_ACTIVATED = set()


def _variant(variant):
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown GDN variant: {variant!r}")


def _source(module, expected):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(expected.resolve()):
        raise RuntimeError(f"Wrong staged source for {module.__name__}: {path}; expected {expected}")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@cache
def _stage_sources():
    # The training bridge adds gift/helpers only; do not rely on editable installs.
    gift = Path(__file__).resolve().parents[3]
    for relative in ("contrib/flash-linear-attention", "contrib/flash-qla"):
        path = str(gift / relative)
        if path not in sys.path:
            sys.path.insert(0, path)


@cache
def _qla():
    _stage_sources()
    module = importlib.import_module("flash_qla.ops.gated_delta_rule.chunk")
    _source(module, Path(__file__).resolve().parents[3] / "contrib/flash-qla")
    return module


def _validate(q, k, v, raw_g, beta, A_log, dt_bias, cu_seqlens, scale):
    tensors = (q, k, v, raw_g, beta, A_log, dt_bias)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("GDN requires q/k/v/raw_g/beta/A_log/dt_bias tensors")
    if q.ndim != 4 or q.shape[-2:] != (16, 128) or k.shape != q.shape:
        raise ValueError("GDN requires matching [B,S,16,128] q/k")
    if v.shape != (*q.shape[:2], 32, 128):
        raise ValueError("GDN requires [B,S,32,128] values (GVA=2)")
    if raw_g.shape != v.shape[:3] or beta.shape != raw_g.shape:
        raise ValueError("GDN gate and beta must have shape [B,S,32]")
    if A_log.shape != (32,) or dt_bias.shape != (32,):
        raise ValueError("GDN gate parameters must have shape [32]")
    if any(t.dtype != torch.bfloat16 for t in tensors[:5]):
        raise ValueError("GDN requires BF16 q/k/v/raw_g/beta")
    if any(t.dtype not in (torch.bfloat16, torch.float32) for t in tensors[5:]):
        raise ValueError("GDN gate parameters must be BF16 or FP32")
    if q.device.type != "cuda" or any(t.device != q.device for t in tensors):
        raise ValueError("GDN requires all tensors on the same CUDA device")
    if cu_seqlens is not None:
        if (q.shape[0] != 1 or cu_seqlens.ndim != 1
                or cu_seqlens.dtype not in (torch.int32, torch.int64)
                or cu_seqlens.device != q.device):
            raise ValueError("GDN varlen requires B=1 and CUDA integer cu_seqlens")
    if scale is not None and scale != 128 ** -0.5:
        raise ValueError("GDN supports only default K**-0.5 scale")


class _FusedGate(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q, k, v, raw_g, beta, A_log, dt_bias, cu_seqlens, scale):
        from fla.ops.gated_delta_rule.gate import gdn_gate_chunk_cumsum

        # Match the pinned public Function input guard before lower-level kernels.
        q, k, v, raw_g, beta, A_log, dt_bias = (
            tensor.contiguous() for tensor in (q, k, v, raw_g, beta, A_log, dt_bias)
        )
        cu_seqlens = cu_seqlens.contiguous() if cu_seqlens is not None else None
        qla = _qla()
        cp = qla._auto_intra_cp_context(k, v, cu_seqlens, qla.CHUNK_SIZE, False, True)
        g = gdn_gate_chunk_cumsum(
            g=raw_g, A_log=A_log, chunk_size=qla.CHUNK_SIZE,
            scale=None, dt_bias=dt_bias, cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
        )
        A = qla.kkt_solve(k=k, b=beta, cu_seqlens=cp.cu_seqlens, chunk_size=qla.CHUNK_SIZE)
        initial_state, cp_cache = qla.cp_preprocess_fwd(
            cp, k=k, v=v, a=A, g=g, beta=beta, initial_state=None,
            state_v_first=False, enable_fwd_cp_cache=False,
        )
        out, _, _ = qla.fused_gdr_fwd(
            q=q, k=k, v=v, a=A, g=g, b=beta, scale=scale,
            initial_state=initial_state, output_final_state=False,
            output_h=False, output_o=True, cu_seqlens=cp.cu_seqlens,
            cp_seq_map=cp.seq_map_c2r, raw_cu_seqlens=cp.cu_seqlens,
            state_v_first=False,
        )
        ctx.save_for_backward(q, k, v, raw_g, beta, A_log, dt_bias, g, A, cu_seqlens)
        ctx.cp, ctx.cp_cache, ctx.scale = cp, cp_cache, scale
        return out.to(q.dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do):
        from fla.ops.gated_delta_rule.gate import gdn_gate_bwd

        q, k, v, raw_g, beta, A_log, dt_bias, g, A, cu = ctx.saved_tensors
        dq, dk, dv, db, dg, _ = _qla().chunk_gated_delta_rule_bwd(
            q=q, k=k, v=v, g=g, beta=beta, A=A, do=do.contiguous(),
            dht=None, scale=ctx.scale, initial_state=None, cu_seqlens=cu,
            state_v_first=False, auto_cp=False,
            cp_cache=ctx.cp_cache, cp_context=ctx.cp,
        )
        # QLA already returns natural-log dg after reverse cumsum and GVA reduction.
        dr, dA, dt = gdn_gate_bwd(g=raw_g, A_log=A_log, dt_bias=dt_bias, dyg=dg)
        return dq.to(q), dk.to(k), dv.to(v), dr, db.to(beta), dA, dt, None, None


def run_gdn(variant, q, k, v, raw_g, beta, A_log, dt_bias,
            cu_seqlens=None, scale=None):
    """Run a raw-gate GDN variant, failing on unsupported contracts."""
    _variant(variant)
    _validate(q, k, v, raw_g, beta, A_log, dt_bias, cu_seqlens, scale)
    scale = 128 ** -0.5 if scale is None else scale
    key = (variant, torch.is_grad_enabled())
    if key not in _ACTIVATED:
        _LOG.warning("GDN_BACKEND_ACTIVATED variant=%s grad_enabled=%s", *key)
        _ACTIVATED.add(key)
    if variant == "fla":
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        # Remove dispatch and compiler wrappers to avoid dispatch recursion.
        return inspect.unwrap(chunk_gated_delta_rule)(
            q=q, k=k, v=v, g=raw_g, beta=beta, A_log=A_log,
            dt_bias=dt_bias, use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=False, output_final_state=False,
            cu_seqlens=cu_seqlens, scale=scale,
        )
    if variant == "external_gate":
        from fla.ops.gated_delta_rule.gate import fused_gdn_gate

        g = fused_gdn_gate(raw_g, A_log, dt_bias, output_dtype=torch.float32)
        return _qla().chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False, cu_seqlens=cu_seqlens, auto_cp=False,
        )
    return _FusedGate.apply(q, k, v, raw_g, beta, A_log, dt_bias, cu_seqlens, scale), None


def _external_gate_backward(leaves, grad_outputs, static, _scalars, _text):
    from fla.ops.gated_delta_rule.gate import gdn_gate_bwd, gdn_gate_fwd

    (do,) = grad_outputs
    (cu,) = static
    q, k, v, raw_g, beta, A_log, dt_bias = (t.contiguous() for t in leaves)
    cu = cu.contiguous() if cu is not None else None
    qla = _qla()
    with torch.cuda.device(q.device):
        # Match the public gate and scan input guards and recompute only the
        # intermediates consumed by the stateless, CP-disabled native backward.
        g = gdn_gate_fwd(g=raw_g, A_log=A_log, dt_bias=dt_bias, output_dtype=torch.float32)
        g = qla.chunk_local_cumsum(g=g, cu_seqlens=cu, chunk_size=qla.CHUNK_SIZE)
        A = qla.kkt_solve(k=k, b=beta, cu_seqlens=cu, chunk_size=qla.CHUNK_SIZE)
        dq, dk, dv, db, dg, _ = qla.chunk_gated_delta_rule_bwd(
            q=q, k=k, v=v, g=g, beta=beta, A=A, do=do.contiguous(),
            scale=k.shape[-1] ** -0.5, cu_seqlens=cu, auto_cp=False,
        )
        # The native backward returns the natural-log gradient in FP32.
        dr, dA, dt = gdn_gate_bwd(g=raw_g, A_log=A_log, dt_bias=dt_bias, dyg=dg)
    return [dq.to(q), dk.to(k), dv.to(v), dr, db.to(beta), dA, dt]


def install_backend(variant: str) -> dict:
    """Explicitly select a strict priority-zero backend for native GDN dispatch."""
    _variant(variant)
    direct_backward = os.environ.get("GIFT_FLASH_QLA_DIRECT_BACKWARD") or "0"
    if direct_backward not in ("0", "1"):
        raise ValueError("GIFT_FLASH_QLA_DIRECT_BACKWARD must be 0 or 1")
    direct_backward = direct_backward == "1"
    if direct_backward and variant != "external_gate":
        raise ValueError("GIFT_FLASH_QLA_DIRECT_BACKWARD=1 requires external_gate")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1":
        raise RuntimeError("FLA_DISABLE_BACKEND_DISPATCH=1 prevents explicit GDN selection")
    _stage_sources()
    import fla.ops.backends as dispatch
    from fla.ops.gated_delta_rule.backends import gdr_registry

    if dispatch._DISPATCH_DISABLED:
        raise RuntimeError("FLA dispatch was disabled when imported")
    from torchtitan.models.common.attention.linear.fla_custom_ops import _DIRECT_VJP

    gift = Path(__file__).resolve().parents[3]
    fla = importlib.import_module("fla.ops.gated_delta_rule.chunk")
    identity = {
        "variant": variant,
        "direct_backward": direct_backward,
        "fla_upstream_revision": "1d2281ff1b7988d8070f00dba5ffa9747ae3a60e",
        "flash_qla_upstream_revision": "a14dbf3474514e1674526431e960325f72081e9f",
        "task_head": subprocess.check_output(
            ["git", "-C", str(gift), "rev-parse", "HEAD"], text=True
        ).strip(),
        "fla": _source(fla, gift / "contrib/flash-linear-attention"),
        "backend": _source(importlib.import_module(__name__), gift),
    }
    if variant != "fla":
        identity["flash_qla"] = _source(_qla(), gift / "contrib/flash-qla")

    class SelectedBackend(dispatch.BaseBackend):
        backend_type = "experimental_flash_qla_gate"
        priority = 0

        def verify(self, func_name, *args, **kwargs):
            # Reject inside the implementation: default FLA verification swallows errors.
            return func_name == "chunk_gated_delta_rule", None

        def chunk_gated_delta_rule(
            self, q, k, v, g, beta, scale=None, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=False,
            use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
            state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None,
            cp_context=None, **kwargs,
        ):
            use_gate = kwargs.pop("use_gate_in_kernel", False)
            A_log = kwargs.pop("A_log", None)
            dt_bias = kwargs.pop("dt_bias", None)
            if (initial_state is not None or output_final_state or
                    use_qk_l2norm_in_kernel or use_beta_sigmoid_in_kernel or
                    allow_neg_eigval or state_v_first or cp_context is not None or
                    not use_gate or kwargs):
                raise ValueError("Explicit GDN backend requires raw gate, upstream qknorm, CP1, no state or extra flags")
            return run_gdn(variant, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, scale)

    gdr_registry.register(SelectedBackend())
    if direct_backward:
        _DIRECT_VJP["gdn_chunk"] = _external_gate_backward
    else:
        _DIRECT_VJP.pop("gdn_chunk", None)
    return identity
