"""Replay only the inputs to the pinned FLA GDN backward."""

import torch


# GIFT uses the pinned FLA default, whose 64-token path has a fused solver.
GDN_FLA_CHUNK_SIZE = 64


@torch.library.custom_op("torchtitan::gdn_fla_backward", mutates_args=())
def gdn_fla_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    grad_output: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    import triton

    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd
    from fla.ops.gated_delta_rule.chunk_fwd import (
        chunk_gated_delta_rule_fwd_kkt_solve_kernel,
    )
    from fla.ops.gated_delta_rule.gate import gdn_gate_chunk_cumsum
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.index import prepare_chunk_indices

    q, k, v, raw_gate, beta, A_log, dt_bias, grad_output = (
        t.contiguous() for t in (q, k, v, raw_gate, beta, A_log, dt_bias, grad_output)
    )
    chunk_size = GDN_FLA_CHUNK_SIZE
    with torch.cuda.device(q.device):
        chunk_indices = None
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.contiguous()
            # Match native forward's cache key for the packed chunk indices.
            chunk_indices = prepare_chunk_indices(
                cu_seqlens, chunk_size, cu_seqlens_cpu=None
            )
        gate = gdn_gate_chunk_cumsum(
            g=raw_gate,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        # Match FLA's 64-token intra-chunk solve. Native backward builds W and U.
        batch, seqlen, heads, key_width = k.shape
        value_heads = v.shape[2]
        num_chunks = (
            triton.cdiv(seqlen, chunk_size)
            if chunk_indices is None
            else len(chunk_indices)
        )
        A = k.new_zeros((batch, seqlen, value_heads, chunk_size))
        chunk_gated_delta_rule_fwd_kkt_solve_kernel[(num_chunks, batch * value_heads)](
            k=k,
            g=gate,
            beta=beta,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=seqlen,
            H=heads,
            HV=value_heads,
            K=key_width,
            BT=chunk_size,
            BC=16,
        )
        dq, dk, dv, db, dg, _, dA_log, ddt_bias = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            A=A,
            scale=key_width**-0.5,
            initial_state=None,
            do=grad_output,
            dht=None,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_gate_in_kernel=True,
            g_input=raw_gate,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=chunk_size,
        )
    return (
        dq.to(q).contiguous(),
        dk.to(k).contiguous(),
        dv.to(v).contiguous(),
        dg.to(raw_gate).contiguous(),
        db.to(beta).contiguous(),
        dA_log.to(A_log).contiguous(),
        ddt_bias.to(dt_bias).contiguous(),
    )


@gdn_fla_backward.register_fake
def _(q, k, v, raw_gate, beta, A_log, dt_bias, grad_output, cu_seqlens=None):
    return tuple(
        torch.empty_like(t, memory_format=torch.contiguous_format)
        for t in (q, k, v, raw_gate, beta, A_log, dt_bias)
    )
