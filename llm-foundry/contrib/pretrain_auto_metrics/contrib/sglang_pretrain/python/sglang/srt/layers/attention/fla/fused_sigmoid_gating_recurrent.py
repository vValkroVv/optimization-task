from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    # Parameters for target_verify support (unused for decode)
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    replay_kv,
    replay_gb,
    stride_replay_kv,
    stride_replay_gb,
    last_correct_step,
    replay_track_indices,
    replay_track_steps,
    stride_layer_kv,
    stride_layer_gb,
    stride_layer_h,
    # ================================================
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    # Optional flags for target_verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
    SAVE_REPLAY_INPUTS: tl.constexpr = False,
    REPLAY_COMMIT: tl.constexpr = False,
    REPLAY_TRACK: tl.constexpr = False,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    if REPLAY_COMMIT:
        i_l = tl.program_id(0).to(tl.int64)
        i_k = 0
        i_v = tl.program_id(1) % ((V + BV - 1) // BV)
        i_hv = tl.program_id(1) // ((V + BV - 1) // BV)
        i_n = tl.program_id(2)
        q += i_l * stride_layer_kv
        k += i_l * stride_layer_kv
        v += i_l * stride_layer_kv
        a += i_l * stride_layer_gb
        b += i_l * stride_layer_gb
        h0_source += i_l * stride_layer_h
    else:
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    if REPLAY_COMMIT:
        commit_step = tl.load(last_correct_step + i_n).to(tl.int32)
        commit_slot = tl.load(h0_indices + i_n).to(tl.int64)
        T = commit_step + 1
        if REPLAY_TRACK:
            track_step = tl.load(replay_track_steps + i_n).to(tl.int32)
            track_slot = tl.load(replay_track_indices + i_n).to(tl.int64)
            T = tl.maximum(commit_step, track_step) + 1

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + bos * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v[None, :] * K
                + o_k[:, None]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Preload tree attention data if needed
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(
            retrieve_parent_token_base, mask=mask_retrieve, other=0
        )

    # Prepare intermediate state cache index if enabled
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES or SAVE_REPLAY_INPUTS:
        cache_idx = tl.load(intermediate_state_indices + i_n)

    step_idx = 0
    for _ in range(0, T):
        # Tree attention: load parent's cached state
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx == 0 uses b_h from USE_INITIAL_STATE
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        if REPLAY_COMMIT:
            b_g = b_a
            b_beta = b_b

        if SAVE_REPLAY_INPUTS:
            if cache_idx >= 0:
                row = cache_idx * cache_steps + step_idx
                p_kv = replay_kv + row * stride_replay_kv
                p_gb = replay_gb + row * stride_replay_gb
                if i_v == 0:
                    if i_hv % (HV // H) == 0:
                        tl.store(
                            p_kv + i_h * K + o_k,
                            b_k.to(p_kv.dtype.element_ty),
                            mask=mask_k,
                        )
                    tl.store(p_gb + i_hv, b_g)
                    tl.store(p_gb + HV + i_hv, b_beta)
                tl.store(
                    p_kv + H * K + i_hv * V + o_v,
                    b_v.to(p_kv.dtype.element_ty),
                    mask=mask_v,
                )

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        if not REPLAY_COMMIT:
            b_o = tl.sum(b_h * b_q[:, None], 0)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if REPLAY_COMMIT:
            if (step_idx == commit_step) & (commit_slot >= 0):
                p_hc = (
                    h0_source
                    + commit_slot * HV * K * V
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_hc, b_h.to(p_hc.dtype.element_ty), mask=mask_h)
            if REPLAY_TRACK:
                if (step_idx == track_step) & (track_slot >= 0):
                    p_ht = (
                        h0_source
                        + track_slot * HV * K * V
                        + i_hv * K * V
                        + o_v[None, :] * K
                        + o_k[:, None]
                    )
                    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        # Cache intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1

        # Update pointers for next timestep
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += stride_a

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE and not REPLAY_COMMIT:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx * HV * K * V
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    # Optional parameters for target_verify support
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[
        int
    ] = None,  # kept for API compat; stride is derived from ``intermediate_states_buffer.shape[1]``
    retrieve_parent_token: Optional[torch.Tensor] = None,
    replay_kv: Optional[torch.Tensor] = None,
    replay_gb: Optional[torch.Tensor] = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    Supports both decode and target_verify modes:
    - decode: standard single-step update with state write-back
    - target_verify: multi-step with intermediate state caching, optional tree attention,
                     and optional state update disable
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]
    # Both paths (KDA/GDN) advance p_a once per token, so use the token-axis stride.
    # For 2D a ([T, ...]) this is stride(0); for 3D a ([B, T, ...]) this is stride(1).
    # Using stride()[-2] covers GDN [T, HV] and KDA layouts ([T, HV*K] / [B, T, HV*K]).
    stride_a = a.stride()[-2]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)

    # Prepare retrieve_parent_token strides
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
        stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)
    else:
        stride_retrieve_parent_token_seq = 0
        stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)

    grid = (NK, NV, N * HV)

    # Per-req stride must match the buffer's allocated dim, not runtime steps
    # (they can differ under --speculative-adaptive).
    cache_stride_steps = (
        intermediate_states_buffer.shape[1]
        if intermediate_states_buffer is not None
        else 0
    )
    if replay_kv is not None:
        assert retrieve_parent_token is None, "GDN spec replay supports topk == 1 only"
        cache_stride_steps = replay_kv.shape[1]

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_stride_steps,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        replay_kv=replay_kv,
        replay_gb=replay_gb,
        stride_replay_kv=replay_kv.stride(-2) if replay_kv is not None else 0,
        stride_replay_gb=replay_gb.stride(-2) if replay_gb is not None else 0,
        last_correct_step=None,
        replay_track_indices=None,
        replay_track_steps=None,
        stride_layer_kv=0,
        stride_layer_gb=0,
        stride_layer_h=0,
        scale=scale,
        T=T,
        stride_a=stride_a,
        stride_q=stride_q,
        stride_k=stride_k,
        stride_v=stride_v,
        stride_b=stride_b,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        SAVE_REPLAY_INPUTS=replay_kv is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o


def fused_sigmoid_gating_delta_rule_replay_commit(
    kv: torch.Tensor,
    gb: torch.Tensor,
    ssm_states: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    last_correct_step: torch.Tensor,
    num_k_heads: int,
    track_indices: Optional[torch.Tensor] = None,
    track_steps: Optional[torch.Tensor] = None,
):
    L, B, T, _ = kv.shape
    _, _, HV, V, K = ssm_states.shape
    H = num_k_heads
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    assert triton.cdiv(K, BK) == 1
    assert kv.shape[-1] == H * K + HV * V and gb.shape[-1] == 2 * HV
    assert kv.stride(1) == T * kv.stride(2) and gb.stride(1) == T * gb.stride(2)
    k, v = kv[..., : H * K], kv[..., H * K :]
    g, beta = gb[..., :HV], gb[..., HV:]
    grid = (L, triton.cdiv(V, BV) * HV, B)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=g,
        a=g,
        dt_bias=g,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=k,
        k=k,
        v=v,
        b=beta,
        o=g,
        h0_source=ssm_states,
        h0_indices=ssm_state_indices,
        cu_seqlens=None,
        intermediate_states_buffer=None,
        intermediate_state_indices=None,
        cache_steps=T,
        retrieve_parent_token_ptr=None,
        stride_retrieve_parent_token_seq=0,
        stride_retrieve_parent_token_token=0,
        replay_kv=None,
        replay_gb=None,
        stride_replay_kv=0,
        stride_replay_gb=0,
        last_correct_step=last_correct_step,
        replay_track_indices=track_indices,
        replay_track_steps=track_steps,
        stride_layer_kv=kv.stride(0),
        stride_layer_gb=gb.stride(0),
        stride_layer_h=ssm_states.stride(0),
        scale=1.0,
        T=T,
        stride_a=gb.stride(2),
        stride_q=kv.stride(2),
        stride_k=kv.stride(2),
        stride_v=kv.stride(2),
        stride_b=gb.stride(2),
        NP2_T=triton.next_power_of_2(T),
        B=1,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=True,
        IS_VARLEN=False,
        IS_KDA=False,
        DISABLE_STATE_UPDATE=True,
        REPLAY_COMMIT=True,
        REPLAY_TRACK=track_indices is not None,
        num_warps=1,
        num_stages=3,
    )
