import threading

import torch
import torch.nn.functional as F
from llmfoundry.models.layers import dsa_monitor

from flash_mla import flash_mla_sparse_fwd
from llmfoundry.models.layers.cudnn_dsa import (
    sparse_attn_score_recompute_wrapper,
    sparse_indexer_score_recompute_wrapper,
    sparse_attention_backward_wrapper,
    indexer_backward_wrapper,
    dense_attn_score_recompute_wrapper,
    dense_indexer_score_recompute_wrapper,
    dense_indexer_backward_wrapper,
)

INDEX_HEADS_OK = (64,)             # cuDNN indexer_backward requires heads>=64
INDEX_DIM_OK = 128
ATTN_HEADS_OK = (32, 64, 128)
ATTN_HEADS_FLASH = (64, 128)
NEG = float("-inf")
WARMUP_KL_TILE = 16384             # key-tile for the (logging-only) warm-up KL scalar

_dense_indexer_backward_cache = dense_indexer_backward_wrapper.__globals__.get(
    "_cache_of_DenseIndexerBackwardObjects"
)
_dense_indexer_backward_cleanup_lock = threading.Lock()
_dense_indexer_backward_cleanup_queued = False


def _queue_dense_indexer_backward_cache_cleanup():
    """Release per-shape dense-backward workspaces after the current backward."""
    cache = _dense_indexer_backward_cache
    if cache is None:
        return

    global _dense_indexer_backward_cleanup_queued
    with _dense_indexer_backward_cleanup_lock:
        if _dense_indexer_backward_cleanup_queued:
            return
        _dense_indexer_backward_cleanup_queued = True

    def cleanup():
        global _dense_indexer_backward_cleanup_queued
        try:
            # The cuDNN wrapper caches one full [Sq, Sk] dummy-topk tensor for
            # every varlen shape. Compiled kernels live in a separate cache.
            cache.clear()
        finally:
            with _dense_indexer_backward_cleanup_lock:
                _dense_indexer_backward_cleanup_queued = False

    try:
        torch.autograd.Variable._execution_engine.queue_callback(cleanup)
    except Exception:
        with _dense_indexer_backward_cleanup_lock:
            _dense_indexer_backward_cleanup_queued = False
        raise


def _check(cond, msg):
    if not cond:
        raise AssertionError("DSA: " + msg)


def _cudnn_bsh_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Return a BSH tensor with a unit-stride sequence dimension for cuDNN DSA.

    cuDNN Frontend transposes BSH to BHS before converting the tensor to a
    CuTe tensor, whose last dimension must have stride 1. For a singleton
    sequence, ``transpose(...).contiguous()`` can be a no-op and preserve the
    original head-size stride instead. An explicit contiguous-format clone
    forces the required layout even when S == 1.
    """
    if tensor.stride(1) == 1:
        return tensor
    return (
        tensor.transpose(1, 2)
        .clone(memory_format=torch.contiguous_format)
        .transpose(1, 2)
    )


def _cu_bounds(cu):
    """Host list of the document boundaries, computed once per cu tensor"""
    bounds = getattr(cu, "_dsa_bounds", None)
    if bounds is None:
        bounds = cu.tolist()
        cu._dsa_bounds = bounds
    return bounds


def _pad_heads(tensor, extra):
    """Zero-pad dimension 2, including the [B, S, H] LSE layout."""
    if not extra:
        return tensor
    return F.pad(tensor, (0, 0) * (tensor.dim() - 3) + (0, extra))


@torch.no_grad()
def select_topk(index_q, index_k, weights, scale, K, positions_q, cu_seqlens=None,
                q_max=None, q_min=None):
    """Fused scoring and selection; valid ids first, followed by -1 padding."""
    from llmfoundry.models.layers.dsa_fused_topk import fused_select_topk
    # Layer callers supply host bounds, avoiding a device-to-host synchronization.
    if q_max is None:
        q_max = int(positions_q.max())
    if q_min is None:
        q_min = int(positions_q.min())
    return fused_select_topk(index_q, index_k, weights, scale, K, positions_q,
                             cu_seqlens, q_max=q_max, q_min=q_min)


class _DSAFunction(torch.autograd.Function):
    """Stage-2 sparse attention and KL over the selected keys.

    Output gradients update Q/KV; KL gradients update only the indexer, with the
    attention target detached. KL is summed over valid queries and selected keys.
    """
    @staticmethod
    def forward(ctx, index_q, index_k, weights, q_attn, kv, topk, topk_length, s_attn, s_idx, d_v, offset, q_real_max=None):
        B, Sq, Hi, Di = index_q.shape
        _, Sk, _, _ = index_k.shape
        _, _, Hq, Dqk = q_attn.shape
        # Skip the corresponding backward when only one of the outputs is used.
        ctx.set_materialize_grads(False)
        _check(B == 1, "operator is packed B=1 (per rank)")
        for t, name in [(index_q, "index_q"), (index_k, "index_k"), (weights, "weights"),
                        (q_attn, "q_attn"), (kv, "kv")]:
            _check(t.dtype == torch.bfloat16 and t.is_contiguous(), f"{name} must be contiguous bf16")
        _check(Hi in INDEX_HEADS_OK and Di == INDEX_DIM_OK, "indexer heads/dim unsupported")
        _check(Hq in ATTN_HEADS_OK, f"attention heads must be in {ATTN_HEADS_OK}, got {Hq}")
        _check(0 < d_v < Dqk, f"d_v must be in (0, Dqk={Dqk}), got {d_v}")
        _check(kv.shape == (1, Sk, Dqk) and index_k.shape == (1, Sk, 1, Di), "key shapes")
        _check(topk.dtype == torch.int32 and topk.shape[:2] == (1, Sq), "topk (1,Sq,K) int32")
        _check(topk_length.shape == (Sq,) and topk_length.dtype == torch.int32, "topk_length (Sq,) int32")
        # Rank-wide selection is sliced into SP blocks. A contiguous int32 slice can
        # start at an unaligned address; CuTe requires 16-byte alignment, and
        # contiguous() alone would leave that slice unchanged.
        if topk_length.data_ptr() % 16:
            topk_length = topk_length.clone()
        K = topk.shape[-1]
        ik = index_k.squeeze(2)
        valid = torch.arange(K, device=topk.device)[None, None, :] < topk_length[None, :, None]

        extra = max(0, ATTN_HEADS_FLASH[0] - Hq)
        o, _mx, lse = flash_mla_sparse_fwd(
            _pad_heads(q_attn, extra)[0], kv[0].unsqueeze(1), topk[0].unsqueeze(1),
            sm_scale=s_attn, d_v=d_v, topk_length=topk_length)
        # FlashMLA returns natural-log LSE. Only real heads contribute to the KL target.
        out, lse = o.unsqueeze(0), lse.unsqueeze(0)
        if extra:
            out, lse = out[:, :, :Hq].contiguous(), lse[:, :, :Hq].contiguous()
        ctx.extra = extra

        # KL: main-attention target vs indexer probability, both recomputed over the top-k by cuDNN
        tl = topk_length.unsqueeze(0).contiguous()
        lse_cudnn = _cudnn_bsh_layout(lse)
        weights_cudnn = _cudnn_bsh_layout(weights)
        _check(lse_cudnn.stride(1) == 1,
               f"cuDNN LSE sequence stride must be 1, got {lse_cudnn.stride()}")
        _check(weights_cudnn.stride(1) == 1,
               f"cuDNN indexer-weight sequence stride must be 1, got {weights_cudnn.stride()}")
        target = sparse_attn_score_recompute_wrapper(
            q_attn, kv, lse_cudnn, topk, softmax_scale=s_attn,
            qhead_per_kv_head=Hq, topk_length=tl)["target"]
        target = torch.where(valid, target, torch.zeros_like(target))
        index_prob = sparse_indexer_score_recompute_wrapper(
            (index_q * s_idx).contiguous(), ik, weights_cudnn, topk,
            qhead_per_kv_head=Hi, topk_length=tl)["predict"]
        # zero the padded slots explicitly
        index_prob = torch.where(valid, index_prob, torch.zeros_like(index_prob))
        if q_real_max is not None:
            # SFT right-padding: pad queries (global position >= q_real_max) must not add
            # KL terms or indexer gradients. Zeroing BOTH saved distributions covers
            # forward and backward (indexer_backward reads the saved tensors).
            rows_real = (torch.arange(Sq, device=target.device) + int(offset)) < q_real_max
            rows_real = rows_real[None, :, None]
            target = torch.where(rows_real, target, torch.zeros_like(target))
            index_prob = torch.where(rows_real, index_prob, torch.zeros_like(index_prob))
        kl_loss = (target * (target.clamp_min(1e-9).log() - index_prob.clamp_min(1e-9).log())).nan_to_num(0.0).sum()

        ctx.save_for_backward(index_q, index_k, weights, q_attn, kv, topk, topk_length, out, lse,
                              target, index_prob)
        ctx.s_attn, ctx.s_idx, ctx.offset = s_attn, s_idx, offset
        return out, kl_loss

    @staticmethod
    def backward(ctx, dout, d_kl):
        (index_q, index_k, weights, q_attn, kv, topk, topk_length, out, lse,
         target, index_prob) = ctx.saved_tensors
        _, Sq, Hq, _ = q_attn.shape
        s_attn, s_idx = ctx.s_attn, ctx.s_idx

        dq_attn = dkv = None
        if dout is not None:    # LM gradient (stage 2 only)
            sink = torch.full((Hq + ctx.extra,), NEG, device=kv.device, dtype=torch.float32)
            r = sparse_attention_backward_wrapper(
                _pad_heads(q_attn, ctx.extra)[0], kv[0],
                _pad_heads(out, ctx.extra)[0], _pad_heads(dout, ctx.extra)[0].contiguous(),
                _pad_heads(lse, ctx.extra)[0], sink,
                topk[0], softmax_scale=s_attn, topk_length=topk_length)
            dq_attn, dkv = r["dq"].unsqueeze(0), r["dkv"].unsqueeze(0)
            if ctx.extra:
                dq_attn = dq_attn[:, :, :Hq].contiguous()

        d_index_q = d_index_k = d_weights = None
        if d_kl is not None:    # KL gradient -> indexer only
            # Invalid slots contribute zero; spread their atomic adds over query rows
            # instead of serializing them all on key row 0. Keep d_kl on the device.
            rows = ctx.offset + torch.arange(Sq, device=topk.device, dtype=topk.dtype)
            spread = torch.where(topk >= 0, topk, rows[None, :, None].expand_as(topk)).contiguous()
            ib = indexer_backward_wrapper(
                index_q, weights, index_k.squeeze(2), target.clone().float(), index_prob.clone().float(),
                spread, sm_scale=s_idx, loss_coeff=float(Sq), grad_loss=d_kl)
            d_index_q, d_index_k, d_weights = ib["d_index_q"], ib["d_index_k"].unsqueeze(2), ib["d_weights"]
        return (d_index_q, d_index_k, d_weights, dq_attn, dkv, None, None, None, None, None, None, None)


def dsa_sparse_attention(index_q, index_k, weights, q_attn, kv, positions_q, cu_seqlens,
                         softmax_scale, index_scale, d_v, topk, q_max=None, offset=0,
                         selection=None, q_min=None, q_real_max=None):
    """Sparse attention and indexer KL with fused, document-aware top-k selection."""
    if selection is None:
        topk_ids, topk_len = select_topk(
            index_q, index_k, weights, index_scale, topk,
            positions_q=positions_q, cu_seqlens=cu_seqlens,
            q_max=q_max, q_min=q_min)
    else:
        topk_ids, topk_len = selection
    return _DSAFunction.apply(index_q, index_k, weights, q_attn, kv, topk_ids, topk_len,
                              softmax_scale, index_scale, d_v, offset, q_real_max)


class _DSAWarmupFunction(torch.autograd.Function):
    """Stage-1 (warm-up) DENSE KL over every causal key."""
    @staticmethod
    def forward(ctx, index_q, index_k, weights, q_attn, kv, lse, s_attn, s_idx, offset, compute_value,
                attention_mass_rows=None, indexer_topk_rows=None, attention_mass_topk_values=None,
                indexer_topk_values=None):
        B, Sq, Hi, Di = index_q.shape
        _, Sk, _, _ = index_k.shape
        _, _, Hq, Dqk = q_attn.shape
        ctx.set_materialize_grads(False)
        _check(B == 1, "operator is packed B=1 (per rank)")
        for t, name in [(index_q, "index_q"), (index_k, "index_k"), (weights, "weights"),
                        (q_attn, "q_attn"), (kv, "kv")]:
            _check(t.dtype == torch.bfloat16 and t.is_contiguous(), f"{name} must be contiguous bf16")
        _check(Hi in INDEX_HEADS_OK and Di == INDEX_DIM_OK, "indexer heads/dim unsupported")
        _check(Hq in ATTN_HEADS_OK, f"attention heads must be in {ATTN_HEADS_OK}, got {Hq}")
        _check(kv.shape == (1, Sk, Dqk) and index_k.shape == (1, Sk, 1, Di), "key shapes")
        _check(lse.shape == (1, Sq, Hq), f"lse must be (1,Sq,Hq), got {tuple(lse.shape)}")
        qco = torch.full((1,), int(offset), device=index_q.device, dtype=torch.int32)

        # dense recompute over all causal keys (heads reduced in-kernel): main-attention target prob
        # (out) + its l1 norm, and raw indexer logits (out, = s_idx * sum_h w*ReLU(q.k)) + lse.
        # NOTE: the wrapper allocates its [Sq,Sk] outputs under torch_stream_context; the image must
        # carry the patch (ExternalStream(0) phantom-stream fix).
        lse_cudnn = _cudnn_bsh_layout(lse)
        weights_cudnn = _cudnn_bsh_layout(weights)
        _check(lse_cudnn.stride(1) == 1,
               f"cuDNN LSE sequence stride must be 1, got {lse_cudnn.stride()}")
        _check(weights_cudnn.stride(1) == 1,
               f"cuDNN indexer-weight sequence stride must be 1, got {weights_cudnn.stride()}")
        a = dense_attn_score_recompute_wrapper(q_attn, kv.unsqueeze(2), lse_cudnn,
                                               softmax_scale=s_attn, qhead_per_kv_head=Hq, q_causal_offsets=qco)
        b = dense_indexer_score_recompute_wrapper(index_q, index_k, weights_cudnn,
                                                  qhead_per_kv_head=Hi, sm_scale=s_idx, q_causal_offsets=qco)
        attn_score, attn_l1norm = a["out"], a["denom"]
        index_score, index_lse = b["out"], b["denom"]

        # raw summed KL(target ‖ softmax(index logits)) for LOGGING only
        dev = index_q.device
        kl_loss = torch.zeros((), device=dev, dtype=torch.float32)
        if compute_value:
            qpos = torch.arange(Sq, device=dev) + int(offset)
            den = attn_l1norm[0].float().clamp_min(1e-9)[:, None]
            ilse = index_lse[0].float()[:, None]
            for k0 in range(0, Sk, WARMUP_KL_TILE):
                k1 = min(k0 + WARMUP_KL_TILE, Sk)
                m = torch.arange(k0, k1, device=dev)[None, :] <= qpos[:, None]
                t = torch.where(m, attn_score[0, :, k0:k1].float(), torch.zeros((), device=dev)) / den
                logp = index_score[0, :, k0:k1].float() - ilse
                kl_loss = kl_loss + (t * (t.clamp_min(1e-9).log() - logp)).nan_to_num(0.0).sum()

        monitor_enabled = attention_mass_rows is not None or indexer_topk_rows is not None
        attention_mass_stats = None
        indexer_topk_stats = None
        if attention_mass_rows is not None:
            attention_mass_stats = dsa_monitor.dense_attention_mass_stats(
                attn_score, attn_l1norm, index_score, attention_mass_topk_values, offset,
                attention_mass_rows)
        if indexer_topk_rows is not None:
            indexer_topk_stats = dsa_monitor.dense_indexer_topk_stats(
                index_score, indexer_topk_values, offset, indexer_topk_rows)

        ctx.save_for_backward(index_q, index_k, weights, attn_score, attn_l1norm, index_score, index_lse)
        ctx.s_idx, ctx.offset, ctx.Sq = s_idx, int(offset), Sq
        ctx.monitor_enabled = monitor_enabled
        if monitor_enabled:
            if attention_mass_stats is None:
                attention_mass_stats = torch.empty(0, device=dev, dtype=torch.float32)
            if indexer_topk_stats is None:
                indexer_topk_stats = torch.empty(0, device=dev, dtype=torch.float32)
            ctx.mark_non_differentiable(attention_mass_stats, indexer_topk_stats)
            return kl_loss, attention_mass_stats, indexer_topk_stats
        return kl_loss

    N_INPUTS = 14   # must match forward()'s signature; a fixed width keeps the two in sync

    @staticmethod
    def backward(ctx, d_kl, _d_attention_mass_stats=None, _d_indexer_topk_stats=None):
        if d_kl is None:
            return (None,) * _DSAWarmupFunction.N_INPUTS
        # the kernel below consumes the saved fp32 scores in-place
        if getattr(ctx, "_scores_consumed", False):
            raise RuntimeError(
                "DSA warmup backward invoked twice on one graph, but its saved scores were consumed "
                "in-place by the first run; retain_graph/double-backward is not supported here.")
        ctx._scores_consumed = True
        index_q, index_k, weights, attn_score, attn_l1norm, index_score, index_lse = ctx.saved_tensors
        qco = torch.full((1,), ctx.offset, device=index_q.device, dtype=torch.int32)
        # analytic KL grad -> indexer only. loss_coeff=Sq cancels the kernel's 1/(B*Sq) so the grad is
        # d_kl*(index_prob - target). attn_score/index_score (fp32) are consumed in-place.
        ib = dense_indexer_backward_wrapper(
            index_q, weights, index_k.squeeze(2), attn_score, attn_l1norm.clone(),
            index_score, index_lse.clone(),
            sm_scale=ctx.s_idx, loss_coeff=float(ctx.Sq), grad_loss=float(d_kl), q_causal_offsets=qco)
        _queue_dense_indexer_backward_cache_cleanup()
        grads = (ib["d_index_q"], ib["d_index_k"].unsqueeze(2), ib["d_weights"])
        return grads + (None,) * (_DSAWarmupFunction.N_INPUTS - len(grads))


def dsa_warmup_kl(index_q, index_k, weights, q_attn, kv, lse, softmax_scale, index_scale,
                  offset=0, cu_seqlens=None, compute_value=True, attention_mass_mask=None,
                  indexer_topk_mask=None, attention_mass_topk_values=None,
                  indexer_topk_values=None):
    """Stage-1 dense-KL warm-up for one rank's query-chunk."""
    end = offset + index_q.shape[1]
    monitor_enabled = attention_mass_mask is not None or indexer_topk_mask is not None
    if cu_seqlens is None:
        # clip keys to the causal horizon: key j > offset+Sq-1 is masked for every query in the chunk,
        # and the recompute retains 2 fp32 [Sq,Sk] scores for backward - clipping is what caps them.
        if not monitor_enabled:
            return _DSAWarmupFunction.apply(index_q, index_k[:, :end].contiguous(), weights,
                                            q_attn, kv[:, :end].contiguous(), lse,
                                            softmax_scale, index_scale, offset, compute_value)
        attention_mass_rows = (attention_mass_mask[offset:end].nonzero(as_tuple=True)[0]
                               if attention_mass_mask is not None else None)
        indexer_topk_rows = (indexer_topk_mask[offset:end].nonzero(as_tuple=True)[0]
                             if indexer_topk_mask is not None else None)
        return _DSAWarmupFunction.apply(index_q, index_k[:, :end].contiguous(), weights,
                                        q_attn, kv[:, :end].contiguous(), lse,
                                        softmax_scale, index_scale, offset, compute_value,
                                        attention_mass_rows, indexer_topk_rows,
                                        attention_mass_topk_values, indexer_topk_values)
    bounds = _cu_bounds(cu_seqlens)
    kl = None
    attention_mass_stats = (
        torch.zeros((len(attention_mass_topk_values), dsa_monitor.NUM_ATTENTION_MASS_STATS),
                    device=index_q.device, dtype=torch.float32)
        if attention_mass_mask is not None else None
    )
    indexer_topk_stats = (
        torch.zeros((len(indexer_topk_values), dsa_monitor.NUM_INDEXER_TOPK_STATS),
                    device=index_q.device, dtype=torch.float32)
        if indexer_topk_mask is not None else None
    )
    for d0, d1 in zip(bounds[:-1], bounds[1:]):
        qs, qe = max(d0, offset), min(d1, end)  # global query range on this rank
        if qs >= qe:
            continue
        lq = slice(qs - offset, qe - offset)    # chunk-local query slice
        if not monitor_enabled:
            seg = _DSAWarmupFunction.apply(
                index_q[:, lq].contiguous(), index_k[:, d0:qe].contiguous(), weights[:, lq].contiguous(),
                q_attn[:, lq].contiguous(), kv[:, d0:qe].contiguous(), lse[:, lq].contiguous(),
                softmax_scale, index_scale, qs - d0, compute_value)  # causal offset within the document
        else:
            attention_mass_rows = (attention_mass_mask[qs:qe].nonzero(as_tuple=True)[0]
                                   if attention_mass_mask is not None else None)
            indexer_topk_rows = (indexer_topk_mask[qs:qe].nonzero(as_tuple=True)[0]
                                 if indexer_topk_mask is not None else None)
            seg, seg_attention_mass_stats, seg_indexer_topk_stats = _DSAWarmupFunction.apply(
                index_q[:, lq].contiguous(), index_k[:, d0:qe].contiguous(), weights[:, lq].contiguous(),
                q_attn[:, lq].contiguous(), kv[:, d0:qe].contiguous(), lse[:, lq].contiguous(),
                softmax_scale, index_scale, qs - d0, compute_value, attention_mass_rows,
                indexer_topk_rows, attention_mass_topk_values, indexer_topk_values)
            if attention_mass_stats is not None:
                attention_mass_stats = attention_mass_stats + seg_attention_mass_stats
            if indexer_topk_stats is not None:
                indexer_topk_stats = indexer_topk_stats + seg_indexer_topk_stats
        kl = seg if kl is None else kl + seg
    if monitor_enabled:
        if attention_mass_stats is None:
            attention_mass_stats = torch.empty(0, device=index_q.device, dtype=torch.float32)
        if indexer_topk_stats is None:
            indexer_topk_stats = torch.empty(0, device=index_q.device, dtype=torch.float32)
        return kl, attention_mass_stats, indexer_topk_stats
    return kl
