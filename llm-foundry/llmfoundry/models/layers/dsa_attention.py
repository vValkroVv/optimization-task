import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from composer.utils import dist as cdist
from composer.utils.profiler_annotation import decorator_forward_backward
from flash_attn import flash_attn_func, flash_attn_varlen_func

from llmfoundry.models.layers.custom_embedding import apply_rotary_emb  # torch.compile-safe wrapper
from llmfoundry.models.layers.triton_rotary_embeddings import apply_rotary_emb_mla
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY
from llmfoundry.models.layers.attention import LlamaLatentAttention, _apply_sigmoid_gate_fp32
from llmfoundry.models.layers.moe.moe_layers import AddAuxiliaryLoss
from llmfoundry.models.utils.dsa_config import validate_dsa_config
from llmfoundry.models.layers import dsa_operator
from llmfoundry.models.layers import dsa_monitor


def freeze_main_for_warmup(model):
    """Stage-1 warm-up trains ONLY the DSA indexers. Returns (trainable, total) tensor counts."""
    n_train = n_total = 0
    for name, p in model.named_parameters():
        is_indexer = ".indexer." in f".{name}"   # module-relative names start with "indexer."
        p.requires_grad_(is_indexer)
        n_total += 1
        n_train += int(is_indexer)
    if n_train == 0:
        raise ValueError("freeze_main_for_warmup: the model has no DSA indexer parameters; "
                         "dsa_stage=1 needs attention_type='LlamaLatentSparseAttention'")
    return n_train, n_total


class _GatherKV(torch.autograd.Function):
    """All-gather keys over seq; reduce-scatter backward so key grads sum across ranks"""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        ctx.dtype = x.dtype
        parts = [torch.empty_like(x) for _ in range(group.size())]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.cat(parts, dim=1)

    @staticmethod
    def backward(ctx, g):
        group = ctx.group
        g = g.movedim(1, 0).contiguous().float() # (S, b, ...), fp32 for the cross-rank sum
        out = torch.empty((g.size(0) // group.size(), *g.shape[1:]), dtype=torch.float32, device=g.device)
        dist.reduce_scatter_tensor(out, g, group=group)
        return out.movedim(0, 1).to(ctx.dtype).contiguous(), None



class _AllToAll(torch.autograd.Function):
    """All-to-all over the SP group on the sequence dim (dim 1)"""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        xt = x.movedim(1, 0).contiguous()
        out = torch.empty_like(xt)
        dist.all_to_all_single(out, xt, group=group)
        return out.movedim(0, 1).contiguous()

    @staticmethod
    def backward(ctx, g):
        gt = g.movedim(1, 0).contiguous()
        out = torch.empty_like(gt)
        dist.all_to_all_single(out, gt, group=ctx.group)
        return out.movedim(0, 1).contiguous(), None


class DSAIndexer(nn.Module):
    """Lightning indexer: per-token scorer whose top-k selects the KV each query attends to.
    Projections consume DETACHED inputs (indexer grad comes only from the KL loss). The key is MQA
    (single head); index_head_dim is a qk_rope_head_dim rope prefix (non-interleaved) + a remainder.
    """
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.indexer_n_heads
        self.head_dim = config.indexer_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.topk = config.indexer_topk
        q_lora_rank = config.q_lora_rank if config.q_lora_rank else config.hidden_size
        dev = config.init_device
        self.wq = nn.Linear(q_lora_rank, self.n_heads * self.head_dim, bias=False, device=dev)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False, device=dev)
        self.k_norm = NORM_CLASS_REGISTRY[getattr(config, "indexer_norm_type", "LlamaRMSNorm")](
            self.head_dim, device=dev)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False, device=dev)
        self.scale = self.head_dim ** -0.5

    def project(self, hidden_states, q_latent):
        """Detached projections. Returns q (b,s,H_idx,head_dim), k (b,s,head_dim), weights (b,s,H_idx)."""
        q = rearrange(self.wq(q_latent.detach()), "b s (h d) -> b s h d", d=self.head_dim)
        k = self.k_norm(self.wk(hidden_states.detach()))
        weights = self.weights_proj(hidden_states.detach()) * self.n_heads ** -0.5
        return q, k, weights

    def apply_rope_q(self, q, cos, sin):
        """Non-interleaved rope on the rope-prefix of the index query (b,s,H,head_dim)."""
        qr, qn = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        return torch.cat([apply_rotary_emb(qr, cos, sin, interleaved=False), qn], dim=-1)

    def apply_rope_k(self, k, cos, sin):
        """Non-interleaved rope on the rope-prefix of the (MQA) index key (b,s,head_dim)."""
        kr, kn = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        kr = apply_rotary_emb(kr.unsqueeze(2), cos, sin, interleaved=False).squeeze(2)
        return torch.cat([kr, kn], dim=-1)



_SEL_OFFLOAD_STREAM = None


def _sel_offload_stream(device):
    """Copy stream for selection-cache offload, shared by all DSA layers."""
    global _SEL_OFFLOAD_STREAM
    if _SEL_OFFLOAD_STREAM is None:
        _SEL_OFFLOAD_STREAM = torch.cuda.Stream(device=device)
    return _SEL_OFFLOAD_STREAM


class LlamaLatentSparseAttention(LlamaLatentAttention):
    """MLA attention with DeepSeek Sparse Attention.
      * stage 1 (warm-up): dense attention output; the indexer learns a DENSE KL over all keys (not
        the top-k), use `freeze_main_for_warmup`.
      * stage 2 (sparse):  sparse top-k attention output; the top-k-restricted KL keeps training it.
    Gradients are decoupled: q_attn/kv <- LM loss only, index_* <- KL loss only.

    Stage-2 checkpoint replay holds one selection per layer. Each training forward
    must finish backward before this layer starts the next training forward;
    overlapping checkpointed forwards on the same layer are not supported.
    """
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        d_qk = self.kv_lora_rank + self.qk_rope_head_dim
        assert not self.apply_qk_norm, "DSA: apply_qk_norm is not supported"
        assert getattr(config, "rope_origin", "custom") != "deepseek", "DSA: 'deepseek' rope_origin not supported"
        assert self.num_krope_heads == 1, f"DSA: num_krope_heads must be 1, got {self.num_krope_heads}"
        assert self.tp_size == 1, "DSA: tensor parallel not supported yet (sequence parallel only)"
        assert self.num_heads in (32, 64, 128), \
            f"DSA: num_heads must be 32/64/128 (32 padded to 64 for FlashMLA), got {self.num_heads}"
        assert d_qk in (512, 576), f"DSA: kv_lora_rank+qk_rope_head_dim must be 512/576 (FlashMLA), got {d_qk}"
        assert self.num_heads != 32 or d_qk == 576, \
            f"DSA: num_heads=32 requires d_qk==576, got {d_qk}"
        # config_overrides are applied after GigaMixConfig.__init__ has run, so this is the
        # first point where an overridden value is checked at all
        validate_dsa_config(config)
        self.indexer = DSAIndexer(config)
        self.dsa_stage = config.dsa_stage
        self.indexer_topk = config.indexer_topk
        self.aux_kl_loss_koeff = config.aux_kl_loss_koeff
        self.index_scale = config.indexer_head_dim ** -0.5
        self.sp_rebalance = getattr(config, "dsa_sp_rebalance", True)  # block-cyclic SP load balancing
        self.block_checkpoint = getattr(config, "dsa_block_checkpoint", False)  # ckpt per-block saves (mem for recompute)
        self.sel_cache_offload = getattr(config, "dsa_sel_cache_offload", False)
        self._sel_cache = None   # this layer's selection for checkpoint replay
        self.loss = 0.0
        self._compute_attention_mass_value = False
        self._compute_indexer_topk_value = False
        self.attention_mass_monitor_stats = None
        self.attention_mass_monitor_num_queries = 128
        self.attention_mass_monitor_q_start_pos = 0
        self.attention_mass_monitor_topk_values = (self.indexer_topk,)
        self.indexer_topk_monitor_stats = None
        self.indexer_topk_monitor_num_queries = 128
        self.indexer_topk_monitor_q_start_pos = 0
        self.indexer_topk_monitor_topk_values = (self.indexer_topk,)

    def _sel_store(self, ids, lengths):
        """Cache selection on GPU or copy it to pinned CPU for checkpoint replay."""
        if not self.sel_cache_offload:
            self._sel_cache = (ids, lengths)
            return
        cached = self._sel_cache
        if cached is None or cached[0].shape != ids.shape:
            if cached is not None:
                cached[2].synchronize()  # the old buffers may still be copying
            ids_cpu = torch.empty(ids.shape, dtype=ids.dtype, pin_memory=True)
            lengths_cpu = torch.empty(lengths.shape, dtype=lengths.dtype, pin_memory=True)
        else:
            ids_cpu, lengths_cpu, _ = cached
        side = _sel_offload_stream(ids.device)
        ready = torch.cuda.Event()
        ready.record()
        side.wait_event(ready)
        with torch.cuda.stream(side):
            ids_cpu.copy_(ids, non_blocking=True)
            lengths_cpu.copy_(lengths, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
        ids.record_stream(side)  # keep the allocator from recycling them before the copy
        lengths.record_stream(side)
        self._sel_cache = (ids_cpu, lengths_cpu, done)

    def _sel_replay(self, device):
        """Return the selection stored by _sel_store (checkpoint recompute path)."""
        entry = self._sel_cache
        assert entry is not None, "no cached selection to replay"
        if not self.sel_cache_offload:
            return entry
        ids_cpu, len_cpu, done = entry
        done.synchronize()
        return ids_cpu.to(device, non_blocking=True), len_cpu.to(device, non_blocking=True)

    def _select_cached(self, iq, ik_op, iw, pos, cu, q_max, q_min):
        """One fused selection for this rank, replayed during checkpoint recompute."""
        from composer.trainer.torch_utils_checkpoint import is_recomputing
        if is_recomputing():
            return self._sel_replay(iq.device)
        ids, ln = dsa_operator.select_topk(
            iq, ik_op, iw, self.index_scale, self.indexer_topk, positions_q=pos, cu_seqlens=cu,
            q_max=q_max, q_min=q_min)
        if self.training and torch.is_grad_enabled():
            self._sel_store(ids, ln)
        return ids, ln

    def _dense(self, q_nope, q_rope, latent, k_rope, offset, cu):
        """Stage-1 dense MLA attention (warm-up), non-absorbed per-head (576-dim absorbed form exceeds flash-attns 256 head-dim cap)."""
        b, Sq = q_nope.shape[0], q_nope.shape[1]
        H, end = self.num_heads, offset + Sq
        q = torch.cat([q_nope, q_rope], dim=-1)
        # expand per-head keys only over the used range [key_lo, end)
        bounds = dsa_operator._cu_bounds(cu) if cu is not None else None
        key_lo = max(d for d in bounds if d <= offset) if bounds is not None else 0
        lat = latent[:, key_lo:end]
        kw = end - key_lo
        k_nope = self.uk_proj(lat).view(b, kw, H, self.qk_nope_head_dim)
        value = self.uv_proj(lat).view(b, kw, H, self.v_head_dim)
        k = torch.cat([k_nope, k_rope[:, key_lo:end].expand(-1, -1, H, -1)], dim=-1)  # MQA rope -> per-head
        value = F.pad(value, [0, self.qk_head_dim - self.v_head_dim])  # pad V to qk dim for flash-attn

        if cu is None:  # single doc: bottom-right causal
            out, lse, _ = flash_attn_func(q, k, value, causal=True, softmax_scale=self.softmax_scale,
                                          return_attn_probs=True)
            return out[..., : self.v_head_dim], lse.transpose(1, 2).contiguous()  # lse (1,H,Sq)->(1,Sq,H)

        # packed docs: one flash_attn_varlen over per-document (q_seg, k_seg)
        q_segs, k_segs, v_segs, cu_q, cu_k = [], [], [], [0], [0]
        for d0, d1 in zip(bounds[:-1], bounds[1:]):
            qs, qe = max(d0, offset), min(d1, end)
            if qs >= qe:
                continue
            q_segs.append(q[0, qs - offset:qe - offset])         # q is chunk-local
            k_segs.append(k[0, d0 - key_lo:qe - key_lo])         # k is [key_lo, end)-local
            v_segs.append(value[0, d0 - key_lo:qe - key_lo])
            cu_q.append(cu_q[-1] + (qe - qs))
            cu_k.append(cu_k[-1] + (qe - d0))
        dev = q.device
        out, lse, _ = flash_attn_varlen_func(
            torch.cat(q_segs), torch.cat(k_segs), torch.cat(v_segs),
            torch.tensor(cu_q, device=dev, dtype=torch.int32),
            torch.tensor(cu_k, device=dev, dtype=torch.int32),
            max(b - a for a, b in zip(cu_q[:-1], cu_q[1:])),
            max(b - a for a, b in zip(cu_k[:-1], cu_k[1:])),
            causal=True, softmax_scale=self.softmax_scale, return_attn_probs=True)
        # segments tile the chunk in local order -> packed lse (H, S_loc)
        lse = lse.transpose(0, 1).reshape(b, Sq, H).contiguous()
        return out[..., : self.v_head_dim].view(b, Sq, H, self.v_head_dim), lse

    @decorator_forward_backward()
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cu_seqlens=None, max_seqlen=None,
                logical_batch_size=None, is_left_padded_eval=False):
        assert not use_cache and past_key_value is None and not output_attentions
        assert not is_left_padded_eval, (
            "DSA does not support left-padded eval batches. Document boundaries and RoPE "
            "positions come from cu_seqlens/position_ids, which left padding shifts; MLA "
            "compensates inside rope, DSA cannot, so attention would be silently wrong. "
            "Use a right-padded flash_attn eval loader with device_eval_batch_size=1.")
        bsz, S_loc = self._get_hidden_states_shapes(hidden_states, logical_batch_size)
        assert bsz == 1, (
            f"DSA requires per-rank batch size 1 (packed sequences): got bsz={bsz}. Use a packing "
            f"dataloader or device_train_microbatch_size=1. A bsz>1 per-sample loop is not implemented.")

        if self.q_lora_rank == 0:
            qr = self.alpha_q * hidden_states   # indexer wq consumes hidden
            query = (self.alpha_q * self.q_proj(hidden_states)).view(bsz, S_loc, self.num_heads, self.qk_head_dim)
        else:
            qr = self.alpha_q * self.q_norm(self.dq_proj(hidden_states))       # query latent, shared with the indexer
            query = self.uq_proj(qr).view(bsz, S_loc, self.num_heads, self.qk_head_dim)
        k_rope = self.kr_proj(hidden_states).view(bsz, S_loc, 1, self.qk_rope_head_dim)
        latent = self.alpha_kv * self.kv_norm(self.dkv_proj(hidden_states))
        idx_q, idx_k, idx_w = self.indexer.project(hidden_states, qr)
        gate = self.gate_proj(hidden_states) if self.gated_attention else None # per-token sigmoid gate (b,s,H*v)
        # SP: this rank owns the contiguous query chunk [offset, offset+S_loc)
        sp_active = self.training and self.sp_size > 1 and not self.is_ring_attn
        offset = cdist.get_sp_group().rank() * S_loc if sp_active else 0
        dev = hidden_states.device
        # RoPE on the LOCAL chunk (global positions [offset, offset+S_loc))
        pos_local = (torch.arange(offset, offset + S_loc, device=dev).unsqueeze(0)
                     if position_ids is None else position_ids.reshape(1, -1)[:, offset:offset + S_loc])
        cu = cu_seqlens["cu_seqlens_q"] if isinstance(cu_seqlens, dict) else cu_seqlens
        cos, sin = self.rotary_emb(k_rope, seq_len=offset + S_loc, position_ids=pos_local, cu_seqlens=cu)
        if self.use_custom_rotary_kernel:
            query = apply_rotary_emb_mla(query, cos, sin, inplace=False, interleaved=True,
                                         head_offset=self.qk_nope_head_dim)
            q_nope, q_rope = torch.split(query, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        else:
            q_nope, q_rope = torch.split(query, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            q_rope = apply_rotary_emb(q_rope, cos, sin, interleaved=True)
        k_rope = apply_rotary_emb(k_rope, cos, sin, interleaved=True)
        idx_q = self.indexer.apply_rope_q(idx_q, cos, sin)  # non-interleaved
        idx_k = self.indexer.apply_rope_k(idx_k, cos, sin)
        if sp_active: # gather the ROTATED keys
            group = cdist.get_sp_group()
            latent = _GatherKV.apply(latent, group)
            k_rope = _GatherKV.apply(k_rope, group)
            idx_k = _GatherKV.apply(idx_k, group)
        # Block-cyclic rebalance: split this rank's contiguous chunk into sp sub-blocks and exchange
        rebalance = sp_active and self.sp_rebalance
        if rebalance:
            assert S_loc % self.sp_size == 0, \
                f"DSA block-cyclic rebalance needs S_loc ({S_loc}) divisible by sp_size ({self.sp_size})"
            blk, sp_rank = S_loc // self.sp_size, offset // S_loc
            q_nope = _AllToAll.apply(q_nope, group)
            q_rope = _AllToAll.apply(q_rope, group)
            idx_q = _AllToAll.apply(idx_q, group)
            idx_w = _AllToAll.apply(idx_w, group)
            blocks = [(s * S_loc + sp_rank * blk, blk) for s in range(self.sp_size)]
            pos_q = torch.cat([torch.arange(o, o + n, device=dev) for o, n in blocks])
        else:
            blocks = [(offset, S_loc)]
            pos_q = torch.arange(offset, offset + S_loc, device=dev)    # global packed idx for the op
        # absorb the (rebalanced) query into latent space; build the operator's q_attn / kv (MQA k_rope)
        uk_w = self.uk_proj.weight.view(self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        q_absorbed = torch.einsum("bshn,hnl->bshl", q_nope, uk_w)
        q_attn = torch.cat([q_absorbed, q_rope], dim=-1).contiguous()
        kv = torch.cat([latent, k_rope[:, :, 0, :]], dim=-1).contiguous()
        idx_q, idx_w = idx_q.contiguous(), idx_w.contiguous()
        idx_k_op = idx_k[:, :, None, :].contiguous()
        compute_value = getattr(self, "_compute_kl_value", True)
        from composer.trainer.torch_utils_checkpoint import is_recomputing

        attention_mass_on = (
            self._compute_attention_mass_value
            and self.dsa_stage == 1
            and self.training
            and not is_recomputing()
        )
        indexer_topk_on = (
            self._compute_indexer_topk_value and self.training and not is_recomputing()
        )
        if attention_mass_on:
            attention_mass_mask, attention_eligible_q_count, attention_total_q_count = (
                dsa_monitor.build_query_sample(
                    kv.shape[1],
                    cu,
                    self.attention_mass_monitor_q_start_pos,
                    self.attention_mass_monitor_num_queries,
                    dev,
                )
            )
            attention_mass_stats = torch.zeros(
                (
                    len(self.attention_mass_monitor_topk_values),
                    dsa_monitor.NUM_ATTENTION_MASS_STATS,
                ),
                device=dev,
                dtype=torch.float32,
            )
        else:
            attention_mass_mask = attention_mass_stats = None
            attention_eligible_q_count = attention_total_q_count = 0
        if indexer_topk_on:
            same_sample = (
                attention_mass_on
                and self.indexer_topk_monitor_q_start_pos
                == self.attention_mass_monitor_q_start_pos
                and self.indexer_topk_monitor_num_queries
                == self.attention_mass_monitor_num_queries
            )
            if same_sample:
                indexer_topk_mask = attention_mass_mask
                indexer_eligible_q_count = attention_eligible_q_count
                indexer_total_q_count = attention_total_q_count
            else:
                indexer_topk_mask, indexer_eligible_q_count, indexer_total_q_count = (
                    dsa_monitor.build_query_sample(
                        kv.shape[1],
                        cu,
                        self.indexer_topk_monitor_q_start_pos,
                        self.indexer_topk_monitor_num_queries,
                        dev,
                    )
                )
            indexer_topk_stats = torch.zeros(
                (
                    len(self.indexer_topk_monitor_topk_values),
                    dsa_monitor.NUM_INDEXER_TOPK_STATS,
                ),
                device=dev,
                dtype=torch.float32,
            )
        else:
            indexer_topk_mask = indexer_topk_stats = None
            indexer_eligible_q_count = indexer_total_q_count = 0
        # SFT right-padding: global count of real (non-pad) queries, attached to the
        # cu tensor by the model
        q_real_max = getattr(cu, "_dsa_q_real_max", None) if cu is not None else None
        # per-block DSA: each block is a contiguous global range [blk_off, blk_off+len)
        outs, kl_loss, loc = [], None, 0
        rank_sel = None
        if self.dsa_stage == 2:
            # Select once per rank; each attention block takes a view below.
            rank_sel = self._select_cached(
                idx_q, idx_k_op, idx_w, pos_q, cu,
                q_max=blocks[-1][0] + blocks[-1][1] - 1, q_min=blocks[0][0])
        for blk_off, blk_len in blocks:
            sl = slice(loc, loc + blk_len)
            loc += blk_len
            if self.dsa_stage == 2:  # sparse output + top-k KL
                sel = (rank_sel[0][:, sl], rank_sel[1][sl])
                out_b, kl_b = dsa_operator.dsa_sparse_attention(
                    idx_q[:, sl],
                    idx_k_op,
                    idx_w[:, sl],
                    q_attn[:, sl],
                    kv,
                    positions_q=None,
                    cu_seqlens=cu,
                    softmax_scale=self.softmax_scale,
                    index_scale=self.index_scale,
                    d_v=self.kv_lora_rank,
                    topk=self.indexer_topk,
                    q_max=blk_off + blk_len - 1,
                    offset=blk_off,
                    selection=sel,
                    q_real_max=q_real_max,
                )
                outs.append(out_b)
                if indexer_topk_on:
                    block_positions = pos_q[sl]
                    indexer_topk_rows = indexer_topk_mask.index_select(
                        0,
                        block_positions.long(),
                    ).nonzero(as_tuple=True)[0]
                    indexer_topk_stats += dsa_monitor.sparse_indexer_topk_stats(
                        sel[0],
                        sel[1],
                        block_positions,
                        indexer_topk_rows,
                        self.indexer_topk_monitor_topk_values,
                        index_q=idx_q[:, sl],
                        index_k=idx_k_op,
                        weights=idx_w[:, sl],
                        index_scale=self.index_scale,
                    )
            else:  # stage 1 warm-up: dense output + cuDNN dense KL

                def _block(qn, qr, iq, iw, qa, lat, kr, kv_, ik, blk_off=blk_off):
                    attn_b, lse_b = self._dense(qn, qr, lat, kr, blk_off, cu)
                    monitor_kwargs = {}
                    if attention_mass_on or indexer_topk_on:
                        active_attention_mass_mask = (
                            attention_mass_mask
                            if attention_mass_on and not is_recomputing()
                            else attention_mass_mask[:0]
                            if attention_mass_on
                            else None
                        )
                        active_indexer_topk_mask = (
                            indexer_topk_mask
                            if indexer_topk_on and not is_recomputing()
                            else indexer_topk_mask[:0]
                            if indexer_topk_on
                            else None
                        )
                        monitor_kwargs = {
                            "attention_mass_mask": active_attention_mass_mask,
                            "indexer_topk_mask": active_indexer_topk_mask,
                            "attention_mass_topk_values": self.attention_mass_monitor_topk_values,
                            "indexer_topk_values": self.indexer_topk_monitor_topk_values,
                        }
                    kl_result = dsa_operator.dsa_warmup_kl(
                        iq,
                        ik,
                        iw,
                        qa,
                        kv_,
                        lse_b,
                        softmax_scale=self.softmax_scale,
                        index_scale=self.index_scale,
                        offset=blk_off,
                        cu_seqlens=cu,
                        compute_value=compute_value,
                        **monitor_kwargs,
                    )
                    if attention_mass_on or indexer_topk_on:
                        kl_b, block_attention_mass_stats, block_indexer_topk_stats = (
                            kl_result
                        )
                        return (
                            attn_b,
                            kl_b,
                            block_attention_mass_stats,
                            block_indexer_topk_stats,
                        )
                    return attn_b, kl_result

                args = (
                    q_nope[:, sl],
                    q_rope[:, sl],
                    idx_q[:, sl],
                    idx_w[:, sl],
                    q_attn[:, sl],
                    latent,
                    k_rope,
                    kv,
                    idx_k_op,
                )
                if self.block_checkpoint:
                    # stage 1 retains 2 fp32 [Sq_blk, end] scores + flashs expanded
                    block_result = torch_checkpoint(_block, *args, use_reentrant=False)
                else:
                    block_result = _block(*args)
                if attention_mass_on or indexer_topk_on:
                    (
                        attn_b,
                        kl_b,
                        block_attention_mass_stats,
                        block_indexer_topk_stats,
                    ) = block_result
                    if attention_mass_on:
                        attention_mass_stats += block_attention_mass_stats
                    if indexer_topk_on:
                        indexer_topk_stats += block_indexer_topk_stats
                else:
                    attn_b, kl_b = block_result
                outs.append(attn_b)
            kl_loss = (
                kl_b if kl_loss is None else kl_loss + kl_b
            )  # raw sum over the rank's S_loc queries
        attn = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        if self.dsa_stage == 2:
            uv_w = self.uv_proj.weight.view(
                self.num_heads, self.v_head_dim, self.kv_lora_rank
            )
            attn = torch.einsum("bshl,hvl->bshv", attn, uv_w)
        if q_real_max is not None:
            # normalize the KL by the number of real queries this rank summed over
            kl_q_count = sum(
                torch.clamp(q_real_max - o, 0, n) for o, n in blocks
            ).clamp_min(1)
        else:
            kl_q_count = S_loc
        if self.dsa_stage == 2 or compute_value:
            self.loss = kl_loss.detach() / kl_q_count
        if attention_mass_on or indexer_topk_on:
            counts_owner = not sp_active or group.rank() == 0
            # accumulate over the microbatches of a step; DSAMonitor resets to None at batch_start
            if attention_mass_on:
                if counts_owner:
                    attention_mass_stats[
                        :, dsa_monitor.ATTENTION_MASS_ELIGIBLE_Q_COUNT
                    ] = attention_eligible_q_count
                    attention_mass_stats[
                        :, dsa_monitor.ATTENTION_MASS_TOTAL_Q_COUNT
                    ] = attention_total_q_count
                prev = self.attention_mass_monitor_stats
                stats = attention_mass_stats.detach()
                self.attention_mass_monitor_stats = stats if prev is None else prev + stats
            if indexer_topk_on:
                if counts_owner:
                    indexer_topk_stats[:, dsa_monitor.INDEXER_TOPK_ELIGIBLE_Q_COUNT] = (
                        indexer_eligible_q_count
                    )
                    indexer_topk_stats[:, dsa_monitor.INDEXER_TOPK_TOTAL_Q_COUNT] = (
                        indexer_total_q_count
                    )
                prev = self.indexer_topk_monitor_stats
                stats = indexer_topk_stats.detach()
                self.indexer_topk_monitor_stats = stats if prev is None else prev + stats
        attn = AddAuxiliaryLoss.apply(attn, kl_loss * (self.aux_kl_loss_koeff / kl_q_count))
        attn = rearrange(attn, "b s h d -> b s (h d)")
        if self.gated_attention:
            # restore contiguous order first, then gate + o_proj
            if rebalance:
                attn = _AllToAll.apply(attn, group)
            attn = self.o_proj(_apply_sigmoid_gate_fp32(attn, gate))
        else:
            attn = self.o_proj(attn)
            if rebalance: # un-rebalance the narrower o_proj output
                attn = _AllToAll.apply(attn, group)
        return attn, None, past_key_value
