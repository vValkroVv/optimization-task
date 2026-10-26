#!/usr/bin/env python
# coding: utf-8
"""
Reverse converter: HuggingFace DeepSeek-V3 checkpoint → local training format.

Handles:
  - FP8 (e4m3) dequantization back to bfloat16 (block-wise, any scale format incl. ue8m0)
  - Weight renaming: HF naming → local training naming
  - Split fused KV projections (kv_a_proj_with_mqa → dkv + kr, kv_b_proj → uk + uv)
  - Merge per-expert MoE weights → fused block_sparse_moe weight1/weight2
  - Two MoE layouts: ggemm (2D), sonicmoe (2D)
  - Multi-node streaming conversion compatible with convertation_manager.py pipeline

Standalone usage:
  python mapper_hf_to_deepseek_moe.py \\
      --input /path/to/DeepSeek-V3.1-Base \\
      --output /path/to/local_checkpoint \\
      --moe-layout ggemm \\
      --global-rank 0 --world-size 4
"""
import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import safetensors
from safetensors.torch import save_file as safe_save_file
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# 1. FP8 block-wise dequantization
# ═══════════════════════════════════════════════════════════════════════════

def dequantize_block_fp8(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: Tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """
    Dequantize a block-wise FP8 tensor back to bfloat16.

    weight_bf16[block] = weight_fp8[block].to(float32) * scale_inv[block_idx]

    Works with any scale dtype (float32, ue8m0/float8_e8m0, etc.) —
    we cast scale_inv to float32 before the multiply.
    """
    bm, bn = block_size
    rows, cols = weight_fp8.shape[-2], weight_fp8.shape[-1]

    pad_rows = (bm - rows % bm) % bm
    pad_cols = (bn - cols % bn) % bn

    if pad_rows or pad_cols:
        weight_fp8 = torch.nn.functional.pad(weight_fp8, (0, pad_cols, 0, pad_rows))

    padded_rows = weight_fp8.shape[-2]
    padded_cols = weight_fp8.shape[-1]
    orig_shape = weight_fp8.shape

    w = weight_fp8.reshape(
        -1,
        padded_rows // bm, bm,
        padded_cols // bn, bn,
    ).permute(0, 1, 3, 2, 4).to(torch.float32)

    s = scale_inv.to(torch.float32)
    s = s.reshape(w.shape[0], w.shape[1], w.shape[2], 1, 1)

    w = w * s

    w = w.permute(0, 1, 3, 2, 4).reshape(orig_shape)

    if pad_rows or pad_cols:
        slices = [slice(None)] * w.dim()
        slices[-2] = slice(0, rows)
        slices[-1] = slice(0, cols)
        w = w[tuple(slices)]

    return w.to(torch.bfloat16)


def dequantize_tensors(
    tensors: Dict[str, torch.Tensor],
    block_size: Tuple[int, int] = (128, 128),
) -> Dict[str, torch.Tensor]:
    """
    Dequantize every FP8 weight that has a matching weight_scale_inv.
    Drop all scale_inv keys.  Non-FP8 tensors cast to bfloat16.
    """
    result: Dict[str, torch.Tensor] = {}
    scale_keys: Set[str] = {k for k in tensors if k.endswith(".weight_scale_inv")}

    for key, tensor in tensors.items():
        if key in scale_keys:
            continue

        scale_key = (
            key.rsplit(".", 1)[0] + ".weight_scale_inv"
            if key.endswith(".weight")
            else None
        )

        if scale_key and scale_key in scale_keys:
            result[key] = dequantize_block_fp8(tensor, tensors[scale_key], block_size)
        else:
            result[key] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 2a. Split fused KV projections
# ═══════════════════════════════════════════════════════════════════════════

def split_kv_a_proj_with_mqa(
    fused_w: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    kv_a_proj_with_mqa = cat([dkv_proj, kr_proj], dim=0).
    Returns (dkv_proj, kr_proj).
    """
    assert fused_w.shape[0] == kv_lora_rank + qk_rope_head_dim
    dkv = fused_w[:kv_lora_rank, :].contiguous()
    kr = fused_w[kv_lora_rank:, :].contiguous()
    return dkv, kr


def split_kv_b_proj(
    fused_w: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    kv_b_proj = interleaved [uk; uv] per head.
    Returns (uk_proj, uv_proj).
    """
    r_kv = fused_w.shape[1]
    head_dim_total = qk_nope_head_dim + v_head_dim
    assert fused_w.shape[0] == num_heads * head_dim_total

    w = fused_w.view(num_heads, head_dim_total, r_kv)
    uk = w[:, :qk_nope_head_dim, :].reshape(num_heads * qk_nope_head_dim, r_kv).contiguous()
    uv = w[:, qk_nope_head_dim:, :].reshape(num_heads * v_head_dim, r_kv).contiguous()
    return uk, uv


# ═══════════════════════════════════════════════════════════════════════════
# 2b. Linear / hybrid attention (GigaChat 3.5) — reverse of the forward
#     transforms in transformers_utils/deepseek.py.
#
# Head layout (Qwen3NextGatedDeltaNet & KimiDeltaAttention):
#   n_h = linear_num_key_heads,  n_v = linear_num_value_heads,  g = n_v // n_h
#   d_k = linear_key_head_dim,   d_v = linear_value_head_dim
#   key_dim = n_h * d_k,         value_dim = n_v * d_v = n_h * g * d_v
#
# The training-time "interleaved per-head" layout packs, per key-head, the
# slice [q(d_k), k(d_k), v(g*d_v)] contiguously; HF "conv order" is the flat
# concatenation [q_all(key_dim), k_all(key_dim), v_all(value_dim)].
# ═══════════════════════════════════════════════════════════════════════════

def _interleave_qkv_per_head(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_h: int,
    d_k: int,
    g: int,
    d_v: int,
) -> torch.Tensor:
    """Pack conv-order q/k/v back into the training-time interleaved per-head
    layout `[h0: q,k,v; h1: q,k,v; ...]`. Works for 2D projections (tail=(H,))
    and 3D conv weights (tail=(1,K)). Inverse of the forward split that slices a
    per-head block into q/k/v."""
    tail = q.shape[1:]
    q_h = q.reshape(n_h, d_k, *tail)
    k_h = k.reshape(n_h, d_k, *tail)
    v_h = v.reshape(n_h, g * d_v, *tail)
    per_head = 2 * d_k + g * d_v
    return torch.cat([q_h, k_h, v_h], dim=1).reshape(n_h * per_head, *tail).contiguous()


def split_in_proj_qkvz(
    fused: torch.Tensor,
    n_h: int,
    n_v: int,
    d_k: int,
    d_v: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `_maybe_accumulate_and_return_qkvz` (deepseek.py).

    Forward packs, per key-head, `[q(d_k), k(d_k), v(g*d_v), z(g*d_v)]` into
    `in_proj_qkvz`.  Split it back into the local `qkv_proj` (q,k,v) and
    `z_proj` (z) projections (non-legacy layout).
    """
    assert n_v % n_h == 0, f"n_v={n_v} must be divisible by n_h={n_h}"
    g = n_v // n_h
    per_head_qkv = 2 * d_k + g * d_v
    per_head_z = g * d_v
    per_head_tot = per_head_qkv + per_head_z
    H = fused.shape[1]
    assert fused.shape[0] == n_h * per_head_tot, (
        f"[in_proj_qkvz] shape {tuple(fused.shape)} != {(n_h * per_head_tot, H)}"
    )
    ph = fused.view(n_h, per_head_tot, H)
    qkv = ph[:, :per_head_qkv, :].reshape(n_h * per_head_qkv, H).contiguous()
    z = ph[:, per_head_qkv:, :].reshape(n_h * per_head_z, H).contiguous()
    return qkv, z


def conv_order_to_interleaved_qkv_conv(
    conv: torch.Tensor,
    n_h: int,
    n_v: int,
    d_k: int,
    d_v: int,
) -> torch.Tensor:
    """Inverse of `_interleaved_qkv_to_conv_order` (deepseek.py).

    `conv` is HF conv order `[q_all(key_dim); k_all(key_dim); v_all(value_dim)]`
    with shape `(2*key_dim + value_dim, *tail)`.  Re-interleave per head into the
    local fused `qkv_conv1d` layout (non-legacy).  Handles `.weight` (tail=(1,K))
    and `.bias` (tail=()).
    """
    assert n_v % n_h == 0, f"n_v={n_v} must be divisible by n_h={n_h}"
    g = n_v // n_h
    key_dim = n_h * d_k
    value_dim = n_v * d_v
    assert conv.shape[0] == 2 * key_dim + value_dim, (
        f"[conv1d] channel dim {conv.shape[0]} != 2*key_dim+value_dim="
        f"{2 * key_dim + value_dim}"
    )
    q = conv[:key_dim]
    k = conv[key_dim:2 * key_dim]
    v = conv[2 * key_dim:]
    return _interleave_qkv_per_head(q, k, v, n_h, d_k, g, d_v)


def split_conv_order_to_separate(
    conv: torch.Tensor,
    n_h: int,
    n_v: int,
    d_k: int,
    d_v: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of `DeepseekV3Mapper.pack_conv_weights` (base_mappers.py), which
    did `cat([q, k, v], dim=0)`.  Split HF conv order back into the separate
    local `q_conv1d`/`k_conv1d`/`v_conv1d` weights (legacy layout)."""
    key_dim = n_h * d_k
    value_dim = n_v * d_v
    assert conv.shape[0] == 2 * key_dim + value_dim, (
        f"[conv1d] channel dim {conv.shape[0]} != 2*key_dim+value_dim="
        f"{2 * key_dim + value_dim}"
    )
    q = conv[:key_dim].contiguous()
    k = conv[key_dim:2 * key_dim].contiguous()
    v = conv[2 * key_dim:].contiguous()
    return q, k, v


def merge_kimi_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_h: int,
    n_v: int,
    d_k: int,
    d_v: int,
) -> torch.Tensor:
    """Inverse of `_interleaved_qkv_to_kimi_order` (deepseek.py).

    Merge KimiDeltaAttention's separate q/k/v (projections or convs) back into a
    single interleaved-per-head tensor (`qkv_proj` / `qkv_conv1d`).  Requires
    `d_k == d_v` (KDA constraint).
    """
    assert d_k == d_v, "KimiDeltaAttention expects linear_key_head_dim == linear_value_head_dim"
    assert n_v % n_h == 0, f"n_v={n_v} must be divisible by n_h={n_h}"
    g = n_v // n_h
    return _interleave_qkv_per_head(q, k, v, n_h, d_k, g, d_v)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Merge per-expert → fused MoE weights  (3 layouts)
# ═══════════════════════════════════════════════════════════════════════════
#
# HF per-expert:  gate_proj (I,H), up_proj (I,H), down_proj (H,I)
#
# ┌────────────────┬──────────────────────────┬─────────────────────────┐
# │  Layout        │  weight1                 │  weight2                │
# ├────────────────┼──────────────────────────┼─────────────────────────┤
# │  ggemm         │  (H, 2*I*E)  2D          │  (I*E, H)   2D          │
# │  sonicmoe      │  (H, 2*I*E)  2D interl.  │  (I*E, H)   2D sonic    │
# └────────────────┴──────────────────────────┴─────────────────────────┘

def merge_experts_to_weight1(
    experts_gate: List[torch.Tensor],
    experts_up: List[torch.Tensor],
    layout: str = "ggemm",
) -> torch.Tensor:
    E = len(experts_gate)
    I, H = experts_gate[0].shape

    chunks = []
    for i in range(E):
        gate_t = experts_gate[i].t().contiguous()
        up_t = experts_up[i].t().contiguous()
        chunks.append(torch.cat([gate_t, up_t], dim=-1))
    w1_g = torch.stack(chunks, dim=0)  # (E, H, 2*I)

    if layout == "ggemm":
        return w1_g.reshape(H, 2 * I * E).contiguous()

    if layout == "sonicmoe":
        gate = w1_g[:, :, :I]
        up = w1_g[:, :, I:]
        interleaved = torch.stack([gate, up], dim=3).reshape(E, H, 2 * I)
        sonic = interleaved.permute(0, 2, 1).contiguous()
        return sonic.reshape(H, 2 * I * E).contiguous()

    raise ValueError(f"Unknown layout: {layout!r}. Supported: 'ggemm', 'sonicmoe'")


def merge_experts_to_weight2(
    experts_down: List[torch.Tensor],
    layout: str = "ggemm",
) -> torch.Tensor:
    E = len(experts_down)
    H, I = experts_down[0].shape

    if layout == "ggemm":
        chunks = [experts_down[i].t().contiguous() for i in range(E)]
        return torch.cat(chunks, dim=0).contiguous()

    if layout == "sonicmoe":
        chunks = [experts_down[i].t().contiguous() for i in range(E)]
        w2_g = torch.stack(chunks, dim=0)
        return w2_g.permute(0, 2, 1).contiguous().reshape(I * E, H)

    raise ValueError(f"Unknown layout: {layout!r}. Supported: 'ggemm', 'sonicmoe'")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Per-chunk reverse mapping  (HF → local)
# ═══════════════════════════════════════════════════════════════════════════

class HFToLocalPerLayerMapper:
    """
    Processes a chunk of HF-named tensors (already dequantized to bf16) for
    a single layer or global tensors, and returns them in local training naming.

    Designed to be called per-chunk in a streaming multi-node pipeline, analogous
    to _DeepseekPerLayerTransformer but in reverse direction.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        moe_layout: str = "ggemm",
        mtp_block_type: str = "moe",
    ):
        self.n_experts = config.get("n_routed_experts", 0)
        self.kv_lora_rank = config.get("kv_lora_rank", 512)
        self.qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
        self.qk_nope_head_dim = config.get("qk_nope_head_dim", 128)
        self.v_head_dim = config.get("v_head_dim", 128)
        self.num_heads = config.get("num_attention_heads", 128)
        self.hidden_size = config.get("hidden_size", 7168)
        self.first_k_dense_replace = config.get("first_k_dense_replace", 1)
        self.num_hidden_layers = config.get("num_hidden_layers", 61)
        self.num_mtp = config.get("num_nextn_predict_layers", 0)
        self.moe_layout = moe_layout
        if mtp_block_type not in ("moe", "dense"):
            raise ValueError(
                f"Unsupported mtp_block_type={mtp_block_type!r}; expected 'moe' or 'dense'."
            )
        self.mtp_block_type = mtp_block_type

        # ── GigaChat 3.5 hybrid linear-attention layout ──
        # `linear_attention_type` selects the reverse path for non-full-softmax
        # layers; `full_attention_layers` lists the (MLA) layers that stay
        # full-softmax. When linear_attention_type is None the model is pure MLA
        # (DeepSeek 3.0 or full-softmax gigachat3_5) and no layer is linear.
        self.linear_attention_type = config.get("linear_attention_type", None)
        _full = config.get("full_attention_layers", None)
        self.full_attention_layers = set(_full) if _full else set()
        self.lin_num_k_heads = config.get("linear_num_key_heads", 0)
        self.lin_num_v_heads = config.get("linear_num_value_heads", 0)
        self.lin_head_k_dim = config.get("linear_key_head_dim", 0)
        self.lin_head_v_dim = config.get("linear_value_head_dim", 0)
        # HF `conv1d`/`in_proj_qkvz` layout vs. legacy separate q/k/v conv +
        # fused qkvz; must match training (see attention.py Qwen3NextGatedDeltaNet).
        self.linear_use_legacy_qkvz_layout = config.get(
            "linear_use_legacy_qkvz_layout", False
        )
        # Accumulator for KimiDeltaAttention: HF stores q/k/v as separate tensors
        # that must be merged into a single interleaved local tensor. Keyed by
        # (layer_idx, local_prefix, kind) so layers never cross-contaminate.
        self._kda_buf: Dict[Tuple[int, str, str], Dict[str, torch.Tensor]] = {}

    def _is_linear_layer(self, layer_idx: int) -> bool:
        """True if this layer uses hybrid linear attention (not full-softmax MLA)."""
        return (
            self.linear_attention_type is not None
            and layer_idx not in self.full_attention_layers
        )

    def map_chunk(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Map a chunk of HF tensors (one or more layers) to local format.
        Expert weights within the same layer are collected and merged automatically.
        """
        out: Dict[str, torch.Tensor] = {}
        expert_buf: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

        for key, tensor in tensors.items():
            if "weight_scale_inv" in key:
                continue

            # Global tensors (embed, lm_head, norm).
            if key in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
                if key == "lm_head.weight":
                    out["model.lm_head.weight"] = tensor
                else:
                    out[key] = tensor
                continue

            layer_idx = self._get_layer_idx(key)
            if layer_idx is None:
                out[key] = tensor
                continue

            # ── MTP layers (layer_idx >= num_hidden_layers) ──
            if self.num_mtp > 0 and layer_idx >= self.num_hidden_layers:
                mtp_mapped = self._map_mtp_key(key, tensor, layer_idx)
                if mtp_mapped is not None:
                    for mk, mv in mtp_mapped:
                        out[mk] = mv
                continue

            prefix = f"model.layers.{layer_idx}."
            suffix = key[len(prefix):]

            # Expert weights → buffer for merge
            expert_match = re.match(
                r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", suffix
            )
            if expert_match:
                ei = int(expert_match.group(1))
                proj = expert_match.group(2)
                expert_buf.setdefault(layer_idx, {}).setdefault(ei, {})[proj] = tensor
                continue

            if self._is_linear_layer(layer_idx):
                lin_mapped = self._map_linear_attention_key(suffix, tensor, layer_idx)
                if lin_mapped is not None:
                    for local_suffix, local_tensor in lin_mapped:
                        out[prefix + local_suffix] = local_tensor
                    continue

            mapped = self._map_attention_key(suffix, tensor)
            if mapped is not None:
                for local_suffix, local_tensor in mapped:
                    out[prefix + local_suffix] = local_tensor
                continue

            if suffix.startswith("input_layernorm."):
                rest = suffix[len("input_layernorm."):]
                out[prefix + "self_attn.input_layernorm." + rest] = tensor
                continue
            if suffix.startswith("post_attention_layernorm."):
                rest = suffix[len("post_attention_layernorm."):]
                out[prefix + "self_attn.post_attention_layernorm." + rest] = tensor
                continue

            if suffix.startswith("post_self_attn_layernorm."):
                out[prefix + "self_attn." + suffix] = tensor
                continue
            if suffix.startswith("post_feedforward_layernorm."):
                if layer_idx >= self.first_k_dense_replace:
                    out[prefix + suffix] = tensor
                else:
                    out[prefix + "self_attn." + suffix] = tensor
                continue

            if suffix == "mlp.shared_expert_sigmoid.weight":
                out[prefix + "self_attn.shared_expert_sigmoid.weight"] = tensor
                continue

            # Dense MLP
            if suffix.startswith("mlp.") \
                    and not suffix.startswith("mlp.experts.") \
                    and not suffix.startswith("mlp.shared_experts.") \
                    and not suffix.startswith("mlp.gate."):
                out[prefix + suffix.replace("mlp.", "self_attn.mlp.", 1)] = tensor
                continue

            # Shared experts
            if suffix.startswith("mlp.shared_experts."):
                out[prefix + suffix.replace("mlp.shared_experts.", "self_attn.shared_experts.", 1)] = tensor
                continue

            # Gate
            if suffix == "mlp.gate.weight":
                out[prefix + "self_attn._gate.gate.weight"] = tensor
                continue
            if suffix == "mlp.gate.e_score_correction_bias":
                out[prefix + "self_attn._gate.expert_bias"] = tensor
                continue

            raise RuntimeError(
                f"Unmapped key: {key}\n"
                f"This key does not match any known DeepSeek-V3 weight pattern.\n"
                f"If this is a hybrid/linear-attention model, add reverse mappings "
                f"for it in HFToLocalPerLayerMapper._map_mtp_key() or map_chunk()."
            )

        # Merge experts (regular layers)
        for layer_idx in sorted(expert_buf.keys()):
            experts = expert_buf[layer_idx]
            prefix = f"model.layers.{layer_idx}."
            n_exp = max(experts.keys()) + 1
            assert n_exp == self.n_experts, (
                f"Layer {layer_idx}: found {n_exp} experts, expected {self.n_experts}"
            )

            gate_list = [experts[i]["gate_proj"] for i in range(n_exp)]
            up_list = [experts[i]["up_proj"] for i in range(n_exp)]
            down_list = [experts[i]["down_proj"] for i in range(n_exp)]

            w1 = merge_experts_to_weight1(gate_list, up_list, layout=self.moe_layout)
            w2 = merge_experts_to_weight2(down_list, layout=self.moe_layout)

            out[prefix + "block_sparse_moe.experts.weight1"] = w1
            out[prefix + "block_sparse_moe.experts.weight2"] = w2

        # Merge MTP expert weights (stored with __mtp_expert__ prefix)
        mtp_expert_buf: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        mtp_keys_to_remove = []
        for k, v in out.items():
            if k.startswith("__mtp_expert__"):
                # __mtp_expert__{layer_idx}__{expert_idx}__{proj}
                parts = k.split("__")
                lid = int(parts[2])
                eid = int(parts[3])
                proj = parts[4]
                mtp_expert_buf.setdefault(lid, {}).setdefault(eid, {})[proj] = v
                mtp_keys_to_remove.append(k)

        for k in mtp_keys_to_remove:
            del out[k]

        for layer_idx in sorted(mtp_expert_buf.keys()):
            mtp_i = layer_idx - self.num_hidden_layers
            experts = mtp_expert_buf[layer_idx]
            n_exp = max(experts.keys()) + 1
            assert n_exp == self.n_experts

            gate_list = [experts[i]["gate_proj"] for i in range(n_exp)]
            up_list = [experts[i]["up_proj"] for i in range(n_exp)]
            down_list = [experts[i]["down_proj"] for i in range(n_exp)]

            w1 = merge_experts_to_weight1(gate_list, up_list, layout=self.moe_layout)
            w2 = merge_experts_to_weight2(down_list, layout=self.moe_layout)

            mtp_prefix = f"model.mtp_block.mtp_layers.{mtp_i}.decoder_layer."
            out[mtp_prefix + "block_sparse_moe.experts.weight1"] = w1
            out[mtp_prefix + "block_sparse_moe.experts.weight2"] = w2

        if self._kda_buf:
            unpaired = {str(k): sorted(v.keys()) for k, v in self._kda_buf.items()}
            self._kda_buf.clear()
            raise RuntimeError(f"Unpaired KimiDeltaAttention q/k/v parts at chunk end: {unpaired}")

        return out

    @staticmethod
    def _get_layer_idx(key: str) -> Optional[int]:
        m = re.match(r"model\.layers\.(\d+)\.", key)
        return int(m.group(1)) if m else None

    def _map_linear_attention_key(
        self, suffix: str, tensor: torch.Tensor, layer_idx: int,
    ) -> Optional[List[Tuple[str, torch.Tensor]]]:
        """Map one HF linear-attention key to local naming.

        Returns a list of (local_suffix, tensor) pairs (possibly empty while a
        KDA q/k/v triple is still being accumulated), or None if `suffix` is not
        an attention key (MLP / norms / gate handled by the caller).
        """
        if suffix.startswith("self_attn."):
            inner = suffix[len("self_attn."):]
            local_prefix = "self_attn._attn."
        elif suffix.startswith("linear_attn."):
            # attention_distillation variant
            inner = suffix[len("linear_attn."):]
            local_prefix = "self_attn._attn.gated_delta_net."
        else:
            return None

        if self.linear_attention_type == "KimiDeltaAttention":
            return self._map_kda_inner(inner, tensor, local_prefix, layer_idx)
        return self._map_gdn_inner(inner, tensor, local_prefix)

    def _map_gdn_inner(
        self, inner: str, tensor: torch.Tensor, local_prefix: str,
    ) -> Optional[List[Tuple[str, torch.Tensor]]]:
        """Qwen3NextGatedDeltaNet: inverse of the string/tensor transforms in
        deepseek.py (in_proj_qkvz, conv1d, in_proj_ba, out_proj, norm, A_log, dt_bias)."""
        n_h, n_v = self.lin_num_k_heads, self.lin_num_v_heads
        d_k, d_v = self.lin_head_k_dim, self.lin_head_v_dim

        if inner == "in_proj_qkvz.weight":
            if self.linear_use_legacy_qkvz_layout:
                return [(local_prefix + "qkvz_proj.weight", tensor)]
            qkv, z = split_in_proj_qkvz(tensor, n_h, n_v, d_k, d_v)
            return [
                (local_prefix + "qkv_proj.weight", qkv),
                (local_prefix + "z_proj.weight", z),
            ]

        if inner == "in_proj_ba.weight":
            return [(local_prefix + "ba_proj.weight", tensor)]

        if inner == "out_proj.weight":
            return [(local_prefix + "o_proj.weight", tensor)]

        if inner in ("conv1d.weight", "conv1d.bias"):
            conv_sub = inner[len("conv1d."):]  # "weight" | "bias"
            if self.linear_use_legacy_qkvz_layout:
                q, k, v = split_conv_order_to_separate(tensor, n_h, n_v, d_k, d_v)
                return [
                    (local_prefix + f"q_conv1d.{conv_sub}", q),
                    (local_prefix + f"k_conv1d.{conv_sub}", k),
                    (local_prefix + f"v_conv1d.{conv_sub}", v),
                ]
            qkv_conv = conv_order_to_interleaved_qkv_conv(tensor, n_h, n_v, d_k, d_v)
            return [(local_prefix + f"qkv_conv1d.{conv_sub}", qkv_conv)]

        if inner in ("A_log", "dt_bias"):
            return [(local_prefix + inner, tensor)]

        if inner.startswith("norm."):
            # inverse of _map_linear_attn_norm_key
            rest = inner[len("norm."):]
            return [(local_prefix + "o_norm." + rest, tensor)]

        return None

    def _map_kda_inner(
        self, inner: str, tensor: torch.Tensor, local_prefix: str, layer_idx: int,
    ) -> Optional[List[Tuple[str, torch.Tensor]]]:
        """KimiDeltaAttention: inverse of deepseek.py 494-535. HF stores q/k/v
        projections and convs separately; merge them back to interleaved fused
        local tensors, and rename the f/g gate projections."""
        m = re.match(r"([qkv])_proj\.weight$", inner)
        if m:
            return self._accumulate_kda(layer_idx, local_prefix, "proj", m.group(1), tensor)

        m = re.match(r"([qkv])_conv1d\.(weight|bias)$", inner)
        if m:
            return self._accumulate_kda(
                layer_idx, local_prefix, f"conv1d.{m.group(2)}", m.group(1), tensor)

        kda_renames = {
            "f_a_proj": "f_proj.0", "f_b_proj": "f_proj.1",
            "g_a_proj": "g_proj.0", "g_b_proj": "g_proj.1",
        }
        for hf_name, local_name in kda_renames.items():
            if inner.startswith(hf_name + "."):
                rest = inner[len(hf_name):]  # ".weight"
                return [(local_prefix + local_name + rest, tensor)]

        # b_proj / o_proj / o_norm keep their inner names; A_log / dt_bias too.
        if inner.startswith(("b_proj.", "o_proj.", "o_norm.")) or inner in ("A_log", "dt_bias"):
            return [(local_prefix + inner, tensor)]

        return None

    def _accumulate_kda(
        self, layer_idx: int, local_prefix: str, kind: str, part: str,
        tensor: torch.Tensor,
    ) -> List[Tuple[str, torch.Tensor]]:
        """Buffer a KDA q/k/v part; once all three are present, merge to the
        interleaved local tensor. `kind` is 'proj' or 'conv1d.weight'/'conv1d.bias'."""
        buf = self._kda_buf.setdefault((layer_idx, local_prefix, kind), {})
        buf[part] = tensor
        if not all(p in buf for p in ("q", "k", "v")):
            return []
        merged = merge_kimi_qkv(
            buf["q"], buf["k"], buf["v"],
            self.lin_num_k_heads, self.lin_num_v_heads,
            self.lin_head_k_dim, self.lin_head_v_dim,
        )
        del self._kda_buf[(layer_idx, local_prefix, kind)]
        if kind == "proj":
            return [(local_prefix + "qkv_proj.weight", merged)]
        conv_sub = kind.split(".", 1)[1]  # weight | bias
        return [(local_prefix + f"qkv_conv1d.{conv_sub}", merged)]

    def _map_mtp_key(
        self, key: str, tensor: torch.Tensor, layer_idx: int,
    ) -> Optional[List[Tuple[str, torch.Tensor]]]:
        """
        Reverse-map MTP layer keys (layer_idx >= num_hidden_layers) back to
        local mtp_block format.

        Forward direction (local → HF) from deepseek.py:
          mtp_block.mtp_layers.{i}.decoder_layer.self_attn.mtp_input_proj  → layers.{61+i}.eh_proj  (halves swapped)
          mtp_block.mtp_layers.{i}.decoder_layer.self_attn.mtp_token_layernorm → layers.{61+i}.enorm
          mtp_block.mtp_layers.{i}.decoder_layer.self_attn.mtp_hidden_layernorm → layers.{61+i}.hnorm
          mtp_block.mtp_norms.{i}.weight → layers.{61+i}.shared_head.norm.weight
          embed_tokens.weight → layers.{61+i}.embed_tokens.weight  (clone, skip on reverse)
          lm_head.weight → layers.{61+i}.shared_head.head.weight  (clone, skip on reverse)
          All attention/MoE keys go through normal rename under decoder_layer prefix.
        """
        mtp_i = layer_idx - self.num_hidden_layers
        prefix = f"model.layers.{layer_idx}."
        suffix = key[len(prefix):]
        mtp_prefix = f"model.mtp_block.mtp_layers.{mtp_i}.decoder_layer."

        # embed_tokens / shared_head.head — these are CLONES of the main tensors, skip
        if suffix == "embed_tokens.weight":
            return []  # skip, already have model.embed_tokens.weight
        if suffix == "shared_head.head.weight":
            return []  # skip, already have lm_head.weight

        # shared_head.norm → mtp_block.mtp_norms.{i}  (may be gated)
        if suffix.startswith("shared_head.norm."):
            rest = suffix[len("shared_head.norm."):]
            return [(f"model.mtp_block.mtp_norms.{mtp_i}." + rest, tensor)]

        # eh_proj → mtp_input_proj  (reverse the half-swap from forward conversion)
        if suffix == "eh_proj.weight":
            W = tensor
            H = self.hidden_size
            if W.shape[1] == 2 * H:
                # Forward did: left,right = W[:,:H], W[:,H:] → cat([right,left])
                # Reverse: left,right = W[:,:H], W[:,H:] → cat([right,left]) (same swap!)
                left, right = W[:, :H], W[:, H:]
                W = torch.cat([right, left], dim=1).contiguous()
            elif W.shape[0] == 2 * H:
                top, bottom = W[:H, :], W[H:, :]
                W = torch.cat([bottom, top], dim=0).contiguous()
            return [(mtp_prefix + "self_attn.mtp_input_proj.weight", W)]

        if suffix.startswith("enorm."):
            rest = suffix[len("enorm."):]
            return [(mtp_prefix + "self_attn.mtp_token_layernorm." + rest, tensor)]

        if suffix.startswith("hnorm."):
            rest = suffix[len("hnorm."):]
            return [(mtp_prefix + "self_attn.mtp_hidden_layernorm." + rest, tensor)]

        # All other MTP layer keys: normal attention/MoE/layernorm reverse mapping,
        # but the output goes under mtp_block.mtp_layers.{i}.decoder_layer.
        # Also, HF "self_attn.*" in MTP layers maps to "self_attn.base_attn.*" locally.

        if self.mtp_block_type == "moe":
            attn_replacement = "self_attn.base_attn._attn."
        else:  # dense
            attn_replacement = "self_attn.base_attn."

        if self._is_linear_layer(layer_idx):
            lin_mapped = self._map_linear_attention_key(suffix, tensor, layer_idx)
            if lin_mapped is not None:
                return [
                    (mtp_prefix + ls.replace("self_attn._attn.", attn_replacement, 1), lt)
                    for ls, lt in lin_mapped
                ]

        # Attention keys (with base_attn prefix for MTP).
        # MoE path keeps the AttnGate `_attn` indirection; dense path drops it
        # because LlamaDecoderLayer holds LatentAttention directly at self_attn.
        mapped = self._map_attention_key(suffix, tensor)
        if mapped is not None:
            result = []
            for local_suffix, local_tensor in mapped:
                local_suffix = local_suffix.replace(
                    "self_attn._attn.", attn_replacement, 1
                )
                result.append((mtp_prefix + local_suffix, local_tensor))
            return result

        if suffix.startswith(("input_layernorm.", "post_attention_layernorm.")):
            base, rest = suffix.split(".", 1)
            if self.mtp_block_type == "moe":
                return [(mtp_prefix + "self_attn.base_attn." + base + "." + rest, tensor)]
            return [(mtp_prefix + base + "." + rest, tensor)]

        if suffix.startswith("post_self_attn_layernorm."):
            if self.mtp_block_type == "moe":
                return [(mtp_prefix + "self_attn.base_attn." + suffix, tensor)]
            return [(mtp_prefix + suffix, tensor)]

        if suffix.startswith("post_feedforward_layernorm."):
            return [(mtp_prefix + suffix, tensor)]

        if suffix == "mlp.shared_expert_sigmoid.weight":
            return [(mtp_prefix + "self_attn.base_attn.shared_expert_sigmoid.weight", tensor)]

        if suffix.startswith("mlp.") \
                and not suffix.startswith("mlp.experts.") \
                and not suffix.startswith("mlp.shared_experts.") \
                and not suffix.startswith("mlp.gate."):
            if self.mtp_block_type == "moe":
                return [(mtp_prefix + suffix.replace("mlp.", "self_attn.mlp.", 1), tensor)]
            return [(mtp_prefix + suffix, tensor)]

        # Shared experts — in MTP, under base_attn
        if suffix.startswith("mlp.shared_experts."):
            return [(mtp_prefix + suffix.replace("mlp.shared_experts.", "self_attn.base_attn.shared_experts.", 1), tensor)]

        # Gate — in MTP, under base_attn
        if suffix == "mlp.gate.weight":
            return [(mtp_prefix + "self_attn.base_attn._gate.gate.weight", tensor)]
        if suffix == "mlp.gate.e_score_correction_bias":
            return [(mtp_prefix + "self_attn.base_attn._gate.expert_bias", tensor)]

        # MoE expert weights — these need special handling for merge
        expert_match = re.match(
            r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", suffix
        )
        if expert_match:
            # Store with a synthetic key that _map_mtp_experts will process
            return [(f"__mtp_expert__{layer_idx}__{expert_match.group(1)}__{expert_match.group(2)}",
                      tensor)]

        raise RuntimeError(
            f"Unmapped MTP key: {key}\n"
            f"This key does not match any known DeepSeek-V3 MTP weight pattern.\n"
            f"Add reverse mapping in HFToLocalPerLayerMapper._map_mtp_key()."
        )

    def _map_attention_key(
        self, suffix: str, tensor: torch.Tensor,
    ) -> Optional[List[Tuple[str, torch.Tensor]]]:

        if suffix == "self_attn.q_a_proj.weight":
            return [("self_attn._attn.dq_proj.weight", tensor)]
        if suffix.startswith("self_attn.q_a_layernorm."):
            rest = suffix[len("self_attn.q_a_layernorm."):]
            return [("self_attn._attn.q_norm." + rest, tensor)]
        if suffix == "self_attn.q_b_proj.weight":
            return [("self_attn._attn.uq_proj.weight", tensor)]
        if suffix == "self_attn.q_proj.weight":
            return [("self_attn._attn.q_proj.weight", tensor)]
        if suffix == "self_attn.o_proj.weight":
            return [("self_attn._attn.o_proj.weight", tensor)]
        if suffix == "self_attn.gate_proj.weight":
            return [("self_attn._attn.gate_proj.weight", tensor)]
        if suffix.startswith("self_attn.kv_a_layernorm."):
            rest = suffix[len("self_attn.kv_a_layernorm."):]
            return [("self_attn._attn.kv_norm." + rest, tensor)]

        if suffix == "self_attn.kv_a_proj_with_mqa.weight":
            dkv, kr = split_kv_a_proj_with_mqa(
                tensor, self.kv_lora_rank, self.qk_rope_head_dim
            )
            return [
                ("self_attn._attn.dkv_proj.weight", dkv),
                ("self_attn._attn.kr_proj.weight", kr),
            ]

        if suffix == "self_attn.kv_b_proj.weight":
            uk, uv = split_kv_b_proj(
                tensor, self.num_heads, self.qk_nope_head_dim, self.v_head_dim
            )
            return [
                ("self_attn._attn.uk_proj.weight", uk),
                ("self_attn._attn.uv_proj.weight", uv),
            ]

        return None


# ═══════════════════════════════════════════════════════════════════════════
# 5. Config conversion  HF → local
# ═══════════════════════════════════════════════════════════════════════════

def _is_gigachat_hf_config(hf_config: Dict[str, Any]) -> bool:
    """Detect the GigaChat3.5 architecture family from an HF config dict.

    Reuses the canonical `is_gigachat_model` detector; falls back to an inline
    copy of its logic when the `utils` package is not importable (e.g. running
    this module fully standalone without the convertation package on the path).
    """
    try:
        from utils.llmfoundry_config_utils import is_gigachat_model
    except ImportError:
        try:
            from llmfoundry_config_utils import is_gigachat_model  # type: ignore
        except ImportError:
            _MARKERS = (
                "gated_attention", "attention_distillation",
                "use_shared_expert_sigmoid", "use_mla_scaling_factor",
            )
            _DEEPSEEK_LN = {"rmsnorm", "LlamaRMSNorm", "pre"}
            if hf_config.get("model_type") == "gigachat3_5":
                return True
            if hf_config.get("linear_attention_type") is not None:
                return True
            if any(hf_config.get(f) for f in _MARKERS):
                return True
            ln = hf_config.get("layernorm_type")
            return ln is not None and ln not in _DEEPSEEK_LN
    return is_gigachat_model(hf_config)


def resolve_mtp_block_type_from_hf_config(hf_config: Dict[str, Any]) -> str:
    """Pick the MTP block flavor ('moe'/'dense') for the target local model.

    `nextn_is_sparse` is written by the forward mapper from the local `mtp_block_type`
    (see `DeepseekV3Mapper._get_nextn_is_sparse`). Its absence is informative rather than
    ambiguous: it means the source local config carried no `mtp_block_type` either, i.e.
    the run used the `GigaMixConfig` default, which is "moe". DeepSeek-V3 checkpoints
    never carry the field and also have a MoE MTP head — so "moe" is the right default
    in both cases, and no manual override is needed.
    """
    nextn_is_sparse = hf_config.get("nextn_is_sparse")
    if nextn_is_sparse is None:
        return "moe"
    return "moe" if nextn_is_sparse else "dense"


def convert_hf_config_to_local(
    hf_config: Dict[str, Any],
    moe_layout: str = "ggemm",
    mtp_block_type: Optional[str] = None,
) -> Dict[str, Any]:
    local = dict(hf_config)
    is_gigachat = _is_gigachat_hf_config(hf_config)

    # `model_type` is required downstream by giga_fsdp
    strip = ["architectures", "auto_map", "quantization_config",
             "_name_or_path", "_auto_map_for_save", "layer_types"]
    if is_gigachat:
        local["model_type"] = "giga_mix"
    else:
        strip.append("model_type")
    for k in strip:
        local.pop(k, None)

    if "moe_intermediate_size" in local and "intermediate_size" in local:
        local["dense_intermediate_size"] = local["intermediate_size"]
        local["intermediate_size"] = local["moe_intermediate_size"]

    if "n_routed_experts" in local:
        local["num_routed_experts"] = local["n_routed_experts"]
    if "routed_scaling_factor" in local:
        local["moe_router_routed_scaling_factor"] = local["routed_scaling_factor"]
    if "n_group" in local:
        local["moe_router_num_groups"] = local["n_group"]
    if "topk_group" in local:
        local["moe_router_group_topk"] = local["topk_group"]
    if "aux_loss_alpha" in local:
        local["router_aux_loss_coef"] = local["aux_loss_alpha"]

    if local.get("q_lora_rank", None) is None:
        local["q_lora_rank"] = 0

    if is_gigachat and local.get("final_logit_softcapping") is not None:
        local["lm_head_logit_softcapping"] = local["final_logit_softcapping"]

    local.setdefault("attention_type", "LlamaLatentAttention")
    local.setdefault("num_krope_heads", 1)
    local.setdefault("scoring_func", "sigmoid")
    local.setdefault("aux_loss_free", True)

    if local.get("num_nextn_predict_layers", 0) > 0:
        local["use_mtp"] = True
        local["mtp_predictor_num"] = local["num_nextn_predict_layers"]
        if mtp_block_type is None:
            mtp_block_type = resolve_mtp_block_type_from_hf_config(local)
        local["mtp_block_type"] = mtp_block_type
        local.pop("nextn_is_sparse", None)
        try:
            from utils.mtp_attention import resolve_mtp_uses_linear_attention
        except ImportError:
            from mtp_attention import resolve_mtp_uses_linear_attention
        local["use_linear_attention_in_mtp"] = resolve_mtp_uses_linear_attention(local)
        local.pop("nextn_is_linear", None)
    else:
        local["use_mtp"] = False
        local.pop("nextn_is_sparse", None)
        local.pop("nextn_is_linear", None)
        local.pop("use_linear_attention_in_mtp", None)

    if moe_layout == "sonicmoe":
        local["moe_kernel"] = "sonic"
        local["moe_impl"] = "SonicGMMMoeBlock"
        local["use_new_expert_weight_layout"] = False
    else:
        local["moe_kernel"] = "g_gemm"
        local["use_new_expert_weight_layout"] = False
        # Clear sonic-only markers that may have been inherited from the HF config
        local.pop("moe_impl", None)

    return local


# ═══════════════════════════════════════════════════════════════════════════
# 6. Multi-node streaming  (compatible with _stream_core / merge_safetensors_ranks)
# ═══════════════════════════════════════════════════════════════════════════

_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
_MTP_LAYER_RE = re.compile(r"model\.mtp_block\.mtp_layers\.(\d+)\.")
_MTP_NORMS_RE = re.compile(r"model\.mtp_block\.mtp_norms\.(\d+)\.")


def _group_keys_by_layer(
    all_keys: List[str],
    num_hidden_layers: int = 0,
) -> Tuple[List[str], Dict[int, List[str]]]:
    """
    Split tensor keys into:
      - global_keys: embed_tokens, lm_head, model.norm, etc.
      - layer_keys: {layer_idx: [keys...]}

    Recognizes:
      - model.layers.N.*  → layer N
      - model.mtp_block.mtp_layers.N.*  → layer (num_hidden_layers + N)
      - model.mtp_block.mtp_norms.N.*   → layer (num_hidden_layers + N)
    """
    global_keys: List[str] = []
    layer_keys: Dict[int, List[str]] = {}
    for k in all_keys:
        m = _LAYER_RE.match(k)
        if m:
            layer_keys.setdefault(int(m.group(1)), []).append(k)
            continue
        m = _MTP_LAYER_RE.match(k)
        if m and num_hidden_layers > 0:
            layer_keys.setdefault(num_hidden_layers + int(m.group(1)), []).append(k)
            continue
        m = _MTP_NORMS_RE.match(k)
        if m and num_hidden_layers > 0:
            layer_keys.setdefault(num_hidden_layers + int(m.group(1)), []).append(k)
            continue
        global_keys.append(k)
    return global_keys, layer_keys


def _assign_layers_to_rank(
    num_layers: int,
    rank: int,
    world_size: int,
    chunk_layers: int = 2,
) -> List[int]:
    """Distribute layer indices across ranks in round-robin chunks."""
    all_layers = list(range(num_layers))
    chunks = [all_layers[i:i + chunk_layers] for i in range(0, len(all_layers), chunk_layers)]
    my_layers = []
    for idx, ch in enumerate(chunks):
        if idx % world_size == rank:
            my_layers.extend(ch)
    return my_layers


def _load_keys_from_hf(
    model_dir: str,
    keys_to_load: List[str],
    weight_map: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """Load only specific keys from HF safetensors shards."""
    keys_set = set(keys_to_load)
    # Group by shard file
    shard_to_keys: Dict[str, List[str]] = {}
    for k in keys_to_load:
        shard_file = weight_map.get(k)
        if shard_file:
            shard_to_keys.setdefault(shard_file, []).append(k)

    result: Dict[str, torch.Tensor] = {}
    for shard_file, shard_keys in shard_to_keys.items():
        shard_path = os.path.join(model_dir, shard_file)
        with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in shard_keys:
                result[k] = f.get_tensor(k)
    return result


def _collect_scale_keys_for(keys: List[str], all_weight_map: Dict[str, str]) -> List[str]:
    """For each weight key, find its matching weight_scale_inv key if it exists."""
    extra = []
    for k in keys:
        if k.endswith(".weight"):
            scale_key = k.rsplit(".", 1)[0] + ".weight_scale_inv"
            if scale_key in all_weight_map:
                extra.append(scale_key)
    return extra


def stream_hf_to_local_safetensors(
    input_dir: str,
    output_dir: str,
    *,
    rank: int = 0,
    world_size: int = 1,
    chunk_layers: int = 2,
    moe_layout: str = "ggemm",
    block_size: Tuple[int, int] = (128, 128),
    hf_config: Optional[Dict[str, Any]] = None,
    mtp_block_type: Optional[str] = None,
) -> None:
    """
    Streaming multi-node conversion: HF safetensors → local training format.

    Each rank:
      1. Reads HF index
      2. Gets assigned a subset of layers
      3. For each chunk: loads HF tensors → dequantizes → reverse-maps → saves shard
      4. Writes rank-index.json

    After all ranks finish, master runs merge_safetensors_ranks.py to flatten.

    Output structure (per rank):
      output_dir/rank{NN}/shard-r{NN}-c{CCCCC}.safetensors
      output_dir/rank{NN}/rank-index.json
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load HF config
    if hf_config is None:
        with open(os.path.join(input_dir, "config.json")) as f:
            hf_config = json.load(f)

    # Load HF index
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            hf_index = json.load(f)
        weight_map = hf_index["weight_map"]
    else:
        # Single-file model
        sf_files = sorted(Path(input_dir).glob("*.safetensors"))
        weight_map = {}
        for sf in sf_files:
            with safetensors.safe_open(str(sf), framework="pt") as f:
                for k in f.keys():
                    weight_map[k] = sf.name

    all_keys = sorted(weight_map.keys())
    num_hidden_layers = hf_config.get("num_hidden_layers", 0)
    global_keys, layer_keys_map = _group_keys_by_layer(all_keys, num_hidden_layers)

    num_layers = max(layer_keys_map.keys()) + 1 if layer_keys_map else 0
    my_layers = _assign_layers_to_rank(num_layers, rank, world_size, chunk_layers)

    # Rank 0 also handles global tensors
    handle_globals = (rank == 0)

    if mtp_block_type is None:
        mtp_block_type = resolve_mtp_block_type_from_hf_config(hf_config)

    mapper = HFToLocalPerLayerMapper(
        hf_config, moe_layout=moe_layout, mtp_block_type=mtp_block_type
    )

    # Save config and composer_state.pt on rank 0
    if rank == 0:
        local_config = convert_hf_config_to_local(
            hf_config, moe_layout=moe_layout, mtp_block_type=mtp_block_type
        )
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(local_config, f, indent=2)
        print(f"[rank {rank}] Saved config.json")

        # Create composer_state.pt — required by gigafsdp/papayafsdp checkpoint loader.
        # Contains model config under state.integrations.huggingface.model.config.content
        composer_state = {
            "state": {
                "integrations": {
                    "huggingface": {
                        "model": {
                            "config": {
                                "content": local_config,
                            }
                        }
                    }
                },
                "optimizers": {},
            }
        }
        composer_state_path = os.path.join(output_dir, "composer_state.pt")
        torch.save(composer_state, composer_state_path)
        print(f"[rank {rank}] Saved composer_state.pt")

    out_rank_dir = os.path.join(output_dir, f"rank{rank:02d}")
    os.makedirs(out_rank_dir, exist_ok=True)

    chunk_idx = 0

    # Process global tensors (rank 0 only)
    if handle_globals and global_keys:
        scale_extras = _collect_scale_keys_for(global_keys, weight_map)
        keys_to_load = global_keys + scale_extras
        raw = _load_keys_from_hf(input_dir, keys_to_load, weight_map)
        deq = dequantize_tensors(raw, block_size=block_size)
        del raw
        mapped = mapper.map_chunk(deq)
        del deq

        shard_file = f"shard-r{rank:02d}-c{chunk_idx:05d}.safetensors"
        shard_path = os.path.join(out_rank_dir, shard_file)
        shard_size = _save_chunk(mapped, shard_path)
        _write_weight_map(out_rank_dir, rank, chunk_idx, mapped, shard_file, shard_size)
        mapped.clear()
        chunk_idx += 1
        print(f"[rank {rank}] Saved global tensors ({len(global_keys)} keys)")

    # Process layers in chunks
    layer_chunks = [my_layers[i:i + chunk_layers] for i in range(0, len(my_layers), chunk_layers)]

    for lc in tqdm(layer_chunks, desc=f"[rank {rank}] Converting layers"):
        # Collect all keys for these layers + their scale keys
        layer_keys: List[str] = []
        for lid in lc:
            layer_keys.extend(layer_keys_map.get(lid, []))

        scale_extras = _collect_scale_keys_for(layer_keys, weight_map)
        keys_to_load = layer_keys + scale_extras

        raw = _load_keys_from_hf(input_dir, keys_to_load, weight_map)
        deq = dequantize_tensors(raw, block_size=block_size)
        del raw
        mapped = mapper.map_chunk(deq)
        del deq

        shard_file = f"shard-r{rank:02d}-c{chunk_idx:05d}.safetensors"
        shard_path = os.path.join(out_rank_dir, shard_file)
        shard_size = _save_chunk(mapped, shard_path)
        _write_weight_map(out_rank_dir, rank, chunk_idx, mapped, shard_file, shard_size)
        mapped.clear()
        chunk_idx += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write rank-index.json (same format as _write_local_index)
    _write_rank_index(out_rank_dir)
    print(f"[rank {rank}] Done. Processed {len(my_layers)} layers, wrote {chunk_idx} shards.")


def _save_chunk(tensors: Dict[str, torch.Tensor], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cpu_tensors = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    safe_save_file(cpu_tensors, out_path)
    return sum(t.numel() * t.element_size() for t in cpu_tensors.values())


def _write_weight_map(
    out_rank_dir: str,
    rank: int,
    chunk_idx: int,
    mapped: Dict[str, torch.Tensor],
    shard_file: str,
    shard_size: int,
) -> None:
    os.makedirs(out_rank_dir, exist_ok=True)
    wm = {k: shard_file for k in mapped.keys()}
    wm_path = os.path.join(out_rank_dir, f"shard-r{rank:02d}-c{chunk_idx:05d}.weight_map.json")
    with open(wm_path, "w") as fh:
        json.dump({"weight_map": wm, "total_size": shard_size}, fh)


def _write_rank_index(out_rank_dir: str) -> None:
    """Write rank-index.json — same format as utils/safetensors_local.py::_write_local_index."""
    import glob as glob_mod
    pattern = os.path.join(out_rank_dir, "shard-r*-c*.weight_map.json")
    files = sorted(glob_mod.glob(pattern))
    weight_map: Dict[str, str] = {}
    total_size = 0
    for f in files:
        with open(f) as fh:
            part = json.load(fh)
        for k, v in part.get("weight_map", {}).items():
            weight_map.setdefault(k, v)
        total_size += int(part.get("total_size", 0))
    local_index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(out_rank_dir, "rank-index.json"), "w") as fh:
        json.dump(local_index, fh, indent=2)
    print(f"[{os.path.basename(out_rank_dir)}] rank-index.json: "
          f"{len(weight_map)} tensors, {total_size} bytes")


# ═══════════════════════════════════════════════════════════════════════════
# 7. CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert HF DeepSeek-V3 checkpoint → local training format (multi-node)"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to HF model directory")
    parser.add_argument("--output", "-o", required=True,
                        help="Path to output directory")
    parser.add_argument("--moe-layout", default="ggemm",
                        choices=["ggemm", "sonicmoe"],
                        help="MoE weight layout (default: old_g_gemm)")
    parser.add_argument("--global-rank", type=int, default=None,
                        help="Node rank (default: from RANK env or 0)")
    parser.add_argument("--world-size", type=int, default=None,
                        help="Number of nodes (default: from WORLD_SIZE env or 1)")
    parser.add_argument("--chunk-layers", type=int, default=2,
                        help="Layers per chunk/shard (default: 2)")
    args = parser.parse_args()

    rank = args.global_rank if args.global_rank is not None else int(os.environ.get("RANK", "0"))
    world_size = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))

    with open(os.path.join(args.input, "config.json")) as f:
        hf_config = json.load(f)

    block_size = tuple(
        hf_config.get("quantization_config", {}).get("weight_block_size", [128, 128])
    )
    print(f"[rank {rank}/{world_size}] block_size={block_size}, moe_layout={args.moe_layout}")

    stream_hf_to_local_safetensors(
        input_dir=args.input,
        output_dir=args.output,
        rank=rank,
        world_size=world_size,
        chunk_layers=args.chunk_layers,
        moe_layout=args.moe_layout,
        block_size=block_size,
        hf_config=hf_config,
    )

    if rank == 0 and world_size == 1:
        # Single-node: run merge immediately
        from utils.merge_ranks import flatten_move_and_reindex
        print("\n=== Merging rank shards ===")
        flatten_move_and_reindex(args.output)

    print(f"[rank {rank}] All done!")


if __name__ == "__main__":
    main()
