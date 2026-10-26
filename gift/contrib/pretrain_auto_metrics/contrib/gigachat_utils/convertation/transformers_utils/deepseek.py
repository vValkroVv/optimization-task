from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import re
from logger import logger
from collections.abc import Mapping

try:
    from utils.mtp_attention import resolve_mtp_uses_linear_attention
except ImportError:
    from mtp_attention import resolve_mtp_uses_linear_attention

def _get_value(d, k, default=None):
    if isinstance(d, Mapping):
        return d.get(k, default)
    return getattr(d, k, default)

def _has_key(d, k):
    if isinstance(d, Mapping):
        return k in d
    return hasattr(d, k)

def _get_int(d, *keys, default=0):
    for k in keys:
        if _has_key(d, k):
            v = _get_value(d, k, None)
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    pass
    return default

def _get_bool(d, *keys, default=False):
    for k in keys:
        if _has_key(d, k):
            v = _get_value(d, k, None)
            if v is not None:
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    return v.lower() in {"1", "true", "yes", "y", "on"}
                try:
                    return bool(int(v))
                except Exception:
                    pass
    return default

def _get_layer_idx_from_key(key: str):
    parts = key.split(".")

    if parts and parts[0] == "model":
        parts = parts[1:]

    for i in range(len(parts) - 1):
        if parts[i] in {"layers", "layer", "blocks", "h"} and parts[i + 1].isdigit():
            return int(parts[i + 1])

    return None

def _get_str(d, *keys, default=""):
    for k in keys:
        if _has_key(d, k):
            v = _get_value(d, k, None)
            if v is not None:
                return str(v)
    return default

class _DeepseekPerLayerTransformer:
    LINEAR_ATTN_NORM_REPLACEMENTS = (
        ("self_attn._attn.o_norm.", "self_attn.norm."),
        ("self_attn._attn.gated_delta_net.o_norm.", "linear_attn.norm."),
        ("self_attn.o_norm.", "self_attn.norm."),
    )

    # Linear-attention (GDN) inner names arrive under three prefixes, mirroring
    # `_ATTN_PREFIX_VARIANTS` for the softmax path:
    #   `.self_attn._attn.<inner>`                 -> `.self_attn.<target>`
    #   `.self_attn.<inner>`                       -> `.self_attn.<target>`   (dense MTP)
    #   `.self_attn._attn.gated_delta_net.<inner>` -> `.linear_attn.<target>` (distillation)
    # Ordered longest-first so the bare prefix never shadows the other two.
    _LINEAR_ATTN_PREFIX_TARGETS = (
        (".self_attn._attn.gated_delta_net.", "linear_attn"),
        (".self_attn._attn.", "self_attn"),
        (".self_attn.", "self_attn"),
    )

    # Inner attention module keys in the checkpoint may live under two prefixes:
    #   `.self_attn._attn.<inner>.` — standard gigamix layers wrap attention in `_attn`.
    #   `.self_attn.<inner>.`       — dense MTP decoder layers inline attention directly.
    # Both forms map to the same HF target; renames below apply to either.
    _ATTN_PREFIX_VARIANTS = (".self_attn._attn.", ".self_attn.")

    # (inner, target_inner): ".self_attn[._attn].{inner}." -> ".self_attn.{target_inner}.".
    _ATTN_INNER_RENAMES = (
        ("dq_proj", "q_a_proj"),
        ("q_norm",  "q_a_layernorm"),
        ("uq_proj", "q_b_proj"),
        ("kv_norm", "kv_a_layernorm"),
        ("indexer.wq_b", "indexer.wq_b"),
        ("indexer.wq", "indexer.wq_b"),
        ("indexer.wk", "indexer.wk"),
        ("indexer.k_norm", "indexer.k_norm"),
        ("indexer.weights_proj", "indexer.weights_proj"),
    )

    # (inner, kind) for kv_a / kv_b accumulation: match `.{inner}.weight`.
    _ATTN_INNER_KV_A = (("dkv_proj", "dkv"), ("kr_proj", "kr"))
    _ATTN_INNER_KV_B = (("uv_proj",  "uv"),  ("uk_proj", "uk"))

    def __init__(self, cfg):
        self.n_experts    = _get_int(cfg, "n_routed_experts", "num_routed_experts", default=0)
        self.n_heads      = _get_int(cfg, "num_attention_heads", "n_head", default=0)
        self.aux_free     = _get_bool(cfg, "aux_loss_free", default=False)
        self.topk_method  = _get_str(cfg, "topk_method", default="noaux_tc")
        self.base_mtp_idx = _get_int(cfg, "num_hidden_layers", default=0)
        self.num_mtp      = _get_int(cfg, "num_nextn_predict_layers",default=0)
        self.hidden_size  = _get_int(cfg, "hidden_size",default=0)
        _moe_kernel = _get_str(cfg, "moe_kernel", default="g_gemm")
        _moe_impl = _get_str(cfg, "moe_impl", default="")
        # Same condition as mapper_deepseek_moe_to_hf (stream safetensor path).
        self.use_sonic_moe_layout = (
            _moe_kernel == "sonic" or _moe_impl == "SonicGMMMoeBlock"
        )

        self.linear_attention_type = _get_value(cfg, "linear_attention_type", default=None)
        self.is_hybrid    = self.linear_attention_type is not None
        self.full_attention_layers = _get_value(cfg, "full_attention_layers", default=[])

        self.mtp_is_linear_attention = (
            self.num_mtp > 0 and resolve_mtp_uses_linear_attention(cfg)
        )

        # Linear-attention (Qwen3NextGatedDeltaNet) head layout — needed to
        # rearrange packed qkv/qkvz tensors between the training-time
        # "interleaved per-head" layout and the HF "conv order" layout.
        self.lin_num_k_heads   = _get_int(cfg, "linear_num_key_heads", default=0)
        self.lin_num_v_heads   = _get_int(cfg, "linear_num_value_heads", default=0)
        self.lin_head_k_dim    = _get_int(cfg, "linear_key_head_dim", default=0)
        self.lin_head_v_dim    = _get_int(cfg, "linear_value_head_dim", default=0)

        missing = []
        if self.n_heads   <= 0: missing.append("num_attention_heads/n_head")
        if self.hidden_size <= 0:  missing.append("hidden_size")
        if missing:
            raise ValueError(f"Config lacks required dims: {', '.join(missing)}")
        self._init_full_attention_dims(cfg)

        self.kv_a_buf: Dict[str, Dict[str, torch.Tensor]] = {}
        self.kv_b_buf: Dict[str, Dict[str, torch.Tensor]] = {}
        self.qkvz_buf: Dict[str, Dict[str, torch.Tensor]] = {}

    def _init_full_attention_dims(self, cfg) -> None:
        """Read the full-attention head layout (MLA)."""
        self.dk_rope      = _get_int(cfg, "qk_rope_head_dim", default=1)
        self.dk_nope      = _get_int(cfg, "qk_nope_head_dim", "k_nope_head_dim", "q_nope_head_dim", default=0)
        self.v_head_dim   = _get_int(cfg, "v_head_dim", "v_head_dim_per_head", default=0)
        self.q_lora_rank  = _get_int(cfg, "q_lora_rank", default=0)

        missing = []
        if self.dk_nope   <= 0: missing.append("qk_nope_head_dim")
        if self.v_head_dim<= 0: missing.append("v_head_dim")
        if self.dk_rope   <= 0: missing.append("qk_rope_head_dim")
        if missing:
            raise ValueError(f"Config lacks required dims: {', '.join(missing)}")

    def _is_full_attention_layer(self, layer_name: str) -> bool:
        return not self.is_hybrid or _get_layer_idx_from_key(layer_name) in self.full_attention_layers

    @staticmethod
    def _assign(out: Dict[str, torch.Tensor], key: str, tensor: torch.Tensor) -> None:
        out[key] = tensor

    @staticmethod
    def _canonicalize_gated_norm_name(layer_name: str) -> str:
        layer_name = layer_name.replace(".gate.0.", ".gate_up_projection.")
        layer_name = layer_name.replace(".gate.2.", ".gate_down_projection.")
        return layer_name

    @classmethod
    def _map_linear_attn_norm_key(cls, layer_name: str) -> str | None:
        for old, new in cls.LINEAR_ATTN_NORM_REPLACEMENTS:
            if old in layer_name:
                return layer_name.replace(old, new)
        return None

    @classmethod
    def _rename_attn_inner(
        cls, layer_name: str, inner: str, target_inner: str
    ) -> Optional[str]:
        """If `layer_name` contains `.self_attn._attn.{inner}.` or
        `.self_attn.{inner}.`, replace it with `.self_attn.{target_inner}.`
        and return the new name; otherwise return None."""
        for prefix in cls._ATTN_PREFIX_VARIANTS:
            needle = f"{prefix}{inner}."
            if needle in layer_name:
                return layer_name.replace(needle, f".self_attn.{target_inner}.")
        return None

    @classmethod
    def _split_linear_attn_key(
        cls, layer_name: str, inner: str, allow_bare: bool = True
    ) -> Optional[tuple[str, str]]:
        """Split a linear-attention key on its `inner` suffix."""
        for prefix, target_prefix in cls._LINEAR_ATTN_PREFIX_TARGETS:
            if not allow_bare and prefix == ".self_attn.":
                continue
            needle = f"{prefix}{inner}"
            if layer_name.endswith(needle):
                return layer_name[: -len(needle)], target_prefix
        return None

    @classmethod
    def _attn_weight_base(cls, layer_name: str, inner: str) -> Optional[str]:
        """For a weight key ending in `.self_attn._attn.{inner}.weight` or
        `.self_attn.{inner}.weight`, return the layer prefix before the match
        (used as the accumulator key for kv_a / kv_b fusing)."""
        for prefix in cls._ATTN_PREFIX_VARIANTS:
            tail = f"{prefix}{inner}.weight"
            if layer_name.endswith(tail):
                return layer_name[: -len(tail)]
        return None

    def _maybe_accumulate_and_return_kv_a(self, base: str, kind: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        buf = self.kv_a_buf.setdefault(base, {})
        buf[kind] = tensor
        if "dkv" in buf and "kr" in buf:
            dkv = buf["dkv"]
            kr  = buf["kr"]

            assert dkv.shape[1] == kr.shape[1], f"[kv_a] hidden_size mismatch for {base}: {dkv.shape} vs {kr.shape}"
            assert kr.shape[0] == self.dk_rope, f"[kv_a] kr[0] must be qk_rope_head_dim={self.dk_rope}, got {kr.shape}"

            fused = torch.cat([dkv, kr], dim=0).contiguous()
            out[base + ".self_attn.kv_a_proj_with_mqa.weight"] = fused
            del self.kv_a_buf[base]
        return out

    def _maybe_accumulate_and_return_kv_b(self, base: str, kind: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        buf = self.kv_b_buf.setdefault(base, {})
        buf[kind] = tensor
        if "uk" in buf and "uv" in buf:
            uk = buf["uk"]  # (n_heads * dk_nope, r_kv)
            uv = buf["uv"]  # (n_heads * v_head_dim, r_kv)

            n_heads = self.n_heads
            dk_nope = self.dk_nope
            vdim    = self.v_head_dim
            r_kv    = uk.shape[1]

            assert uk.shape == (n_heads * dk_nope, r_kv), f"[kv_b.uk] shape {uk.shape} != {(n_heads * dk_nope, r_kv)}"
            assert uv.shape == (n_heads * vdim,   r_kv), f"[kv_b.uv] shape {uv.shape} != {(n_heads * vdim,   r_kv)}"

            ukv = torch.cat([
                uk.view(n_heads, dk_nope, r_kv),
                uv.view(n_heads, vdim,   r_kv)
            ], dim=1).reshape(n_heads * (dk_nope + vdim), r_kv).contiguous()

            out[base + ".self_attn.kv_b_proj.weight"] = ukv
            del self.kv_b_buf[base]
        return out

    def _maybe_accumulate_and_return_qkvz(self, base: str, target_prefix: str, kind: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        buf_key = f"{base}.{target_prefix}"
        buf = self.qkvz_buf.setdefault(buf_key, {})
        buf[kind] = tensor
        if "qkv" in buf and "z" in buf:
            qkv = buf["qkv"]
            z = buf["z"]

            assert qkv.shape[1] == z.shape[1], f"[qkvz] hidden_size mismatch for {base}: {qkv.shape} vs {z.shape}"

            # Training emits `qkv_proj` / `z_proj` in an *interleaved per-head* layout
            # (per-head chunks packed along dim 0). HF `in_proj_qkvz` also expects
            # interleaved per-head ordering, but with `z` appended inside every head:
            # [h0: q, k, v, z; h1: q, k, v, z; ...].
            n_h     = self.lin_num_k_heads
            n_v     = self.lin_num_v_heads
            d_k     = self.lin_head_k_dim
            d_v     = self.lin_head_v_dim
            assert n_h > 0 and n_v > 0 and d_k > 0 and d_v > 0, (
                "Linear-attention head dims are required to fuse qkv_proj+z_proj; "
                "set linear_num_key_heads/linear_num_value_heads/"
                "linear_key_head_dim/linear_value_head_dim in config."
            )
            assert n_v % n_h == 0, (
                f"linear_num_value_heads={n_v} must be divisible by linear_num_key_heads={n_h}"
            )
            g = n_v // n_h  # v/z heads per k head
            key_dim = n_h * d_k
            v_group = g * d_v
            per_head_qkv = 2 * d_k + v_group
            per_head_z   = v_group
            H = qkv.shape[1]
            assert qkv.shape == (n_h * per_head_qkv, H), (
                f"[qkvz] qkv_proj shape {qkv.shape} != {(n_h * per_head_qkv, H)}"
            )
            assert z.shape == (n_h * per_head_z, H), (
                f"[qkvz] z_proj shape {z.shape} != {(n_h * per_head_z, H)}"
            )

            qkv_ph = qkv.view(n_h, per_head_qkv, H)
            z_ph   = z.view(n_h,   per_head_z,   H)
            fused = torch.cat([qkv_ph, z_ph], dim=1).reshape(
                n_h * (per_head_qkv + per_head_z), H
            ).contiguous()
            out[f"{base}.{target_prefix}.in_proj_qkvz.weight"] = fused
            del self.qkvz_buf[buf_key]
        return out

    def _interleaved_qkv_to_conv_order(self, w: torch.Tensor) -> torch.Tensor:
        """Reorder `qkv_conv1d` weight/bias from training's interleaved per-head
        layout ([h0: q(d_k), k(d_k), v(g*d_v); h1: ...; ...]) into HF's conv order
        ([q_all(key_dim); k_all(key_dim); v_all(value_dim)]) expected by
        `conv1d.weight`, since HF does `torch.cat((q,k,v), dim=-1)` before conv."""
        n_h = self.lin_num_k_heads
        n_v = self.lin_num_v_heads
        d_k = self.lin_head_k_dim
        d_v = self.lin_head_v_dim
        assert n_h > 0 and n_v > 0 and d_k > 0 and d_v > 0, (
            "Linear-attention head dims are required to rearrange qkv_conv1d; "
            "set linear_num_key_heads/linear_num_value_heads/"
            "linear_key_head_dim/linear_value_head_dim in config."
        )
        assert n_v % n_h == 0, (
            f"linear_num_value_heads={n_v} must be divisible by linear_num_key_heads={n_h}"
        )
        g = n_v // n_h
        per_head = 2 * d_k + g * d_v
        assert w.shape[0] == n_h * per_head, (
            f"[qkv_conv1d] channel dim {w.shape[0]} != num_heads*per_head = {n_h * per_head}"
        )
        tail = w.shape[1:]
        ph = w.view(n_h, per_head, *tail)
        q_h = ph[:, :d_k]
        k_h = ph[:, d_k:2 * d_k]
        v_h = ph[:, 2 * d_k:]
        q = q_h.reshape(n_h * d_k, *tail)
        k = k_h.reshape(n_h * d_k, *tail)
        v = v_h.reshape(n_h * g * d_v, *tail)
        return torch.cat([q, k, v], dim=0).contiguous()

    def _interleaved_qkv_to_kimi_order(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_h = self.lin_num_k_heads
        n_v = self.lin_num_v_heads
        d_k = self.lin_head_k_dim
        d_v = self.lin_head_v_dim
        assert d_k == d_v, (
            "KimiDeltaAttention export expects linear_key_head_dim == "
            "linear_value_head_dim."
        )
        assert n_v % n_h == 0, (
            f"linear_num_value_heads={n_v} must be divisible by linear_num_key_heads={n_h}"
        )
        g = n_v // n_h
        per_head = 2 * d_k + g * d_v
        assert w.shape[0] == n_h * per_head, (
            f"[kda_qkv] channel dim {w.shape[0]} != num_heads*per_head = {n_h * per_head}"
        )
        tail = w.shape[1:]
        ph = w.view(n_h, per_head, *tail)
        q = ph[:, :d_k].reshape(n_h * d_k, *tail)
        k = ph[:, d_k:2 * d_k].reshape(n_h * d_k, *tail)
        v = ph[:, 2 * d_k:].reshape(n_h, g, d_v, *tail).reshape(n_v * d_v, *tail)
        return q.contiguous(), k.contiguous(), v.contiguous()

    def rename_and_split(self, merged_layer: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        for orig_name, weights in merged_layer.items():
            layer_name = self._canonicalize_gated_norm_name(orig_name)
            weights = weights.to(torch.bfloat16)
            
            is_mtp_key = layer_name.startswith("model.mtp_block.mtp_layers.")
            if is_mtp_key:
                is_full_softmax = not self.mtp_is_linear_attention
            else:
                is_full_softmax = self._is_full_attention_layer(layer_name)

            if is_mtp_key:
                m = re.match(r"model\.mtp_block\.mtp_layers\.(\d+)\.(.*)", layer_name)
                if m:
                    mtp_i = int(m.group(1))
                    tail = m.group(2)

                    if tail.startswith("decoder_layer."):
                        tail = tail[len("decoder_layer."):]

                    layer_id = self.base_mtp_idx + mtp_i
                    layer_name = f"model.layers.{layer_id}." + tail

                    layer_name = layer_name.replace(".self_attn.base_attn.", ".self_attn.")

                    layer_name = layer_name.replace(".self_attn.mtp_input_proj.", ".eh_proj.")
                    layer_name = layer_name.replace(".self_attn.mtp_token_layernorm.", ".enorm.")
                    layer_name = layer_name.replace(".self_attn.mtp_hidden_layernorm.", ".hnorm.")
                else:
                    raise RuntimeError(f"Unexpected MTP key: {layer_name}")

            if ".eh_proj." in layer_name:
                W = weights
                H = self.hidden_size
                if W.shape[1] == 2 * H:
                    left, right = W[:, :H], W[:, H:]
                    W = torch.cat([right, left], dim=1).contiguous()
                elif W.shape[0] == 2 * H:
                    top, bottom = W[:H, :], W[H:, :]
                    W = torch.cat([bottom, top], dim=0).contiguous()
                else:
                    raise ValueError(
                        f"eh_proj weight has shape {tuple(W.shape)}"
                    )
                self._assign(out, layer_name, W)
                continue
                
            if layer_name.startswith("model.mtp_block.mtp_norms."):
                m = re.match(r"model\.mtp_block\.mtp_norms\.(\d+)\.(.*)", layer_name)
                if m:
                    mtp_i = int(m.group(1))
                    layer_id = self.base_mtp_idx + mtp_i
                    tail = m.group(2)
                    final = f"model.layers.{layer_id}.shared_head.norm.{tail}"
                else:
                    raise RuntimeError(f"Unexpected MTP key: {layer_name}")
                self._assign(out, final, weights)
                continue

            if layer_name.startswith("model.") and 'lm_head' in layer_name:
                layer_name = layer_name[len("model."):]

            if layer_name.endswith(".self_attn._attn.q_proj.weight"):
                final = layer_name.replace(".self_attn._attn.q_proj.", ".self_attn.q_proj.")
                self._assign(out, final, weights)
                continue

            # Inner attention keys handle both `.self_attn._attn.<inner>.` (standard
            # gigamix) and `.self_attn.<inner>.` (dense MTP) prefixes; see class attrs.
            renamed = None
            for inner, target_inner in self._ATTN_INNER_RENAMES:
                renamed = self._rename_attn_inner(layer_name, inner, target_inner)
                if renamed is not None:
                    break
            if renamed is not None:
                self._assign(out, renamed, weights)
                continue

            if layer_name.endswith(".self_attn._attn.o_proj.weight") and is_full_softmax:
                final = layer_name.replace(".self_attn._attn.o_proj.", ".self_attn.o_proj.")
                self._assign(out, final, weights)
                continue

            if layer_name.endswith(".self_attn._attn.gate_proj.weight") and is_full_softmax:
                final = layer_name.replace(".self_attn._attn.gate_proj.", ".self_attn.gate_proj.")
                self._assign(out, final, weights)
                continue

            kv_handled = False
            for inner, kind in self._ATTN_INNER_KV_A:
                base = self._attn_weight_base(layer_name, inner)
                if base is not None:
                    for k, v in self._maybe_accumulate_and_return_kv_a(base, kind, weights).items():
                        self._assign(out, k, v)
                    kv_handled = True
                    break
            if kv_handled:
                continue

            for inner, kind in self._ATTN_INNER_KV_B:
                base = self._attn_weight_base(layer_name, inner)
                if base is not None:
                    for k, v in self._maybe_accumulate_and_return_kv_b(base, kind, weights).items():
                        self._assign(out, k, v)
                    kv_handled = True
                    break
            if kv_handled:
                continue

            if ".self_attn.input_layernorm." in layer_name:
                final = layer_name.replace(".self_attn.input_layernorm.", ".input_layernorm.")
                self._assign(out, final, weights)
                continue

            if ".self_attn.post_attention_layernorm." in layer_name:
                final = layer_name.replace(".self_attn.post_attention_layernorm.", ".post_attention_layernorm.")
                self._assign(out, final, weights)
                continue

            if ".self_attn.post_self_attn_layernorm." in layer_name:
                final = layer_name.replace(".self_attn.post_self_attn_layernorm.", ".post_self_attn_layernorm.")
                self._assign(out, final, weights)
                continue

            if ".self_attn.post_feedforward_layernorm." in layer_name:
                final = layer_name.replace(".self_attn.post_feedforward_layernorm.", ".post_feedforward_layernorm.")
                self._assign(out, final, weights)
                continue

            if ".self_attn.mlp." in layer_name and layer_name.endswith(".weight"):
                final = layer_name.replace(".self_attn.mlp.", ".mlp.")
                self._assign(out, final, weights)
                continue

            if ".self_attn.shared_experts." in layer_name and layer_name.endswith(".weight"):
                final = layer_name.replace(".self_attn.shared_experts.", ".mlp.shared_experts.")
                self._assign(out, final, weights)
                continue

            if layer_name.endswith(".self_attn.shared_expert_sigmoid.weight"):
                final = layer_name.replace(".self_attn.shared_expert_sigmoid.", ".mlp.shared_expert_sigmoid.")
                self._assign(out, final, weights)
                continue

            if layer_name.endswith(".self_attn._gate.gate.weight"):
                final = layer_name.replace(".self_attn._gate.gate.", ".mlp.gate.")
                self._assign(out, final, weights)
                continue

            if layer_name.endswith((
                ".self_attn._gate.expert_bias",
                ".self_attn.base_attn._gate.expert_bias",
            )):
                final = layer_name.replace(".self_attn._gate.expert_bias", ".mlp.gate.e_score_correction_bias")
                final = final.replace(".self_attn.base_attn._gate.expert_bias", ".mlp.gate.e_score_correction_bias")
                self._assign(out, final, weights)
                continue

            if "embed_tokens" in layer_name:
                self._assign(out, layer_name, weights)
                for mtp_i in range(self.num_mtp):
                    layer_id = self.base_mtp_idx + mtp_i
                    layer_name = f"model.layers.{layer_id}.embed_tokens.weight"
                    self._assign(out, layer_name, weights.contiguous().clone())
                continue

            if "lm_head" in layer_name:
                self._assign(out, layer_name, weights)
                for mtp_i in range(self.num_mtp):
                    layer_id = self.base_mtp_idx + mtp_i
                    layer_name = f"model.layers.{layer_id}.shared_head.head.weight"
                    self._assign(out, layer_name, weights.contiguous().clone())
                continue
            
            # Linear Attention Part

            if self.linear_attention_type == "KimiDeltaAttention" and not is_full_softmax:
                final = (
                    self._rename_attn_inner(layer_name, "conv1d", "conv1d")
                    or self._rename_attn_inner(layer_name, "qkv_conv1d", "conv1d")
                )
                if final is not None and final.endswith(".conv1d.weight"):
                    q_w, k_w, v_w = self._interleaved_qkv_to_kimi_order(weights)
                    base = final[: -len(".conv1d.weight")]
                    self._assign(out, f"{base}.q_conv1d.weight", q_w)
                    self._assign(out, f"{base}.k_conv1d.weight", k_w)
                    self._assign(out, f"{base}.v_conv1d.weight", v_w)
                    continue

                qkv_renamed = self._rename_attn_inner(layer_name, "qkv_proj", "qkv_proj")
                if qkv_renamed is not None and qkv_renamed.endswith(".qkv_proj.weight"):
                    q_w, k_w, v_w = self._interleaved_qkv_to_kimi_order(weights)
                    base = qkv_renamed[: -len(".qkv_proj.weight")]
                    self._assign(out, f"{base}.q_proj.weight", q_w)
                    self._assign(out, f"{base}.k_proj.weight", k_w)
                    self._assign(out, f"{base}.v_proj.weight", v_w)
                    continue

                kda_replacements = (
                    ("f_proj.0", "f_a_proj"),
                    ("f_proj.1", "f_b_proj"),
                    ("g_proj.0", "g_a_proj"),
                    ("g_proj.1", "g_b_proj"),
                    ("b_proj", "b_proj"),
                    ("o_norm", "o_norm"),
                    ("o_proj", "o_proj"),
                    ("A_log", "A_log"),
                    ("dt_bias", "dt_bias"),
                )
                for inner, target_inner in kda_replacements:
                    kda_renamed = self._rename_attn_inner(layer_name, inner, target_inner)
                    if kda_renamed is not None:
                        self._assign(out, kda_renamed, weights)
                        break
                else:
                    kda_renamed = None
                if kda_renamed is not None:
                    continue

            # qkv_proj + z_proj are fused into HF `in_proj_qkvz`
            split = self._split_linear_attn_key(
                layer_name, "qkv_proj.weight", allow_bare=not is_full_softmax
            )
            if split is not None:
                base, target_prefix = split
                qkvz = self._maybe_accumulate_and_return_qkvz(
                    base, target_prefix, "qkv", weights
                )
                for k, v in qkvz.items():
                    self._assign(out, k, v)
                continue

            split = self._split_linear_attn_key(
                layer_name, "z_proj.weight", allow_bare=not is_full_softmax
            )
            if split is not None:
                base, target_prefix = split
                qkvz = self._maybe_accumulate_and_return_qkvz(
                    base, target_prefix, "z", weights
                )
                for k, v in qkvz.items():
                    self._assign(out, k, v)
                continue

            # qkvz_proj -> in_proj_qkvz
            split = self._split_linear_attn_key(
                layer_name, "qkvz_proj.weight", allow_bare=not is_full_softmax
            )
            if split is not None:
                base, target_prefix = split
                self._assign(
                    out, f"{base}.{target_prefix}.in_proj_qkvz.weight", weights
                )
                continue

            # ba_proj -> in_proj_ba
            split = self._split_linear_attn_key(
                layer_name, "ba_proj.weight", allow_bare=not is_full_softmax
            )
            if split is not None:
                base, target_prefix = split
                self._assign(out, f"{base}.{target_prefix}.in_proj_ba.weight", weights)
                continue

            # o_proj -> out_proj
            if not is_full_softmax:
                split = self._split_linear_attn_key(layer_name, "o_proj.weight")
                if split is not None:
                    base, target_prefix = split
                    self._assign(
                        out, f"{base}.{target_prefix}.out_proj.weight", weights
                    )
                    continue

            # o_norm -> norm
            final = self._map_linear_attn_norm_key(layer_name)
            if final is not None:
                self._assign(out, final, weights)
                continue

            # conv1d: `qkv_conv1d.*` arrives in the training-time interleaved
            # per-head layout; `conv1d.*` is already in conv order (produced
            # either by `pack_conv_weights` concatenating separate q/k/v convs,
            # or by the legacy path). Only the former needs reordering.
            split = self._split_linear_attn_key(
                layer_name, "conv1d.weight", allow_bare=not is_full_softmax
            )
            if split is None:
                split = self._split_linear_attn_key(
                    layer_name, "qkv_conv1d.weight", allow_bare=not is_full_softmax
                )
                if split is not None:
                    weights = self._interleaved_qkv_to_conv_order(weights)
            if split is not None:
                base, target_prefix = split
                self._assign(out, f"{base}.{target_prefix}.conv1d.weight", weights)
                continue

            # A_log / dt_bias keep their names.
            linear_param_handled = False
            for inner in ("A_log", "dt_bias"):
                split = self._split_linear_attn_key(
                    layer_name, inner, allow_bare=not is_full_softmax
                )
                if split is not None:
                    base, target_prefix = split
                    self._assign(out, f"{base}.{target_prefix}.{inner}", weights)
                    linear_param_handled = True
                    break
            if linear_param_handled:
                continue

            if "block_sparse_moe" not in layer_name:
                self._assign(out, layer_name, weights)
                continue

            # === MoE split ===
            # If checkpoint was trained with sonic kernel, convert weights to g_gemm convention first.
            # Both formats use the same 2D shapes: weight1 (H, 2*I*E), weight2 (I*E, H).
            # The difference is element ordering due to different 3D view conventions.
            if layer_name.endswith("block_sparse_moe.experts.weight2"):
                if self.use_sonic_moe_layout:
                    # sonic view(E, H, I) -> g_gemm view(E, I, H): per-expert transpose
                    E = self.n_experts
                    I = weights.shape[0] // E
                    H = weights.shape[1]
                    weights = weights.view(E, H, I).permute(0, 2, 1).contiguous().view(I * E, H)

                assert weights.ndim == 2, (
                    f"{layer_name}: expected fused 2D weight2 (I*E, H), got shape {tuple(weights.shape)}; "
                    "a 3D tensor means the checkpoint stores experts in the new weight layout "
                    "and was not normalized upstream (see use_new_expert_weight_layout handling in mappers.py)"
                )
                experts_ffn_dim, hidden_dim = weights.shape
                assert self.n_experts > 0, "meta.num_routed_experts must be > 0 for MoE split"
                ffn_dim = experts_ffn_dim // self.n_experts
                assert ffn_dim * self.n_experts == experts_ffn_dim, "not divisible experts in weight2"
                for i in range(self.n_experts):
                    w_i = weights[i * ffn_dim:(i + 1) * ffn_dim, :].t().contiguous()
                    final = layer_name.replace("block_sparse_moe.experts.weight2", f"mlp.experts.{i}.down_proj.weight")
                    self._assign(out, final, w_i)
                continue

            if layer_name.endswith("block_sparse_moe.experts.weight1"):
                if self.use_sonic_moe_layout:
                    # sonic view(E, 2*I, H) interleaved -> g_gemm view(E, H, 2*I) chunked
                    E = self.n_experts
                    H = weights.shape[0]
                    I = weights.shape[1] // (2 * E)
                    w = weights.view(E, 2 * I, H).permute(0, 2, 1).contiguous()  # (E, H, 2*I) interleaved
                    gate = w[:, :, 0::2]   # (E, H, I)
                    up   = w[:, :, 1::2]   # (E, H, I)
                    weights = torch.cat([gate, up], dim=-1).contiguous().view(H, 2 * I * E)  # chunked

                hidden_dim, total_ffn = weights.shape
                assert self.n_experts > 0, "meta.num_routed_experts must be > 0 for MoE split"
                D = total_ffn // (2 * self.n_experts)
                assert 2 * self.n_experts * D == total_ffn, "MoE weight1 not divisible"
                w1 = weights.view(self.n_experts, hidden_dim, 2 * D)
                for i in range(self.n_experts):
                    gate_part, up_part = w1[i].chunk(2, dim=-1)
                    self._assign(
                        out,
                        layer_name.replace("block_sparse_moe.experts.weight1", f"mlp.experts.{i}.gate_proj.weight"),
                        gate_part.t().contiguous(),
                    )
                    self._assign(
                        out,
                        layer_name.replace("block_sparse_moe.experts.weight1", f"mlp.experts.{i}.up_proj.weight"),
                        up_part.t().contiguous(),
                    )
                continue

            if "block_sparse_moe.post_self_attn_layernorm." in layer_name:
                final = layer_name.replace(
                    "block_sparse_moe.post_self_attn_layernorm.",
                    "post_self_attn_layernorm.",
                )
                self._assign(out, final, weights)
                continue

            if "block_sparse_moe.post_feedforward_layernorm." in layer_name:
                final = layer_name.replace(
                    "block_sparse_moe.post_feedforward_layernorm.",
                    "post_feedforward_layernorm.",
                )
                self._assign(out, final, weights)
                continue

            raise RuntimeError(f"Unexpected MoE key: {layer_name}")

        return out

    def flush_pending(self) -> Dict[str, torch.Tensor]:
        if self.kv_a_buf:
            missing = {k: list(v.keys()) for k, v in self.kv_a_buf.items()}
            raise RuntimeError(f"Unpaired kv_a parts remained at chunk end: {missing}")

        if self.kv_b_buf:
            missing = {k: list(v.keys()) for k, v in self.kv_b_buf.items()}
            raise RuntimeError(f"Unpaired kv_b parts remained at chunk end: {missing}")

        if self.qkvz_buf:
            missing = {k: list(v.keys()) for k, v in self.qkvz_buf.items()}
            raise RuntimeError(f"Unpaired qkvz parts remained at chunk end: {missing}")

        return {}



class _LoraDeepseekPerLayerTransformer(_DeepseekPerLayerTransformer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.SIMPLE_REPLACEMENTS = {
            ".self_attn._attn.q_proj.": ".self_attn.q_proj.",
            ".self_attn._attn.dq_proj.": ".self_attn.q_a_proj.",
            ".self_attn._attn.q_norm.": ".self_attn.q_a_layernorm.",
            ".self_attn._attn.uq_proj.": ".self_attn.q_b_proj.",
            ".self_attn._attn.o_proj.": ".self_attn.o_proj.",
            ".self_attn._attn.kv_norm.": ".self_attn.kv_a_layernorm.",
            ".self_attn.input_layernorm.": ".input_layernorm.",
            ".self_attn.post_attention_layernorm.": ".post_attention_layernorm.",
            ".self_attn.mlp.": ".mlp.",
            ".self_attn.mlp.down_proj.": ".mlp.down_proj.",
            ".self_attn.mlp.gate_proj.": ".mlp.gate_proj.",
            ".self_attn.mlp.up_proj.": ".mlp.up_proj.",
            ".self_attn.shared_experts.": ".mlp.shared_experts.",
            ".self_attn.shared_experts.gate_proj.": ".mlp.shared_experts.gate_proj.",
            ".self_attn.shared_experts.up_proj.": ".mlp.shared_experts.up_proj.",
            ".self_attn.shared_experts.down_proj.": ".mlp.shared_experts.down_proj.",
            ".self_attn._gate.gate.": ".mlp.gate.",
        }
        
        self.PROJ_SUFFIXES = [
            ".weight",
            ".lora_A.default.weight", 
            ".lora_B.default.weight",
            ".base_layer.weight"
        ]

        # KV буферные паттерны
        self.KV_A_PATTERNS = {
            ".self_attn._attn.dkv_proj.weight": "dkv",
            ".self_attn._attn.kr_proj.weight": "kr",
            ".self_attn._attn.dkv_proj.base_layer.weight": "dkv_base",
            ".self_attn._attn.dkv_proj.lora_A.default.weight": "dkv_lora_A",
            ".self_attn._attn.dkv_proj.lora_B.default.weight": "dkv_lora_B",
            ".self_attn._attn.kr_proj.base_layer.weight": "kr_base",
            ".self_attn._attn.kr_proj.lora_A.default.weight": "kr_lora_A",
            ".self_attn._attn.kr_proj.lora_B.default.weight": "kr_lora_B"
        }


        self.KV_B_PATTERNS = {
            ".self_attn._attn.uv_proj.base_layer.weight": "uv_base",
            ".self_attn._attn.uv_proj.lora_A.default.weight": "uv_lora_A",
            ".self_attn._attn.uv_proj.lora_B.default.weight": "uv_lora_B",
            ".self_attn._attn.uk_proj.base_layer.weight": "uk_base",
            ".self_attn._attn.uk_proj.lora_A.default.weight": "uk_lora_A",
            ".self_attn._attn.uk_proj.lora_B.default.weight": "uk_lora_B",
            ".self_attn._attn.uv_proj.weight": "uv",
            ".self_attn._attn.uk_proj.weight": "uk"
        }

    @staticmethod
    def _normalize_super_input_key(layer_name: str) -> str:
        if layer_name.startswith("model.model."):
            return "model." + layer_name[len("model.model."):]
        return layer_name

    @staticmethod
    def _restore_super_output_key(layer_name: str, source_name: str) -> str:
        if source_name.startswith("model.model.") and layer_name.startswith("model.layers."):
            return "model." + layer_name
        return layer_name

    @staticmethod
    def _restore_base_layer_output_key(layer_name: str) -> str:
        assert layer_name.endswith(".weight"), f"Unexpected base MoE output key: {layer_name}"
        return layer_name[:-len(".weight")] + ".base_layer.weight"

    def _should_route_through_super(self, layer_name: str) -> bool:
        if "embed_tokens" in layer_name or "lm_head" in layer_name:
            return False

        if layer_name.endswith((
            ".base_layer.weight",
            ".lora_A.default.weight",
            ".lora_B.default.weight",
            "block_sparse_moe.experts.lora_A1",
            "block_sparse_moe.experts.lora_B1",
            "block_sparse_moe.experts.lora_A2",
            "block_sparse_moe.experts.lora_B2",
        )):
            return False

        return True

    def _route_single_weight_through_super(
        self,
        out: Dict[str, torch.Tensor],
        source_name: str,
        weights: torch.Tensor,
        *,
        synthetic_name: str | None = None,
        as_base_layer: bool = False,
    ) -> None:
        normalized_name = self._normalize_super_input_key(synthetic_name or source_name)
        routed = super().rename_and_split({normalized_name: weights})
        for key, tensor in routed.items():
            restored_key = self._restore_super_output_key(key, source_name)
            if as_base_layer:
                restored_key = self._restore_base_layer_output_key(restored_key)
            self._assign(out, restored_key, tensor)

    def _maybe_accumulate_and_return_kv_a(self, base: str, kind: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        buf = self.kv_a_buf.setdefault(base, {})
        buf[kind] = tensor

        pairs_mapping = {
            ("dkv", "kr"): "weight",
            ("dkv_base", "kr_base"): "base_layer.weight",
            ("dkv_lora_A", "kr_lora_A"): "lora_A.default.weight",
            ("dkv_lora_B", "kr_lora_B"): "lora_B.default.weight",
         }

        for (dkv_key, kr_key), suffix in pairs_mapping.items():
            if dkv_key in buf and kr_key in buf:
                dkv = buf[dkv_key]
                kr  = buf[kr_key]
                if suffix in ["weight", "base_layer.weight"]:
                    assert dkv.shape[1] == kr.shape[1], f"[kv_a] hidden_size mismatch for {base}: {dkv.shape} vs {kr.shape}"
                    assert kr.shape[0] == self.dk_rope, f"[kv_a] kr[0] must be qk_rope_head_dim={self.dk_rope}, got {kr.shape}"
                    fused = torch.cat([dkv, kr], dim=0).contiguous()
                elif suffix == "lora_A.default.weight":
                    fused = torch.cat([dkv, kr], dim=0).contiguous()
                else:
                    dkv_rows, dkv_cols = dkv.shape
                    kr_rows, kr_cols = kr.shape
                    fused = dkv.new_zeros((dkv_rows + kr_rows, dkv_cols + kr_cols))
                    fused[:dkv_rows, :dkv_cols] = dkv
                    fused[dkv_rows:, dkv_cols:] = kr
                    fused = fused.contiguous()

                out[base + ".self_attn.kv_a_proj_with_mqa." + suffix] = fused

                del buf[dkv_key]
                del buf[kr_key]
        # Очищаем буфер, если все пары обработаны
        if not buf:
            self.kv_a_buf.pop(base, None)
        
        return out

    def _maybe_accumulate_and_return_kv_b(self, base: str, kind: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        buf = self.kv_b_buf.setdefault(base, {})
        buf[kind] = tensor

        pairs_mapping = {
            ("uv", "uk"): "weight",
            ("uv_base", "uk_base"): "base_layer.weight",
            ("uv_lora_A", "uk_lora_A"): "lora_A.default.weight",
            ("uv_lora_B", "uk_lora_B"): "lora_B.default.weight",
        }

        for (uv_key, uk_key), suffix in pairs_mapping.items():
            if uv_key in buf and uk_key in buf:
                uv = buf[uv_key]
                uk = buf[uk_key]
                n_heads = self.n_heads
                dk_nope = self.dk_nope
                vdim    = self.v_head_dim
                r_kv    = uk.shape[1]
                if suffix in ["weight", "base_layer.weight"]:
                    assert uk.shape == (n_heads * dk_nope, r_kv), f"[kv_b.uk] shape {uk.shape} != {(n_heads * dk_nope, r_kv)}"
                    assert uv.shape == (n_heads * vdim,   r_kv), f"[kv_b.uv] shape {uv.shape} != {(n_heads * vdim,   r_kv)}"
                    fused = torch.cat([
                        uk.view(n_heads, dk_nope, r_kv),
                        uv.view(n_heads, vdim,   r_kv)
                    ], dim=1).reshape(n_heads * (dk_nope + vdim), r_kv).contiguous()
                elif suffix == "lora_A.default.weight":
                    fused = torch.cat([uk, uv], dim=0).contiguous()
                else:
                    uk_rows, uk_cols = uk.shape
                    uv_rows, uv_cols = uv.shape
                    fused = uk.new_zeros((uk_rows + uv_rows, uk_cols + uv_cols))
                    fused[:uk_rows, :uk_cols] = uk
                    fused[uk_rows:, uk_cols:] = uv
                    fused = fused.contiguous()
                
                out[base + ".self_attn.kv_b_proj." + suffix] = fused
                del buf[uv_key]
                del buf[uk_key]
        if not buf:
            self.kv_b_buf.pop(base, None)
        
        return out

    def rename_and_split(self, merged_layer: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for orig_name, weights in merged_layer.items():
            weights = weights.to(torch.bfloat16)

            if orig_name.endswith("block_sparse_moe.experts.base_layer.weight1"):
                synthetic_name = orig_name.replace(
                    "block_sparse_moe.experts.base_layer.weight1",
                    "block_sparse_moe.experts.weight1",
                )
                self._route_single_weight_through_super(
                    out,
                    orig_name,
                    weights,
                    synthetic_name=synthetic_name,
                    as_base_layer=True,
                )
                continue

            if orig_name.endswith("block_sparse_moe.experts.base_layer.weight2"):
                synthetic_name = orig_name.replace(
                    "block_sparse_moe.experts.base_layer.weight2",
                    "block_sparse_moe.experts.weight2",
                )
                self._route_single_weight_through_super(
                    out,
                    orig_name,
                    weights,
                    synthetic_name=synthetic_name,
                    as_base_layer=True,
                )
                continue

            if self._should_route_through_super(orig_name):
                self._route_single_weight_through_super(out, orig_name, weights)
                continue

            layer_name = orig_name
            processed = False
            if layer_name.startswith("model.model.mtp_block.mtp_layers."):
                m = re.match(r"model.model\.mtp_block\.mtp_layers\.(\d+)\.(.*)", layer_name)
                if m:
                    mtp_i = int(m.group(1))
                    tail = m.group(2)

                    if tail.startswith("decoder_layer."):
                        tail = tail[len("decoder_layer."):]

                    layer_id = self.base_mtp_idx + mtp_i
                    layer_name = f"model.model.layers.{layer_id}." + tail
                    layer_name = layer_name.replace(".self_attn.base_attn.", ".self_attn.")
                    layer_name = layer_name.replace(".self_attn.mtp_input_proj.", ".eh_proj.")
                    layer_name = layer_name.replace(".self_attn.mtp_token_layernorm.", ".enorm.")
                    layer_name = layer_name.replace(".self_attn.mtp_hidden_layernorm.", ".hnorm.")
                else:
                    raise RuntimeError(f"Unexpected MTP key: {layer_name}")

            # Обработка eh_proj
            if ".eh_proj." in layer_name:
                W = weights
                H = self.hidden_size
                if W.shape[1] == 2 * H:
                    left, right = W[:, :H], W[:, H:]
                    W = torch.cat([right, left], dim=1).contiguous()
                elif W.shape[0] == 2 * H:
                    top, bottom = W[:H, :], W[H:, :]
                    W = torch.cat([bottom, top], dim=0).contiguous()
                else:
                    raise ValueError(f"eh_proj weight has shape {tuple(W.shape)}")
                self._assign(out, layer_name, W)
                continue

            # Обработка MTP norms
            if layer_name.startswith("model.model.mtp_block.mtp_norms."):
                m = re.match(r"model.model\.mtp_block\.mtp_norms\.(\d+)\.(.*)", layer_name)
                if m:
                    mtp_i = int(m.group(1))
                    layer_id = self.base_mtp_idx + mtp_i
                    final = f"model.model.layers.{layer_id}.shared_head.norm.weight"
                else:
                    raise RuntimeError(f"Unexpected MTP key: {layer_name}")
                self._assign(out, final, weights)
                continue

            final = self._map_linear_attn_norm_key(layer_name)
            if final is not None:
                self._assign(out, final, weights)
                continue

            # Обработка простых замен
            for old_pattern, new_pattern in self.SIMPLE_REPLACEMENTS.items():
                old_base = old_pattern[:-1]   
                new_base = new_pattern[:-1]   
            
                for suffix in self.PROJ_SUFFIXES:
                    if layer_name.endswith(old_base + suffix):
                        final = layer_name.replace(old_base, new_base)
                        self._assign(out, final, weights)
                        processed = True
                        break           
                if processed:
                    break            
            if processed:
                continue
                    
            # Обработка KV_A буфера
            for pattern, key in self.KV_A_PATTERNS.items():
                if layer_name.endswith(pattern):
                    base = layer_name.split(pattern)[0]
                    kv = self._maybe_accumulate_and_return_kv_a(base, key, weights)
                    for k, v in kv.items():
                        self._assign(out, k, v)
                    processed = True
                    break
            if processed:
                continue

            # Обработка KV_B буфера
            for pattern, key in self.KV_B_PATTERNS.items():
                if layer_name.endswith(pattern):
                    base = layer_name.split(pattern)[0]
                    kv = self._maybe_accumulate_and_return_kv_b(base, key, weights)
                    for k, v in kv.items():
                        self._assign(out, k, v)
                    processed = True
                    break
            if processed:
                continue
            # Обработка expert_bias
            if layer_name.endswith((".self_attn._gate.expert_bias", ".self_attn.base_attn._gate.expert_bias")):
                final = layer_name.replace(".self_attn._gate.expert_bias", ".mlp.gate.e_score_correction_bias")
                final = final.replace(".self_attn.base_attn._gate.expert_bias", ".mlp.gate.e_score_correction_bias")
                self._assign(out, final, weights)
                continue

            # Обработка embed_tokens
            if "embed_tokens" in layer_name:
                self._assign(out, layer_name, weights)
                for mtp_i in range(self.num_mtp):
                    layer_id = self.base_mtp_idx + mtp_i
                    embed_name = f"model.layers.{layer_id}.embed_tokens.weight"
                    self._assign(out, embed_name, weights.contiguous().clone())
                continue

            # Обработка lm_head
            if "lm_head" in layer_name:
                self._assign(out, layer_name, weights)
                for mtp_i in range(self.num_mtp):
                    layer_id = self.base_mtp_idx + mtp_i
                    head_name = f"model.layers.{layer_id}.shared_head.head.weight"
                    self._assign(out, head_name, weights.contiguous().clone())
                continue

            # Обработка остальных слоев (не block_sparse_moe)
            if "block_sparse_moe" not in layer_name:
                self._assign(out, layer_name, weights)
                continue

             # === MoE split ===
            if layer_name.endswith("block_sparse_moe.experts.weight2"):
                assert weights.ndim == 2, (
                    f"{layer_name}: expected fused 2D weight2 (I*E, H), got shape {tuple(weights.shape)}; "
                    "a 3D tensor means the checkpoint stores experts in the new weight layout "
                    "and was not normalized upstream (see use_new_expert_weight_layout handling in mappers.py)"
                )
                experts_ffn_dim, hidden_dim = weights.shape
                assert self.n_experts > 0, "meta.num_routed_experts must be > 0 for MoE split"
                ffn_dim = experts_ffn_dim // self.n_experts
                assert ffn_dim * self.n_experts == experts_ffn_dim, "not divisible experts in weight2"
                for i in range(self.n_experts):
                    w_i = weights[i * ffn_dim:(i + 1) * ffn_dim, :].t().contiguous()
                    final = layer_name.replace("block_sparse_moe.experts.weight2", f"mlp.experts.{i}.down_proj.weight")
                    self._assign(out, final, w_i)
                continue

            if layer_name.endswith("block_sparse_moe.experts.weight1"):
                hidden_dim, total_ffn = weights.shape
                assert self.n_experts > 0, "meta.num_routed_experts must be > 0 for MoE split"
                D = total_ffn // (2 * self.n_experts)
                assert 2 * self.n_experts * D == total_ffn, "MoE weight1 not divisible"
                w1 = weights.view(self.n_experts, hidden_dim, 2 * D)
                for i in range(self.n_experts):
                    gate_part, up_part = w1[i].chunk(2, dim=-1)
                    self._assign(
                        out,
                        layer_name.replace("block_sparse_moe.experts.weight1", f"mlp.experts.{i}.gate_proj.weight"),
                        gate_part.t().contiguous(),
                    )
                    self._assign(
                        out,
                        layer_name.replace("block_sparse_moe.experts.weight1", f"mlp.experts.{i}.up_proj.weight"),
                        up_part.t().contiguous(),
                    )
                continue

            # === Expert LoRA: lora_A1 / lora_B1 (gate_proj + up_proj) ===
            # Expert LoRA is stored in NT layout for GroupedLlamaMLP:
            # lora_A1: [E, r, H], lora_B1: [E, 2*D, r]
            # HF/PEFT nn.Linear expects A [r, in_features], B [out_features, r],
            # so no extra transpose is needed here.
            if layer_name.endswith("block_sparse_moe.experts.lora_A1"):
                E, r, H = weights.shape
                assert E == self.n_experts
                for i in range(self.n_experts):
                    lora_a = weights[i].contiguous()  # [r, H]
                    self._assign(
                        out,
                        layer_name.replace("block_sparse_moe.experts.lora_A1", f"mlp.experts.{i}.gate_proj.lora_A.default.weight"),
                        lora_a,
                    )
                    self._assign(
                        out,
                        layer_name.replace("block_sparse_moe.experts.lora_A1", f"mlp.experts.{i}.up_proj.lora_A.default.weight"),
                        lora_a.clone(),
                    )
                continue

            # lora_B1: [E, 2*D, r] → split into gate [D, r] and up [D, r]
            if layer_name.endswith("block_sparse_moe.experts.lora_B1"):
                E, two_D, r = weights.shape
                assert E == self.n_experts
                D = two_D // 2
                assert 2 * D == two_D, "lora_B1 last dim must be 2*D"
                for i in range(self.n_experts):
                    gate_B = weights[i, :D, :].contiguous()  # [D, r]
                    up_B = weights[i, D:, :].contiguous()    # [D, r]
                    self._assign(
                        out,
                        layer_name.replace("block_sparse_moe.experts.lora_B1", f"mlp.experts.{i}.gate_proj.lora_B.default.weight"),
                        gate_B,
                    )
                    self._assign(
                        out,
                        layer_name.replace("block_sparse_moe.experts.lora_B1", f"mlp.experts.{i}.up_proj.lora_B.default.weight"),
                        up_B,
                    )
                continue

            # === Expert LoRA: lora_A2 / lora_B2 (down_proj) ===
            # lora_A2: [E, r, D] → per-expert [r, D]
            if layer_name.endswith("block_sparse_moe.experts.lora_A2"):
                E, r, D = weights.shape
                assert E == self.n_experts
                for i in range(self.n_experts):
                    lora_a = weights[i].contiguous()  # [r, D]
                    self._assign(
                        out,
                        layer_name.replace("block_sparse_moe.experts.lora_A2", f"mlp.experts.{i}.down_proj.lora_A.default.weight"),
                        lora_a,
                    )
                continue

            # lora_B2: [E, H, r] → per-expert [H, r]
            if layer_name.endswith("block_sparse_moe.experts.lora_B2"):
                E, H, r = weights.shape
                assert E == self.n_experts
                for i in range(self.n_experts):
                    lora_b = weights[i].contiguous()  # [H, r]
                    self._assign(
                        out,
                        layer_name.replace("block_sparse_moe.experts.lora_B2", f"mlp.experts.{i}.down_proj.lora_B.default.weight"),
                        lora_b,
                    )
                continue

            raise RuntimeError(f"Unexpected MoE key: {layer_name}")

        return out
