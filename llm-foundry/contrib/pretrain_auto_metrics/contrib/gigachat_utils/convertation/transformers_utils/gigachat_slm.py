from __future__ import annotations

import re
from typing import Dict, Optional

import torch

from transformers_utils.deepseek import _get_bool, _get_int


class _GigachatSLMPerLayerTransformer:
    """llm-foundry SLM (`LlamaAttention`, dense MTP) checkpoint keys -> `gigachat_slm` HF keys, one chunk at a time."""

    _MTP_LAYER_RE = re.compile(r"^model\.mtp_block\.mtp_layers\.(\d+)\.(.*)$")
    _MTP_NORM_RE = re.compile(r"^model\.mtp_block\.mtp_norms\.(\d+)\.(.*)$")

    _ATTN_INNER_MODULES = frozenset({
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
        "q_norm", "k_norm",
        "q_sconv", "k_sconv", "v_sconv", "attn_out_sconv",
    })

    _LAYER_RENAMES = (
        (".self_attn.input_layernorm.", ".input_layernorm."),
        (".self_attn.post_attention_layernorm.", ".post_attention_layernorm."),
        (".self_attn.mlp.", ".mlp."),
    )

    _MTP_ADAPTER_RENAMES = (
        ("self_attn.mtp_input_proj.", "eh_proj."),
        ("self_attn.mtp_token_layernorm.", "enorm."),
        ("self_attn.mtp_hidden_layernorm.", "hnorm."),
    )

    def __init__(self, cfg):
        self.hidden_size = _get_int(cfg, "hidden_size", default=0)
        self.n_heads = _get_int(cfg, "num_attention_heads", default=0)
        self.n_kv_heads = _get_int(cfg, "num_key_value_heads", default=self.n_heads)
        self.head_dim = _get_int(cfg, "head_dim", default=self.hidden_size // max(self.n_heads, 1))
        self.base_mtp_idx = _get_int(cfg, "num_hidden_layers", default=0)
        self.tie_word_embeddings = _get_bool(cfg, "tie_word_embeddings", default=False)

        missing = [
            name for name, value in (
                ("hidden_size", self.hidden_size),
                ("num_attention_heads", self.n_heads),
                ("num_hidden_layers", self.base_mtp_idx),
            ) if value <= 0
        ]
        if missing:
            raise ValueError(f"Config lacks required dims: {', '.join(missing)}")

    def rename_and_split(self, merged_layer: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, weight in merged_layer.items():
            weight = weight.to(torch.bfloat16)
            for key, value in self._map_key(name, weight).items():
                if "._attn." in key or "base_attn." in key:
                    raise ValueError(
                        f"Unmapped attention wrapper key {name!r} -> {key!r}; "
                        "extend _GigachatSLMPerLayerTransformer"
                    )
                out[key] = value

        if self.tie_word_embeddings and "lm_head.weight" in out:
            embedding = out.get("model.embed_tokens.weight")
            if embedding is None:
                raise ValueError(
                    "Cannot export tied embeddings: model.embed_tokens.weight "
                    "must be in the same chunk as lm_head.weight."
                )
            if not torch.equal(embedding, out["lm_head.weight"]):
                raise ValueError(
                    "Cannot export tied embeddings: model.embed_tokens.weight and "
                    "lm_head.weight differ after conversion to bfloat16."
                )
            del out["lm_head.weight"]
        return out

    def flush_pending(self) -> Dict[str, torch.Tensor]:
        return {}

    def _map_key(self, name: str, weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        mtp_layer = self._MTP_LAYER_RE.match(name)
        if mtp_layer:
            return self._map_mtp_layer_key(int(mtp_layer.group(1)), mtp_layer.group(2), weight)

        mtp_norm = self._MTP_NORM_RE.match(name)
        if mtp_norm:
            layer_id = self.base_mtp_idx + int(mtp_norm.group(1))
            return {f"model.layers.{layer_id}.shared_head.norm.{mtp_norm.group(2)}": weight}

        if name in ("lm_head.weight", "model.lm_head.weight"):
            return {"lm_head.weight": weight}

        for old, new in self._LAYER_RENAMES:
            if old in name:
                return {name.replace(old, new): weight}

        if ".self_attn." in name:
            mapped = self._map_attention_key(name.replace(".self_attn._attn.", ".self_attn."), weight)
            return mapped if mapped is not None else {name: weight}

        return {name: weight}

    def _map_mtp_layer_key(self, mtp_i: int, tail: str, weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        if tail.startswith("decoder_layer."):
            tail = tail[len("decoder_layer."):]
        layer_prefix = f"model.layers.{self.base_mtp_idx + mtp_i}"

        for training_name, hf_name in self._MTP_ADAPTER_RENAMES:
            if tail.startswith(training_name):
                param = tail[len(training_name):]
                if hf_name == "eh_proj." and param == "weight":
                    weight = self._swap_eh_proj_halves(weight)
                return {f"{layer_prefix}.{hf_name}{param}": weight}

        name = f"{layer_prefix}.{tail}"
        if ".self_attn." in name:
            unwrapped = name.replace(".self_attn.base_attn.", ".self_attn.").replace(".self_attn._attn.", ".self_attn.")
            mapped = self._map_attention_key(unwrapped, weight)
            return mapped if mapped is not None else {name: weight}

        return {name: weight}

    def _map_attention_key(self, name: str, weight: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """`<layer>.self_attn.<inner>.<param>` with the wrapper already removed; None for an unknown inner module."""
        inner = name.split(".self_attn.", 1)[1].split(".", 1)[0]
        if inner == "qkv_proj":
            return self._split_fused_qkv(name, weight)
        if inner in self._ATTN_INNER_MODULES:
            return {name: weight}
        return None

    def _split_fused_qkv(self, name: str, weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        if weight.shape[0] != q_size + 2 * kv_size:
            raise ValueError(
                f"{name}: expected fused qkv with {q_size + 2 * kv_size} rows "
                f"(heads={self.n_heads}, kv_heads={self.n_kv_heads}, head_dim={self.head_dim}), "
                f"got {tuple(weight.shape)}"
            )
        prefix, param = name.rsplit(".qkv_proj.", 1)
        parts = torch.split(weight, [q_size, kv_size, kv_size], dim=0)
        return {f"{prefix}.{proj}_proj.{param}": part.contiguous() for proj, part in zip("qkv", parts)}

    def _swap_eh_proj_halves(self, weight: torch.Tensor) -> torch.Tensor:
        """llm-foundry feeds mtp_input_proj with [prev_hidden, embedding]; sglang DeepSeek-style eh_proj expects [embedding, prev_hidden]."""
        hidden = self.hidden_size
        if weight.shape[1] == 2 * hidden:
            prev_hidden, embedding = weight[:, :hidden], weight[:, hidden:]
            return torch.cat([embedding, prev_hidden], dim=1).contiguous()
        if weight.shape[0] == 2 * hidden:
            prev_hidden, embedding = weight[:hidden, :], weight[hidden:, :]
            return torch.cat([embedding, prev_hidden], dim=0).contiguous()
        raise ValueError(
            f"MTP input projection has shape {tuple(weight.shape)}, "
            f"expected one dimension to be 2 * hidden_size={2 * hidden}"
        )
