from __future__ import annotations
from typing import Dict
import torch

from transformers_utils.deepseek import _DeepseekPerLayerTransformer, _get_int


class _Gigachat35DensePerLayerTransformer(_DeepseekPerLayerTransformer):
    """Dense GQA variant: splits the fused q/k/v projection, the rest of the layout is shared with the parent."""

    def _init_full_attention_dims(self, cfg) -> None:
        self.n_kv_heads = _get_int(cfg, "num_key_value_heads", default=self.n_heads)
        self.head_dim = _get_int(cfg, "head_dim", default=self.hidden_size // self.n_heads)

    def _is_fused_gqa_qkv(self, layer_name: str) -> bool:
        # The dense MTP block (`base_attn`) is always full attention, `_validate_dense_gqa` rejects a linear last layer
        if layer_name.endswith(".self_attn.base_attn.qkv_proj.weight"):
            return True
        return layer_name.endswith(".self_attn._attn.qkv_proj.weight") and self._is_full_attention_layer(layer_name)

    def _split_fused_qkv(self, layer_name: str, weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        if weight.shape[0] != q_size + 2 * kv_size:
            raise ValueError(
                f"{layer_name}: expected fused qkv with {q_size + 2 * kv_size} rows "
                f"(heads={self.n_heads}, kv_heads={self.n_kv_heads}, head_dim={self.head_dim}), "
                f"got {tuple(weight.shape)}"
            )
        base = layer_name.rsplit(".self_attn.", 1)[0]
        parts = torch.split(weight, [q_size, kv_size, kv_size], dim=0)
        return {f"{base}.self_attn.{name}_proj.weight": part.contiguous() for name, part in zip("qkv", parts)}

    def rename_and_split(self, merged_layer: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        unfused: Dict[str, torch.Tensor] = {}
        for layer_name, weight in merged_layer.items():
            if self._is_fused_gqa_qkv(layer_name):
                unfused.update(self._split_fused_qkv(layer_name, weight))
            else:
                unfused[layer_name] = weight
        return super().rename_and_split(unfused)
