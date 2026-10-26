# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This script is adapted from torchtitan/models/llama3/model/state_dict_adapter.py.

We can use this script to adapt the checkpoint from HF to the format that we can load into the torchtitan model and vice versa.
This can enable us to do a parity test with the HF implementation and make sure that our results are
aligned with the HF implementation.

"""

import re
from typing import Any

import torch

from torchtitan.config.schema.components.lora import unwrap_frozen

from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    GQAttentionConfig,
)
from torchtitan.config.schema.models.common.decoder_blocks.layer_group import (
    placed_configs,
)

from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.models.utils import MoEStateDictAdapter
from .model import Qwen3Model


class Qwen3StateDictAdapter(MoEStateDictAdapter[Qwen3Model.Config]):
    def __init__(self, model_config: Qwen3Model.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.attention_config = placed_configs(
            [model_config.placed_layers()[0].attention],
            GQAttentionConfig,
            path="layers.0.attention",
        )[0]
        self.fuse_qkv = isinstance(
            unwrap_frozen(self.attention_config.qkv_linear), FusedQKVLinear.Config
        )

        qkv_map: dict[str, str | None]
        if self.fuse_qkv:
            qkv_map = {
                "model.layers.{}.self_attn.q_proj.weight": None,
                "model.layers.{}.self_attn.k_proj.weight": None,
                "model.layers.{}.self_attn.v_proj.weight": None,
            }
        else:
            qkv_map = {
                "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
                "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
                "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            }

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention module
            **qkv_map,
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # MLP module for non-MoE
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Transformer layer
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1_EFD",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3_EFD",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2_EDF",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def _get_attention_dims(self) -> tuple[int, int, int]:
        """Return (n_heads, n_kv_heads, head_dim) from model config."""
        attn = self.attention_config
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
        head_dim = (
            attn.head_dim
            if attn.head_dim is not None
            else self.model_config.dim // n_heads
        )
        return n_heads, n_kv_heads, head_dim

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Split the GroupedExperts' weight into separate expert's wegiht.
        """
        if self.fuse_qkv:
            to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
            n_heads, n_kv_heads, head_dim = self._get_attention_dims()
        else:
            to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                hf_state_dict.update(
                    self._grouped_experts_to_hf(
                        value,
                        abstract_key,
                        to_hf_map,
                        layer_num,
                        self._num_routed_experts(),
                    )
                )

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)

                if (
                    self.fuse_qkv
                    and abstract_key == "layers.{}.attention.qkv_linear.wqkv.weight"
                ):
                    wq, wk, wv = self.fused_to_separate_qkv(
                        value,
                        n_heads,  # pyrefly: ignore [unbound-name]
                        n_kv_heads,  # pyrefly: ignore [unbound-name]
                        head_dim,  # pyrefly: ignore [unbound-name]
                    )
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.q_proj.weight"
                    ] = wq
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.k_proj.weight"
                    ] = wk
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.v_proj.weight"
                    ] = wv
                    continue

                if abstract_key not in to_hf_map:
                    continue
                new_key = to_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value

            else:
                if key not in to_hf_map:
                    continue
                if self.model_config.enable_weight_tying and key == "lm_head.weight":
                    continue
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Concate separate expert's wegiht into GroupedExperts' weight.
        """

        state_dict = {}
        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}
        # Collect Q/K/V per layer for fusing (only used when fuse_qkv=True)
        pending_qkv: dict[str, dict[str, torch.Tensor]] = {}

        if self.fuse_qkv:
            n_heads, n_kv_heads, head_dim = self._get_attention_dims()

        if (
            self.model_config.enable_weight_tying
            and "lm_head.weight" not in hf_state_dict
        ):
            assert "model.embed_tokens.weight" in hf_state_dict
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        for key, value in hf_state_dict.items():
            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)
                titan_abstract_key = self.from_hf_map[abstract_key]
                assert titan_abstract_key is not None
                converted = self._grouped_experts_from_hf(
                    expert_weights_by_layer,
                    titan_abstract_key,
                    layer_num,
                    expert_num,
                    value,
                    self._num_routed_experts(),
                )
                if converted is not None:
                    new_key, new_value = converted
                    state_dict[new_key] = new_value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)

                if self.fuse_qkv and abstract_key in (
                    "model.layers.{}.self_attn.q_proj.weight",
                    "model.layers.{}.self_attn.k_proj.weight",
                    "model.layers.{}.self_attn.v_proj.weight",
                ):
                    if layer_num not in pending_qkv:
                        pending_qkv[layer_num] = {}
                    proj = abstract_key.split(".")[-2]  # q_proj, k_proj, v_proj
                    pending_qkv[layer_num][proj] = value
                    if len(pending_qkv[layer_num]) == 3:
                        fused = self.separate_to_fused_qkv(
                            pending_qkv[layer_num]["q_proj"],
                            pending_qkv[layer_num]["k_proj"],
                            pending_qkv[layer_num]["v_proj"],
                            n_heads,  # pyrefly: ignore [unbound-name]
                            n_kv_heads,  # pyrefly: ignore [unbound-name]
                            head_dim,  # pyrefly: ignore [unbound-name]
                        )
                        state_dict[
                            f"layers.{layer_num}.attention.qkv_linear.wqkv.weight"
                        ] = fused
                        del pending_qkv[layer_num]
                    continue

                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value

            else:
                new_key = self.from_hf_map[key]
                if new_key is None:
                    continue
                state_dict[new_key] = value

        if self.fuse_qkv and pending_qkv:
            raise ValueError(
                f"Incomplete Q/K/V projections for layers: {list(pending_qkv.keys())}"
            )

        self._validate_grouped_experts_from_hf(expert_weights_by_layer)
        return state_dict
