# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.tools.logging import logger

from .model import BaseModel


class BaseStateDictAdapter(ABC):
    """Abstract base class for state dict transformations.

    This class defines the interface for converting between native model
    state dict format and other model state dict formats.
    Args:
        model_config: for initializing the model's memory space
        hf_assets_path: path to HF assets folder containing tokenizer, model weights, etc.
    """

    fqn_to_index_mapping: dict[Any, int] | None
    hf_assets_path: str | None

    @abstractmethod
    def __init__(
        self,
        model_config: BaseModel.Config,
        hf_assets_path: str | None,
    ):
        pass

    @abstractmethod
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.

        Args:
            state_dict: The native model state dict

        Returns:
            The converted HuggingFace format state dict
        """
        pass

    @abstractmethod
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Obtain native model state dict from HuggingFace format.

        Args:
            hf_state_dict: The HuggingFace format state dict

        Returns:
            The converted native model state dict
        """
        pass

    @abstractmethod
    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        """Returns hf storage reader to read HF checkpoint

        Args:
            path: the path to read HF checkpoint

        Returns:
            The HuggingFace storage reader to read from HF checkpoint

        """
        pass

    def check_hf_export(self, state_dict: dict[str, Any]) -> None:
        """Raise if an HF export would drop or reshape an entry of ``state_dict``.

        Round-trips the keys through ``to_hf``/``from_hf`` on meta tensors, so
        the key tables are the only source of truth and no weights are touched.
        Loading from HF tolerates such entries (they keep their init); an
        export must not lose them.
        """
        meta = {
            k: torch.empty(v.shape, dtype=v.dtype, device="meta")
            for k, v in state_dict.items()
            # A legacy RoPE cache, rebuilt from config.
            if not k.endswith(".rotary_emb.inv_freq")
        }
        try:
            back = self.from_hf(self.to_hf(dict(meta)))
        except Exception as exc:
            raise ValueError(
                f"HF export cannot map this model's state: {type(exc).__name__}: {exc}"
            ) from exc
        lost = [k for k, v in meta.items() if k not in back or back[k].shape != v.shape]
        if lost:
            # The first entry of each kind, layer index folded.
            kinds: dict[str, str] = {}
            for key in lost:
                kinds.setdefault(re.sub(r"\.\d+\.", ".{}.", key, count=1), key)
            shown = list(kinds.values())[:8]
            more = (
                f" (+{len(lost) - len(shown)} more)" if len(lost) > len(shown) else ""
            )
            raise ValueError(
                f"HF export cannot represent {len(lost)} state entries: "
                f"{', '.join(shown)}{more}. Merge LoRA adapters, disable "
                "expert-bias load balancing (load_balance_coeff=0) and drop "
                "heads the HF format lacks (e.g. MTP) before an HF export, or "
                "use native checkpoints."
            )

    @staticmethod
    def hf_config(model_config: BaseModel.Config) -> dict[str, Any]:
        """The ``config.json`` describing what ``to_hf`` writes."""
        raise NotImplementedError(
            f"{model_config.__class__.__qualname__} has no HF config writer: "
            "implement hf_config() on its state dict adapter to export a "
            "directory a serving stack can resolve."
        )

    @staticmethod
    def default_hf_remote_code_dir() -> str | None:
        """Where the remote-code files live when the user configured no dir."""
        return None


class StateDictAdapter(BaseStateDictAdapter):
    """State dict adapter base class which provides convenient default behavior to build fqn_to_index_mapping"""

    def __init__(
        self,
        model_config: BaseModel.Config,
        hf_assets_path: str | None,
    ):
        self.hf_assets_path = hf_assets_path
        if hf_assets_path:
            mapping_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
            try:
                with open(mapping_path, "r") as f:
                    hf_safetensors_indx = json.load(f)
            except FileNotFoundError:
                logger.warning(
                    f"model.safetensors.index.json not found at hf_assets_path: {mapping_path}. \
                    Defaulting to saving a single safetensors file if checkpoint is saved in HF format"
                )
                hf_safetensors_indx = None

            if hf_safetensors_indx:
                self.fqn_to_index_mapping = {}
                for hf_key, raw_indx in hf_safetensors_indx["weight_map"].items():
                    # pyrefly: ignore [missing-attribute]
                    indx = re.search(r"\d+", raw_indx).group(0)
                    self.fqn_to_index_mapping[hf_key] = int(indx)
            else:
                self.fqn_to_index_mapping = None
        else:
            self.fqn_to_index_mapping = None

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            logger.warning(
                "Loading from quantized checkpoint format is not supported for this model."
            )
        return HuggingFaceStorageReader(path)

    @staticmethod
    def fused_to_separate_qkv(
        fused_weight: torch.Tensor,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split fused wqkv weight [n_kv * R * hd, dim] into separate Q, K, V."""
        heads_per_kv = n_heads // n_kv_heads
        r_dim = heads_per_kv + 2
        dim = fused_weight.shape[1]
        # [n_kv_heads, R, head_dim, dim]
        w = fused_weight.view(n_kv_heads, r_dim, head_dim, dim)
        wq = w[:, :heads_per_kv, :, :].reshape(n_heads * head_dim, dim)
        wk = w[:, heads_per_kv, :, :].reshape(n_kv_heads * head_dim, dim)
        wv = w[:, heads_per_kv + 1, :, :].reshape(n_kv_heads * head_dim, dim)
        return wq, wk, wv

    @staticmethod
    def separate_to_fused_qkv(
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Combine separate Q, K, V weights into fused wqkv layout."""
        heads_per_kv = n_heads // n_kv_heads
        r_dim = heads_per_kv + 2
        dim = wq.shape[1]
        # Reshape to per-KV-group: [n_kv_heads, heads_per_kv, head_dim, dim]
        q = wq.view(n_kv_heads, heads_per_kv, head_dim, dim)
        k = wk.view(n_kv_heads, 1, head_dim, dim)
        v = wv.view(n_kv_heads, 1, head_dim, dim)
        # [n_kv_heads, R, head_dim, dim]
        fused = torch.cat([q, k, v], dim=1)
        return fused.reshape(n_kv_heads * r_dim * head_dim, dim)

    @staticmethod
    def fused_to_separate_w13(
        fused_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split fused W13 [n_groups, 2, w_out, w_in] into W1 (gate) and W3 (up)."""
        if fused_weight.ndim != 4 or fused_weight.shape[1] != 2:
            raise ValueError(
                "W13 must have shape [num_experts, 2, hidden_dim, dim], "
                f"got {tuple(fused_weight.shape)}."
            )
        w1, w3 = torch.unbind(fused_weight, dim=1)
        return w1, w3

    @staticmethod
    def separate_to_fused_w13(
        w1: torch.Tensor,
        w3: torch.Tensor,
    ) -> torch.Tensor:
        """Stack W1 (gate) and W3 (up) [n_groups, w_out, w_in] into W13 [n_groups, 2, w_out, w_in]."""
        if w1.ndim != 3 or w3.ndim != 3:
            raise ValueError(
                "W1 and W3 must have shape [num_experts, hidden_dim, dim], "
                f"got {tuple(w1.shape)} and {tuple(w3.shape)}."
            )
        if w1.shape != w3.shape:
            raise ValueError(
                "W1 and W3 must have identical shapes, "
                f"got {tuple(w1.shape)} and {tuple(w3.shape)}."
            )
        if w1.dtype != w3.dtype:
            raise ValueError(
                f"W1 and W3 must have the same dtype, got {w1.dtype} and {w3.dtype}."
            )
        if w1.device != w3.device:
            raise ValueError(
                "W1 and W3 must be on the same device, "
                f"got {w1.device} and {w3.device}."
            )
        w13 = torch.stack([w1, w3], dim=1)
        return w13
