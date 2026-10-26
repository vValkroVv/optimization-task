from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from transformers import PretrainedConfig

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)
from sglang.srt.distributed import divide
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.utils import is_cpu

FULL_ATTENTION = "attention"
FULL_ATTENTION_ALIASES = {FULL_ATTENTION, "full_attention"}
LINEAR_ATTENTION = "linear_attention"

_CACHE_ATTR = "_sglang_hybrid_gdn_config_view"
_REQUIRED_LINEAR_ATTRS = (
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
)


def _validate_layer_id(layer_id: int, num_hidden_layers: int, field_name: str) -> None:
    if not 0 <= layer_id < num_hidden_layers:
        raise ValueError(
            f"{field_name} contains out-of-range layer id {layer_id} for "
            f"{num_hidden_layers} hidden layers."
        )


def _normalize_layer_types(config: PretrainedConfig) -> Optional[list[str]]:
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        return None

    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                "layer_types must have length equal to num_hidden_layers "
                f"({num_hidden_layers}), got {len(layer_types)}."
            )

        normalized_layer_types: list[str] = []
        has_linear_attention = False
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == LINEAR_ATTENTION:
                normalized_layer_types.append(LINEAR_ATTENTION)
                has_linear_attention = True
            elif layer_type in FULL_ATTENTION_ALIASES:
                normalized_layer_types.append(FULL_ATTENTION)
            else:
                raise ValueError(
                    f"Unsupported layer type {layer_type!r} at index {layer_idx}. "
                    f"Supported values are {sorted(FULL_ATTENTION_ALIASES | {LINEAR_ATTENTION})}."
                )
        return normalized_layer_types if has_linear_attention else None

    linear_layer_ids = getattr(config, "linear_layer_ids", None)
    full_attention_layer_ids = getattr(config, "full_attention_layer_ids", None)
    if linear_layer_ids is None and full_attention_layer_ids is None:
        return None

    normalized_layer_types = [FULL_ATTENTION] * num_hidden_layers
    has_linear_attention = False

    if full_attention_layer_ids is not None:
        explicit_full_attention_layer_ids = set(full_attention_layer_ids)
        for layer_id in explicit_full_attention_layer_ids:
            _validate_layer_id(layer_id, num_hidden_layers, "full_attention_layer_ids")
    else:
        explicit_full_attention_layer_ids = set(range(num_hidden_layers))

    if linear_layer_ids is not None:
        for layer_id in linear_layer_ids:
            _validate_layer_id(layer_id, num_hidden_layers, "linear_layer_ids")
            if layer_id in explicit_full_attention_layer_ids and full_attention_layer_ids is not None:
                raise ValueError(
                    f"Layer {layer_id} is present in both linear_layer_ids and "
                    "full_attention_layer_ids."
                )
            normalized_layer_types[layer_id] = LINEAR_ATTENTION
            has_linear_attention = True
    elif full_attention_layer_ids is not None:
        for layer_id in range(num_hidden_layers):
            if layer_id not in explicit_full_attention_layer_ids:
                normalized_layer_types[layer_id] = LINEAR_ATTENTION
                has_linear_attention = True

    return normalized_layer_types if has_linear_attention else None


def _validate_linear_attention_config(config: PretrainedConfig) -> None:
    missing_attrs = [
        attr_name for attr_name in _REQUIRED_LINEAR_ATTRS if not hasattr(config, attr_name)
    ]
    if missing_attrs:
        raise ValueError(
            "Hybrid GDN configuration is missing required linear-attention fields: "
            + ", ".join(sorted(missing_attrs))
        )


def _get_linear_num_key_heads(config: PretrainedConfig) -> int:
    if is_cpu():
        return getattr(
            config,
            "linear_num_key_heads_cpu",
            getattr(config, "linear_num_key_heads"),
        )
    return getattr(config, "linear_num_key_heads")


def _get_linear_num_value_heads(config: PretrainedConfig) -> int:
    if is_cpu():
        return getattr(
            config,
            "linear_num_value_heads_cpu",
            getattr(config, "linear_num_value_heads"),
        )
    return getattr(config, "linear_num_value_heads")


@dataclass(frozen=True)
class HybridGDNConfigView:
    config: PretrainedConfig
    layer_types: list[str]

    @property
    def linear_layer_ids(self) -> list[int]:
        return [
            layer_idx
            for layer_idx, layer_type in enumerate(self.layer_types)
            if layer_type == LINEAR_ATTENTION
        ]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            layer_idx
            for layer_idx, layer_type in enumerate(self.layer_types)
            if layer_type == FULL_ATTENTION
        ]

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams | KimiLinearCacheParams:
        num_key_heads = _get_linear_num_key_heads(self.config)
        num_value_heads = _get_linear_num_value_heads(self.config)
        if getattr(self.config, "linear_attention_type", None) == "KimiDeltaAttention":
            tp_world_size = get_attention_tp_size()
            key_dim = self.config.linear_key_head_dim * num_key_heads
            value_dim = self.config.linear_value_head_dim * num_value_heads
            conv_kernel = self.config.linear_conv_kernel_dim
            shape = KimiLinearStateShape(
                conv=[
                    (conv_kernel - 1, divide(key_dim, tp_world_size)),
                    (conv_kernel - 1, divide(key_dim, tp_world_size)),
                    (conv_kernel - 1, divide(value_dim, tp_world_size)),
                ],
                temporal=(
                    divide(num_value_heads, tp_world_size),
                    self.config.linear_key_head_dim,
                    self.config.linear_value_head_dim,
                ),
                num_heads=num_value_heads,
                head_dim=self.config.linear_value_head_dim,
                num_k_heads=num_key_heads,
                head_k_dim=self.config.linear_key_head_dim,
                conv_kernel=conv_kernel,
                num_spec=0,
            )
            return KimiLinearCacheParams(
                shape=shape,
                layers=self.linear_layer_ids,
                dtype=mamba2_state_dtype(self.config),
            )

        shape = Mamba2StateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=self.config.linear_value_head_dim * num_value_heads,
            n_groups=num_key_heads,
            num_heads=num_value_heads,
            head_dim=self.config.linear_value_head_dim,
            state_size=self.config.linear_key_head_dim,
            conv_kernel=self.config.linear_conv_kernel_dim,
        )

        return Mamba2CacheParams(
            shape=shape,
            layers=self.linear_layer_ids,
            dtype=mamba2_state_dtype(self.config),
        )


def get_hybrid_gdn_config_view(
    config: PretrainedConfig,
) -> Optional[HybridGDNConfigView]:
    if hasattr(config, _CACHE_ATTR):
        return getattr(config, _CACHE_ATTR)

    layer_types = _normalize_layer_types(config)
    if layer_types is None:
        setattr(config, _CACHE_ATTR, None)
        return None

    _validate_linear_attention_config(config)
    view = HybridGDNConfigView(config=config, layer_types=layer_types)
    setattr(config, _CACHE_ATTR, view)
    return view


def is_linear_attention_layer(config: PretrainedConfig, layer_id: int) -> bool:
    view = get_hybrid_gdn_config_view(config)
    return view is not None and view.layer_types[layer_id] == LINEAR_ATTENTION


def get_gigachat35_mtp_attention_layer_id(config: PretrainedConfig) -> int:
    return max(getattr(config, "num_hidden_layers", 0) - 1, 0)


def is_gigachat35_linear_attention_layer(
    config: PretrainedConfig,
    layer_id: int,
) -> bool:
    """Return True if layer_id uses linear (GDN) attention rather than full MLA.

    Priority order:
      1. ``layer_types`` list
      2. ``linear_layer_ids`` set
      3. ``full_attention_layer_ids`` set (complement)
      4. ``linear_attention_type`` + ``full_attention_layers`` (complement)
      5. Delegate to ``is_linear_attention_layer()`` for in-range layers
    """
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and layer_id < len(layer_types):
        return layer_types[layer_id] == "linear_attention"

    linear_layer_ids = getattr(config, "linear_layer_ids", None)
    if linear_layer_ids is not None:
        return layer_id in set(linear_layer_ids)

    full_attention_layer_ids = getattr(config, "full_attention_layer_ids", None)
    if full_attention_layer_ids is not None:
        return layer_id not in set(full_attention_layer_ids)

    linear_attention_type = getattr(config, "linear_attention_type", None)
    if linear_attention_type is not None:
        full_attention_layers = set(getattr(config, "full_attention_layers", []))
        return layer_id not in full_attention_layers

    if layer_id < getattr(config, "num_hidden_layers", 0):
        return is_linear_attention_layer(config, layer_id)

    return False


def gigachat35_mtp_uses_linear_attention(config: PretrainedConfig) -> bool:
    """Return True if the MTP (NextN) block uses linear (GDN) attention."""
    flag = getattr(config, "nextn_is_linear", None)
    if flag is not None:
        return bool(flag) and getattr(config, "linear_attention_type", None) is not None

    return is_gigachat35_linear_attention_layer(
        config, get_gigachat35_mtp_attention_layer_id(config)
    )
