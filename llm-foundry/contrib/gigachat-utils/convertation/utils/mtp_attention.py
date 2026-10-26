from __future__ import annotations

from collections.abc import Mapping
from typing import Any

HF_FIELD = "nextn_is_linear"
HF_SPARSE_FIELD = "nextn_is_sparse"
LLMFOUNDRY_FIELD = "use_linear_attention_in_mtp"

__all__ = [
    "HF_FIELD",
    "HF_SPARSE_FIELD",
    "LLMFOUNDRY_FIELD",
    "mtp_attention_anchor_layer_idx",
    "resolve_grafted_mtp_fields",
    "resolve_mtp_uses_linear_attention",
    "resolve_mtp_uses_sparse_ffn",
]


def _get(cfg: Any, key: str, default=None):
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def mtp_attention_anchor_layer_idx(cfg: Any) -> int:
    return max(int(_get(cfg, "num_hidden_layers") or 0) - 1, 0)


def resolve_mtp_uses_linear_attention(cfg: Any) -> bool:
    linear_attention_type = _get(cfg, "linear_attention_type")
    flag = _get(cfg, HF_FIELD)
    if flag is None:
        flag = _get(cfg, LLMFOUNDRY_FIELD)
    if flag is not None:
        if bool(flag) and linear_attention_type is None:
            raise ValueError(
                f"{HF_FIELD}/{LLMFOUNDRY_FIELD}=True requires linear_attention_type to be set."
            )
        return bool(flag)
    if linear_attention_type is None:
        return False
    full_attention_layers = _get(cfg, "full_attention_layers") or []
    return mtp_attention_anchor_layer_idx(cfg) not in full_attention_layers


def resolve_mtp_uses_sparse_ffn(cfg: Any) -> bool:
    flag = _get(cfg, HF_SPARSE_FIELD)
    return True if flag is None else bool(flag)


def resolve_grafted_mtp_fields(base_cfg: Any, donor_cfg: Any, *, keep_base_heads: bool) -> dict:
    donor_linear = resolve_mtp_uses_linear_attention(donor_cfg)
    donor_sparse = resolve_mtp_uses_sparse_ffn(donor_cfg)
    if keep_base_heads:
        base_linear = resolve_mtp_uses_linear_attention(base_cfg)
        base_sparse = resolve_mtp_uses_sparse_ffn(base_cfg)
        mismatch = []
        if base_linear != donor_linear:
            mismatch.append(f"{HF_FIELD}: base={base_linear}, donor={donor_linear}")
        if base_sparse != donor_sparse:
            mismatch.append(f"{HF_SPARSE_FIELD}: base={base_sparse}, donor={donor_sparse}")
        if mismatch:
            raise ValueError(
                "Cannot graft: the kept base MTP heads and the donor heads run in different "
                "modes, and one HF config describes every head (" + "; ".join(mismatch) + "). "
                "Replace every base head (replace_base_heads over all slots) or pick a donor "
                "trained in the base's mode."
            )
    if donor_linear:
        base_type = _get(base_cfg, "linear_attention_type")
        donor_type = _get(donor_cfg, "linear_attention_type")
        if base_type != donor_type:
            raise ValueError(
                "Cannot graft a linear-attention MTP head: linear_attention_type differs "
                f"(base={base_type!r}, donor={donor_type!r})."
            )
    return {HF_FIELD: donor_linear, HF_SPARSE_FIELD: donor_sparse}
