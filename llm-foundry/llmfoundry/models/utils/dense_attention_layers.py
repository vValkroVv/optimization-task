"""Single source of truth for the DSA ``dense_attention_layers`` setting.

It lists decoder-layer indices that keep dense MLA inside a sparse-attention model. The rules
are enforced when a ``GigaMixConfig`` is built and again when a block resolves its attention
class, because configs can reach the block without passing through ``GigaMixConfig`` (e.g.
``PretrainedConfig.from_dict``), so both call sites share this module.
"""
from __future__ import annotations

from typing import Iterable, Optional

DSA_ATTENTION_TYPE = "LlamaLatentSparseAttention"
DENSE_MLA_ATTENTION_TYPE = "LlamaLatentAttention"
RECURRENT_MTP_ATTENTION_TYPE = "LlamaLatentAttention"

__all__ = [
    "DSA_ATTENTION_TYPE",
    "DENSE_MLA_ATTENTION_TYPE",
    "RECURRENT_MTP_ATTENTION_TYPE",
    "normalize_dense_attention_layers",
    "resolve_attention_type",
]


def normalize_dense_attention_layers(
    dense_attention_layers,
    *,
    attention_type: str,
    num_hidden_layers: int,
    linear_attention_type: Optional[str],
    full_attention_layers: Optional[Iterable[int]],
) -> list[int]:
    """Validate ``dense_attention_layers`` and return it as a plain ``list[int]``.

    Rules (each raises ``ValueError``):
      * must be a list (``None`` means ``[]``) of unique, non-bool integers;
      * non-empty only with ``attention_type == 'LlamaLatentSparseAttention'``;
      * every index must be in ``range(num_hidden_layers)``;
      * in a hybrid model (``linear_attention_type`` set) every index must also be a
        full-attention layer, since only those own an MLA block at all.
    """
    normalized = [] if dense_attention_layers is None else dense_attention_layers
    if not isinstance(normalized, list):
        raise ValueError(
            "dense_attention_layers must be a list of unique layer indices, "
            f"got {dense_attention_layers!r}."
        )
    if any(isinstance(idx, bool) or not isinstance(idx, int) for idx in normalized):
        raise ValueError(
            "dense_attention_layers must contain only integer layer indices, "
            f"got {normalized!r}."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"dense_attention_layers must contain unique layer indices, got {normalized!r}."
        )
    if normalized and attention_type != DSA_ATTENTION_TYPE:
        raise ValueError(
            "dense_attention_layers is supported only with "
            f"attention_type='{DSA_ATTENTION_TYPE}'."
        )
    num_hidden_layers = int(num_hidden_layers or 0)
    invalid = set(normalized) - set(range(num_hidden_layers))
    if invalid:
        raise ValueError(
            "dense_attention_layers contains out-of-range layer indices: "
            f"{sorted(invalid)} for {num_hidden_layers} layers."
        )
    full_ids = (
        set(full_attention_layers or [])
        if linear_attention_type is not None
        else set(range(num_hidden_layers))
    )
    non_full = set(normalized) - full_ids
    if non_full:
        raise ValueError(
            "dense_attention_layers must be a subset of full_attention_layers; "
            f"invalid indices: {sorted(non_full)}."
        )
    return list(normalized)


def resolve_attention_type(
    config, attention_layer_idx: int, *, is_mtp_layer: bool = False
) -> str:
    """Attention class name a decoder block at ``attention_layer_idx`` must build.

    Order: linear attention for non-full layers of a hybrid model, then dense MLA for
    indices listed in ``dense_attention_layers``, otherwise ``config.attention_type``.
    Validation is re-run here (cheap) because the block can receive configs that never
    passed through ``GigaMixConfig.__init__``.

    MTP layers (``is_mtp_layer=True``) do not follow ``full_attention_layers``: they use
    ``linear_attention_type`` only when ``config.use_linear_attention_in_mtp`` is set,
    otherwise full attention. ``attention_layer_idx`` (the MTP anchor layer) still
    decides between DSA and dense MLA for the full-attention case.
    """
    linear_attention_type = getattr(config, "linear_attention_type", None)
    full_attention_layers = getattr(config, "full_attention_layers", None) or []
    dense_attention_layers = normalize_dense_attention_layers(
        getattr(config, "dense_attention_layers", None),
        attention_type=config.attention_type,
        num_hidden_layers=config.num_hidden_layers,
        linear_attention_type=linear_attention_type,
        full_attention_layers=full_attention_layers,
    )
    if is_mtp_layer:
        if linear_attention_type is not None and bool(
            getattr(config, "use_linear_attention_in_mtp", False)
        ):
            return linear_attention_type
    elif linear_attention_type is not None and attention_layer_idx not in full_attention_layers:
        return linear_attention_type
    if attention_layer_idx in dense_attention_layers:
        return DENSE_MLA_ATTENTION_TYPE
    return config.attention_type
