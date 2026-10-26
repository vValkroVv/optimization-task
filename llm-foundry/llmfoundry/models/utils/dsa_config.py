"""One place for the DSA configuration rules.

`GigaMixConfig.__init__` used to hold them inline, but `config_overrides` are applied
with `setattr` *after* the constructor has run, so a value that arrives that way never
met the rules -- `indexer_topk: 3000` reached the selector kernel and failed there
instead of at build time. The layer re-checked only two of the rules, so the rest could
still slip through. Both the config and the layer call this now.
"""
from typing import Any


_MISSING = object()


def _defaulted(config: Any, name: str, default: Any) -> Any:
    """The value the layer will see: its own default when the config omits the field."""
    value = getattr(config, name, _MISSING)
    return default if value is _MISSING else value


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError("DSA config: " + msg)


def validate_dsa_config(config: Any) -> None:
    """Check the DSA knobs on `config`; a no-op for every other attention type."""
    attention_type = getattr(config, "attention_type", None)
    indexer_n_heads = getattr(config, "indexer_n_heads", None)

    if attention_type != "LlamaLatentSparseAttention":
        if indexer_n_heads is not None:
            raise ValueError(
                "indexer_* params are set but attention_type is not 'LlamaLatentSparseAttention'"
            )
        return

    indexer_head_dim = getattr(config, "indexer_head_dim", None)
    indexer_topk = getattr(config, "indexer_topk", None)
    aux_kl_loss_koeff = getattr(config, "aux_kl_loss_koeff", None)
    dsa_stage = getattr(config, "dsa_stage", None)
    # Fields the layer reads through getattr with a default: a config that never mentions
    # them is valid and gets the default. A value that IS set is checked, so an explicit
    # None still fails here rather than reaching the kernel.
    dsa_sel_cache_offload = _defaulted(config, "dsa_sel_cache_offload", False)
    dsa_block_checkpoint = _defaulted(config, "dsa_block_checkpoint", False)
    sp_split_type = _defaulted(config, "sp_split_type", None)

    _require(indexer_n_heads == 64,
             f"indexer_n_heads must be 64: cuDNN indexer_backward requires heads>=64, "
             f"got {indexer_n_heads}")
    _require(indexer_head_dim == 128,
             f"indexer_head_dim must be 128 (fused selection and cuDNN indexer backward), "
             f"got {indexer_head_dim}")
    _require(type(indexer_topk) is int and indexer_topk > 0,
             f"indexer_topk must be a positive int, got {indexer_topk}")
    _require(indexer_topk <= 4096,
             f"indexer_topk must be <= 4096 (Hopper shared memory), got {indexer_topk}")
    _require(indexer_topk >= 128 and (indexer_topk & (indexer_topk - 1)) == 0,
             f"indexer_topk must be a power of two >= 128, got {indexer_topk}")
    # Older checkpoint configs can contain this removed option. Never silently run bf16
    # when the user has requested fp8 scoring.
    _require(getattr(config, "indexer_fp8_scoring", False) is False,
             "indexer_fp8_scoring is no longer supported; fused selection uses bf16")
    _require(isinstance(dsa_sel_cache_offload, bool),
             f"dsa_sel_cache_offload must be a bool, got {dsa_sel_cache_offload!r}")
    _require(aux_kl_loss_koeff is not None,
             "aux_kl_loss_koeff must be set for DSA")
    _require(dsa_stage in (1, 2),
             f"dsa_stage must be 1 (dense warm-up) or 2 (sparse), got {dsa_stage}")
    _require(not (dsa_block_checkpoint and dsa_stage == 2),
             "dsa_block_checkpoint is stage-1-only. Nested under the module-level "
             "activation checkpoint, a per-block checkpoint RETAINS its inputs/outputs - OOM")
    _require(sp_split_type in (None, "equal"),
             "DSA requires a contiguous sequence-parallel split (sp_split_type='equal'), "
             f"got {sp_split_type}")
