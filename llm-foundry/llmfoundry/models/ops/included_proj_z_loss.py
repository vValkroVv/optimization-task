"""Config helpers for the per-projection hidden z-loss.

Kept in `llmfoundry/models/ops` (a leaf module) so that both the config
(`giga_mix.configuration_giga_mix`) and the layer modules
(`layers.attention` / `layers.ffn` / `layers.moe.experts`) can import it
without a circular import (importing `giga_mix.*` from a layer module would
trigger `giga_mix/__init__` -> `modelling` -> `blocks` -> `attention`).

The regularizing gradient is spliced in-graph by `apply_hidden_z_loss_hook`
(see `hidden_z_loss_hook.py`); the per-projection scalar collected here is
logging-only.
"""

import warnings

# Attention classes with a single combined QKV projection that the QKV hidden
# z-loss hook is wired into (see `attention.py`). Layers using any other class
# silently ignore the `qkv` coef — the model emits a warning naming those layers.
QKV_Z_LOSS_SUPPORTED_ATTENTION = frozenset({
    "LlamaPackedAttention",
    "LlamaPackedRingAttention",
    "Qwen3NextGatedDeltaNet",
})

# Projections on which a hidden z-loss can be spliced (each keyed in the
# `hidden_z_loss_included_proj_coef` config dict):
#   qkv       -> attention QKV projection output (combined-QKV classes only)
#   up_mlp    -> dense MLP `up_proj` output (ParallelMLP / LlamaMLP, incl. shared experts)
#   gate_mlp  -> dense MLP `gate_proj` output
#   up_moe    -> up-half of the fused gate+up grouped-GEMM output in routed MoE experts
#   gate_moe  -> gate-half of that fused output
INCLUDED_PROJECTIONS = ("qkv", "up_mlp", "up_moe", "gate_mlp", "gate_moe")


def _included_proj_spec(config, projection: str):
    specs = getattr(config, "hidden_z_loss_included_proj_coef", None) or {}
    return specs.get(projection)


def resolve_included_proj_coef(config, projection: str, layer_idx) -> float:
    """Effective hidden z-loss coefficient for ``projection`` at ``layer_idx``.

    Per-projection spec is either a bare scalar (applies to all layers) or a
    ``{base: float, layerwise: {layer_idx: float}}`` mapping where ``layerwise``
    overrides ``base`` for the listed layers. Reads LIVE config attributes and
    tolerates the raw (un-normalized) form that HF `config_overrides` leaves —
    OmegaConf dicts, string layer keys, missing ``base``.
    """
    spec = _included_proj_spec(config, projection)
    if spec is None:
        return 0.0
    if not hasattr(spec, "get"):  # bare scalar
        return float(spec)
    base = float(spec.get("base", 0.0) or 0.0)
    layerwise = spec.get("layerwise") or {}
    if layer_idx is not None:
        if layer_idx in layerwise:
            return float(layerwise[layer_idx])
        if str(layer_idx) in layerwise:
            return float(layerwise[str(layer_idx)])
    return base


def included_proj_active(config, projection: str) -> bool:
    """Whether ``projection``'s hidden z-loss is enabled anywhere (base or any
    layerwise coef > 0). Live read; not cached (config_overrides mutate late)."""
    spec = _included_proj_spec(config, projection)
    if spec is None:
        return False
    if not hasattr(spec, "get"):
        return float(spec) > 0
    base = float(spec.get("base", 0.0) or 0.0)
    layerwise = spec.get("layerwise") or {}
    return base > 0 or any(float(v) > 0 for v in layerwise.values())


def collect_included_proj_z_losses(model) -> dict:
    """Walk ``model`` and sum, per projection, the hidden z-loss scalar each
    module stashed on ``_included_proj_z_loss`` during the forward pass. Returns
    ``{projection: summed_scalar}`` for projections that produced a value this
    step. Logging-only aggregation — the stashed scalars are detached and the
    regularizer gradient is already spliced in-graph at each hook site, so this
    walk never participates in backward. Per-rank (MoE experts are sharded)."""
    sums: dict = {}
    for module in model.modules():
        stash = getattr(module, "_included_proj_z_loss", None)
        if not stash:
            continue
        # MTP heads namespace their keys via `_z_loss_log_prefix` (e.g. "mtp0_")
        # so their scalars are logged separately from the trunk instead of being
        # summed into the shared per-projection metric.
        prefix = getattr(module, "_z_loss_log_prefix", "") or ""
        for projection, value in stash.items():
            if value is None:
                continue
            key = f"{prefix}{projection}"
            sums[key] = value if key not in sums else sums[key] + value
    return sums


def normalize_included_proj_coef(raw):
    """Validate + canonicalize the `hidden_z_loss_included_proj_coef` dict to
    ``{proj: {base: float, layerwise: {int: float}}}``. Returns None when empty
    so HF `config_overrides` can replace it wholesale. Only runs for values
    passed directly to `GigaMixConfig.__init__` (overrides bypass it — the
    resolver tolerates the raw form)."""
    if not raw:
        return None
    norm = {}
    for proj, spec in raw.items():
        if proj not in INCLUDED_PROJECTIONS:
            raise ValueError(
                f"Unknown projection {proj!r} in hidden_z_loss_included_proj_coef; "
                f"valid projections are {list(INCLUDED_PROJECTIONS)}."
            )
        if hasattr(spec, "items"):
            base = float(spec.get("base", 0.0) or 0.0)
            layerwise = {
                int(k): float(v) for k, v in (spec.get("layerwise") or {}).items()
            }
        else:
            base, layerwise = float(spec), {}
        norm[proj] = {"base": base, "layerwise": layerwise}
    return norm


# Each MTP z-loss component -> the attribute the corresponding layer module
# reads at forward time for its coefficient. Covers both block-level components
# (attn / moe / dense_mlp) and the per-projection ones. Used to force per-MTP-
# head coefs onto the already-built MTP submodules: MTP heads carry
# `layer_idx=-1` and so cannot be addressed via the trunk's `layerwise` map.
MTP_Z_LOSS_COEF_ATTR = {
    "attn": "_hidden_z_loss_attn_coef",
    "moe": "_hidden_z_loss_moe_coef",
    "dense_mlp": "_hidden_z_loss_dense_mlp_coef",
    "qkv": "_qkv_z_loss_coef",
    "up_mlp": "_up_mlp_z_loss_coef",
    "gate_mlp": "_gate_mlp_z_loss_coef",
    "up_moe": "_up_moe_z_loss_coef",
    "gate_moe": "_gate_moe_z_loss_coef",
}
MTP_Z_LOSS_COMPONENTS = tuple(MTP_Z_LOSS_COEF_ATTR)


def resolve_mtp_z_loss(config, component: str, predictor_idx: int):
    """Per-MTP-head coefficient for a hidden z-loss ``component``, read from
    ``config.hidden_z_loss_mtp_coef``.

    Mirrors ``hidden_z_loss_included_proj_coef``: each component maps to either a
    bare scalar (all MTP heads) or a ``{base, layerwise: {mtp_idx: coef}}`` spec
    where ``layerwise`` is keyed by MTP predictor index. Returns ``None`` when the
    component is absent (no override — keep the module's default), else the float
    coefficient. Tolerates the raw form HF ``config_overrides`` leaves."""
    specs = getattr(config, "hidden_z_loss_mtp_coef", None)
    if not specs or not hasattr(specs, "get"):
        return None
    spec = specs.get(component)
    if spec is None:
        return None
    if not hasattr(spec, "get"):  # bare scalar
        return float(spec)
    base = float(spec.get("base", 0.0) or 0.0)
    layerwise = spec.get("layerwise") or {}
    if predictor_idx in layerwise:
        return float(layerwise[predictor_idx])
    if str(predictor_idx) in layerwise:
        return float(layerwise[str(predictor_idx)])
    return base


def apply_mtp_z_loss_coefs(decoder_layer, config, predictor_idx: int) -> None:
    """Force per-MTP-head hidden z-loss coefs (``config.hidden_z_loss_mtp_coef``)
    onto an already-built MTP decoder layer, and namespace the scalars each of
    its submodules logs under ``mtp<idx>_`` (see ``collect_included_proj_z_losses``).

    Only the coef attribute each hook reads at forward time is overridden — the
    hooks themselves live in the layer / attention / ffn / experts modules — so
    the regularizer gradient is spliced in-graph exactly like a trunk layer.
    Components with no target on this head (e.g. ``moe`` on a dense head) are
    skipped silently, mirroring how trunk block-level coefs no-op on layers that
    lack the sublayer; ``qkv`` additionally warns (like the trunk) when the head's
    attention class does not implement the QKV hook."""
    prefix = f"mtp{predictor_idx}_"
    for module in decoder_layer.modules():
        module._z_loss_log_prefix = prefix

    specs = getattr(config, "hidden_z_loss_mtp_coef", None)
    if not specs:
        return
    if hasattr(specs, "keys"):
        unknown = set(specs.keys()) - set(MTP_Z_LOSS_COMPONENTS)
        if unknown:
            warnings.warn(
                f"[MTP head {predictor_idx}] hidden_z_loss_mtp_coef has unknown "
                f"components {sorted(unknown)}; valid: {sorted(MTP_Z_LOSS_COMPONENTS)}.",
                stacklevel=2,
            )

    # qkv is the only component whose applicability depends on the attention
    # class — warn (do not raise) exactly like the trunk, using the same
    # `QKV_Z_LOSS_SUPPORTED_ATTENTION` predicate. It is then skipped below (no
    # `_qkv_z_loss_coef` attr to set).
    base_attn = getattr(decoder_layer, "self_attn", None)
    base_attn = getattr(base_attn, "base_attn", base_attn)  # unwrap MTPAttnAdapter
    real_attn = getattr(base_attn, "_attn", base_attn)  # AttnGate wraps `._attn`
    if resolve_mtp_z_loss(config, "qkv", predictor_idx) and (
        type(real_attn).__name__ not in QKV_Z_LOSS_SUPPORTED_ATTENTION
    ):
        warnings.warn(
            f"[MTP head {predictor_idx}] QKV hidden z-loss is configured but not "
            f"supported for this head's attention class {type(real_attn).__name__}. "
            f"No qkv z-loss will be added on it (supported classes: "
            f"{sorted(QKV_Z_LOSS_SUPPORTED_ATTENTION)}).",
            stacklevel=2,
        )

    for component, attr in MTP_Z_LOSS_COEF_ATTR.items():
        coef = resolve_mtp_z_loss(config, component, predictor_idx)
        if not coef:  # None or 0.0 -> no override
            continue
        for module in decoder_layer.modules():
            if hasattr(module, attr):
                setattr(module, attr, coef)
                if getattr(module, "_included_proj_z_loss", None) is None:
                    module._included_proj_z_loss = {}
