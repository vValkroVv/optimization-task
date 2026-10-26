"""Derived / computed metrics built on top of collected raw series."""

from __future__ import annotations

import math

from .collector import Variables, add_series, snap_value


def _layer_amax_of_group(
    variables: Variables, name_substrings: list[str], output_name: str
) -> None:
    """Per-step max across all series whose name contains any of name_substrings.

    Fast path: when all sources for a layer share identical steps (the common
    case within one run), uses direct indexing instead of snap_value.
    """
    sources = [
        series
        for name, series in variables.items()
        if any(sub in name for sub in name_substrings)
    ]
    if not sources:
        return

    layers = sorted({layer for src in sources for layer in src})
    for layer in layers:
        layer_series = [
            (src[layer][0], src[layer][1]) for src in sources if layer in src
        ]
        if not layer_series:
            continue
        ref_steps = layer_series[0][0]
        n = len(ref_steps)

        # Fast path: all sources have the same step list
        if all(s is ref_steps or s == ref_steps for s, _ in layer_series):
            result = []
            for i in range(n):
                cur = [
                    v[i] for _, v in layer_series if i < len(v) and math.isfinite(v[i])
                ]
                result.append(max(cur) if cur else math.nan)
        else:
            # Fallback: sources differ in steps — use snap_value
            result = []
            for step in ref_steps:
                cur = []
                for steps, values in layer_series:
                    v = snap_value(steps, values, step)
                    if v is not None and math.isfinite(v):
                        cur.append(v)
                result.append(max(cur) if cur else math.nan)

        add_series(variables, output_name, layer, ref_steps, result)


def _add_ratio(
    variables: Variables, numerator: str, denominator: str, output_name: str
) -> None:
    if numerator not in variables or denominator not in variables:
        return
    layers = sorted(set(variables[numerator]) & set(variables[denominator]))
    for layer in layers:
        n_steps, n_vals = variables[numerator][layer]
        d_steps, d_vals = variables[denominator][layer]
        # Fast path: identical steps
        if n_steps is d_steps or n_steps == d_steps:
            vals = [
                n / d if d and math.isfinite(d) and math.isfinite(n) else math.nan
                for n, d in zip(n_vals, d_vals)
            ]
        else:
            vals = []
            for step, n_val in zip(n_steps, n_vals):
                d_val = snap_value(d_steps, d_vals, step)
                vals.append(
                    n_val / d_val
                    if d_val not in (None, 0)
                    and math.isfinite(d_val)
                    and math.isfinite(n_val)
                    else math.nan
                )
        add_series(variables, output_name, layer, n_steps, vals)


def _add_module_ratios(variables: Variables) -> None:
    modules = sorted(
        name.removeprefix("update_norm__")
        for name in variables
        if name.startswith("update_norm__")
    )
    for module in modules:
        # Prefer the native monitors (effective_lr / update_weight_ratio / update_rate,
        # collected under these same names) when the run logged them — they already account
        # for per-projection FSDP reduction and the optimizer eps. Only synthesize the
        # derivative when the native series is absent.
        if f"eff_lr__{module}" not in variables:
            _add_ratio(
                variables,
                f"update_norm__{module}",
                f"grad_norm__{module}",
                f"eff_lr__{module}",
            )
        if f"upd_over_weight__{module}" not in variables:
            _add_ratio(
                variables,
                f"update_norm__{module}",
                f"weight_norm__{module}",
                f"upd_over_weight__{module}",
            )
        if f"grad_over_weight__{module}" not in variables:
            _add_ratio(
                variables,
                f"grad_norm__{module}",
                f"weight_norm__{module}",
                f"grad_over_weight__{module}",
            )


# Residual-connection contribution — how much each sublayer writes into the residual.
# Native monitors log, per layer, the sublayer's contribution into the residual
# (``*_residual_contribution``) and the residual stream right after it (``*_residual``);
# their L2 ratio is the *exact* contribution share. When those native slots are absent
# (older runs that didn't track residual connections), fall back to a proxy: the
# sublayer's output L2 over the layer's final residual-stream norm (``layer_output``).
_RESIDUAL_NATIVE = (
    ("attn", "attn_residual_contribution__act_l2norm", "attn_residual__act_l2norm"),
    ("mlp", "mlp_residual_contribution__act_l2norm", "mlp_residual__act_l2norm"),
)
_RESIDUAL_STREAM = "layer_output__act_l2norm"
_RESIDUAL_WRITERS = (
    "block_sparse_moe_output",
    "mlp_output",
    "shared_experts_output",
    "attn_output_last",
)


def _add_residual_contributions(variables: Variables) -> None:
    have_native = False
    for name, contrib, stream in _RESIDUAL_NATIVE:
        if contrib in variables and stream in variables:
            _add_ratio(variables, contrib, stream, f"residual_contrib__{name}")
            have_native = True

    # Proxy fallback only when the native residual-connection monitors are missing.
    if not have_native and _RESIDUAL_STREAM in variables:
        for writer in _RESIDUAL_WRITERS:
            numerator = f"{writer}__act_l2norm"
            if numerator in variables:
                _add_ratio(
                    variables,
                    numerator,
                    _RESIDUAL_STREAM,
                    f"residual_contrib__{writer}",
                )

    _layer_amax_of_group(
        variables, ["residual_contrib__"], "layer_amax__residual_contrib"
    )


def _add_norm_by_run_amean(
    variables: Variables, source_name: str, output_name: str
) -> None:
    if source_name not in variables:
        return
    for layer, (steps, values) in variables[source_name].items():
        finite_vals = [v for v in values if math.isfinite(v)]
        if not finite_vals:
            continue
        amean = sum(finite_vals) / len(finite_vals)
        if amean == 0:
            continue
        add_series(
            variables,
            output_name,
            layer,
            steps,
            [v / amean if math.isfinite(v) else math.nan for v in values],
        )


def compute_derivatives(variables: Variables) -> None:
    _layer_amax_of_group(variables, ["__act_amax"], "layer_amax__activations")
    _layer_amax_of_group(variables, ["__act_amean"], "layer_amean__activations")
    _layer_amax_of_group(variables, ["__grad_amax"], "layer_amax__act_gradients")
    _layer_amax_of_group(variables, ["update_norm__"], "layer_amax__update_norm")
    _layer_amax_of_group(variables, ["weight_norm__"], "layer_amax__weight_norm")
    _layer_amax_of_group(variables, ["grad_norm__"], "layer_amax__grad_norm")

    _add_module_ratios(variables)
    _add_residual_contributions(variables)

    _layer_amax_of_group(variables, ["eff_lr__"], "layer_amax__eff_lr")
    _layer_amax_of_group(
        variables, ["upd_over_weight__"], "layer_amax__upd_over_weight"
    )
    _layer_amax_of_group(
        variables, ["grad_over_weight__"], "layer_amax__grad_over_weight"
    )

    _add_ratio(
        variables, "layer_amax__update_norm", "layer_amax__grad_norm", "layer_eff_lr"
    )
    _add_ratio(
        variables,
        "layer_amax__update_norm",
        "layer_amax__weight_norm",
        "layer_upd_over_weight",
    )

    for metric in [
        "layer_amax__activations",
        "layer_amax__act_gradients",
        "layer_amax__update_norm",
        "layer_amax__weight_norm",
        "layer_amax__eff_lr",
        "layer_amax__upd_over_weight",
        "layer_amax__grad_over_weight",
        "layer_amax__residual_contrib",
        "layer_eff_lr",
        "layer_upd_over_weight",
    ]:
        _add_norm_by_run_amean(variables, metric, f"{metric}__norm_by_run_amean")
