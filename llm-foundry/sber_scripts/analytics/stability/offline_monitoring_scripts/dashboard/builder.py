"""Plotly figure construction and HTML dashboard generation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import plotly.graph_objects as go

from .collector import RunDict, _timer, snap_value
from .patterns import MAXVIO_COLOR, color_map


_GROUP_DOCS: dict[str, str] = {
    "Layer amax": "Per-step max value across all modules within each layer",
    "Layer amax / run mean": "Layer amax divided by its mean across the whole run — highlights deviation from baseline",
    "Layer ratios": "Layer-level effective LR (update/grad) and update/weight ratio, computed via layer amaxes",
    "Activations": "Raw per-tensor activation statistics: amax, amean, l2norm",
    "Gradients": "Raw per-tensor activation gradient statistics: amax, amean, l2norm",
    "Grad norm": "L2 norm of gradients per module per layer  (from optimizer state)",
    "Update norm": "L2 norm of parameter updates per module per layer",
    "Weight norm": "L2 norm of weights per module per layer",
    "Eff LR": "Effective learning rate per module = update_norm / grad_norm (native effective_lr monitor when logged, else derivative)",
    "Upd / Weight": "Update-to-weight ratio per module = update_norm / weight_norm (native update_weight_ratio monitor when logged, else derivative)",
    "Grad / Weight": "Gradient-to-weight ratio per module = grad_norm / weight_norm (native update_rate monitor when logged, else derivative)",
    "Residual contribution": "Sublayer output L2 relative to the residual-stream norm (layer_output) — how much each block writes into the residual",
    "Pre-clip grad": "L2 norm of gradients before gradient clipping (pre_clip_grad monitor)",
    "Moments": "Optimizer state: L2 norm and amax of Adam first/second moments (moment / moment_sq)",
    "Absolute massive activations": "Fraction of activations satisfying an absolute compare, e.g. actabs>1000",
    "Relative massive activations": "Fraction of activations satisfying a relative-to-mean compare, e.g. actrel>100",
    "Max vio": "MoE dispatcher max load violation per layer",
    "Experts": "All MoE expert metrics: gmm_down, shared_experts activations/gradients, load violations, and cross-expert imbalance",
    "Other": "Uncategorised metrics",
}

_GROUPS: list[tuple[str, Any]] = [
    (
        "Layer amax",
        lambda n: (
            n.startswith("layer_amax__") and not n.endswith("__norm_by_run_amean")
        ),
    ),
    ("Layer amax / run mean", lambda n: n.endswith("__norm_by_run_amean")),
    ("Layer ratios", lambda n: n in ("layer_eff_lr", "layer_upd_over_weight")),
    # Per-metric groups intentionally duplicate layer_amax__* entries so a metric is
    # findable both in "Layer amax" and in its own semantic group.
    (
        "Activations",
        lambda n: (
            "__act_" in n
            or n in ("layer_amax__activations", "layer_amean__activations")
        ),
    ),
    (
        "Gradients",
        lambda n: (
            ("__grad_" in n and "grad_norm" not in n)
            or n == "layer_amax__act_gradients"
        ),
    ),
    ("Grad norm", lambda n: "grad_norm" in n and "pre_clip" not in n),
    ("Pre-clip grad", lambda n: "pre_clip_grad" in n),
    ("Update norm", lambda n: "update_norm" in n),
    ("Weight norm", lambda n: "weight_norm" in n),
    ("Eff LR", lambda n: "eff_lr" in n),
    ("Upd / Weight", lambda n: "upd_over_weight" in n),
    ("Grad / Weight", lambda n: "grad_over_weight" in n),
    (
        "Residual contribution",
        lambda n: (
            n.startswith("residual_contrib__")
            or n.startswith("layer_amax__residual_contrib")
            # native residual-connection activations (contribution + the stream itself)
            or n.startswith("attn_residual")
            or n.startswith("mlp_residual")
        ),
    ),
    ("Moments", lambda n: "moment" in n),
    ("Absolute massive activations", lambda n: "actabs" in n),
    ("Relative massive activations", lambda n: "actrel" in n),
    ("Max vio", lambda n: n.startswith("max_vio__")),
    (
        "Experts",
        lambda n: (
            n.startswith("expert_act__")
            or n.startswith("expert_activation__")
            or n.startswith("max_vio__")
            or n.startswith("moe__")
            or "gmm_down" in n
            or "shared_experts" in n
        ),
    ),
]


def _x_for_layers(layers: list[int], max_real_layer: int) -> list[float]:
    denom = max(max_real_layer, 1)
    return [1.04 if layer > max_real_layer else layer / denom for layer in layers]


def _initial_visible(name: str) -> bool:
    return (
        name.startswith("layer_amax__")  # includes layer_amax__residual_contrib
        or name in ("layer_eff_lr", "layer_upd_over_weight")
        or name.startswith("max_vio__")
        or name.startswith("expert_activation__")
        or name in ("moe__load_balance", "moe__expert_utilization", "moe__maxvio")
        or "__amax_to_amean" in name
    )


def _file_label(text: str) -> str:
    return re.sub(r"[^\w\-.]+", "__", text).strip("_")


def _subsample_steps(steps: list[int], max_steps: int) -> list[int]:
    n = len(steps)
    if n <= max_steps:
        return steps
    stride = math.ceil(n / max_steps)
    result = steps[::stride]
    if not result or result[-1] != steps[-1]:
        result = list(result) + [steps[-1]]
    return result


def build_run_figure(
    run: RunDict,
    variable_names: list[str],
    colors: dict[str, str],
    init_step: int,
    all_steps_sub: list[int],
) -> tuple[go.Figure, list[dict]]:
    """One Plotly figure for one run. Trace index == variable index in variable_names."""
    layers = run["layers"]
    max_real_layer = run["max_real_layer"]
    nonlinear = run["nonlinear"]
    x = _x_for_layers(layers, max_real_layer)
    custom_layers = [[layer] for layer in layers]
    symbols = [
        "star"
        if layer > max_real_layer
        else "square"
        if layer in nonlinear
        else "circle"
        for layer in layers
    ]
    sizes = [
        12 if layer > max_real_layer else 9 if layer in nonlinear else 7
        for layer in layers
    ]

    tick_layers = [layer for layer in layers if layer <= max_real_layer]
    tick_stride = max(1, math.ceil(len(tick_layers) / 12))
    tick_layers_shown = tick_layers[::tick_stride]
    tick_vals = _x_for_layers(tick_layers_shown, max_real_layer)

    fig = go.Figure()
    trace_payloads: list[dict] = []

    for var in variable_names:
        per_layer = run["variables"].get(var, {})
        visible: bool | str = True if _initial_visible(var) else "legendonly"

        y_init: list = []
        series_payload: list = []
        for layer in layers:
            if layer not in per_layer:
                y_init.append(None)
                series_payload.append(None)
                continue
            steps, values = per_layer[layer]
            y_init.append(snap_value(steps, values, init_step))
            series_payload.append([snap_value(steps, values, s) for s in all_steps_sub])

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_init,
                customdata=custom_layers,
                mode="lines+markers",
                name=var,
                legendgroup=var,
                showlegend=True,
                visible=visible,
                line=dict(color=colors[var], width=1.4),
                marker=dict(
                    color=colors[var], symbol=symbols, size=sizes, line=dict(width=0)
                ),
                connectgaps=False,
                hovertemplate=f"<b>{var}</b><br>L=%{{customdata[0]}}  %{{y:.4g}}<extra></extra>",
            )
        )
        trace_payloads.append({"layers": layers, "series": series_payload})

    fig.update_layout(
        template="plotly_white",
        height=530,
        margin=dict(l=72, r=32, t=16, b=64),
        hovermode="closest",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif",
            size=12,
            color="#1d1d1f",
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
            tracegroupgap=4,
            font=dict(size=10),
        ),
        xaxis=dict(
            title_text="normalized layer  (★=MTP  ■=nonlinear)",
            range=[-0.02, 1.07],
            tickmode="array",
            tickvals=tick_vals,
            ticktext=[str(L) for L in tick_layers_shown],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            linecolor="rgba(0,0,0,0.10)",
        ),
        yaxis=dict(
            title_text="Y1",
            type="linear",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            linecolor="rgba(0,0,0,0.10)",
        ),
        yaxis2=dict(
            title_text="Y2",
            overlaying="y",
            side="right",
            type="linear",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            ticks="",
            linecolor="rgba(230,126,34,0.5)",
            tickfont=dict(color="#e67e22"),
            title_font=dict(color="#e67e22"),
            visible=False,
        ),
    )

    return fig, trace_payloads


def _run_container_html(i: int, label: str, graph_html: str) -> str:
    short = label[:48] + "…" if len(label) > 48 else label
    return (
        f'<div id="run-container-{i}" class="run-container">'
        f'<div class="run-header">'
        f'<span class="run-label" title="{label}">{short}</span>'
        f'<div class="yrange-section">'
        f'<div class="yrange-row">'
        f'<span class="yrange-axis-label">Y1</span>'
        f'<input type="number" id="y-min-{i}" class="yrange-input" placeholder="min" step="any">'
        f'<span class="yrange-sep">–</span>'
        f'<input type="number" id="y-max-{i}" class="yrange-input" placeholder="max" step="any">'
        f'<button class="btn yrange-btn" id="y-set-{i}" title="Apply Y1 range">Set</button>'
        f'<button class="btn yrange-btn" id="y-auto-{i}" title="Auto Y1 range">↺</button>'
        f"</div>"
        f'<div class="yrange-row" id="yrange2-row-{i}" style="display:none">'
        f'<span class="yrange-axis-label y2-label">Y2</span>'
        f'<input type="number" id="y2-min-{i}" class="yrange-input" placeholder="min" step="any">'
        f'<span class="yrange-sep">–</span>'
        f'<input type="number" id="y2-max-{i}" class="yrange-input" placeholder="max" step="any">'
        f'<button class="btn yrange-btn" id="y2-set-{i}" title="Apply Y2 range">Set</button>'
        f'<button class="btn yrange-btn" id="y2-auto-{i}" title="Auto Y2 range">↺</button>'
        f"</div>"
        f"</div>"
        f'<button class="btn hide-btn" id="hide-btn-{i}" onclick="toggleRun({i})">Hide</button>'
        f"</div>"
        f"{graph_html}"
        f"</div>"
    )


def _empty_run_container_html(i: int, label: str, n_tags: int) -> str:
    short = label[:48] + "…" if len(label) > 48 else label
    tag_word = "tag" if n_tags == 1 else "tags"
    return (
        f'<div id="run-container-{i}" class="run-container run-empty">'
        f'<div class="run-header">'
        f'<span class="run-label" title="{label}">{short}</span>'
        f'<button class="btn hide-btn" id="hide-btn-{i}" onclick="toggleRun({i})">Hide</button>'
        f"</div>"
        f'<div class="empty-run-body">'
        f'<div class="empty-run-icon">&#9888;</div>'
        f'<div class="empty-run-title">No per-layer metrics</div>'
        f'<div class="empty-run-detail">'
        f"{n_tags} scalar {tag_word} loaded — none matched per-layer patterns.<br>"
        f"The run may have crashed before the first checkpoint or is still initializing."
        f"</div>"
        f"</div>"
        f"</div>"
    )


def write_dashboard_html(
    output_path: Path,
    run_containers_html: str,
    graphs_init: str,
    trace_payloads_per_run: list[list[dict]],
    initial_visible: dict[str, bool],
    all_steps: list[int],
    run_labels: list[str],
    variable_names: list[str],
) -> None:
    n_runs = len(run_labels)
    n_vars = len(variable_names)

    all_predicates = [pred for _, pred in _GROUPS]

    def _make_items(names: list[str], group: str) -> str:
        return "".join(
            f'<div class="var-item" data-group="{group}">'
            f'<label class="var-check-label">'
            f'<input type="checkbox" data-var="{n}"'
            f" {'checked' if initial_visible.get(n) else ''}>"
            f'<span title="{n}">{n}</span>'
            f"</label>"
            f'<button class="y2-btn" data-var="{n}" onclick="toggleVarAxis(this)" title="Move to Y2 axis">Y2</button>'
            f"</div>\n"
            for n in names
        )

    def _group_cb(glabel: str, names: list[str]) -> str:
        all_checked = all(initial_visible.get(n, False) for n in names)
        any_checked = any(initial_visible.get(n, False) for n in names)
        checked_attr = " checked" if all_checked else ""
        indet_attr = ' data-indet="true"' if (any_checked and not all_checked) else ""
        return (
            f'<input type="checkbox" class="group-cb" data-group="{glabel}"'
            f"{checked_attr}{indet_attr}"
            f' onclick="event.stopPropagation()" title="Select/deselect all in group">'
        )

    groups_html = ""
    for glabel, predicate in _GROUPS:
        group_names = [n for n in variable_names if predicate(n)]
        if not group_names:
            continue
        any_visible = any(initial_visible.get(n, False) for n in group_names)
        open_attr = " open" if any_visible else ""
        doc = _GROUP_DOCS.get(glabel, "")
        title_attr = f' title="{doc}"' if doc else ""
        groups_html += (
            f'<details class="var-group"{open_attr}>'
            f"<summary{title_attr}>{_group_cb(glabel, group_names)}{glabel}</summary>"
            f'<div class="group-content">{_make_items(group_names, glabel)}</div>'
            f"</details>\n"
        )

    remaining = [
        n for n in variable_names if not any(pred(n) for pred in all_predicates)
    ]
    if remaining:
        doc = _GROUP_DOCS.get("Other", "")
        groups_html += (
            f'<details class="var-group">'
            f'<summary title="{doc}">{_group_cb("Other", remaining)}Other</summary>'
            f'<div class="group-content">{_make_items(remaining, "Other")}</div>'
            f"</details>\n"
        )

    max_idx = max(len(all_steps) - 1, 0)
    initial_idx = max_idx
    panel_width = 300

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{output_path.stem}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg:           #f5f5f7;
      --surface:      #ffffff;
      --surface-blur: rgba(255,255,255,0.80);
      --text:         #1d1d1f;
      --text-2:       #6e6e73;
      --text-3:       #aeaeb2;
      --accent:       #0071e3;
      --border:       rgba(0,0,0,0.08);
      --border-med:   rgba(0,0,0,0.13);
      --shadow:       0 2px 12px rgba(0,0,0,0.07), 0 1px 3px rgba(0,0,0,0.05);
      --radius:       10px;
      --radius-sm:    7px;
      --panel-w:      {panel_width}px;
      --font: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
    }}
    html, body {{ height:100%; font-family:var(--font); background:var(--bg); color:var(--text); -webkit-font-smoothing:antialiased; }}

    .page  {{ display:flex; height:100vh; overflow:hidden; }}
    .plot  {{ flex:1 1 auto; min-width:0; height:100vh; overflow-y:auto; background:#fff; }}
    .panel {{
      flex: 0 0 var(--panel-w);
      max-width: var(--panel-w);
      height: 100vh;
      background: var(--surface-blur);
      backdrop-filter: blur(28px) saturate(180%);
      -webkit-backdrop-filter: blur(28px) saturate(180%);
      border-left: 1px solid var(--border);
      display: flex; flex-direction: column;
      padding: 16px 12px 12px;
      gap: 9px; overflow: hidden;
      transition: max-width 0.26s cubic-bezier(.4,0,.2,1),
                  padding   0.26s cubic-bezier(.4,0,.2,1);
    }}
    .panel.collapsed {{
      max-width: 0; padding-left: 0; padding-right: 0; border-left: none;
    }}

    .panel-tab {{
      position: fixed; right: var(--panel-w); top: 50%;
      transform: translateY(-50%); z-index: 200;
      width: 20px; height: 48px;
      background: var(--surface);
      border: 1px solid var(--border-med); border-right: none;
      border-radius: 6px 0 0 6px;
      cursor: pointer; display: flex; align-items: center; justify-content: center;
      color: var(--text-2); font-size: 12px;
      box-shadow: -2px 0 8px rgba(0,0,0,0.06);
      transition: right 0.26s cubic-bezier(.4,0,.2,1), color 0.12s;
    }}
    .panel-tab:hover {{ color: var(--text); }}
    .panel-tab.collapsed {{
      right: 0; border-radius: 0 6px 6px 0;
      border-right: 1px solid var(--border-med); border-left: none;
      box-shadow: 2px 0 8px rgba(0,0,0,0.06);
    }}

    .section-label {{
      font-size: 10.5px; font-weight: 600; color: var(--text-3);
      text-transform: uppercase; letter-spacing: 0.09em; padding: 0 2px;
    }}

    /* ── Hidden runs bar ── */
    #hidden-bar {{
      display: none;
      align-items: center; flex-wrap: wrap; gap: 6px;
      padding: 7px 14px;
      background: #fff8e6;
      border-bottom: 1px solid rgba(200,160,0,0.22);
      font-size: 12px; color: var(--text-2);
      position: sticky; top: 0; z-index: 20;
    }}
    .hidden-bar-label {{
      font-size: 10px; font-weight: 600; color: var(--text-3);
      text-transform: uppercase; letter-spacing: .07em; flex-shrink: 0;
    }}
    .show-chip {{
      font: 12px var(--font); padding: 3px 10px;
      border-radius: 980px;
      border: 1.5px solid var(--border-med);
      background: var(--surface); color: var(--accent);
      cursor: pointer; transition: background .12s;
    }}
    .show-chip:hover {{ background: rgba(0,113,227,0.07); }}

    /* ── Run container ── */
    .run-container {{ border-bottom: 1px solid var(--border); }}
    .run-header {{
      display: flex; align-items: center; gap: 10px;
      padding: 8px 14px;
      background: var(--bg);
      border-bottom: 1px solid var(--border);
    }}
    .run-label {{
      font-size: 13px; font-weight: 600; color: var(--text);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      flex: 0 0 auto; max-width: 280px;
    }}
    .hide-btn {{ flex: 0 0 auto; margin-left: auto; padding: 5px 12px; font-size: 12px; }}

    /* ── Empty run placeholder ── */
    .run-container.run-empty .run-header {{ background: #fffbf0; border-bottom-color: rgba(200,150,0,0.2); }}
    .run-container.run-empty .run-label  {{ color: #92670a; }}
    .empty-run-body {{
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 8px; height: 160px;
      background: #fffdf5;
    }}
    .empty-run-icon  {{ font-size: 28px; line-height: 1; }}
    .empty-run-title {{ font-size: 14px; font-weight: 600; color: #92670a; }}
    .empty-run-detail {{
      font-size: 12px; color: var(--text-2); text-align: center;
      max-width: 520px; line-height: 1.6;
    }}

    /* ── Step card ── */
    .card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 10px 12px 12px; box-shadow: var(--shadow);
    }}
    .card-label {{ font-size: 11px; color: var(--text-2); margin-bottom: 2px; }}
    .step-value {{
      font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.2;
      width: 100%; border: none; outline: none; background: transparent;
      color: var(--text); font-family: var(--font); padding: 0;
      border-bottom: 1.5px solid transparent; transition: border-color .15s;
    }}
    .step-value:hover, .step-value:focus {{
      border-bottom-color: var(--accent);
    }}
    .step-value::-webkit-inner-spin-button, .step-value::-webkit-outer-spin-button {{ opacity: 0; }}

    input[type=range] {{
      -webkit-appearance: none; appearance: none;
      width: 100%; height: 4px; margin-top: 10px;
      border-radius: 2px; background: #dde0e6; outline: none; cursor: pointer;
    }}
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 18px; height: 18px; border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 1px 5px rgba(0,113,227,0.38);
      transition: transform 0.1s;
    }}
    input[type=range]::-webkit-slider-thumb:hover {{ transform: scale(1.18); }}

    .seg {{
      display: flex; border: 1px solid var(--border-med);
      border-radius: var(--radius-sm); overflow: hidden; background: var(--bg);
    }}
    .seg-btn {{
      flex: 1; font: 13px/1 var(--font);
      padding: 7px 0; border: none; background: transparent;
      color: var(--text-2); cursor: pointer;
      transition: background .14s, color .14s;
    }}
    .seg-btn + .seg-btn {{ border-left: 1px solid var(--border-med); }}
    .seg-btn.active {{
      background: var(--surface); color: var(--accent);
      font-weight: 600; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}

    .play-row {{ display: flex; gap: 5px; align-items: center; }}
    .btn {{
      font: 13px var(--font); padding: 7px 0; text-align: center;
      border-radius: var(--radius-sm); border: 1px solid var(--border-med);
      background: var(--surface); color: var(--text);
      cursor: pointer; white-space: nowrap; transition: background .12s;
    }}
    .btn:hover {{ background: var(--bg); }}
    .btn-play {{ flex: 1; }}
    .btn-pause {{ flex: 1; }}
    .btn-clear {{ flex: 0 0 auto; padding: 7px 12px; color: var(--text-2); font-size: 12px; }}
    .btn-clear:hover {{ color: var(--text); }}

    select.speed-sel {{
      font: 12px var(--font); padding: 5px 6px;
      border-radius: var(--radius-sm); border: 1px solid var(--border-med);
      background: var(--surface); color: var(--text-2);
      cursor: pointer; outline: none;
    }}

    .search-wrap {{ position: relative; }}
    .search-wrap .ico {{
      position: absolute; left: 9px; top: 50%;
      transform: translateY(-50%); pointer-events: none; color: var(--text-3);
    }}
    .search-input {{
      width: 100%; font: 13px var(--font); padding: 7px 10px 7px 29px;
      border: 1px solid var(--border-med); border-radius: var(--radius-sm);
      background: var(--surface); color: var(--text); outline: none;
      transition: border-color .15s, box-shadow .15s;
    }}
    .search-input::placeholder {{ color: var(--text-3); }}
    .search-input:focus {{
      border-color: var(--accent); box-shadow: 0 0 0 3px rgba(0,113,227,0.14);
    }}

    .vars {{ flex: 1; overflow-y: auto; padding-right: 1px; }}
    .vars::-webkit-scrollbar {{ width: 3px; }}
    .vars::-webkit-scrollbar-thumb {{ background: rgba(0,0,0,0.12); border-radius: 2px; }}

    .var-group {{ border-radius: var(--radius-sm); margin-bottom: 1px; }}
    .var-group summary {{
      list-style: none; display: flex; align-items: center; gap: 6px;
      padding: 5px 7px; border-radius: var(--radius-sm);
      cursor: pointer; user-select: none;
      font-size: 10.5px; font-weight: 600; color: var(--text-2);
      text-transform: uppercase; letter-spacing: 0.07em;
      transition: background .1s;
    }}
    .group-cb {{
      flex-shrink: 0; width: 14px; height: 14px;
      accent-color: var(--accent); cursor: pointer;
    }}
    .var-group summary::-webkit-details-marker {{ display: none; }}
    .var-group summary::before {{
      content: ''; flex-shrink: 0; display: inline-block;
      width: 11px; height: 11px;
      background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236e6e73' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 18l6-6-6-6'/%3E%3C/svg%3E") no-repeat center/contain;
      transition: transform .18s ease;
    }}
    .var-group[open] summary::before {{ transform: rotate(90deg); }}
    .var-group summary:hover {{ background: rgba(0,0,0,0.045); }}
    .group-content {{ padding: 1px 4px 4px 18px; }}
    .var-group[open] .group-content {{ animation: groupOpen .14s ease; }}
    @keyframes groupOpen {{
      from {{ opacity: 0; transform: translateY(-3px); }}
      to   {{ opacity: 1; transform: translateY(0); }}
    }}

    .var-item {{
      display: flex; align-items: center; gap: 4px;
      padding: 1px 4px 1px 2px; border-radius: 5px;
      font-size: 11.5px; color: var(--text); line-height: 1.4;
      transition: background .08s; overflow: hidden;
    }}
    .var-item:hover {{ background: rgba(0,0,0,0.04); }}
    .var-item.hidden {{ display: none !important; }}
    .var-check-label {{
      display: flex; align-items: center; gap: 7px;
      flex: 1; min-width: 0; cursor: pointer; overflow: hidden;
    }}
    .var-check-label input {{ flex-shrink: 0; width: 14px; height: 14px; accent-color: var(--accent); cursor: pointer; }}
    .var-check-label span {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-variant-numeric: tabular-nums; }}
    .y2-btn {{
      flex-shrink: 0;
      font: 10px/1 var(--font); padding: 2px 5px;
      border-radius: 4px; border: 1px solid var(--border-med);
      background: var(--surface); color: var(--text-3);
      cursor: pointer; opacity: 0; transition: opacity .1s, background .1s, color .1s, border-color .1s;
    }}
    .var-item:hover .y2-btn {{ opacity: 1; }}
    .y2-btn.on-y2 {{
      background: #fff3e0; color: #e67e22; border-color: rgba(230,126,34,0.5); opacity: 1;
    }}

    /* ── Y-range row (in run header) ── */
    .yrange-section {{ display: flex; flex-direction: column; gap: 2px; flex: 1; min-width: 0; }}
    .yrange-row {{ display: flex; gap: 4px; align-items: center; min-width: 0; }}
    .yrange-axis-label {{
      font-size: 10px; font-weight: 700; color: var(--text-3);
      flex-shrink: 0; width: 16px; text-align: center;
    }}
    .yrange-axis-label.y2-label {{ color: #e67e22; }}
    .yrange-input {{
      width: 0; flex: 1; min-width: 0;
      font: 12px var(--font); padding: 4px 5px;
      border: 1px solid var(--border-med); border-radius: var(--radius-sm);
      background: var(--surface); color: var(--text); outline: none;
    }}
    .yrange-input:focus {{ border-color: var(--accent); box-shadow: 0 0 0 2px rgba(0,113,227,0.12); }}
    .yrange-input::placeholder {{ color: var(--text-3); }}
    .yrange-sep {{ color: var(--text-3); flex-shrink: 0; font-size: 13px; }}
    .yrange-btn {{ flex: 0 0 auto; padding: 4px 8px; font-size: 12px; }}

    /* ── Y-scale rows in panel ── */
    .axis-scale-row {{ display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }}
    .axis-scale-row:last-child {{ margin-bottom: 0; }}
    .yaxis-scale-label {{ font-size: 11px; font-weight: 700; color: var(--text-2); flex-shrink: 0; width: 18px; }}
    .yaxis-scale-label.y2-label {{ color: #e67e22; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="plot">
      <div id="hidden-bar"><span class="hidden-bar-label">Hidden:</span></div>
      {run_containers_html}
    </div>

    <button class="panel-tab" id="panel-tab" title="Toggle controls">&#8250;</button>

    <aside class="panel" id="panel">
      <span class="section-label">Training step</span>
      <div class="card">
        <div class="card-label">Current step</div>
        <input type="number" class="step-value" id="step-label" value="{all_steps[initial_idx] if all_steps else 0}" min="{all_steps[0] if all_steps else 0}" max="{all_steps[-1] if all_steps else 0}">
        <input id="step-slider" type="range" min="0" max="{max_idx}" value="{initial_idx}" step="1">
      </div>

      <div>
        <div class="section-label" style="margin-bottom:6px">Y axis scale</div>
        <div class="axis-scale-row">
          <span class="yaxis-scale-label">Y1</span>
          <div class="seg" style="flex:1">
            <button class="seg-btn active" id="y-linear">Linear</button>
            <button class="seg-btn"        id="y-log">Log</button>
          </div>
        </div>
        <div class="axis-scale-row">
          <span class="yaxis-scale-label y2-label">Y2</span>
          <div class="seg" style="flex:1">
            <button class="seg-btn active" id="y2-linear">Linear</button>
            <button class="seg-btn"        id="y2-log">Log</button>
          </div>
        </div>
      </div>

      <div class="play-row">
        <button class="btn btn-play"  id="play">&#9654;&thinsp; Play</button>
        <button class="btn btn-pause" id="pause">&#9646;&#9646;&thinsp; Pause</button>
        <select class="speed-sel" id="speed" title="Playback speed">
          <option value="360">0.5&times;</option>
          <option value="180" selected>1&times;</option>
          <option value="90">2&times;</option>
          <option value="45">4&times;</option>
          <option value="22">8&times;</option>
          <option value="11">16&times;</option>
        </select>
        <button class="btn btn-clear" id="btn-clear" title="Hide all variables">&#10005;</button>
      </div>

      <div class="search-wrap">
        <svg class="ico" width="14" height="14" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="11" cy="11" r="7"/><path d="m21 21-4.35-4.35"/>
        </svg>
        <input class="search-input" id="search" type="text" placeholder="Search variables…" autocomplete="off">
      </div>

      <div class="vars">{groups_html}</div>
    </aside>
  </div>

  <script>
    const graphs   = [{graphs_init}];
    const nRuns    = {n_runs};
    const nVars    = {n_vars};
    const varNames = {json.dumps(variable_names)};
    const allSteps = {json.dumps(all_steps)};
    const runLabels = {json.dumps(run_labels)};
    const tracePayloadsPerRun = {json.dumps(trace_payloads_per_run, allow_nan=True)};

    const runVisible = new Array(nRuns).fill(true);
    const varActive  = {{}};
    document.querySelectorAll('input[data-var]').forEach(cb => {{
      varActive[cb.dataset.var] = cb.checked;
    }});

    /* ── Run hide / show ── */
    function updateHiddenBar() {{
      const bar = document.getElementById('hidden-bar');
      const chips = runLabels
        .map((lbl, ri) => runVisible[ri] ? '' :
          `<button class="show-chip" onclick="toggleRun(${{ri}})">${{lbl}}</button>`)
        .join('');
      bar.style.display = chips.trim() ? 'flex' : 'none';
      const label = bar.querySelector('.hidden-bar-label');
      // Remove old chips, re-add fresh ones
      [...bar.children].forEach(el => {{ if (!el.classList.contains('hidden-bar-label')) el.remove(); }});
      bar.insertAdjacentHTML('beforeend', chips);
    }}

    const yRangeFixed  = new Array(nRuns).fill(null);   // [lo, hi] or null
    const y2RangeFixed = new Array(nRuns).fill(null);

    function _restepRun(ri, idx) {{
      const payloads = tracePayloadsPerRun[ri];
      const ys = payloads.map(p => p.series.map(s => {{
        if (!s) return null;
        const v = s[idx];
        return (v !== null && Number.isFinite(v)) ? v : null;
      }}));
      Plotly.restyle(graphs[ri], {{y: ys}}, payloads.map((_, vi) => vi));
      // Re-apply fixed ranges — restyle resets Plotly's internal range state
      if (yRangeFixed[ri] || y2RangeFixed[ri]) {{
        const upd = {{}};
        if (yRangeFixed[ri])  {{ upd['yaxis.range']  = yRangeFixed[ri];  upd['yaxis.autorange']  = false; }}
        if (y2RangeFixed[ri]) {{ upd['yaxis2.range'] = y2RangeFixed[ri]; upd['yaxis2.autorange'] = false; }}
        Plotly.relayout(graphs[ri], upd);
      }}
    }}

    function toggleRun(ri) {{
      runVisible[ri] = !runVisible[ri];
      document.getElementById(`run-container-${{ri}}`).style.display = runVisible[ri] ? '' : 'none';
      if (runVisible[ri] && graphs[ri]) {{
        _restepRun(ri, +document.getElementById('step-slider').value);
      }}
      updateHiddenBar();
    }}

    /* ── Step slider ── */
    function updateStep(idx) {{
      idx = Math.max(0, Math.min(allSteps.length - 1, idx));
      const step = allSteps[idx];
      document.getElementById('step-label').value = step;
      for (let ri = 0; ri < nRuns; ri++) {{
        if (!graphs[ri] || !runVisible[ri]) continue;
        _restepRun(ri, idx);
      }}
    }}
    let _stepDebounce = null;
    document.getElementById('step-slider').addEventListener('input', e => {{
      clearTimeout(_stepDebounce);
      _stepDebounce = setTimeout(() => updateStep(+e.target.value), 40);
    }});

    /* ── Step label: manual entry ── */
    const stepLabel = document.getElementById('step-label');
    function snapToStep(rawValue) {{
      const target = +rawValue;
      if (!isFinite(target)) return;
      let best = 0, bestD = Math.abs(allSteps[0] - target);
      for (let i = 1; i < allSteps.length; i++) {{
        const d = Math.abs(allSteps[i] - target);
        if (d < bestD) {{ best = i; bestD = d; }}
      }}
      document.getElementById('step-slider').value = best;
      updateStep(best);
    }}
    stepLabel.addEventListener('keydown', e => {{
      if (e.key === 'Enter') {{ e.preventDefault(); stepLabel.blur(); }}
    }});
    stepLabel.addEventListener('blur', () => snapToStep(stepLabel.value));

    /* ── Y-axis scale ── */
    function setYType(type) {{
      const isLog = type === 'log';
      const upd = {{}};
      upd['yaxis.type']      = type;
      upd['yaxis.autorange'] = true;
      if (isLog) {{
        upd['yaxis.tickmode']       = 'linear';
        upd['yaxis.dtick']          = 1;
        upd['yaxis.exponentformat'] = 'power';
      }} else {{
        upd['yaxis.tickmode']       = 'auto';
        upd['yaxis.nticks']         = 8;
        upd['yaxis.dtick']          = null;
        upd['yaxis.exponentformat'] = 'e';
      }}
      for (let ri = 0; ri < nRuns; ri++) {{
        if (!graphs[ri]) continue;
        Plotly.relayout(graphs[ri], upd);
        document.getElementById(`y-min-${{ri}}`).value = '';
        document.getElementById(`y-max-${{ri}}`).value = '';
      }}
      document.getElementById('y-linear').classList.toggle('active', !isLog);
      document.getElementById('y-log').classList.toggle('active',     isLog);
    }}
    document.getElementById('y-linear').onclick = () => setYType('linear');
    document.getElementById('y-log').onclick    = () => setYType('log');

    function setY2Type(type) {{
      const isLog = type === 'log';
      const upd = {{}};
      upd['yaxis2.type']      = type;
      upd['yaxis2.autorange'] = true;
      if (isLog) {{
        upd['yaxis2.tickmode']       = 'linear';
        upd['yaxis2.dtick']          = 1;
        upd['yaxis2.exponentformat'] = 'power';
      }} else {{
        upd['yaxis2.tickmode']       = 'auto';
        upd['yaxis2.nticks']         = 8;
        upd['yaxis2.dtick']          = null;
        upd['yaxis2.exponentformat'] = 'e';
      }}
      for (let ri = 0; ri < nRuns; ri++) {{
        if (!graphs[ri]) continue;
        Plotly.relayout(graphs[ri], upd);
        document.getElementById(`y2-min-${{ri}}`).value = '';
        document.getElementById(`y2-max-${{ri}}`).value = '';
      }}
      document.getElementById('y2-linear').classList.toggle('active', !isLog);
      document.getElementById('y2-log').classList.toggle('active',     isLog);
    }}
    document.getElementById('y2-linear').onclick = () => setY2Type('linear');
    document.getElementById('y2-log').onclick    = () => setY2Type('log');

    /* ── Y2 axis toggle ── */
    const varAxis = {{}};
    varNames.forEach(n => {{ varAxis[n] = 'y'; }});

    function syncY2Visibility() {{
      const hasY2 = varNames.some(n => varAxis[n] === 'y2');
      for (let ri = 0; ri < nRuns; ri++) {{
        if (!graphs[ri]) continue;
        Plotly.relayout(graphs[ri], {{
          'yaxis2.visible':          hasY2,
          'yaxis2.showticklabels':   hasY2,
          'yaxis2.showline':         hasY2,
          'yaxis2.ticks':            hasY2 ? 'outside' : '',
          'margin.r':                hasY2 ? 72 : 32,
        }});
        const row2 = document.getElementById(`yrange2-row-${{ri}}`);
        if (row2) row2.style.display = hasY2 ? '' : 'none';
      }}
    }}

    function toggleVarAxis(btn) {{
      const varName = btn.dataset.var;
      const goingY2 = varAxis[varName] !== 'y2';
      varAxis[varName] = goingY2 ? 'y2' : 'y';
      btn.classList.toggle('on-y2', goingY2);
      const vi = varNames.indexOf(varName);
      if (vi >= 0) {{
        for (let ri = 0; ri < nRuns; ri++) {{
          if (!graphs[ri]) continue;
          Plotly.restyle(graphs[ri], {{ yaxis: [varAxis[varName]] }}, [vi]);
        }}
      }}
      syncY2Visibility();
    }}

    /* ── Variable checkboxes ── */
    function setVarActive(varName, active) {{
      varActive[varName] = active;
      document.querySelectorAll(`input[data-var="${{varName}}"]`).forEach(cb => {{ cb.checked = active; }});
      const vi = varNames.indexOf(varName);
      if (vi < 0) return;
      const vis = active ? true : 'legendonly';
      for (let ri = 0; ri < nRuns; ri++) {{
        if (!graphs[ri]) continue;
        Plotly.restyle(graphs[ri], {{visible: [vis]}}, [vi]);
      }}
    }}
    /* ── Clear all variables ── */
    document.getElementById('btn-clear').onclick = () => {{
      varNames.forEach(n => {{ varActive[n] = false; varAxis[n] = 'y'; }});
      document.querySelectorAll('input[data-var]').forEach(cb => {{ cb.checked = false; }});
      document.querySelectorAll('.group-cb').forEach(gcb => {{ gcb.checked = false; gcb.indeterminate = false; }});
      document.querySelectorAll('.y2-btn').forEach(btn => {{ btn.classList.remove('on-y2'); }});
      const vis = new Array(nVars).fill('legendonly');
      const axes = new Array(nVars).fill('y');
      for (let ri = 0; ri < nRuns; ri++) {{
        if (!graphs[ri]) continue;
        Plotly.restyle(graphs[ri], {{visible: vis, yaxis: axes}}, vis.map((_, vi) => vi));
      }}
      syncY2Visibility();
    }};

    /* ── Y range (per run, in run header) ── */
    for (let ri = 0; ri < nRuns; ri++) {{
      if (!graphs[ri]) continue;
      (function(ri) {{
        const applyRange = () => {{
          let lo = parseFloat(document.getElementById(`y-min-${{ri}}`).value);
          let hi = parseFloat(document.getElementById(`y-max-${{ri}}`).value);
          const cur = graphs[ri].layout.yaxis.range || [];
          if (isNaN(lo) && isNaN(hi)) return;
          if (isNaN(lo)) lo = cur[0] ?? -Infinity;
          if (isNaN(hi)) hi = cur[1] ??  Infinity;
          if (lo >= hi) return;
          yRangeFixed[ri] = [lo, hi];
          Plotly.relayout(graphs[ri], {{'yaxis.range': [lo, hi], 'yaxis.autorange': false}});
        }};
        document.getElementById(`y-set-${{ri}}`).onclick = applyRange;
        [`y-min-${{ri}}`, `y-max-${{ri}}`].forEach(id =>
          document.getElementById(id).addEventListener('keydown', e => {{ if (e.key === 'Enter') applyRange(); }})
        );
        document.getElementById(`y-auto-${{ri}}`).onclick = () => {{
          document.getElementById(`y-min-${{ri}}`).value = '';
          document.getElementById(`y-max-${{ri}}`).value = '';
          yRangeFixed[ri] = null;
          Plotly.relayout(graphs[ri], {{'yaxis.autorange': true}});
        }};

        const applyRange2 = () => {{
          let lo = parseFloat(document.getElementById(`y2-min-${{ri}}`).value);
          let hi = parseFloat(document.getElementById(`y2-max-${{ri}}`).value);
          const cur = graphs[ri].layout.yaxis2.range || [];
          if (isNaN(lo) && isNaN(hi)) return;
          if (isNaN(lo)) lo = cur[0] ?? -Infinity;
          if (isNaN(hi)) hi = cur[1] ??  Infinity;
          if (lo >= hi) return;
          y2RangeFixed[ri] = [lo, hi];
          Plotly.relayout(graphs[ri], {{'yaxis2.range': [lo, hi], 'yaxis2.autorange': false}});
        }};
        document.getElementById(`y2-set-${{ri}}`).onclick = applyRange2;
        [`y2-min-${{ri}}`, `y2-max-${{ri}}`].forEach(id =>
          document.getElementById(id).addEventListener('keydown', e => {{ if (e.key === 'Enter') applyRange2(); }})
        );
        document.getElementById(`y2-auto-${{ri}}`).onclick = () => {{
          document.getElementById(`y2-min-${{ri}}`).value = '';
          document.getElementById(`y2-max-${{ri}}`).value = '';
          y2RangeFixed[ri] = null;
          Plotly.relayout(graphs[ri], {{'yaxis2.autorange': true}});
        }};
      }})(ri);
    }}

    /* ── Play / Pause ── */
    let playTimer = null;
    document.getElementById('play').onclick = () => {{
      if (playTimer) return;
      const sl    = document.getElementById('step-slider');
      const speed = () => +document.getElementById('speed').value;
      const tick  = () => {{
        const next = +sl.value + 1 > +sl.max ? 0 : +sl.value + 1;
        sl.value = next; updateStep(next);
      }};
      tick();
      playTimer = setInterval(tick, speed());
      document.getElementById('speed').onchange = () => {{
        if (!playTimer) return;
        clearInterval(playTimer); playTimer = setInterval(tick, speed());
      }};
    }};
    document.getElementById('pause').onclick = () => {{ clearInterval(playTimer); playTimer = null; }};

    /* ── Group checkboxes ── */
    function updateGroupCb(group) {{
      const gcb = document.querySelector(`.group-cb[data-group="${{group}}"]`);
      if (!gcb) return;
      const items = [...document.querySelectorAll(`.var-item[data-group="${{group}}"] input[data-var]`)];
      const visible = items.filter(cb => !cb.closest('.var-item').classList.contains('hidden'));
      const target = visible.length ? visible : items;
      const nChecked = target.filter(cb => cb.checked).length;
      gcb.indeterminate = nChecked > 0 && nChecked < target.length;
      gcb.checked = nChecked === target.length && target.length > 0;
    }}

    document.querySelectorAll('.group-cb').forEach(gcb => {{
      if (gcb.dataset.indet === 'true') gcb.indeterminate = true;
      gcb.addEventListener('change', () => {{
        const group = gcb.dataset.group;
        const active = gcb.checked;
        document.querySelectorAll(`.var-item[data-group="${{group}}"] input[data-var]`).forEach(cb => {{
          if (cb.checked !== active) {{
            cb.checked = active;
            setVarActive(cb.dataset.var, active);
          }}
        }});
        gcb.indeterminate = false;
      }});
    }});

    document.querySelectorAll('input[data-var]').forEach(cb =>
      cb.addEventListener('change', () => {{
        const group = cb.closest('.var-item')?.dataset.group;
        if (group) updateGroupCb(group);
        setVarActive(cb.dataset.var, cb.checked);
      }})
    );

    /* ── Search ── */
    document.getElementById('search').addEventListener('input', e => {{
      const q = e.target.value.toLowerCase().trim();
      document.querySelectorAll('.var-item').forEach(el => {{
        const n = (el.querySelector('input')?.dataset.var || '').toLowerCase();
        el.classList.toggle('hidden', !!q && !n.includes(q));
      }});
      document.querySelectorAll('.var-group').forEach(grp => {{
        const has = [...grp.querySelectorAll('.var-item')].some(el => !el.classList.contains('hidden'));
        grp.style.display = has ? '' : 'none';
        if (q && has) grp.open = true;
        const group = grp.querySelector('.group-cb')?.dataset.group;
        if (group) updateGroupCb(group);
      }});
    }});

    /* ── Panel collapse ── */
    const panelEl = document.getElementById('panel');
    const tabEl   = document.getElementById('panel-tab');
    let panelOpen = true;
    tabEl.addEventListener('click', () => {{
      panelOpen = !panelOpen;
      panelEl.classList.toggle('collapsed', !panelOpen);
      tabEl.classList.toggle('collapsed', !panelOpen);
      tabEl.innerHTML = panelOpen ? '&#8250;' : '&#8249;';
    }});
  </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def build_dashboard_for_paths(
    output_dir: Path,
    run_paths: list[Path],
    labels: list[str] | None = None,
    max_steps: int = 500,
    max_workers: int | None = None,
) -> Path:
    import hashlib

    t0 = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    from .collector import load_runs

    all_runs = load_runs(run_paths, labels, max_workers=max_workers)

    run_labels = [r["label"] for r in all_runs]
    slug = "__vs__".join(_file_label(lbl) for lbl in run_labels)
    stem = f"dashboard__{slug}"
    if len(stem) > 200:
        h = hashlib.md5(slug.encode()).hexdigest()[:8]
        stem = f"dashboard__{slug[:180]}__{h}"
    output_path = output_dir / f"{stem}.html"

    normal_runs = [r for r in all_runs if not r.get("empty")]
    n_empty = len(all_runs) - len(normal_runs)
    if n_empty:
        print(
            f"  ⚠  {n_empty} run(s) have no per-layer metrics — shown as placeholders",
            flush=True,
        )

    variable_names = sorted({name for run in normal_runs for name in run["variables"]})
    colors = color_map(variable_names)
    for name in colors:
        if name.startswith("max_vio__"):
            colors[name] = MAXVIO_COLOR

    all_steps = sorted(
        {
            step
            for run in normal_runs
            for per_layer in run["variables"].values()
            for steps, _ in per_layer.values()
            for step in steps
        }
    )
    all_steps_sub = _subsample_steps(all_steps, max_steps)
    if len(all_steps_sub) < len(all_steps):
        print(
            f"  step subsampling: {len(all_steps)} → {len(all_steps_sub)} steps "
            f"(--max-steps {max_steps})",
            flush=True,
        )
    init_step = all_steps_sub[-1] if all_steps_sub else 0

    print(
        f"── Building figures ({len(normal_runs)} normal + {n_empty} empty runs, "
        f"{len(variable_names)} variables, {len(all_steps_sub)} steps) ──",
        flush=True,
    )

    # Build containers and JS arrays in original run order.
    containers: list[str] = []
    graphs_parts: list[str] = []
    trace_payloads_per_run: list[list[dict]] = []
    first_graph = True

    for i, run in enumerate(all_runs):
        if run.get("empty"):
            containers.append(
                _empty_run_container_html(i, run["label"], run.get("n_tags", 0))
            )
            graphs_parts.append("null")
            trace_payloads_per_run.append([])
        else:
            with _timer(f"build figure     [{run['label']}]"):
                fig, payloads = build_run_figure(
                    run, variable_names, colors, init_step, all_steps_sub
                )
            graph_html = fig.to_html(
                include_plotlyjs=True if first_graph else False,
                full_html=False,
                div_id=f"graph-run-{i}",
                config={"responsive": True, "displaylogo": False},
            )
            first_graph = False
            containers.append(_run_container_html(i, run["label"], graph_html))
            graphs_parts.append(f'document.getElementById("graph-run-{i}")')
            trace_payloads_per_run.append(payloads)

    run_containers_html = "\n".join(containers)
    graphs_init = ", ".join(graphs_parts)
    initial_visible = {name: _initial_visible(name) for name in variable_names}

    with _timer("write HTML"):
        write_dashboard_html(
            output_path,
            run_containers_html,
            graphs_init,
            trace_payloads_per_run,
            initial_visible,
            all_steps_sub,
            run_labels,
            variable_names,
        )
    print(f"\n✓ wrote {output_path}", flush=True)
    print(f"  total wall time: {perf_counter() - t0:.2f}s", flush=True)
    return output_path
