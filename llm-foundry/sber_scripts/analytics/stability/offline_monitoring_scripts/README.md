# LLM Training Dashboard

Interactive per-layer metrics dashboard for monitoring LLM pre-training runs.
Built on Plotly + TensorBoard event files, outputs a single self-contained HTML file.

## Quick start

```bash
# Activate venv
source venv/bin/activate

# Two-run comparison
python3 build_dashboard.py out/ pretrain-v1/ pretrain-v2/

# Single run
python3 build_dashboard.py out/ pretrain-v1/

```


---

## Metric groups

### Layer amax
Maximum value per layer at each step, computed as `max` across all modules/tensors of that type within the layer.

| Metric | Formula |
|---|---|
| `layer_amax__activations` | max(act_amax over all activation patterns) |
| `layer_amax__act_gradients` | max(grad_amax over all gradient patterns) |
| `layer_amax__update_norm` | max(update_norm over all modules) |
| `layer_amax__weight_norm` | max(weight_norm over all modules) |
| `layer_amax__grad_norm` | max(grad_norm over all modules) |
| `layer_amax__eff_lr` | max(eff_lr over all modules) |
| `layer_amax__upd_over_weight` | max(upd/weight over all modules) |
| `layer_amax__weight_tensor_l2norm` | max(L2 norm of weight tensors) |

### Layer amax / run mean
Same as above but divided by the per-run average of that metric (per layer). Values > 1 mean the layer is above its run average at that step.

### Layer ratios
Derived ratios computed from layer amaxes:

| Metric | Formula |
|---|---|
| `layer_eff_lr` | `layer_amax__update_norm / layer_amax__grad_norm` |
| `layer_upd_over_weight` | `layer_amax__update_norm / layer_amax__weight_norm` |

### Per-module metrics
Raw per-module per-layer series:

| Group | Variables |
|---|---|
| **Activations** | `{pattern}__act_{amax,amean}` |
| **Gradients** | `{pattern}__grad_{amax,amean}` |
| **Grad norm** | `grad_norm__{module}` |
| **Update norm** | `update_norm__{module}` |
| **Weight norm** | `weight_norm__{module}` |
| **Eff LR** | `eff_lr__{module}` = update_norm / grad_norm |
| **Upd / Weight** | `upd_over_weight__{module}` = update_norm / weight_norm |

### Max vio
MoE dispatcher max load violation per layer (`dispatcher_layer_experts_max_vio_batch_avg`).

---

## Layer marker legend (in chart)

| Symbol | Meaning |
|---|---|
| ● circle | Standard transformer block layer |
| ■ square | Nonlinear layer (no QKV conv1d) |
| ★ star | MTP (Multi-Token Prediction) block |

---

## Project structure

```
monitorings/
├── dashboard/
│   ├── patterns.py     # Activation/gradient pattern definitions + auto-discovery
│   ├── collector.py    # TensorBoard loading + per-layer metric collection
│   ├── derivatives.py  # Derived/computed metrics (ratios, amaxes, normalizations)
│   └── builder.py      # Plotly figure construction + HTML generation
└── build_dashboard.py  # CLI entry point
```

## Extending

**Add a new metric group** — edit `_GROUPS` in `dashboard/builder.py`.

**Add a new derived metric** — add computation to `compute_derivatives()` in `dashboard/derivatives.py`.

**Add a new raw metric source** — add a `_collect_*` function in `dashboard/collector.py` and call it from `collect_run()`.

**Add new activation patterns** — extend `ACTIVATION_PATTERN_TAILS` in `dashboard/patterns.py`, or they will be auto-discovered from TensorBoard tags.
