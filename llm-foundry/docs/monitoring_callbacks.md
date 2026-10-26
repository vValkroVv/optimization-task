# New Training Monitoring Callbacks

This document describes three new monitoring callbacks added to LLM Foundry for tracking training health and catching problems early.

## 1. Optimizer Health Monitor

**File:** `llmfoundry/callbacks/optimizer_health_monitor.py`

**Config:**
```yaml
optimizer_health_monitor:
  batch_log_interval: 500
```

### What it does

Tracks the internal state of the Adam optimizer to detect stale or unhealthy training dynamics. Parameters are grouped into semantic categories: `embeddings`, `attn_q/k/v/o`, `mlp_gate_up`, `mlp_up`, `mlp_down`, `norm`, `lm_head`.

### Metrics

For each parameter group:

- **`optim_health/frac_eps/{group}`** — Fraction of elements where `sqrt(v) < eps`. When this is high, the second moment estimate has decayed to near-zero, meaning the optimizer is effectively ignoring gradient magnitude and applying a fixed-size update. This signals stale optimizer state (e.g., after loading an old checkpoint for a different data distribution).

- **`optim_health/frac_10eps/{group}`** — Same but with `10 * eps` threshold. A softer version of frac_eps; useful for catching problems earlier.

- **`optim_health/update_mean/{group}`** — Mean of `|m| / (sqrt(v) + eps)` across all elements. This is the effective update magnitude before learning rate scaling. Sudden changes indicate shifts in training dynamics.

- **`optim_health/Q_wd_over_grad/{group}`** — Ratio `||wd * w||_2 / ||m / (sqrt(v) + eps)||_2`. When Q >> 1, weight decay dominates the gradient signal — the optimizer is mostly shrinking weights rather than learning. This can happen when resuming from a checkpoint with mismatched optimizer state.

### How it works

On each logging step, iterates over all parameters, reads Adam's `exp_avg` (m) and `exp_avg_sq` (v) from the optimizer state, computes local statistics, packs them into a single tensor, and does one `all_reduce(SUM)` across the FSDP group. Handles GigaFSDP's `hybrid_group_count` for correct normalization.

### What problems it catches

- **Stale checkpoints**: After loading an old checkpoint, optimizer moments no longer match the current gradient distribution. `frac_eps` will be abnormally high and `update_mean` will be too large.
- **Weight decay domination**: Q >> 1 means the model is forgetting faster than it learns.
- **Learning rate issues**: Sudden jumps in `update_mean` across groups can indicate LR is too high or a scheduler misconfiguration.

### Optional: epsilon sensitivity analysis

Set `compute_sensitivity: true` to also compute how much the effective updates would change with different epsilon values. This helps determine if the training is sensitive to the eps hyperparameter.


## 2. Top-k Accuracy Monitor

**File:** `llmfoundry/callbacks/topk_accuracy_monitor.py`

**Config:**
```yaml
topk_accuracy_monitor:
  k_values: [1, 5, 20]
  batch_log_interval: 200
```

### What it does

Computes prediction accuracy during training: what fraction of tokens have their correct next-token in the model's top-k predictions.

### Metrics

- **`accuracy/top1`** — Standard next-token prediction accuracy.
- **`accuracy/top5`** — Fraction of tokens where the correct answer is in the top 5.
- **`accuracy/top20`** — Fraction of tokens where the correct answer is in the top 20.

### How it works

The computation is injected directly into the model's `forward()` method (in `modelling_giga_mix.py` and `modelling_gigar.py`), after loss computation but before `delete_logits`. On logging steps, the callback reads the results from `model._topk_results` and logs them. With sequence parallelism (SP), an `all_reduce(SUM)` across the SP group aggregates correct counts and token counts before computing the final accuracy.

The callback sets `model._topk_values` during `fit_start` to tell the forward method which k values to compute. The forward method only computes top-k on steps when the callback will read the results (controlled by `batch_log_interval`).

### Limitations

- **Disabled with `use_liger=True`**: Liger fuses the LM head and cross-entropy loss, so logits are never materialized.
- **Disabled with `tp_size > 1`**: Logits are partitioned along the vocabulary dimension across TP ranks. Computing top-k would require TP communication, which we chose to skip to avoid overhead.
- **Works with `delete_logits=True`**: The computation runs before logits are deleted.
- **Works with sequence parallelism**: Each SP rank computes top-k on its local tokens, then results are aggregated.

### What problems it catches

- **Training quality**: Top-1 accuracy is a more interpretable measure of model capability than cross-entropy loss. A model at loss 2.5 might have top-1 accuracy of 45% or 55% depending on the distribution — accuracy tells you directly.
- **Curriculum shifts**: When switching datasets or mixing ratios, accuracy may change differently than loss.
- **Saturation**: If top-20 accuracy is already 95%+ but top-1 is only 40%, the model "knows" the answer but isn't confident — this pattern is informative for understanding model behavior.


## 3. Expert Balancing Monitor (enhancements)

**File:** `llmfoundry/callbacks/expert_balancing_monitor.py`

The existing `ExpertBalancingMonitor` was extended with three new feature groups, each controlled by a separate flag.

**Config:**
```yaml
expert_balancing_monitor:
  batch_log_interval: 100
  activate_load_balance: true
  activate_expert_weight_norms: true
  activate_activation_norms: true
  weight_norm_log_interval: 500
```

### A. Load Balance Metrics (`activate_load_balance`)

#### Metrics (per MoE layer)

- **`moe_load_balance/{layer}`** — Load balance factor `LB = N * sum(f_i * P_i)`, where `f_i` is the normalized token count for expert i and `P_i` is the mean routing probability. LB = 1 means perfect balance, LB > 1 means imbalance (some experts get more tokens AND higher routing scores). From the Switch Transformer paper.

- **`moe_expert_utilization/{layer}`** — Fraction of experts that received at least one token in this batch. 1.0 = all experts active, < 1.0 = some experts are dead (receiving no tokens).

- **`moe_load_balance/mean`**, **`moe_expert_utilization/mean`** — Averages across all MoE layers.

#### What problems it catches

- **Expert collapse**: If `moe_expert_utilization` drops below 1.0, some experts are dead — they receive no tokens and stop learning.
- **Load imbalance**: High `moe_load_balance` means the router is concentrating tokens on a few experts, reducing effective model capacity.

### B. Expert Weight Norms (`activate_expert_weight_norms`)

#### Metrics (per MoE layer)

- **`expert_weights/w1_norm_std/{layer}`** — Standard deviation of per-expert L2 norms for the first expert weight matrix (gate/up projection). Should be small relative to mean.
- **`expert_weights/w1_norm_max_min_ratio/{layer}`** — Ratio of largest to smallest expert weight norm. Should be close to 1.
- **`expert_weights/w1_norm_min_median_ratio/{layer}`** — Ratio of smallest expert norm to median. A declining trend indicates an expert is dying (its weights are shrinking). From Step-3.5-Flash.
- Same metrics for `w2` (down projection).

#### How it works

Reads `weight1` and `weight2` from each expert module (`GroupedLlamaMLP`), computes per-expert L2 norms, and with expert parallelism (EP) does an `all_gather` across the EP group to collect norms from all experts. This is lightweight since only N_experts scalar norms are gathered per layer.

Logged at a separate (typically slower) interval via `weight_norm_log_interval` since it's heavier than routing statistics.

**Note:** With GigaFSDP, weight parameters may be deleted from modules after the forward pass. If `weight1`/`weight2` are not available, this metric is silently skipped for that layer.

#### What problems it catches

- **Expert dying**: The min/median ratio for a specific expert dropping over time means that expert's weights are decaying — it's receiving fewer tokens and weight decay is shrinking it.
- **Expert divergence**: A large max/min ratio or high std means experts are developing at very different rates, which can hurt model quality.

### C. Activation Norms (`activate_activation_norms`)

#### Metrics (per MoE layer)

- **`expert_activation/max_median_ratio/{layer}`** — Ratio of the maximum per-expert activation norm to the median. Should be close to 1. A high ratio means one expert produces much larger activations than others, which can cause training instability. Key stability metric from Step-3.5-Flash.
- **`expert_activation/mean_norm/{layer}`** — Mean activation norm across experts. Useful for tracking overall activation scale.

#### How it works

Registers forward hooks on expert modules (`GroupedLlamaMLP`) during `fit_start`. The hook captures the output tensor, splits it by expert using `tokens_per_expert`, and computes per-expert L2 norms. With EP, norms are gathered across the EP group via `all_gather`.

#### What problems it catches

- **Activation explosion**: If one expert's activation norm is much larger than others (high max/median ratio), it will dominate the final output and may cause gradient explosion.
- **Expert health monitoring**: Combined with weight norms, activation norms give a complete picture of whether experts are healthy — both in terms of what they've learned (weights) and how they behave (activations).


## Overhead

Measured on a small GigaMix model (hidden=64, 4 layers, 8 experts, 1 GPU) with 200 training steps:

**Throughput**: No measurable impact on this small model. The monitoring runs infrequently (every 100-500 steps) and the per-step cost is dominated by model computation. At production scale (120B parameters), the overhead should be well under 1% since the monitoring only adds a few all-reduce operations on small tensors (parameter counts, scalar norms) at widely-spaced intervals.

**Memory**: Negligible. The optimizer health monitor accesses existing optimizer state without creating copies. The top-k monitor computes argmax/topk on logits that already exist in memory. The expert weight norms operate on existing weight tensors. The activation norm hooks store only N_experts floats per MoE layer.

The main potential cost is the all-reduce in the optimizer health monitor, which packs ~66 float64 values (11 param groups * 6 accumulators) into a single collective — this is negligible compared to gradient all-reduces during training.
