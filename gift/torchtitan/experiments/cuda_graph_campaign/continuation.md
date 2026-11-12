# CUDA graph campaign RNG continuation

The `cuda_graph_campaign` trainer uses the `campaign_rng_continuation` callback
for the exact `hf_text` / `c4_test` loader in
`configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml`. The config rejects
other loader types, nonzero worker counts, pinned memory, finite iteration, or
a different dataset because the verified restore boundary depends on this
loader's deterministic fetch path. Existing config validation cannot express
this callback/trainer pairing, so decode and pre-submit accept the callback
even with the native trainer. At runtime, the observer requires binding to a
`CampaignTrainer` at initialization and rejects an unbound `fit_start` or RNG
restore before the first training step. The supplied overlay selects that
trainer and binds the same observer that native DCP registered.

Each process registers one native DCP state under
`continuation_rng_rank_<global-rank>`. Its schema contains version 1, global
rank and world size, CUDA device type/index, a pickled Python and NumPy RNG
snapshot, and cloned Torch CPU and current-device CUDA RNG tensors. DCP load
validates the schema and ownership, then stages the snapshot without changing
process RNG. The trainer restores it once after native lazy iterator creation
and the first deterministic target fetch, before yielding that batch to the
training loop. A fresh run has no pending state and is unchanged.

Python and NumPy states use pickle and therefore must come only from trusted
native checkpoints. This state covers Python, NumPy, and the process's default
Torch CPU and current CUDA generators. It does not save per-module generators,
other CUDA devices, data-loader worker state, model or optimizer state, FP8
backend state, graph executables, or arbitrary extension state. The native
checkpoint manager continues to own model, optimizer, scheduler, loader, and
trainer state. Native saves remain after the optimizer update: the existing
optimizer pre-hook clears nonpersistent MoE expert counts before that boundary,
while persistent expert bias remains in model state. This extension changes
neither. Actual CUDA, HybridEP, FP8, compiled-path and multi-rank continuation
still require the campaign's GPU validation gate.

Runtime overhead is confined to validation checkpoint boundaries: one Python
and NumPy pickle, cloned RNG state tensors, and their DCP serialization. No
state capture, serialization, collective, reset, or extra batch fetch occurs on
ordinary training steps. A resumed run performs one Python-side restore at the
first batch boundary; all model, loss, optimizer, activation-checkpointing,
compilation, precision and parallelism paths remain native.
