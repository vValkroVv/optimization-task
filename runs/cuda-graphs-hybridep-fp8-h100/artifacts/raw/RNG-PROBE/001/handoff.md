# RNG continuation probe 001

## Acceptance and result

Resolved: actual target loader RNG boundary; native two-rank DCP rank-local payloads; callback registration/load fit; pinned compiler preservation boundaries; MoE postupdate counter state. No GIFT writes, GPU access, installs, remote access, or model training. Scripts are diagnostic evidence, not a production implementation.

Source root revision: `446431dfa191ef0300f116257ae918bc85e38c06`. Runtime: `/opt/torchtitan-venv/bin/python`, Python 3.12.3, Torch `2.15.0.dev20260822+cu130`, Torch git `8b353065ce260336c5fdfa74d754e8bd9b91b062`, torchdata `0.11.0+cpu`, datasets `4.7.0`. Probe topology: local CPU only; DCP uses two Gloo processes, file-init and loopback interface. No CUDA numerical or performance result follows from these probes.

## 001a: actual target loader

Question: does actual iterator recreation change global CPU RNG, and can a supported restore boundary preserve actual continuation inputs and RNG? Outcomes distinguish an early restore from a postiterator restore; they determine observer/subclass hook placement.

`loader_probe.py` imports actual `HuggingFaceTextDataLoader`, `HuggingFaceTextDataLoaderConfig`, and `HuggingFaceTokenizer`. It uses c4_test, supplied tokenizer, infinite=True, DP world size 8/rank 0, workers=0, pin_memory=False, batch 8, sequence 4096. Seed 713 for Python/NumPy/Torch. Consume four batches; deepcopy native loader state and RNG; compare four additional batches with restored loader. Compare both labels and every input field, including positions. This is a single DP-rank loader component, not eight-rank training.

Results (`loader.log`, extended `loader-with-callback.log`):

- Fresh and resumed `iter(loader)` both advance global Torch CPU RNG.
- Restoring before iterator construction produces identical input batches but different remaining CPU RNG.
- Restoring after iterator construction, before first next, produces identical input batches and Python/NumPy/Torch CPU RNG.
- Restoring immediately after the first resumed batch fetch (the native `after_dataloader` boundary), before further batches, also produces identical inputs and remaining CPU RNG for this actual deterministic loader.

Cause: Torch `utils/data/dataloader.py:690-693` samples `_base_seed` with `loader.generator` (None here). Torchdata `_StatefulSingleProcessDataLoaderIter.__init__` first runs its superclass, then applies saved state (`stateful_dataloader.py:467-479`). Thus restored loader state does not undo the global CPU RNG consumption.

Native boundary: `Trainer.batch_generator`, `trainer.py:500-521`, is a lazy Python generator. Calling it in `train()` does not construct `iter(self.dataloader)` until first next, *after* `batch_start` and first `before_dataloader`. `after_load`, `fit_start`, `batch_start`, and first `before_dataloader` are too early for final RNG restore.

Minimal integration contract:

1. `stateful_objects()` registers unique `continuation_rng_rank_<global_rank>` key before checkpointer construction (`trainer.py:423-425`). Snapshot Python, NumPy, cloned Torch CPU and current-rank CUDA generator state; validate schema, global rank/world size and device ownership. Serialize Python/NumPy using trusted local checkpoint data, e.g. pickle bytes. Clone CPU tensors rather than retaining mutable state aliases.
2. `load_state_dict()` only stages pending state. DCP may restore other objects afterward. Do not set global RNG inside DCP load.
3. Exact boundary option: experiment subclass `batch_generator` materializes actual underlying iterator, restores pending RNG once, then starts native deterministic fetch. Preserve native timing and exhaustion handling if overriding.
4. Callback-only option for the *verified exact loader/preset*: restore pending RNG once at first `after_dataloader`; this is after lazy iterator creation and before forward. This probe establishes that this target's fetch/tokenization is deterministic and does not otherwise consume these global RNG streams. Do not generalize this hook to stochastic preprocessing/workers. CallbackHandler groups mutators before observers/postmutators/aggregators; preserve existing configured ordering.
5. Never apply restore on an unloaded fresh run. Keep capture/warmup state restoration distinct from checkpoint-load restoration. Do not reset state on every batch.
6. RNG mutation has no named `Mutation` axis in current callback API; an experiment subclass gives the cleanest explicit control boundary. An Observer state extension can stage checkpoint data without pretending RNG restore is mere metric observation.

Dataset identity SHA256: data.json `07a274d2f6c903fd1c491260be302f99383930d31656f50de33ef467af6643b1`; tokenizer.json `fe58d9f888054e03f0b202a00df9f6fc05559d8125fe188aa2f0ff5dee142b6c`; tokenizer_config.json `1ccbfda3bb357c2706747d3a22bac82029baac0e4677c812058c7f4a763e8188`.

## 001b: native DCP

Question: does qualification preserve independently differing rank-local states, and does shared-key dedup lose state? This determines native callback key design.

`dcp_probe.py`: real `Observer`/`CallbackHandler.stateful_objects()`, native `dcp.save`/`dcp.load`, two ranks with different seeds 400+rank, each payload containing rank int, cloned RNG tensor, and pickled Python+NumPy bytes. Replace process RNG before load (900+rank); assert load stages exactly once and does not restore process CPU RNG early.

Final `dcp.log` PASS:

- Qualified keys preserve both ranks' full payloads exactly.
- Shared key does **per-field** dedup: both ranks receive rank and Python/NumPy bytes from rank 0, but Torch CPU RNG tensor from rank 1. Consequently neither complete rank payload survives.
- Native Stateful callback registration and native DCP pending-load behavior execute successfully; full model checkpointer orchestration is source-inspected, not instantiated by this CPU probe.

Failures retained: `dcp-first-failure.log` had an overly strong diagnostic assumption that one whole payload would win (corrected to per-field dedup); `dcp-stale-rendezvous.log` reused the failed file-store. After two failures the method was reassessed; final run uses a fresh `gloo-init-v3`, new checkpoint paths and loopback interface. No failed check is counted as a pass.

## Compiler and capture RNG: inspected, not GPU-executed

Pinned installed source:

- `_dynamo/utils.py:2721` `preserve_rng_state`: clones/restores Torch CPU plus current CUDA/XPU generator if available. It does not preserve Python/NumPy generators, all CUDA devices, model buffers, optimizer/FP8 state, callback state or loader state.
- `_inductor/compile_fx.py:1564` wraps codegen/compilation in that helper. `:2244` wraps lazy cudagraphify setup, then invokes the compiled function outside the helper.
- `_inductor/select_algorithm.py:6031` preserves RNG for example-value generation (`rand_strided`). This is evidence about that path, not a blanket guarantee for every external autotuner/custom kernel.
- `_inductor/cudagraph_trees.py:1426,1469` uses preservation around kernel-free warmup and graph recording; `run_graph()` calls replay. This does not itself undo model-side updates during warmup/capture.

Remaining GPU gate: prove all-rank current-device RNG snapshots match across fresh-process compilation/capture; establish correct replay RNG progression with actual FP8/HybridEP/SAC compiled path; restore complete mutated model/FP8/routing state after custom warmup. No verified dropout or random routing is established for this target, so missing native RNG save alone is not proof of an existing numerical defect.

## MoE counter continuation boundary

Actual preset `load_balance_coeff=0.001` (`golden_configs/...gift_moe_3b_hybridep_fp8_compile.yaml:118,422`). `MoE.tokens_per_expert_E` is nonpersistent (`models/common/moe/__init__.py:95-99`); each forward adds counts (`:162-171`), including AC recomputation caveat. `register_moe_load_balancing_hook` is selected by model build (`config/build.py:194-213`) and registered after optimizer build (`trainer.py:355`). `models/common/moe/load_balance.py:109-122` updates persistent expert bias and zeros each count; hook registration at `:125-129` uses optimizer **pre-hook**, not post-hook. `Trainer.train_step:807-810` executes optimizers, scheduler, then `after_optimizer_step`; train-level save follows entire train_step (`:942`). Therefore native postupdate checkpoint boundary should have zero counts for this preset, and persistent bias is model checkpoint state. Validation observer should explicitly assert zero at saved/restored postupdate boundary. If a custom capture/warmup saves at a nonzero boundary, counts must be included/restored; do not discard them on the assumption they are only metrics.

## Reproduction commands

From task root; output directory was assigned before execution:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME="$PWD/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/cache/hf" HF_DATASETS_CACHE="$PWD/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/cache/hf/datasets" OMP_NUM_THREADS=1 python runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/RNG-PROBE/001/loader_probe.py
GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 python runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/RNG-PROBE/001/dcp_probe.py
```

DCP reproduction requires a *new* rendezvous/checkpoint suffix when repeating; final script names v3 paths to preserve failed evidence. HF offline mode uses task-owned cache only. Existing native validation overlay still needs full-state save/load (`initial_load_model_only=False`, `last_save_model_only=False`) and native DCP checkpoint format. These diagnostics create no training runtime cost; optional production state capture would add rank-local CPU serialization and current CUDA RNG access only at validation save/restore boundaries.
