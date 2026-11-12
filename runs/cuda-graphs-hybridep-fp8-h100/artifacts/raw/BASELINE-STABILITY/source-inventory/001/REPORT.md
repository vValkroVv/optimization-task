# Native graph-off stability contract (proposed)

## Result and finite acceptance

Use the exact original preset with the native Trainer, the existing pinned HybridEP scan correction, and the existing CudaCherry graph-safety wrapper. Remove the performance observer overlay and entry wrappers. Keep `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0`. No source changes are necessary to express this baseline.

This inventory accepts: (1) unchanged source010 identity, (2) exact native entry and configuration overrides, (3) disposition of active modifications, and (4) proof requirements using standard logs. It does not launch GPUs or accept the eventual stability result. The parent coordinates independent fault diagnosis and launch admission.

## Frozen source

- Repository base: `446431dfa191ef0300f116257ae918bc85e38c06`; dirty edits preserved.
- `SOURCE-SNAPSHOT/010/manifest.json`: SHA-256 `b2e9144af50e8673e0b8646a4f03f629bc240e6fc3ed1aaccfe6814491fb7d20`.
- Archive SHA-256: `b815a34f5fd6f12ee2d223458d7d6a08b59149e22e1dca6dc8d9d52aa8600556`.
- All 98 listed source files matched their recorded SHA-256 in this inspection.
- Exact preset `gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml`: SHA-256 `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`.

## Native launch payload

Run only after root admission, using the verified existing remote runtime and an unused persistent attempt directory. Reuse the previous approved eight-device mapping and existing offline caches, without printing private connection details. The launch below is a payload, not an executed runner. The parent supplies detached ownership, resource checks, source pins, bounded deadline and any separately reviewed CUDA coredump flags. A second accepted run uses a new attempt directory and eight fresh rank processes.

```bash
TASK_ROOT=/workspace-SR008.nfs2/vkropotin/optimization-task
GIFT_DIR="$TASK_ROOT/gift"
ATTEMPT_DIR="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/attempts/001"
CACHE_DIR="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NATIVE-SMOKE/001/cache"
CACHE_ROOT="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/cache"
PYTHON=/opt/torchtitan-venv/bin/python3.12
cd "$GIFT_DIR"
export PYTHONPATH="$GIFT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$CACHE_DIR/huggingface/datasets"
export HF_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export DG_JIT_SHARED_CACHE_DIR="$CACHE_ROOT/deepgemm_shared"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export GIFT_HYBRID_EP_SCAN_RUNTIME_DIR="$ATTEMPT_DIR/scan-runtime"
export TORCHINDUCTOR_CUDAGRAPHS=0
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
export TORCHFT_LIGHTHOUSE=http://localhost:29510
export GIFT_LOG_RANK=all
unset TORCH_LOGS TORCH_COMPILE_DEBUG TORCH_TRACE CUDA_LAUNCH_BLOCKING
unset TITAN_STRUCT_LOGGER_HANDLERS
"$PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
  --log-dir="$ATTEMPT_DIR/torchrun" --redirects=3 --tee=3 \
  -m torchtitan.experiments.cuda_graph_campaign.train_scan_fix \
  configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml \
  --training.steps 200 \
  --lr_scheduler.total_steps 30000 \
  --debug.seed 42 \
  --checkpoint.enable false \
  --validator.enable false \
  --profiler.enable_profiling false \
  --profiler.enable_memory_snapshot false \
  --dump_folder "$ATTEMPT_DIR/native"
```

The prior launch already used graph-off, mixed-order reduction off, seed 42, checkpoint off, the same allocator and these caches. Explicit validator/profiler overrides match their existing false defaults. Standard `metrics.log_freq=10`, structured logging true, TensorBoard/WandB false remain defaults. No force-balanced routing, strict deterministic mode, launch blocking or new synchronizations are added. No performance timer, memory sampler, numerical, continuation or completion adapter is selected. `GIFT_LOG_RANK=all` exposes existing per-rank completion messages; it does not change training callbacks.

`training.steps=200` stops the native while loop. The separate scheduler horizon 30000 prevents the existing scheduler from clamping the 2000-step warmup to the shortened stop. This keeps the first 200 updates on the original schedule. The exact preset preserves model/MoE, EP8, FSDP2, dense and sparse FP8, FlexAttention, model and loss compilation, selective AC, 8-chunk CE, batch 8/global 256/sequence 4096/accumulation 4, fused AdamW and c4_test. Start from zero; checkpointing is disabled, not resumed or initialized from weights.

## Active and inactive changes

| Item | Native graph-off disposition |
| --- | --- |
| `train_scan_fix.main` | Active. Initializes ordinary logging, installs pinned correction, then calls `torchtitan.train.main`. Native Trainer and optimizer path are unchanged. |
| HybridEP scan correction | Active. A fresh rank-local runtime copy changes only `thread_local_all_block_local_experts_sum[NUM_OF_EXPERTS_PER_RANK - 1]` to `[NUM_OF_EXPERTS_PER_RANK]`; preserves installed original. Required even with graphs off. Exact wrapper/header hashes are validated by adapter, with standard activation log and runtime manifest. |
| CudaCherry graph-safety wrapper | **Active even with graphs off.** `_cudacherry.py` imports `_cudacherry_graph_safe.fp8_gemm`; adds an outer opaque CUDA custom-op dispatch per dense GEMM. Both return and mutate forms carry `cudagraph_unsafe`; unpack/repack public operand tuples and delegate all GEMM parameters to installed `cudacherry.gemm.gemm`. Mutating variant declares `out` mutation. No replacement GEMM arithmetic or extra explicit synchronization/copy is introduced. Its dispatch/compile boundary survives graph-off; do not call it inert or silently remove it. |
| Mixed-order reduction workaround | Active compiler policy from environment before Torch import; retain false exactly as failed007. It affects generated code independently of graphs. No new compiler wrapper is needed. |
| Flex attention compiled kernel | Unchanged tracked source explicitly sets `triton.cudagraphs=False`, retains tuning and compiled-region options. No standalone Flex lifetime patch is installed. |
| Compiled Flex mask producer | Unchanged `attention/const.py` uses cached `torch.compile(create_block_mask)` and inherits graph-off global policy. The prior graph-on lifetime remedy was the explicit step-boundary callback, not a mask clone/source edit. |
| `campaign_cudagraph_step_boundary` | Absent from the original preset. Do not select this callback; its unconditional `cudagraph_mark_step_begin()` call would still run if selected with graphs off. |
| `compile.cudagraph_exclude_blocks` | Defaults empty. Diff validates at setup, then uses the original `.compile(backend=..., fullgraph=True)` calls. Prior observer overlay listed all 18 blocks and supplied per-block graph-off options; redundant under global graph-off, unnecessary here. Model compilation remains enabled for all blocks. |
| Registry/schema changes | Campaign trainer/callback schemas can be discovered, but original preset selects default Trainer and an empty user callback map. No campaign runtime callback is built. |
| Timing/phase-memory/combine-completion/numerical/continuation adapters | Unused. No imports from their entries or callback selections. Their files remain preserved in source010. Native built-in callbacks still build normally. |
| Config resolver | `config/loader.py` is unchanged versus repository base. Its YAML 1.2 core float resolver already parses scientific literals; preserve it. Typed config checks show AdamW epsilon remains numeric 1e-20. The dirty config change is the graph-block exclusion schema, not a resolver replacement. |

## Standard evidence for 200 updates / 800 microbatches on every rank

`Trainer.train_step` loops four accumulated microbatches, each through decorated `forward_backward_step` (`fwd_bwd` span), then clipping, `OptimizersContainer.step()` (calls each native optimizer once), scheduler step and `optim` span. The outer native loop emits `step` spans. Ordinary per-rank JSONL logging is sufficient; metrics need not be forced to every step.

For **each** of ranks 0–7 require: one fresh process; contiguous step IDs 1–200; 200 paired `step_start/end`; exactly four paired `fwd_bwd_start/end` per step (800 total); one paired `optim_start/end` per step (200 total); no corresponding `_error` events; one successful process-group teardown; standard `Training starts at step 1`, `Training completed`, and `Process group destroyed`; and launcher exit 0. Reject data-exhaustion cancellation, missing ranks, extra steps or any error/failure. Preserve resolved configuration, startup batch/accumulation logs and scan activation/runtime manifests. Rank-zero configuration plus rank initialization/group membership must establish the intended eight-H100 NVLink topology; degrees alone do not prove it.

Span ends alone do **not** prove success: native span `__exit__` emits `_error` and then `_end` on exceptions. CUDA failures can surface asynchronously; require clean complete process teardown and exit as well. This is stability evidence, not numerical equivalence, convergence, capture or a speedup.

Existing PERFORMANCE-ASSESSMENT/006 standard logs were read offline as a proof of coverage: all eight ranks contain 41 paired steps/optimizers, 164 paired fwd_bwd spans and one teardown end. No new observer is required to count 200/800. Earlier attempt007 failure remains evidence; this inventory does not diagnose it.

## Executed verification and remaining gates

1. SHA-256 verification: all 98 source010 paths match; preset hash matches.
2. Read native entry, Trainer loop, optimizer container, scheduler schema/build path, logging span implementation, Flex kernel/mask producers, scan adapter, CudaCherry wrapper and dirty diffs.
3. CPU-only command below exits 0, reports `OK (0 warning(s), model=gift_moe/3B)` and numeric epsilon 1e-20:

```bash
python gift/scripts/config_check.py gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml --training.steps 200 --lr_scheduler.total_steps 30000 --debug.seed 42 --checkpoint.enable false --validator.enable false --profiler.enable_profiling false --profiler.enable_memory_snapshot false --dump_folder /workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/attempts/001/native --print
```

No GPU run, source edit, package change or cache cleanup occurred. Runtime cost of this document is zero. Removing optional observers and marker changes only instrumentation/graph iteration ownership as described; the retained CudaCherry wrapper remains a material boundary. Remaining gates are root review of this frozen payload and diagnostics, runtime identity/compiler-policy confirmation in the existing authorized environment, all-rank ownership/topology/availability checks, diagnosis and repair of the illegal instruction, then two fresh native 200-update successes under the same frozen repaired contract. Current source010 is not claimed stable or production accepted.
