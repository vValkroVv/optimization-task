# FlashQLA gate variants: bounded experiment report

**Status:** The bounded block, native, and trace comparisons are complete. The owner subsequently selected `external_gate` for the current final task baseline, explicitly through `--gdn-backend external_gate`. Both implementations remain task-copy code; the recipe selection does not change a generic backend default, establish a production default, or prove a general throughput gain or convergence. See the canonical [baseline decision](../../docs/baseline.md#current-final-task-baseline).

## Scope and implementation

The isolated worktree started at task source `0f32623d5e0e7544b20c1f047d94236cb6697758`, with FLA pinned at `1d2281ff1b7988d8070f00dba5ffa9747ae3a60e` and FlashQLA at `a14dbf3474514e1674526431e960325f72081e9f`. Workload: B=2, S=4096, Hq=16, Hv=32, K/V=128, CP=1, no initial/final state. `external_gate` combines the FLA gate with FlashQLA; `fused_gate` uses FLA gate/cumsum plus thin QLA orchestration. The native FLA control includes TileLang `common.chunk_bwd_dqkwg`, not an all-Triton baseline. Gathered `A_log`/`dt_bias` leaves are BF16 with FP32 masters/intermediate arithmetic. See the [implementation record](implementation/README.md), [task-copy patch](implementation/implementation.patch), [pinned source archive](implementation/flashqla-source.tar.gz), and [source manifest](implementation/manifest.json).

## Block comparison

Six actual VJP fixtures were replayed for all three variants. Across 18 reports, outputs and seven gradients were finite with exact eager/compiled and zero-upstream parity. Against FLA, output relative RMS was 0.5095% median (0.4408–0.5184%); the largest parameter-gradient relative RMS was 0.7569% (`A_log`) and 0.7267% (`dt_bias`). Direct fused-versus-external differences were smaller (output median 0.02118%; worst gradient 0.1732%). The FP64 reference is a zero-state 64-token prefix, not a full-sequence oracle. Synthetic CUDA-graph 3F+B ratios were 1.903× external and 1.915× fused; these are component measurements, not production speed or fused superiority. See [block results](block-results.md) and [machine results](block-results.json).

## Native training

All three eight-rank H100 runs completed 140 updates with unchanged source/config receipts, finite loss/grad-norm records, and zero allocator retry/OOM deltas; updates 41–140 were measured. Global physical-input TPS was 156,452.36 (FLA), 160,090.72 (external), and 159,735.21 (fused), single-run differences of +2.326%, +2.098%, and −0.222% fused versus external. Each window's first-to-last-20 cadence median shortened about 10.4%. Fixed order, one observation, and this drift prevent causal or stationary TPS conclusions. See [native results](native-results.md), [config comparison](native-config-comparison.json), and [native plan](native-plan.json).

## Trace comparison

The two update-21 captures compare with the accepted combined-four control on ranks 0/1. All four new MCP imports passed event/flow parity with no import errors. Each rank covered 18 calls per phase (original, SAC, VJP recompute, native backward); 288 calls were matched. Both captures exited 0, finished on all ranks, and left the GPUs idle. Whole-GDN 3F+B work fell from 59.2–59.7 ms to 30.9–31.4 ms (1.90–1.92×), removing 28.2–28.6 ms or 2.25–2.28% of the three-microbatch device span. This is not exposed critical-path savings, optimizer-step latency, or TPS evidence. Captures omit 29 accumulation microbatches and the optimizer; overlap is not causality. See [trace analysis](trace-results.md), [machine results](trace-results.json), and the [capture provenance and history](trace-capture-results.md) with its [JSON receipt](trace-capture-results.json).

## Experiment conclusion and subsequent baseline decision

At experiment close, the bounded evidence supported the task-copy mechanism and
local numerical contract, but did not justify production integration or default
promotion. On 14 November 2026 the owner subsequently selected `external_gate`
for the current final task recipe, using the explicit backend flag and the
combined-four foundation. This owner decision is recorded in the canonical
[baseline](../../docs/baseline.md#current-final-task-baseline). It is not a
conclusion that the native throughput differences are causal: the fixed-order
windows drifted by about 10.4%, versus roughly 2% between runs. EMA updates,
checkpoint/resume, long-run convergence, whole-training numerical parity, CP
warmup, and production PP4/EP32 behavior remain unverified. No further
experiment is part of this assignment.

Failure history remains in the [plan](plan.md) and detailed reports. The first capture stopped at a guard before fixture saving; the initial backward graph pilot hit a stale-default-stream error. The native foreground controller raised `BrokenPipeError` after FLA/external passed; fused later completed detached. Trace acquisition retained a gzip/census writer exit 1 from a wrong activation-log path (JSONL had zero markers; TorchRun stdout had two per rank), a preflight wrapper exit 1 of unknown cause with its full body unavailable and no GPU side effects, and a local macOS `chmod` error. The root-saved script then launched the final capture without a GPU restart. These were harness/controller incidents, not kernel failures. The [mirror receipt](mirror-receipt.json) records evidence SHA-256 parity; raw traces/caches remain remote-only.
