# FlashQLA native analysis preparation

Prepared from the frozen `/workspace-SR008.nfs2/vkropotin/opt-fq` sources and actual artifacts. Analyzer lives in the original task, outside the running worktree. It imports only the Python standard library, writes JSON to stdout, and never imports training code.

## Use

```sh
python3 -B /workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/analyze_native.py \
  /workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-native-fla-001 \
  /workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-native-external-001 \
  /workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-native-fused-001
```

Completion requires PASS completion, zero process exit, unchanged source/config receipts, accepted existing native-parser summary, eight complete rank sidecars with contiguous updates, eight matching bootstraps/source identities, accepted resolved-config hash, complete unique rank0 numerical logs, and native forward plus grad-enabled activation on every rank. Missing/partial runs emit incomplete with reasons; integrated throughput is withheld. Cadence-only capture can be complete while measurement is disabled.

## Actual schema and identity

- Root: `metadata.json`, `completion.json`, `summary.json`, `exit-code.txt`, `flash-qla-bridge.json`.
- Rank records: `measurement/rank-00.json` through `rank-07.json`, schema_version1. Fields include training_status/completed_steps, cadence, measurement_status, measurement.
- Bootstrap records: `measurement/flash-qla-bootstrap-rank-XX.json`. Installer source identity is installation evidence; actual `GDN_BACKEND_ACTIVATED variant=... grad_enabled=False/True` log markers establish route invocation.
- Resolved configuration: `native/config/resolved_config.yaml`; raw config hash, launch argv, source snapshots, bridge hashes, installer identities, cache root and runtime are carried into output.
- Native stdout: `logs/torchrun/*/attempt_0/RANK/stdout.log`. Loss, grad_norm and displayed TPS are rounded text; analyzer retains rank0 series, not TensorBoard precision. Multiple attempt/log paths are rejected as ambiguous.
- Plan source HEAD: `0f32623d5e0e7544b20c1f047d94236cb6697758`; FLA upstream `1d2281ff1b7988d8070f00dba5ffa9747ae3a60e`; FlashQLA upstream `a14dbf3474514e1674526431e960325f72081e9f`.
- Three native runs use 140 updates, measurement warmup40/window100, fresh seed2048 and common completed untimed capture compiler cache. TileLang remains attempt-local/rank-local. LR warmup2000/total65000 is separate. EMA recipe starts parameter updates at10000, so these 140 updates do not exercise EMA updates (configuration finding provided by coordinator).
- FLA control preserves the combined-four foundation, including TileLang dispatch for common.chunk_bwd_dqkwg; do not label it an all-Triton baseline.

## Timer and denominator evidence

All references below are inside the frozen worktree.

- `runs/overnight-streaming-perf/measure.py:490–528`: CUDA synchronize before start timestamp after update40; start timestamp precedes start-side allocator/token reads. CUDA synchronize before end timestamp after update140. End-side allocator/token reads follow the end timestamp. This is integrated host elapsed with drained device boundaries, including training, host submission, data loading, optimizer and logging plus window instrumentation; not isolated GPU compute.
- `measure.py:549–600`: native Profiler.step wrapper records host monotonic timestamps at update boundaries. Cadence41..140 covers100 intervals, with no per-update adapter drain; final interval includes end drain. Medians, min/max, p10/p90, MAD and first/last20 drift are descriptive host-cadence statistics.
- `runs/overnight-streaming-perf/parse_results.py:590–620,665–705`: primary elapsed = max(rank end monotonic timestamp) minus min(rank start), valid for this same-host job. Sum physical local deltas over8 ranks. Supervised token delta is already global and counted once. Global TPS = count / envelope seconds; per-GPU TPS divides by8.
- `gift/torchtitan/trainer.py:747–762,839`: physical count uses labels.numel(); supervised uses non-IGNORE_INDEX labels globally reduced. Existing parser geometry is262144 physical/rank/update,2097152 physical/global/update,2096640 supervised/global/update. The analyzer checks these actual recipe denominators rather than inferring from displayed TPS.
- `gift/torchtitan/components/metrics.py:332–352`: native displayed TPS = local interval physical tokens / (perf_counter interval * non_data_parallel_size). It is a separate, rounded per-device logging metric and must not substitute for integrated-window TPS.
- `measure.py:451–478,634–664,1100–1116`: allocated/active/reserved peaks in bytes are sampled at boundaries, each update and before native peak reset; report per rank and GiB (2^30 bytes). Native metrics resets peaks at `metrics.py:515` each logged update; this recipe logs every update. These are allocator observations, not NVML total-device peaks. Retry/OOM deltas compare cumulative counters at start/end; missing fields remain null.

Device compute, host work attribution, host synchronization/GPU idle attribution, and copies/communication attribution are unavailable from these unprofiled sidecars. Their durations overlap integrated latency and cannot be reconstructed by subtraction. Native sequence statistics do not establish paired noise, a general speedup, convergence, elementwise gradient parity, or checkpoint/EMA-update behavior.

## Executed checks (CPU only)

- Syntax compiled in memory.
- Completed `fqla-capture-002`: eight rank sidecars and bootstraps, status complete, measurement disabled, no throughput.
- Current partial `fqla-native-fla-001`: eight bootstrap/activation records, no final rank sidecars; status incomplete and no throughput. Rank0 had19 logged updates at this one-off validation read. This is not a monitoring record.
- Absent `fqla-native-external-001`: status incomplete with missing-artifact reasons and no throughput.
- In-memory valid140-update/eight-rank window: exact physical TPS2097152 and supervised TPS2096640 for100 seconds, cadence median1 second, byte-to-GiB conversion; removing rank7 switches to incomplete and withholds throughput.
- Statistics known series1..100: median50.5, first20 median10.5, last20 median90.5.
- Synthetic checks validate schema/arithmetic only. No completed140-update native window existed at preparation; actual final-window results remain for later invocation. No GPU work, installations, runner edits or source/config changes were performed.
