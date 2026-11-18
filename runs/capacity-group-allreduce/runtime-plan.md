# Capacity all-reduce runtime plan

The owner subsequently requested reuse of the already saved Triton cache during
the first startup. The [current cache policy](report.md#owner-correction-reuse-the-saved-compiler-cache)
supersedes this initial plan's fresh-cache timing commands: both timing variants
use the completed production-data D cache through `--compiler-cache-root`, and
the diagnostic reuses that root after candidate completion. Native receipts own
the final unique run IDs and exact executed commands. The interrupted fresh-cache
startup is retained and does not supply a valid timing result.

## Acceptance and live route

This worker verifies the route, existing inputs/runner and concrete bounded commands; it does not execute GPU work or change source. Live receipt: [runtime-preflight.json](runtime-preflight.json).

Direct execution is available inside the same recorded owner allocation, hostname `lm-mpi-job-23027817-4bfe-4730-926f-5c2de590b32b-mpimaster-0`. On 15 November at 10:52 UTC, all eight H100 80GB GPUs had 0 MiB used, no compute applications and no task training processes. The task is the actual SR008 NFS directory. Python `/opt/torchtitan-venv/bin/python3.12` reports torch `2.15.0.dev20260822+cu130`, CUDA 13.0; `/etc/profile.d/wm.sh` exists. The coordinator accepted this already-recorded local route; the owner then explicitly confirmed, “you are on server”. The `sr008-gpu` SSH alias failed name resolution; no other host was probed. No new allocation is needed.

Starting task revision is `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`, `main`, initially clean. Later task-worker edits must be preserved. Recheck idle GPUs immediately before GPU execution. Existing native preflight checks all eight H100s, runtime package/header pins, installed backends and task-local import path.

## Reused input evidence

The existing runner's `_verify_ready_inputs(TASK_ROOT)` passed read-only. It checks tokenizer hashes, data-002 index and row-manifest hashes, file sizes, receipt linkage and completed depth-0/depth-32 CPU loader receipts. No recensus is needed. Dataset SHA256 is `1ce62620b6dd2cf5fde6715ed5776a247d1cc1bd2df725f5c1e5f2b0effc3a70`; data receipt SHA256 is `b244e47a1fd0e7a0440b666cf225e09905e4e211dfa62ac6170da51c22685037`. Canonical inputs come from `runs/production-data-abcd/data_contract.py`; tokenizer is the verified `dclm-7b-128k_digit_tokenizer_FIM` production asset. Data-002 and existing D timing cache are present.

## Minimal control and guard strategy

Keep one byte-exact original `capacity.py` snapshot plus candidate SHA256 and a small reversible aggregation-order patch in this evidence directory. Do not reset the tree or replay historical repository revisions. Before each native run, verify every tracked/untracked GIFT source byte against the frozen candidate manifest, allowing exactly the original/candidate `capacity.py` hash and affected test hashes. Execute control by temporarily restoring only this one file while no training is live; restore candidate after control exits, before candidate timing. Verify all other source hashes remain equal. Keep candidate unchanged through its timing and diagnostic.

`runs/flash-qla/run.py` itself has no historical fixed capacity-source hash to loosen: `_source_identity` hashes FlashQLA backend, staged package and bridge files. Its base runner `source_snapshot()` hashes every tracked/untracked GIFT file before/after each run and rejects any change while running; keep this guard intact. Bridge config hashes also remain guarded. Therefore no guard disabling is required. A task-owned launch wrapper should add explicit per-variant expected hashes/one-file allowlist before invoking the existing runner; it must not mask the base source-freeze check. Cache reuse validates completed producer status, stopped process group, runtime fingerprint and canonical config hash. Use the candidate timing cache only for candidate diagnostic.

Native cleanup is restricted to the child process group created with `start_new_session=True`: SIGTERM, bounded wait, then SIGKILL for that group only. No broad cleanup command is present. Preserve Perfetto services and unrelated processes.

## Component benchmark to implement once, then run

No existing capacity NCCL microbenchmark was found. Reuse functional `torch.distributed._functional_collectives.all_reduce` from `capacity.py`, existing int64 count oracle/mocks in `test_token_drop.py` and `test_token_drop_fixed_capacity_group.py`, actual Gloo collective cases in `test_token_drop.py`, and compile/SAC checks in `test_token_drop_compile.py` and `test_moe_checkpointing.py`. Existing production component fixtures capture embedding/FFN tensors and add no useful coverage for this integer metadata fragment; do not reacquire them.

Add a small task-owned `component.py` only, with one NCCL process group of all eight ranks and rank-local device selection. Use deterministic bounded nonnegative int64 `[128]` counts varying by rank; expected global loads are an exact integer oracle. Setup/warmup are outside timing. Warm both variants (e.g. 50 rounds) then 200 paired rounds, alternating A/B versus B/A order each round. Report complete raw per-rank microseconds, medians/p10/p90 and slowest-rank paired series.

Measure separately: (1) collective A on precomputed `[128]` versus B on precomputed `[8]`; (2) complete fragment A `wait(all_reduce(counts)).view(8,16).sum(1)` versus B `wait(all_reduce(counts.view(8,16).sum(1)))`. Do not mutate fixture counters through in-place APIs. Both use the exact functional collective path and group; call functional `wait_tensor` before the end event so lazy async results cannot escape the timed interval. Use CUDA events and end-event synchronization to establish actual device completion. Apply identical barrier/device-sync arrival preparation before every measured sample; keep barrier outside the event interval. Record wall completion separately if desired; do not label CPU enqueue timing as collective latency. Verify exact equal group loads outside timing for every arm. No data/model download, training preset or source override is needed.

Planned command, after `component.py` exists and passes focused review:

```bash
cd /workspace-SR008.nfs2/vkropotin/optimization-task
source /etc/profile.d/wm.sh
PYTHONPATH="$PWD/gift" /opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max_restarts=0 runs/capacity-group-allreduce/component.py --output runs/capacity-group-allreduce/artifacts/raw/component-001 --warmup 50 --pairs 200
```

## Native timing and final diagnostic

Use D argv verbatim from `../production-data-abcd/campaign-plan.json`, replacing run ID/output only. Commands below are the original runner invocation; the planned explicit source-hash wrapper may surround them. Execute serially, control then candidate, with fresh cache for each. No extra A/A or repeats are requested.

```bash
/opt/torchtitan-venv/bin/python3.12 runs/flash-qla/run.py --gdn-backend external_gate --routing-metadata fused --cross-entropy stable --num-prefetch-batches 32 --epoch-size 102400 --gfsdp-reuse-bf16-expert-weights --gfsdp-split-mtp-ffn --gfsdp-split-layer0-ffn --gfsdp-row-owner-embeddings --run-id capacity-control-timing-001 --output-root /workspace-SR008.nfs2/vkropotin/optimization-task/runs/capacity-group-allreduce/artifacts/raw --steps 20 --warmup 10 --window 10 --observe-compiler-cache
/opt/torchtitan-venv/bin/python3.12 runs/flash-qla/run.py --gdn-backend external_gate --routing-metadata fused --cross-entropy stable --num-prefetch-batches 32 --epoch-size 102400 --gfsdp-reuse-bf16-expert-weights --gfsdp-split-mtp-ffn --gfsdp-split-layer0-ffn --gfsdp-row-owner-embeddings --run-id capacity-candidate-timing-001 --output-root /workspace-SR008.nfs2/vkropotin/optimization-task/runs/capacity-group-allreduce/artifacts/raw --steps 20 --warmup 10 --window 10 --observe-compiler-cache
/opt/torchtitan-venv/bin/python3.12 runs/flash-qla/run.py --gdn-backend external_gate --routing-metadata fused --cross-entropy stable --num-prefetch-batches 32 --epoch-size 102400 --gfsdp-reuse-bf16-expert-weights --gfsdp-split-mtp-ffn --gfsdp-split-layer0-ffn --gfsdp-row-owner-embeddings --run-id capacity-candidate-diagnostic-001 --output-root /workspace-SR008.nfs2/vkropotin/optimization-task/runs/capacity-group-allreduce/artifacts/raw --steps 21 --warmup 20 --window 0 --profile-update 21 --profile-microbatches 3 --observe-compiler-cache --compiler-cache-root /workspace-SR008.nfs2/vkropotin/optimization-task/runs/capacity-group-allreduce/artifacts/raw/capacity-candidate-timing-001/cache
```

Keep fixture acquisition environment unset (`ABCD_COMPONENT_FIXTURE_DIR`, FlashQLA capture); no BF16 counters are needed. Both timings retain all eight ranks, B2/S4096/global512/accum32, original seed2048, LR65000/2000, FP8/compile/SAC/MTP/EMA/optimizer/capacity and monitoring-off controls. Verify actual child args, resolved config and feature activations before interpreting results. Collect complete updates11–20, physical/supervised TPS, wall elapsed, cadence drift, allocated/reserved peaks and failures. Diagnostic captures only ranks0/1, max three complete F/B microbatches of update21 while all eight complete accum32 and optimizer. Preserve raw trace bytes/hashes. MCP Perfetto must reconcile import coverage and show eight int64 counters with preserved order/compile/SAC. Historical matching D rank0 trace is `artifacts/4fix_qla_embed.json` (SHA256 `a75ba588a23695b68fbbd575d381017cd0db6cf2d89ad9dcf6cf1da4dd460d90`); original rank0/1 files remain in `runs/production-data-abcd/artifacts/raw/proddata-d-diagnostic-001/native/profiling/traces/iteration_21/`.

## Remaining risks and stop

Reduced bytes may leave NCCL latency unchanged. Local sum placement can alter fusion/kernel count; separate fragment timing is required. Component synchronized arrival does not establish native peer-arrival latency. One native pair is provisional. Source must remain frozen while children live; control restoration must touch only the expected file. Existing preflight may expose a real environment mismatch; retain the first failure and repair only a concrete task-owned harness defect. No compute was executed by this worker. Stop after the component, exactly two timing runs and one final diagnostic plus required correctness/review/report gates.
