# Short native performance preparation

## Correction after the first launch

Attempt001 failed before native updates because the launch did not provide the
three required allocator-observer identity variables. Its artifacts and all
preparation001 bytes remain frozen. Original002–005 are not run and superseded.
New006–010 restart the five-run sequence. No CUDA fault occurred in001.

Every new launch exports these constructor-required identities:

- TORCHTITAN_PHASE_MEMORY_RUN_ID: PERFORMANCE-ASSESSMENT-attempt-NNN
- TORCHTITAN_PHASE_MEMORY_SOURCE_ID: the common Source010 manifest SHA256
- TORCHTITAN_PHASE_MEMORY_CONFIG_ID: the common observer YAML SHA256

These are also in the per-rank environment allowlist. Run IDs identify each
attempt without changing the off/on work. Source/config IDs share identical
meaning across both variants; compiler receipts record graph activation.
RANK/LOCAL_RANK/WORLD_SIZE come from torchrun; CUDA_VISIBLE_DEVICES is explicitly
pinned. Inspected wrapper startup also needs GIFT_HYBRID_EP_SCAN_RUNTIME_DIR and
RANK for the scan correction, both already supplied. No further required
identity exists on this startup path.

The CPU smoke reproduces the original missing-RUN_ID failure, reads all three
actual export values from each new launch, invokes the real observer constructor
and installer using the existing accepted fake native modules, performs native
metrics1/10/20/30/40, checks complete output, and restores bindings/environment.
It imports no real torch and initializes no GPU. The normal config check from
preparation001 is reused: YAML and common CLI settings are unchanged except
attempt-local dump paths. Only identities, allowlist, attempt/controller scope
and their frozen hashes change. The three host environment reads at startup
and allowlist JSON entries add no per-update/device/collective work.

## Acceptance and status

Ready for root review and per-attempt release; no GPU query, CUDA initialization,
training, benchmark, trace or process termination was performed by this preparation.
The root separately owns the fresh preflight receipt and source freeze.

Five attempts run manually and serially: 006 off-AA1, 007 off-AA2, 008 on-B1,
009 off-A2, 010 on-B2. Stop after any invalid attempt. Each launch uses native
Trainer for 41 optimizer updates, closes the twenty-update timer at the first
batch-start of update 41, and preserves scheduler horizon 30000 and seed42.
Expected completion is164 microbatches/rank; useful window tokens are20,971,520
already global. The timing output remains the authority on validity.

`config-check-report.yaml` records the successful torch-free pre_submit call
with the complete common CLI overrides and world-size8. `config-resolved.yaml`
contains typed defaults followed by the CLI success line. `cpu-identity-smoke.json` records actual observer installation, completion and
restoration using the accepted CPU native fixture for every corrected attempt.
`artifact-hashes.json` freezes runnable preparation/attempt artifacts.

The phase entry passes bounds startup20/final-metrics40 to the accepted observer,
uses its existing try/flush/restore behavior and rejects non-complete phase JSON.
MetricsProcessor.should_log emits step1 and multiples of10; native41 ends after
metrics40. Inspected keys match phase_memory._document. Native Trainer,
train_scan_fix, backend correction and compiler entry are reused unchanged.
All attempts explicitly pin visibility to the preflight's eight UUIDs in physical
index order, unset CUDA_LAUNCH_BLOCKING and record its unset state per rank.

## Release and one detached launch

Root must first write this attempt's release.json with these fields:

- status: COORDINATOR_CONFIRMED_FOR_LAUNCH
- scope: PERFORMANCE-ASSESSMENT/attempts/006 (change only the attempt suffix)
- source_manifest_sha256: b2e9144af50e8673e0b8646a4f03f629bc240e6fc3ed1aaccfe6814491fb7d20
- config_sha256: 01675fbc967e2e0a7e3798d333edc117d18bf5ad3746d999a86f566a3ff554f8

On the verified runtime host, launch exactly one released attempt:

```bash
TASK=/workspace-SR008.nfs2/vkropotin/optimization-task
PERF="$TASK/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-ASSESSMENT"
ATTEMPT="$PERF/attempts/006"
(set -C; EXPECTED_SOURCE_MANIFEST_SHA256=b2e9144af50e8673e0b8646a4f03f629bc240e6fc3ed1aaccfe6814491fb7d20 \
  nohup /opt/torchtitan-venv/bin/python3.12 "$PERF/preparation/002/controller.py" 006 \
  > "$ATTEMPT/controller.log" 2>&1 < /dev/null &)
```

The controller remains the live parent until the exact accepted process owner
returns. The owner bounds the attempt at1800seconds and records/cleans only
identified task descendants. Its --entrypoint is the exact phase_entry.py path
and working directory is the verified gift directory. The controller does not
launch another attempt. Canonical full torchrun argv is in each command.txt.
Launch rejects old runtime/native/log outputs; controller rejects old ownership
outputs; exclusive redirection rejects an old controller.log.

Read controller-complete.json, owner-status.json (including recorded identities
absent), exit-status.json, full launcher/torchrun/native logs, all8 compiler
receipts, all8 environment/phase-memory records, timing validity, finite metrics,
counts, sampler gaps and stderr before releasing the next attempt. These are
execution gates owned by root; a zero launcher exit alone is insufficient.

## Scoped runtime-cost review

Inherited native fetching,18compiled SAC model blocks, chunked compiled loss,
accumulation4, clipping, fused optimizer, scheduler and native metrics remain on
the integrated path. All model graphs are excluded. Only agreed graph activation
changes between variants; both preserve mix_order_reduction=false and partitioning.
No numerical/CP/recorder/checkpoint/restore work is added. Existing callbacks and
accepted phase-memory wrappers run identically on each side. Phase-memory copies
existing host allocator results and native reset metadata; it adds no per-update
allocator query, device copy, collective or synchronization. Environment recording
is once per rank before native setup; completion JSON/checks are after native exit.
The accepted owner polls host process identities every0.25s; the existing external
nvidia-smi sampler runs every500ms. These matched host costs remain unmeasured.
Persistent graph pools and staging may change device memory; root's matched
window and allocator envelope measurements decide actual cost.

Primary throughput is global window tokens divided by maximum completed-window
wall time across ranks; wall/20 is a completed mean. Per-update host boundary and
current-stream event spans are not complete-update medians or pure compute time.
Reuse A008 all-rank replay proof21–23; replay/recapture21–40 stays unobserved.
The prior native fault at48 remains unresolved. This preparation establishes
runnable artifacts and static/configuration checks, not GPU correctness or a
performance result. No new diagnostic or long-run gate is introduced.
