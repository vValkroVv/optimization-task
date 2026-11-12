# SANITIZER-ORIGIN / attempts/001

## Status

Prepared for root review. No watcher or runner was started, no GPU/NVML query
was made, and no training or sanitizer target was launched. No GIFT source was
changed. The attempt directory contains only the pinned entrypoint/config,
runner, watcher, copied helper, CPU fixtures, pins and this handoff.

## Pinned run plan

- SOURCE-SNAPSHOT/007: 92 files; manifest SHA-256
  `2585e504350276cb4689ae9ea45a33706e9f898bb61969abe5d08f2650c55620`, archive
  SHA-256 `0fc5ddad83c5f99b4036a47dd7f485ba2f42d1284c42f086d5f051cebad66082`,
  base `446431dfa191ef0300f116257ae918bc85e38c06`. The runner checks each
  manifest file and requires `git diff --name-only HEAD -- gift` plus untracked
  GIFT files to equal the same 92 paths.
- Preset SHA-256: `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`.
  Scan-fix entry and implementation SHA-256 values are pinned in `pins.json`.
- Configuration keeps the 3B FlexAttention/HybridEP FP8 preset, FSDP2/EP8,
  selective AC, model/loss compilation, batch 8/256, sequence 4096, accumulation
  4, seed 42, native 60-update horizon and total schedule 30000. All 18 model
  layers remain compiled and graph-excluded. The public
  `campaign_cudagraph_step_boundary` marker remains. Completion/timing observers
  and profiling are absent. Graphs are off, mix-order reduction is zero, graph
  partitioning remains true, and the scan-fix entrypoint is retained.
- The one `torch.distributed.run` process tree is wrapped by the installed
  Compute Sanitizer v2026.1.0.0 build 37182542, SHA-256
  `5cba659e1cd603aa2370e103425f184368e6ed405da90ce85743c4399c324dbc`. Verified
  local help accepts `--tool memcheck`, `--target-processes all`,
  `--error-exitcode 86`, `--show-backtrace yes`, `--log-file`,
  `--print-limit 50`, and `--launch-timeout 180` (seconds). Local help documents
  `%p` as the Compute Sanitizer frontend PID substitution for the log filename.
  The shipped manual documents `NV_COMPUTE_SANITIZER_BINARY_PATCHING`; the
  runner sets it to `0`. It does not use a historical `--binary-patching` flag,
  forced synchronization, padding, filters, or coredump generation. Its shell
  limit sets core size to zero for this task process only.
- The runner uses one 20-minute timeout, 60 native updates, and torchrun restart
  count zero. HOME is inherited, existing caches are reused, and the HybridEP
  runtime path is fresh. The runner refuses to proceed without a live READY
  receipt and a reviewed `launch-gate.json`.

## READY and launch gate

Start `watch_binary_provenance.py run --attempt-dir <this attempt>` and wait for
`watcher-ready.json` before invoking `launch.sh`. READY binds the watcher PID and
start ticks, exact attempt path, watcher/helper hashes, and the required process
discovery rule. The launch script calls `verify-ready` and the receipt writer
revalidates the live watcher before creating an exclusive launch receipt.

After review, the root-owned `launch-gate.json` must carry these values:

- `status=COORDINATOR_CONFIRMED_FOR_LAUNCH`, exact attempt path, current
  `pins.json` SHA-256 `58b307cf17bec3dc46dcdceb7942abe8e79374dd8ea17959d0ae3acc9b8f3c16`,
  SOURCE007 manifest/archive hashes, sanitizer hash, and `compute_release=true`.
- `gpu_preflight_status=PASS`, `gpu_count=8`, eight unique GPU UUIDs,
  `topology=single_node_nvlink`, `gpu_free_bytes` with at least 4 GiB on each
  device, and timestamp `gpu_checked_at_utc`.
- `initial_storage_free_bytes` at least 4 GiB plus the 64 MiB output allowance,
  and timestamp `storage_checked_at_utc`.

The watcher samples free space every three seconds after launch. Below 4 GiB or
on a watcher storage-write failure, it signals only the runner PID and rank PIDs
whose `/proc` start ticks and attempt path still match. It does not enumerate
and kill unrelated processes. It records the stop action when storage permits;
there is no storage reservation or cleanup.

The 64 MiB allowance covers native logs, sanitizer output, and small watcher/
helper receipts only. It is estimated from the larger quiet MATCHED-RECAPTURE
attempt footprint (16,287,481 bytes over 26 updates), scaled to 60 updates
(37,586,494 bytes), rounded to a 40 MiB native allowance, plus 24 MiB for up to
50 sanitizer reports. `--print-limit 50` limits report count, not bytes per
report, so this is a planning estimate rather than a hard cap. Fresh scan-runtime
and existing compiler-cache growth were not measured and are outside the budget;
the active 4 GiB floor is the stop condition, not a storage guarantee.

## CPU checks completed

- `verify_watcher_cpu.py`: 7 fixture groups passed. These cover fail-closed live
  READY/launch gates, exclusive receipts, all-eight-rank mapping and one-shot
  trigger, sequence gaps, a missing rank, error-before-update-1, dead PIDs,
  storage stop targeting and write failure. The real-record fixture reads only
  prefixes through `optim_end` step 1 from the prior COMBINE-BWD-PROBE structured
  logs, then uses mocked `/proc`, helper and storage; it does not inspect old
  process IDs or call the helper on them.
- The existing LOADED-BINARY-RECEIPT CPU suite was reused against this attempt's
  helper copy: 10 focused checks passed. No helper tests were duplicated.
- `bash -n launch.sh` and syntax compilation of the five Python files passed.
  The local source-changed set was 92/92. Sanitizer CLI help/version and shipped
  manual/release-note hashes were checked locally; no sanitizer target ran.

## Remaining gates

Root review, live READY creation, fresh eight-GPU NVML and storage records,
explicit compute release, and actual launch remain pending. Compatibility,
memory sufficiency, eight-rank sanitizer tracking and exception attribution are
unverified. This mode tests attribution only; a clean or nonreproducing run does
not establish memory correctness or clear the fault. No performance claim or
automatic retry is selected.
