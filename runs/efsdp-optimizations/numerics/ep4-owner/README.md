# Native EP4 compact BF16 owner numerical diagnostic

Run two baseline repeats and one compact arm. Each runs one untimed native
optimizer update with 32 microbatches on eight H100s. The runner derives the
recipe from the verified original A receipt. All arms use the original data,
FP8, model and loss compilation, selective activation checkpointing, native
HybridEP, optimizer, and all four Expert GFSDP switches. The diagnostic enables
strict deterministic mode.

## Stage and plan

Use dedicated task-owned source roots. Stage the [deferred dW overlay](../../experiments/deferred-dw/deferred-dw.patch)
in both roots, then stage the [compact BF16 slot overlay](../../experiments/owner-bf16-slot/owner-bf16-slot.patch)
in the compact root. Record both overlay SHA256s and the full GIFT snapshot
SHA256 of each staged root. Apply both launcher patches in each staged root.

```sh
common=runs/efsdp-optimizations/numerics/native-numerics.patch
owner=runs/efsdp-optimizations/numerics/ep4-owner/native-ep4-owner.patch
git apply --check "$common"
git apply "$common"
git apply --check "$owner"
git apply "$owner"
```

Print the GIFT SHA with the native driver's snapshot algorithm. The hash covers
tracked and untracked GIFT paths and file modes.

```sh
python -B - <<'PY'
import runpy
from pathlib import Path

runner = runpy.run_path("runs/overnight-streaming-perf/run.py")
print(runner["source_snapshot"](Path.cwd())["tracked_and_untracked_sha256"])
PY
```

Run the planner for both roles. With the reference supplied, the launcher
checks configuration and input provenance and passes the digest to each rank.
A parser-only plan without a reference has an empty digest.

```sh
python -B runs/efsdp-optimizations/run.py plan "$run_id" \
  --ep 4 --numerical-ep4-owner "$role" \
  --optimization bf16-cache --optimization fp8-cache \
  --optimization direct-dw --optimization tail-retention \
  --direct-dw-build "$build_root" --reference "$original_a"
```

After native preflight, replace plan with run and add
--expected-gift-source "$staged_source_sha256". Use baseline for the deferred
source and compact for the compact BF16 source. Use distinct run IDs and
fresh compiler caches, the same hardware and runtime, and the same direct
dW build manifest and observer bytes. Keep the complete native raw attempts.

## Capture and compare

Each reduction audits membership in one of the pairs [0,4], [1,5], [2,6],
and [3,7]. Each microbatch must reduce layers 7 through 1 in order. A
baseline rank observes 224 native FP32 reduce-scatter calls. A compact rank
observes 448 BF16 all-to-all calls and checks the peer-owned source offset,
two chunk lengths, disjoint receive storage, and FP32 destination. Native
completion follows all 32 microbatches.

All ranks sample 4,096 elements of each full produced projection in layers
1, 4, and 7 just before reduction. The observer widens the selected compact
BF16 values exactly to FP32. Owner-local weight, gradient, and Adam samples
use independent indices. Flattened W13 precedes W2. Ranks 0–3 own 301,989,888
W13 elements and zero W2 elements. Ranks 4–7 own 100,663,296 W13 and
201,326,592 W2 elements. The receipts record empty owner-local parameter and
gradient state explicitly. An existing zero-shard Adam counter is retained
and zero-length moments are recorded. Full produced W2 remains sampled on
every rank.

The shared observer hashes bounded initial model, optimizer, and RNG state
and all native inputs. It records unrounded losses, unclipped grad norm,
router samples, scheduler state, and counters. Samples fit the inherited
4-MiB raw array budget and transfer after the update. The diagnostic changes
launch pressure, so measure throughput in the separate native runs.

```sh
python -B runs/efsdp-optimizations/numerics/ep4-owner/ep4.py \
  "$a1_attempt" "$a2_attempt" "$b_attempt" \
  --baseline-source-sha256 "$a_sha256" --compact-source-sha256 "$b_sha256" \
  > ep4-owner-numerics.json
```

The comparator validates the receipts and reports all three pairs with
exact equality, maximum and p99 absolute error, and relative L2 per field.
A nonexact result exits 1 with metrics preserved. One A/A pair cannot supply
a tolerance. Report the observed variation and first differing sampled
boundary before assigning a cause.

## Local checks

```sh
python3.12 -B runs/efsdp-optimizations/numerics/ep4-owner/test_ep4.py
ruff check --select F821,F811,F841 --no-cache runs/efsdp-optimizations/numerics/ep4-owner
```

Local checks cover both real untrained rank plans, flat owner ranges,
empty or absent Adam state, peer-view offsets, exact BF16 widening, and
invalid shape and dtype rejection. The compiled eight-H100 update remains
the native gate.
