# GFSDP scheduling implementation

## Scope and acceptance

This note owns the GFSDP and parallelism-schema portion of the 15 November
schedule/embedding assignment. The implementation target is the task-local
`gift/` snapshot at initial revision
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`. The independent capacity source,
test, report and planning edits are preserved.

Acceptance is limited to:

- Head backward prefetch submits both the MTP parent and nested FFN groups;
  the later parent-to-FFN call remains safe and idempotent.
- Construction-time transition setup remaps the last trunk layer's matching
  dense data group and norm's matching dense gradient group to bounded views of
  the existing MTP FFN regions. Runtime sizes, dtype buckets, process groups,
  alignment, offsets and capacities determine the aliases.
- Alias ownership and release events remain visible to overlapping views and
  whole-storage borrowers. Retained-weight and unreduced-gradient conflicts
  are checked over overlapping contexts, with existing recursive-live checks
  and mode guards retained.
- The post-backward transition runs only after a real MTP FFN backward has
  enqueued gradient reduction, resharded/released its data view, and when
  reduction and `reshard_after_backward` are enabled. Unused callbacks and
  retained gradients do not trigger it.
- Both controls decode default-off. The row-owner optimization flag is rejected
  unless row-owner embeddings are enabled and is passed to the embedding
  factory. The transition rejects unsupported configuration before setup.
- Focused unit/integration tests and scoped Ruff pass. GPU/NCCL execution is
  held by the root agent; no GPU is used here.

## Pre-implementation path and runtime review

In `parallelize.py`, `apply_gfsdp` wraps the split MTP FFN before its parent,
then builds one forward shard order and an explicit reverse backward chain.
`GFSDPState._init_shared_state` deals the original data and gradient slots,
partitions MTP parent/FFN regions, and only then binds `GFSDPParamGroup` buffers
in `lazy_init`. `GFSDPParamGroup.unshard` already waits for release events on
the all-gather stream; post-backward reduction records a release event and
clears the gradient owner. `GFSDPBufferContext` already tracks overlap, owner
conflicts and release events in the shared storage registry. The initial
`validate_shared_state` uses exact contexts for mode grouping but interval
overlap for recursive simultaneously-live checks.

Construction costs in the Python root lazy-init path are: create
two aliases with `narrow` views, register them in the existing storage
contexts, validate sizes/alignment, and emit one activation log with actual
pool/region dtype, offset and capacity. This adds no GPU allocation, data
copy, host/device synchronization or collective. The post-backward path adds
one entry-state capture, reduction-enqueued boolean and guarded Python call;
the existing layer-7 all-gather is submitted earlier on the existing
communication stream and waits on the FFN release event. Head prefetch moves
the existing MTP parent and FFN all-gathers before head reduction; the existing
later parent-to-FFN prefetch remains idempotent. No communication payload,
collective count, arithmetic, normalization or precision changes are intended.
The default-off path must retain its prior ordering.

Expected cost risks include overlap scans during root setup and the
event-wait/launch ordering of the moved all-gather. The additional registered
aliases also enlarge inherited runtime overlap scans during unshard, gradient
packing and embedding borrowing, as reconciled in the
[combined review](review.md#final-runtime-cost-assessment).
A queued all-gather may affect later backward waits;
GPU traces are required to determine this. CPU/Gloo checks cannot establish
NCCL ordering, device-side overlap, production pool peaks or native performance.

## Findings and verification

The implementation is in the assigned GFSDP/config/test scope. The two
default-off controls live on `ParallelismConfig` in
`gift/torchtitan/config/configs.py`. `parallelize.py` validates the transition
requirements, passes the row-owner optimization flag to the factory, configures
the source/data/gradient modules before lazy init, and installs the two-target
head prefetch while retaining the generic parent-to-FFN link. `_state.py`
creates root-storage aliases from absolute source ranges after original buffer
dealing, records the target process-group matches and emits one INFO activation
record containing module names plus per-dtype byte offset, alias size, source
capacity and pool size. `_param_group.py` only submits the layer-target gather
after a real backward reduction releases the source data region.

The runtime path creates two bounded `narrow` views and registers them in the
existing storage contexts. No communication pool allocation, buffer copy,
host/device synchronization, collective, payload, reduction arithmetic or
precision change was added. Retained-mode validation now checks pairwise
context overlap; the recursive simultaneous-live check remains in place. The
runtime-path additions are two alias/context registrations, root-init size and
alignment checks, an init-only INFO record, pairwise Python overlap validation,
and a short guarded post-backward branch. The gather is an existing layer
all-gather moved earlier on the current communication stream. GPU timing and
event ordering remain unmeasured here.

## Executed checks

Source revision before these edits: `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`.
All commands ran from the task checkout with `/opt/torchtitan-venv`, Python
3.12, and no GPU visibility.

1. `py_compile` on all nine owned changed Python files passed.
2. `ruff check --select F821,F811,F841 --no-cache` on those files passed with
   `All checks passed!`; `git diff --check` passed.
3. The scoped CPU/Gloo regression command was:

   ```sh
   CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=gift \
     /opt/torchtitan-venv/bin/python3.12 -m pytest -q \
     gift/tests/unit_tests/test_ffn_overlap.py \
     gift/tests/unit_tests/test_gfsdp_buffer_order.py \
     gift/tests/unit_tests/test_gfsdp_ffn_overlap.py \
     gift/tests/unit_tests/test_gfsdp_row_owner_integration.py \
     -k 'not baseline_checkpoint_loads_split_layout_and_continues' \
     --disable-warnings --tb=short
   ```

   Result: **69 passed, 1 skipped, 1 deselected, 4 warnings in 52.59 s**.
   The skipped case requires CUDA streams/NCCL. The deselected case is the
   accumulation-32 SAC/DCP exact-state continuation test; its CPU/Gloo attempt
   fails during the unchanged baseline update before candidate construction
   with `CPUBFloat16Type [64] version 8 expected 4`. Existing
   `runs/ffn-row-owner/buffer-regions.md` records why CPU/Gloo shared-view
   version counters do not validate this path. The test remains configured to
   load the baseline DCP state into the transition candidate and compare the
   next update on the GPU/NCCL route.
4. The alias-layout test passes on two CPU/Gloo ranks. It confirms bounded
   absolute storage offsets, byte alignment, existing storage pointers, equal
   unique pool storage between control/candidate, and validation of overlap
   modes. The fixture raises only the tiny MTP FFN hidden width to make its
   source region large enough for the target group; the default tiny fixture
   is correctly rejected at lazy init when the target requires 82,176 bytes
   but the FFN region has only 49,152 bytes. Runtime capacity is checked from
   the actual parameter groups, not a configured crop size.
5. The factory spy confirms `optimize=True` reaches
   `RowOwnerEmbedding.from_embedding` through the distributed parallelize path.
   CPU tests also cover both flags decoding default-off, invalid optimization
   without row-owner embeddings, unsupported transition requests, head
   multi-prefetch targets, generic parent-to-child prefetch idempotence,
   unused/stale post-backward callbacks, no reduction, and no reshard.

## Required root-owned gates

The root agent owns GPU execution and native training. The focused NCCL suite
to run serially is:

```sh
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=gift \
  /opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  gift/tests/unit_tests/test_ffn_overlap.py \
  gift/tests/unit_tests/test_gfsdp_buffer_order.py \
  gift/tests/unit_tests/test_gfsdp_ffn_overlap.py \
  gift/tests/unit_tests/test_gfsdp_row_owner_integration.py \
  --disable-warnings --tb=short
```

This must include the pending-reader/whole-storage borrow test and the
accumulation-32 SAC/DCP baseline-to-transition continuation. Before interpreting
native performance, confirm that the one-time `GFSDP buffer region transition
active` record appears with the production source and target FQNs and actual
byte capacity, then use the task's required matched native runs and Perfetto
profiles to verify ordering, event waits, pool peaks, unchanged collectives and
downstream stalls. This agent did not access GPUs, submit jobs or report a
speedup. Those checks remain open; component and native performance acceptance
is outside this implementation handoff.

## Independent review R1 follow-up

The transition is restricted to exactly one MTP layer in both
`_validate_mtp_transition_request` and the direct `apply_gfsdp` entry point.
When the transition flag is false, the public validator still returns before
inspecting the MTP container, preserving the existing multiple-head path. The
focused tests cover enabled multiple-head rejection at both entry points, the
flag-off multi-head validator no-op, and acceptance of a valid MTP1 request.

On 2026-11-15, the affected CPU validation command returned **9 passed, 13
deselected, 1 warning in 3.37 s**:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=gift \
  /opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  gift/tests/unit_tests/test_gfsdp_row_owner_integration.py \
  -k mtp_transition --disable-warnings --tb=short
```

Scoped `ruff check --select F821,F811,F841 --no-cache` on `parallelize.py` and
`test_gfsdp_row_owner_integration.py`, plus `git diff --check`, passed. The
root-owned 71-test CUDA/NCCL suite already passed for MTP1 before this
validation-only guard; it was not rerun. No GPU test or native run was done for
R1.
