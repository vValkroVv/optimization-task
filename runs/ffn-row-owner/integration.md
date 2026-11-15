# Row-owner configuration and GFSDP integration

## Scope and source state

This note owns the root-side integration of the existing row-owner module into
the GIFT MoE config and GFSDP parallelizer. Source base is
`2ada87125e433e1f76c4cb2bfad244b641a78a93`; the shared task checkout also has
the completed FFN overlap edits described in
[`ffn-implementation.md`](ffn-implementation.md). Integration edits are limited
to `gift/torchtitan/config/configs.py`,
`gift/torchtitan/models/gift_moe/_base/parallelize.py`,
`gift/docs/fsdp.md`, focused integration/config tests, and this task record.
The embedding module, backend EMA/DCP state paths, and direct owner correctness
test remain with their existing owner.

## Pre-integration design and runtime-cost review

The public setting is an independent `ParallelismConfig` boolean defaulting to
false. Runtime validation uses the actual `ParallelDims` degrees and model
parameters; the GFSDP entry point also checks the actual dense mesh dimensionality.
The selected implementation converts `model.tok_embeddings` once with that
dense mesh before any GFSDP call, then shares one `ignored_params` set containing
the final `owner.weight` across every nested and root GFSDP call. The ordinary
embedding wrapper is skipped, so the existing unit and FFN prefetch chain keeps
its order with the embedding naturally absent. After root construction and
chain installation, the owner binds to the root communication context and
checks that no GFSDP parameter group captured it. The default-off path keeps
the existing wrapper and chain.

The owner module's inspected contract is in
[`embedding-implementation.md`](embedding-implementation.md): eager forward
deduplicates signed int64 row IDs and exchanges only requested rows; backward
uses native dense BF16 embedding coalescing, transports the BF16 row results,
then widens each source before FP32 owner summation. The per-microbatch path
includes a dynamic count exchange and one explicit host metadata read, sorting
and inverse mapping, request/value and gradient exchanges, native full-vocab
BF16 gradient materialization (1002 MiB output plus scratch at the crop, with a
low-parallelism FP32 temporary of 2004 MiB), and a dense 250.5 MiB local FP32
owner-gradient allocation. EMA retains another 250.5 MiB local FP32 table.
No persistent row cache or full-table gather/reduce-scatter is intended. These
are source-derived costs, not GPU timings or allocator measurements.

The root-side wiring adds no per-microbatch copies, reductions, or host reads;
it constructs the owner at model setup and changes wrapper membership and
prefetch order only by removing the embedding unit. All later costs and stream
lifetime behavior belong to the owner implementation and remain subject to its
H100/NCCL component and native checks. The earlier private-parent-buffer design
in [`ffn-implementation.md`](ffn-implementation.md) is superseded by the user's
updated no-extra-pool FFN contract. This integration preserves the nested
parent/FFN chain; the dedicated buffer owner supplies the final storage-view
and completion review. No speedup is inferred from this wiring.

## Final review and verification

The final wiring converts the input embedding only in the GFSDP path and only
before the first `g_fully_shard` call. It passes the same ignored-parameter set
to every nested wrapper and root, omits the old embedding GFSDP unit from the
chain, sets the remaining chain in its existing order, and attaches the owner
after the root state exists. The default-off integration test verifies that the
ordinary wrapper remains first in the chain; the enabled two-rank test verifies
the owner is absent from every actual GFSDP parameter group while the selected
layer-0 parent/FFN entries remain adjacent and ordered.

Verification used the existing interpreter
`/opt/torchtitan-venv/bin/python` and process-scoped `CUDA_VISIBLE_DEVICES=''`.
The first integration-test attempt failed before wrapping because
`ParallelDims` falls back to a CUDA mesh name when no accelerator is detected.
The test now explicitly patches that mesh selector to CPU for its actual
two-rank Gloo mesh. The rerun passed: **10 tests** in
`tests/unit_tests/test_gfsdp_row_owner_integration.py`; ranks used CPU/Gloo and
the command hid CUDA with a process-scoped environment. This establishes
configuration guards and wrapper/chain structure, not GPU/NCCL lifecycle or
performance.

Scoped Ruff `F821,F811,F841` passed for the changed Python files, and
`git diff --check` passed. The first config-golden check correctly exposed four
missing additive defaults: the three task controls and the already-existing
`gfsdp_reuse_bf16_expert_weights`. The reviewed snapshot delta adds only those
four `false` fields (two rows in four RL snapshots); 60 experiment snapshots
changed, with no preset snapshot changes. After pinning that documented semantic
delta, `test_experiment_parity.py` and `test_model_arch_presets.py` passed: **112
tests**.

Still open are the worker-owned H100/NCCL row-owner route and component results,
the source-level FFN view/completion review under the updated contract, and all
production FP8/SAC, eight-rank, performance, checkpoint-continuation, and
training gates. The integration checks did not run CUDA, allocate GPUs, or
measure performance.
