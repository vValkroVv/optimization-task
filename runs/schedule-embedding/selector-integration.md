# Selector integration preflight

Status: source and harness checks pass. The seven timing runs and four diagnostics have not been launched.

## Native selector

`ParallelismConfig.gfsdp_row_owner_embedding_optimization_stages` is an optional tuple. `None` preserves the existing defaults: aggregate optimization disabled, or all four stages enabled when the existing aggregate flag is `True`. An explicit selector conflicts with that aggregate flag, even when the selector is empty. A nonempty selector requires row-owner embeddings. The existing stage resolver rejects unknown and duplicate names and restores canonical stage order.

The selector passes through `parallelize_gift_moe` and `apply_gfsdp` to `RowOwnerEmbedding.from_embedding`. No embedding algorithm, CUDA kernel, precision path, or communication path changed for this deliverable. The selector is resolved as construction metadata; it adds no per-update work, device buffer, copy, synchronization, collective, or compilation step.

The harness records the resolved module names and stage tuples once per rank after `Trainer` initialization and before the first update. That receipt performs a CPU module walk and one JSON write before training updates. The temporary `WindowMeasurement.__init__` wrapper preserves the base constructor's keyword-only `trainer` argument and restores the original initializer after receipt creation. It reads no device values.

## Campaign protocol

The seven timing variants are `A`, `B`, `C`, `D0`, `S`, `SE`, and `SE-fused`. A/B/C use the historical recipes from `runs/production-data-abcd/campaign-plan.json`, with later controls explicitly false and the selector null. D0/S/SE/SE-fused use the accepted D recipe. SE selects `compact`, `owner_add`, and `owner_lookup`; SE-fused adds `routing`. The native parser round trip confirms that SE and SE-fused differ only in this selector.

Timing commands use 140 steps, 40 warmup updates, and a 100 update window. Diagnostics for D0/S/SE/SE-fused use update 21, 20 warmup updates, and no timing window. The executor requires all seven timing runs in order before diagnostics, continues independent timing runs after a run failure, skips diagnostics if any timing run did not pass, and refuses attempt reuse. New run IDs use the `140-003` namespace and receipts use `artifacts/raw/campaign-140-receipts.jsonl`; the historical 10+10 outputs remain untouched.

The source freeze covers the GIFT snapshot; base configuration, input, and tokenizer hashes; FlashQLA bridge files; schedule run/measure/campaign files; data contract, overnight runner/measure/parser, compiler observer, streaming file lock, historical plan, and original capacity snapshot. It checks both the original snapshot and its tracked Git HEAD source against SHA-256 `2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120`. The existing compiler observer remains boundary-only; no per-step hooks or NFS scans were added. All seven timing commands reuse the same accepted D compiler cache at its existing absolute path; the in-place cache may gain specialization entries during warmup or measurement.

## Verification

- `campaign.py --dry-check` passed with CUDA hidden. It checked all 7 timing and 4 diagnostic child commands, actual native config decoding, all 7 stage-receipt variants, all 7 runner main hooks, capacity-overlay identity, and SE/SE-fused selector differences.
- `pytest -q tests/unit_tests/test_row_owner_embedding_stage_config.py` passed: 9 tests. This covers omitted/null/empty/3-stage/4-stage CLI decoding, aggregate compatibility and conflicts, row-owner requirements, and unknown/duplicate/canonical stage resolution.
- `pytest -q tests/unit_tests/test_gfsdp_row_owner_integration.py::TestGFSDPRowOwnerIntegration::test_row_owner_stage_selector_reaches_embedding_factory` passed on two-rank Gloo. It exercised the actual `parallelize_gift_moe` to `apply_gfsdp` to `RowOwnerEmbedding.from_embedding` path.
- `pytest -q runs/schedule-embedding/test_campaign_diagnostic_gate.py` passes three stubbed executor checks: all four diagnostics after seven timing passes, diagnostics-only with accepted prior timing receipts, and a failed timing run blocking diagnostics.
- On GPU 0 (H100), `test_routing_kernels_clamp_signed_limits_and_count_empty_intervals` passed. The optimized routing metadata matched the fallback for signed int64 extremes, empty IDs, duplicates, skew, and N=8192. All eight GPUs were idle at the preflight check. The test used personal cache roots `/workspace-SR008.nfs2/vkropotin/.triton/cache` and `/workspace-SR008.nfs2/vkropotin/.sr008-gpu/.cache/torchinductor_user`; the saved D cache was not used.
- `py_compile`, full configured Ruff checks on the three schedule-embedding harness files, scoped `F821,F811,F841` checks on all changed Python files, and `git diff --check` passed.
- The protected capacity source and fixed-capacity test hashes still match `164af05bd72ceb0675f0d9f756ecf04926ced17430a58dd2e38bc160c514dfdc` and `2abffc21929ed2dd1c4731f67a8602d051866ea496aecc20abdeaf4c476260b6`.

The final preflight freeze is `3b3244ae1e759111b8240af929715074b751c946881202a9ff95e9f3b9918ffd`. The full compact command, stage, capacity, cache, helper-hash, and preflight-failure record is in `selector-integration.json`.

Initial preflight findings are retained there with their fixes: the receipt dry check reused a patched class; the actual constructor has a keyword-only `trainer`; the capacity snapshot is task-untracked and must be compared with the tracked source path; the CLI test fixture needed `config_version: 1`; a timing-receipt helper was missing during initial lint; and the diagnostic gate compared the full completion set rather than verified timing passes. The final checks cover the corrected behavior.

## Remaining gate

No matrix run has started. The live eight-rank stage receipts, exact 140-update rank cadence and token counters, all-rank memory and process-group receipts, and diagnostic captures remain to be verified by the authorized campaign launch after independent review and root GO. The executor's exact round-3 command is recorded in `selector-integration.json` and unsets `CUDA_VISIBLE_DEVICES` so the child sees all eight GPUs.
