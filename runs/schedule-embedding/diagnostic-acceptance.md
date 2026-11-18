# Frozen update-21 diagnostic acceptance contract

Design only; no launch, parser execution, trace read, source change or GPU work
was performed for this note. Applies to the four diagnostic entries in
`artifacts/raw/newlaunchrecords/repair-004-plan.json`, pinned SHA256
`1a193606963e2f3c7086a6afb1d6d56ad78692dae54121e257c05c18fc8a5968`, and
combined campaign freeze
`3b3244ae1e759111b8240af929715074b751c946881202a9ff95e9f3b9918ffd`.

## Decision and finite acceptance

Use each exact direct `run.py` argv from the immutable repair plan, serially
D0, S, SE, SE-fused. Start only after the retained A003 and six004 timing rows
have separate valid offline acceptance records against the same freeze. Keep
the frozen campaign/run/measure and overnight runner unchanged. Record the
direct-launch provenance and timing prerequisites outside original raw files.
Do not route these runs through frozen `campaign._execute`: its timing-receipt
and raw `completion.json` PASS requirements cannot consume offline recovery.

Accept each diagnostic through a mode-specific offline proof that reuses frozen
native `parse_results.summarize_run` plus the existing strict source/config,
recipe, nullable-selector, bootstrap and stage checks. Extend the existing
offline adjudicator narrowly if implementation is requested; do not build a
second controller. Current timing adjudication explicitly rejects 21/20/0 and
profiling, so it cannot be applied unchanged. Store a distinct offline summary
and acceptance receipt; preserve every original status, failure and completion.
Do not create a synthetic raw completion or insert receipts into the frozen
campaign as if it had accepted the runs.

Acceptance is complete when all four have (1) provenance/config/activation,
(2) clean native completion, (3) native parser capture acceptance and (4) original
trace tail coverage. A failed gate stops acceptance of that diagnostic. A
successful earlier diagnostic is retained; a raw nullable FAIL alone does not
justify another training run.

## Indispensable checks

1. **Provenance and recipe.** Pin immutable plan, freeze and executed helper
   identities; compare pre/final source and configuration receipts rather than
   rescanning trees while timing runs execute. Preserve saved D cache producer
   identity and verified real-data/tokenizer input identity. Parse the actual
   child argv: eight ranks, schedule measure, fresh 21 updates, 20 warmup,
   window 0, profile update 21, profile microbatches 3. Compare actual child
   overrides/resolved configuration with the planned parent recipe, including
   accumulation 32, LR 65000/2000, precision/compile/SAC/MTP, and disabled
   monitoring, validation, checkpoint saving, memory snapshots and inventory.
   Require metadata's full native profile spec, not just its target update.
2. **Native exit and wrapper status.** `process.json` must show native exit 0,
   no timeout, no orphan and stopped process group. Require unchanged source
   and config finals plus successful task-tmpdir cleanup. For original PASS,
   require consistent exit/status/completion, resolved-config hash and native
   summary. Permit FAIL only for D0/S with wrapper exit 1, exact known
   `RuntimeError: resolved config lacks parallelism.gfsdp_row_owner_embedding_optimization_stages`,
   matching failure receipt, no completion and no other failure. Because this
   guard precedes parsing, missing raw `summary.json` is expected only here.
   Independently validate skipped post-guard checks; native exit 0 is insufficient.
3. **Nullable exception and activation.** Match every expected resolved field;
   permit omission only of the named selector when its typed default and
   annotation are nullable, explicit child override is null, native typed decode
   is None, and all eight initialized module receipts show one row-owner module
   with empty stages. Other controls must match. SE requires exactly compact,
   owner_add, owner_lookup; SE-fused additionally routing. Check all eight
   original-capacity overlay/bootstrap and final identity receipts. Frozen
   `campaign._validate_finished` also has an independent missing-field rejection;
   bypassing only the overnight guard would not fix that route.
4. **Full native diagnostic parser.** Run the frozen `summarize_run` offline,
   require status complete, mode profile_update, training_steps 21 and no errors;
   compare with raw summary when present. Its common checks require all eight
   same-host rank records, correct run IDs, completed training and updates 1..21.
   Its profile checks require every rank to report 32 complete F+B calls, one
   train step and one optimizer call at update 21, one profiler build call, and
   actual profiler creation only on ranks 0/1. For ranks 0/1 require collection
   of exactly three, one CUDA drain, stopped/export update 21, one token reduction;
   ranks 2..7 collect zero and do not stop/export a profiler. Timing stays disabled.
5. **Original trace structure and tails.** Native parser requires exactly
   iteration_21/rank0_trace.json and rank1_trace.json; valid JSON; exactly three
   complete CPU microbatch markers and one train-step prefix; no optimizer marker;
   complete global-token reduction before microbatches; prefix covering input
   and microbatches; backward CPU coverage within each microbatch; GPU kernels,
   at least three complete GPU microbatch projections and an NCCL all-reduce.
   Retain original trace hashes/bytes and use those originals for acceptance.
   For the required tail claim, additionally reconcile CPU launches/flows or
   correlation IDs with GPU kernels/copies/communication for the three complete
   F+B regions; verify third-microbatch asynchronous work is present through the
   completed device drain before capture closes, with no truncated/missing final
   device events. CUDA work may extend beyond its CPU microbatch annotation.
   Missing links or evidence of clipping leave coverage unresolved; counts or
   metadata alone do not close it. Preserve any annotation/stream derivatives as
   derivatives, with event/count/hash reconciliation to the originals. A future
   compressed artifact must round-trip losslessly. Perfetto mechanism analysis
   and timings must distinguish CPU scopes from device spans and overlap.

## Source basis and limits

- `campaign.py:_validate_finished` checks recipe, identities, per-rank cadence,
  initialized stages and profile metadata, but does not itself parse traces.
- `../overnight-streaming-perf/run.py:_verify_resolved_config` rejects absent
  default-None before `main` calls `summarize_run`; source/config finals already
  exist at that point. Cleanup then writes original FAIL/status/exit receipts.
- `../overnight-streaming-perf/measure.py:NativeAdapter.__enter__` wraps the
  actual F+B call. After the third return it calls `torch.cuda.synchronize`,
  closes the prefix, sets export step 21 and exits the active profiler, then
  lets the remaining 29 microbatches and optimizer finish. Per-rank profile
  counters continue after collection stops.
- `../overnight-streaming-perf/parse_results.py:_profile_update_summary` checks
  real trace events and counters. It does not reconcile launch correlations,
  assert a recorded final synchronization, or establish that each asynchronous
  tail survived export; that is the targeted structural check above.
- `adjudicate.py:_validate`, `_validate_rank_cadence` and
  `_validate_timing_summary` are timing-specific. Its nullable helpers and
  provenance checks are reusable; its current output schema assumes timing
  tokens/window/memory and needs a diagnostic branch rather than fabricated
  timing fields.

The concrete implementation gap is a narrow offline diagnostic branch and its
receipt schema; the immutable native launch and capture route is available.
No runtime cost is added to training by this design. Parse/hash/trace analysis
runs after training/timing has ended. This note establishes no executed capture,
GPU correctness result, speedup or completed diagnostic acceptance.
