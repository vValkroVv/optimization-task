# Per-block CUDA Graph policy: independent integration and runtime-cost review

## Scope and finite acceptance

Review the opt-in `CompileConfig.cudagraph_exclude_blocks` policy only. Acceptance:
(1) exact block-name matching and fail-before-compile validation; (2) canonical
empty default and Inductor/fullgraph preservation; (3) actual 3B/SAC wrapper
selection of 17 HybridEP MoE blocks with one dense block inheriting global policy;
(4) config serialization/default/torch-free/snapshot impact and unchanged loss and
checkpoint tensor ownership; (5) scoped runtime-cost review and reconciled owner
checks. All five source-review acceptance items are complete. Both material findings were fixed and verified; no material finding remains open for the source policy. Production acceptance is separate and remains open below.

Read code-rules and perf-experiments, GIFT AGENTS/CONTRIBUTING/REVIEW and applicable
config/distributed/models/integer-overflow/experiments rules, plus accepted
`LANE-A-PARTITION-POOL-DESIGN/001/handoff.md`. No source edits, CUDA discovery,
initialization, GPU runs, packages, publication, or subagents were used. Evidence
writes are confined to this review directory.

## Material findings

1. **Test CUDA state leaked during collection.** The initial new test assigned
   `torch.cuda.is_available`, `device_count`, `init`, and `_lazy_init` at module
   level without restoration. This contaminates later tests. Resolved: the module saves the exact four borrowed APIs, restores them in a
   finally block after import, and uses an autouse monkeypatch fixture during
   tests. Owner same-interpreter restoration check passes.
2. **Experiment golden snapshots stale.** `CompileConfig` gains a serialized,
   compare=True default field. `config_golden.resolve('experiments/...')` includes
   trainer config, and structural_dump visits every compare field. A guarded CPU
   check of the exact 3B campaign preset failed with only the expected
   `cudagraph_exclude_blocks: []` addition. See golden-regression-command.txt and
   golden-regression-result.txt. All experiment goldens need the reviewed default
   insertion; model preset goldens should remain unchanged. Resolved: 58 tracked experiment snapshots add only the empty field; the
   canonical continuation snapshot was added; model preset snapshots are unchanged.
   Owner golden repin suite passes 110 tests, with structural audit of the diff.

No additional material production-policy defect was found in the bounded review.

## Integration conclusions

- `config/configs.py:CompileConfig.__post_init__` checks list/value/duplicate and
  enable/model-component/Inductor requirements without importing torch. Runtime
  `distributed/compile.py:apply_compile` repeats validation for mutable configs;
  it resolves every selected FQN among is_block segments before setting compiler
  globals or invoking any module compile. A late unmatched name cannot leave
  earlier blocks partially compiled.
- Empty exclusions preserve the existing `module.compile(backend=..., fullgraph=True)`
  call exactly. Selected names add only `options={'triton.cudagraphs': False}`.
  There is no inherited kwargs/options dict to overwrite in this checkout.
- `models/gift_moe/_base/parallelize.py:parallelize_gift_moe` applies model EP/TP
  parallelization, token reservation, SAC, compile, then FSDP. The new policy does
  not reorder these. `protocols/blocks.py:walk_segments` recognizes `Block` behind
  the SAC wrapper via `peel`, while yielding that wrapper as segment.module.
  `distributed/activation_checkpoint.py:apply_ac` rebinds each block once, before
  apply_compile selects the names. No wrapper is removed or replaced by the policy.
- The actual meta-built production preset test uses real HybridEP dispatcher
  class nesting (constructor replaced to avoid CUDA runtime), confirms all 18
  FQNs, 17 MoE blocks and one dense block, applies selective AC, then records calls
  at the public torch.compile boundary. It checks all 18 remain fullgraph compiled,
  only layers.1 through layers.17 receive graph-off options, and logs match.
  This establishes selection/plumbing, not actual GPU compiler execution.
- Installed nn.Module.compile at `/opt/torchtitan-venv/lib/python3.12/site-packages/torch/nn/modules/module.py:3057`
  only assigns `_compiled_call_impl = torch.compile(self._call_impl, ...)`. The
  existing wrapper, registered parameters, buffers, and state keys retain identity.
  Subsequent FSDP receives the same module objects through named_segments.
- `components/loss.py:BaseLoss._maybe_compile` remains its independent
  `torch.compile(self.fn, backend=compile_config.backend)` path; the test records
  no graph exclusion kwargs there. Data, objective, optimizer, schedule, precision,
  and SAC save/recompute policies are unchanged by this policy.
- `config/save.py:get_typed_document` serializes the field through to_plain and
  existing round-trip/run-bundle checks. This intentionally changes resolved run
  provenance and experiment config snapshots. CheckpointManager still registers
  MODEL/OPTIMIZER/DATALOADER/LR_SCHEDULER and training state, with no new tensor
  state or ownership changes. Native continuation remains a required execution
  gate; source inspection is not its proof.

## Runtime-cost disposition

Validation, block-list/set creation, matching, and activation logging occur once
at model construction, using Python names. For the selected preset there are 18
blocks and 17 excluded names; small list membership is bounded startup work.
The change introduces no per-microbatch or per-update tensor allocation,
computation, device-value read, host/device synchronization, transfer, or
collective. Parameter/state lifetime and FSDP communication ordering are unchanged
at the source boundary. Existing compiler globals and SAC behavior are inherited.

Excluding whole MoE blocks removes capture eligibility for their attention/shared
FFN too, as accepted by the existing block compilation granularity. This can
increase host launch work compared with hypothetical working full capture, alter
compilation choices/cache identity and graph memory lifetime, and forgo most
model graph coverage. Neither memory magnitude nor integrated performance is
established. Dense/loss capture eligibility does not establish actual capture or
replay; A006 must demonstrate surviving regions. No speedup is claimed.

## Checks and open gates

Reviewer executed exactly one new bounded CPU behavior check: the pre-fix exact
campaign config golden comparison, which failed as expected above. Final affected checks reuse owner evidence below; passing tests were not
repeated without a changed contract or inconclusive evidence. Source HEAD at
review start: `446431dfa191ef0300f116257ae918bc85e38c06`.

Final reconciliation (all paths under `MOE-GRAPH-REGION-IMPLEMENTATION/001/`):

- `cuda-guard-restoration-001.log`: exact borrowed API restoration passes in the
  same interpreter.
- `pytest-focused-003.log`: 13 passed after the test isolation fix, covering real
  preset meta/SAC selection, compile options/default, validation, loss path, and
  torch-free config default/round-trip.
- `pytest-sensitivity-001.log`: deliberate removal of graph-off options failed
  the real-preset selection assertion as intended; implementation was restored
  and focused tests passed afterward. This failure remains retained evidence.
- `pytest-golden-repin-001.log`: 110 passed.
- `golden-diff-audit-001.log`: 58 tracked experiment snapshots add only the empty
  list; new canonical continuation snapshot is empty/default/no personal paths;
  no model-preset golden changed. Reviewer inspected the 58 one-line diff sizes.
- `ruff-focused-002.log`: required F821/F811/F841 selection passed.
- `diff-check-001.log`: clean, empty output.
- `docs-links-001.log`: 28 inline links checked; all local targets exist.

The coordinator's `LANE-A/attempts/006/config-preflight.json` reports guarded native
load_trainer_config in WORLD_SIZE=8 context: exactly layers.1 through layers.17,
model and loss Inductor compilation retained, other non-graph configuration
unchanged, only campaign_timing/campaign_step_boundary callbacks added, and CUDA
uninitialized. This resolves the actual prepared overlay producer route at decode
level. It was inspected and reused, not rerun.

`source-pins.json` pins final policy/config/test/doc files, every experiment
snapshot and affected inherited helpers, plus installed nn.Module.compile source.
Production policy files remained unchanged during findings fixes. No broad test
suite, pre-commit run over all source, GPU or distributed runtime test was run by
the reviewer; these focused results do not imply those gates passed.

Production acceptance remains open: native A006 exact 25 completed optimizer
updates on every rank (four microbatches, forward/backward/SAC and next-update
transition), proof of excluded MoE compilation and actual dense/loss capture and
replay, numerical elementwise gradients/clipping/optimizer/update parity,
checkpoint continuation, historical update-33/native-71 failure gate, and
controlled performance/memory comparison. A005 remains failed. No framework,
package, precision, backend, or unsafe-op adapter change is recommended here.
