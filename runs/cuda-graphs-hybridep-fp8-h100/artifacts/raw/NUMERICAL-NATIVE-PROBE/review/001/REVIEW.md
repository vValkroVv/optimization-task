# Native one-update launch preparation review

**HOLD for execution. Preflight preparation may proceed after R4 is corrected.**
Review scope is preparation/001, not the independently reviewed numerical
producer patch. No launch, GPU query, install, source/preparation edit, or
subagent was performed. Exact reviewed file identities are recorded in
[reviewed-sha256.json](reviewed-sha256.json).

## Finite acceptance checklist

1. Preserve the exact native target and serial step-1/full-S0 then fresh-process
   load-only absolute-step-2 path.
2. Fail closed on source/config/adapter/release identity, fresh idle eight-H100
   topology/software proof, NFS admission before native S0 save, and output reuse.
3. Bound execution and release every owned rank on interruption/deadline/failure,
   with precise ownership and persistent status.
4. Label readiness sampling costs correctly and leave production evidence gates open.

## Consolidated findings

### R1 — launch fails before config validation (material)

`launch.sh:53-54` uses `mkdir -m 0700 "$CONFIG_CHECK_DIR"` where the directory is
`preparation/001/config-checks/<stamp>`. The `config-checks` parent does not exist
in the reviewed preparation (`test -d` returned 1). The first otherwise released
launch exits here under `set -e`. Create the task-owned parent explicitly or use
an appropriately guarded recursive creation. Verify the fresh-parent path;
do not create the attempt/checkpoint directories while resolving configs.

### R2 — timeout identity does not own or prove rank release (material safety HOLD)

`launch.sh:151-167,322-349` tracks and signals only the timeout PID. Once its wait
returns it clears all active identity, with no surviving-rank check. Installed
PyTorch's `torch/distributed/elastic/multiprocessing/subprocess_handler/
subprocess_handler.py:63-76` uses `Popen(start_new_session=True)` for each rank.
Ranks therefore have sessions/process groups distinct from timeout/torchrun.
Normal torchrun teardown signals rank groups, but the installed
`multiprocessing/api.py:1023-1065` expressly permits abandoning workers that
remain uninterruptible after TERM/KILL waits. A killed/exited/hung agent also
cannot be relied upon to forward signals to its rank groups.

Concrete failure: torchrun exits/is killed while an independently sessioned
rank survives; timeout finishes, `ACTIVE_PID` is cleared, and `finish` has no
rank identity left to inspect or release. The controller can publish an exit
status or even enter the next phase while an owned GPU rank remains. A timeout
PID/start-tick/ancestry check proves only timeout ownership. There is also an
early interruption window between `setsid ... &` and acquisition of start ticks:
cleanup refuses an empty/mismatched start identity.

Require an independent owner/release path with recorded PID/start-tick ancestry
for every rank/descendant, bounded TERM/KILL escalation and post-phase/deadline/
controller-interruption survivor verification. It must survive controller loss,
handle exec/exit races, avoid matching mutable argv as sole ownership evidence,
and never touch unrelated PIDs. A surviving uninterruptible process must produce
an explicit unresolved release failure and prevent the next phase; do not claim
successful release from timeout exit. A focused CPU process-tree regression
should exercise separate-session children, agent early exit, controller TERM,
and deadline before GPU execution. No GPU reproduction is needed to establish
this code-level defect.

### R3 — claimed host RSS excludes the training ranks (material evidence gap)

`launch.sh:196` sums RSS only for the timeout session. Because rank subprocesses
start new sessions, this excludes their model, optimizer, probe and serialization
RSS. README calls this host RSS for the isolated training session, but it cannot
answer the required host-memory readiness question. Sum precise recorded owned
descendants across sessions, or relabel the existing sample as controller/agent
RSS and supply actual rank RSS separately. Reuse the lifecycle ownership fix;
do not add a second broad process matcher. Ten-second samples remain sampled
observations, not true instantaneous host peaks.

### R4 — fresh receipt lacks required topology and software validation (material)

`preflight.sh` checks UUID order, H100 names, idle memory/utilization, no compute
apps, NFS mount type/target, inodes and write/fsync/readback. It never checks
NV18 topology or actual software identity/import origin. `verify_release.py`
trusts that receipt plus source-file hashes; the compiler adapter records a
PyTorch version later but does not compare it with an approved build. A changed
runtime or topology can pass this release unchanged. Add bounded read-only
topology proof for the eight authorized UUIDs and expected software/build/import
identity, with root review/pinned expectations and receipt validation. Preserve
the existing no-active-app and storage gates. No need to rerun numerical-patch
producer review for this correction.

### R5 — three hours per phase lacks a task-specific budget (operational decision)

`launch.sh:322` permits six hours of training-side execution for two single
updates, excluding unbounded config checks and synchronous sampler commands.
The exact prior loss-only route's first step took about 39 seconds after its
training-start marker (`LOSS-ONLY-PERF/attempts/001/supervisor.log:629,650`). That
does not predict full checkpoint/probe I/O. The nominal probe is 97.35 GiB, so
the material unknown is storage/readback time, not steady-state update time.

Recommend a **30-minute hard operational deadline per phase** with separately
bounded controller/sampler operations and the independent release path above.
This is a conservative stop policy, not a measured runtime guarantee: reserving
five minutes for startup leaves 25 minutes for approximately 100 GiB, equivalent
to approximately 68 MiB/s of aggregate payload progress. Include additional
native-save/readback traffic when documenting the final budget. If existing
NFS evidence cannot justify that allowance, root should choose a concrete
measured-I/O budget before release; a deadline expiry stops this attempt without
automatic extension/retry. Three hours currently has no supporting explanation.

## Accepted inspection findings

- Phase configs import the exact continuation/preset, preserve FlexAttention,
  HybridEP, FP8, selective AC, model/loss compilation, batch 8/global 256/sequence
  4096, derived accumulation 4, seed 42 and 30,000/2,000 schedule. The model preset
  has one dense plus seventeen MoE layers. Graph capture is explicitly off.
- Config-check uses the native torch-free pre-submit path and does not construct
  a trainer/checkpointer. Both checks precede attempt creation. Config resolution
  remains gated on numerical-source integration, as intended.
- S0 is exactly `create_s0/checkpoint/step-1`. Observe uses its absolute path,
  requires absent destination `checkpoint/`, full native load with no exclusions,
  and `load_only=true`. Native `CheckpointManager.load` selects that initial path
  when the destination folder is absent; step-1 avoids the step-0 model-only
  special case. Native save policy suppresses writes under load-only.
- Compiler adapter copies match the established adapter SHA256
  `ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065`.
- Root release is deliberately absent/unconfirmed until the next integrated
  source pin and fresh receipt are ready. That is a valid intended gate, not
  missing user authorization. Existing manifest/archive/file hashes and exact
  artifact set, preset/continuation pins, receipt age and fsync-probe readback
  fail closed. Root must freeze those files throughout both phases.
- The 732-GiB free admission occurs before either native process and therefore
  before S0 save. It reserves the 700-GiB cap plus 32 GiB. Reused numerical-patch
  review establishes later live-payload accounting/reserve checks; native S0's
  earlier save is why this launch admission is necessary.
- Serial commands, fresh phase output, per-rank persistent logs, compiler-option
  records, resolved-config and manifest checks are appropriate. No broad kills,
  checkpoint pruning, retries or unrelated cleanup are present.

## Runtime cost and verification

The numerical observer's substantial D2H, hashing, serialization, storage and
collective costs belong to its separate accepted targeted readiness review.
This review reuses that evidence. Launch sampling adds a GPU query every ten
seconds and NFS directory walks every thirty seconds, plus process queries and
persistent logs. These operations can perturb this diagnostic execution and
their wall time is unmeasured. Rank RSS attribution must be fixed per R3.
No device-compute, host-work, host-wait, communication/copy timing or integrated
performance result was measured here. A readiness run with these observers is
not a throughput benchmark.

Executed: `bash -n launch.sh preflight.sh`; AST parsing of three preparation
Python files; JSON parsing of the release template; adapter SHA equality; local
parent-directory existence check. All syntax/JSON/hash checks passed; the absent
parent exposed R1. Inspected native configuration/pre-submit/checkpoint consumers
and installed torchrun rank-launch/teardown behavior. No CPU process harness was
needed: separate-session launch and explicit abandoned-worker teardown are
visible in the actual installed consumer. No actual config-resolution run was
attempted against the not-yet-integrated numerical schema.

## Exact remaining launch gates

1. Resolve R1-R4 and root's R5 deadline decision; verify only affected behavior,
   including the process-tree failure paths and corrected rank RSS attribution.
2. Integrate the separately accepted numerical patch, complete root integration
   checks, and pin the next complete source snapshot; retain preset and adapter
   identity. Resolve both configs with world size eight and prove no checkpoint
   directories were created.
3. Run fresh corrected preflight, including topology/software, all eight UUIDs
   idle/no GPU apps, NFS admission and fsync/readback. Root confirms reviewed
   receipt and exact final script/config/source hashes in root-release.json.
4. Execute only the two serial native jobs and verify owned-process release,
   full S0 save/load/readback, one absolute-step-2 observed update, all eight
   manifests/compiler records and actual storage/host-memory observations.

Even a successful readiness result leaves production runtime, calibrated A/A
numerical equivalence, full comparison payloads, ten-update generation, graph-on
behavior, prior CUDA fault clearance and campaign acceptance open. It establishes
none of those properties by itself.
