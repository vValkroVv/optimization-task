# Streaming performance campaign: cancellation and cleanup

Status: ORIGINAL_CAMPAIGN_CANCELLED_INCOMPLETE; cleanup assignment COMPLETE.
The [active plan](../../planning/overnight-streaming-perf.md) owns the finite
cleanup checklist. No candidate is accepted. Do not resume G0–G7 or launch any
training, benchmark, trace or local numerical block check during cleanup.

## Revised protocol and cleanup receipt

The last campaign execution source was `e9048f65`. Remove the campaign full-model
numerical modules and exclusive integrations/tests/config, and disable detailed
monitoring callbacks/hooks in the performance configuration. Ordinary two-chunk
loss, lightweight loss/time/TPS/errors, EMA and training losses remain. Preserve
the fixed MDS/model, FP8, compile, CuTe, SAC, MTP, optimizer, routing and batch.
Validation, checkpoint saves, snapshots, numerical capture and CUDA Graphs stay off.

Future separately authorized measurements start with 10 warmup + 10 measured
optimizer updates, accumulation 32 and LR horizon 65,000/warmup 2,000 unchanged.
Extend warmup only for actual compilation/runtime settling affecting the window;
a cache entry alone is insufficient. No mandatory 110/50, exhaustive-cache or large
A/A or A/B/A/B matrix remains. Begin with a matched short baseline/candidate pair and
label it provisional; repeat promising/ambiguous effects when needed.

Performance comes first. A candidate worth retaining then receives a local check
of the smallest affected block on matched boundary inputs and needed weights/
state, plus affected backward with matched incoming gradients if changed. Assess
accumulation precision, casts, rounding, reductions and fusion, using a higher-
precision local reference when needed. Baseline output is not assumed exact;
there is no universal tolerance, assumed bitwise equality or whole-model
parameter/gradient/Adam capture. See [validation](../../planning/validation.md).

Historical monitoring-inclusive G2 median input TPS was 124454.81175967702 global
and 15556.851469959627 per GPU. **It is not directly comparable to future
monitoring-off measurements.** MDS preparation, G2 measurements, G3 Perfetto
SQL/results/compressed traces, candidate code and useful compact evidence remain.

C1 B2 passed; compact history is retained. C2 A reached terminal PASS export
at 08:34:48 UTC; child 418189/process group stopped and controller absent at 08:45.
C2 B and C50 were cancelled without launch. Full-model validation was abandoned,
not completed. V2 source `21c1891d`/`vkropotin-v2-prefetch` remains preserved and
default 0; actual eight-owner Gloo/DCP and performance gates were not executed or
completed. Its source/CPU handoff is not runtime acceptance.

| Current cleanup acceptance | Receipt |
| --- | --- |
| Numerical removal/import/parser checks | PASS: modules, full-model probe and exclusive tests/config removed; imports/help and removed-mode rejection verified |
| Short window/disabled monitoring/preserved config checks | PASS: 100 scoped CPU/config tests plus 1 canonical golden; one unrelated native-DCP test deselected; independent integration/runtime-cost review passed |
| Exact stopped jobs/heavy numerical payload removal | PASS: [receipt](cancellation-cleanup.json); zero owned numerical processes and reference blobs; 40,650 files / 381,915,112,855 logical bytes removed, plus two obsolete clean worktrees; no NFS reclaimed-byte claim |
| Affected document links | PASS: 176 local links across 20 affected documents resolve; 2 fragment links pass; 3 unchanged upstream root-relative links are excluded and recorded in results; no new broken link; JSON/whitespace checks pass |
| Commit/push | Cleanup commit `aa45e3cd8d0c8856d383e72a9ad5f5d60c0530ae` pushed to `origin/main`; this receipt update records completion |

Verification commands, intermediate failed checks and their resolutions are retained
in `artifacts/raw/cleanup-verification/runner/CHECKS.md`; the independent review is
`artifacts/raw/cleanup-verification/integration-review.md`. The new configuration
regression fails on the original monitored YAML and passes on the final YAML.
No full suite or GPU execution is claimed. The final path removes numerical
exports and diagnostic hooks/reductions; it retains native logging, EMA and two
timing-boundary synchronizations, with no new model computation or communication.
The GIFT operating instructions also replace their universal bitwise-loss rule
with the local, precision-aware post-performance policy.

The unchanged historical smoke launcher requires its recorded original source
and configuration; current MDS/monitoring-off defaults do not reproduce its
receipt. MDS index and both canonical compressed G3 trace hashes still match;
V2 source remains clean at `21c1891d`. The existing untracked V2 loader-check helper
is preserved without adding it to this cleanup change set.

## Archived original campaign record

**Everything below is dated historical evidence, including its proposed next
steps, numerical prerequisites, monitoring-inclusive policy and old 110/50/cache
method. Those instructions are retired and do not authorize or constrain future
runs. The current cleanup plan and validation policy above replace them.**
Original failures and incomplete gates are preserved; the archive does not claim
the original campaign finished. Heavy numerical payload locations may describe
removed historical files; compact evidence is retained where useful.

## Entry and environment

Execution started 13 November 2026 02:18 Europe/Moscow (12 November 23:18 UTC)
at clean task `main`, revision `a2a59e14545b55ca777f94a28e332759e7c06087`.
The existing single master exposes eight idle H100 80 GB GPUs. The task path
is on SR008 NFS with approximately 2.5 TiB free; no new allocation was made.
Runtime preflight verified `/opt/torchtitan-venv/bin/python3.12`, torch
`2.15.0.dev20260822+cu130`, CUDA 13.0, NCCL 2.30.7, streaming 0.13.0,
transformers 4.57.1, CUTLASS DSL 4.4.1 and TVM FFI 0.1.11 after sourcing
`/etc/profile.d/wm.sh`. Shared packages remain unchanged.

Perfetto MCP successfully loaded a synthetic Chrome JSON at the actual task
absolute path and executed `SELECT name, ts, dur FROM slice`; it returned
`campaign_route_check`, 1000000 ns, 100000 ns. This verifies tool/path access,
not the later required training-trace analysis. No private server is needed.

## Scope and interpretation

Preserve the fixed crop, precision, SAC, MTP1, EMA, optimizer, seed and scheduler.
MDS is a new data baseline because HF and MDS packing differ. Performance will
include input and monitoring at interval 50, excluding validation and checkpoint
saves. Short runs remain in optimizer LR warmup. No convergence, continuation,
multi-node or production-scale performance conclusion is implied.

## Results

Native MDS smoke retry `g1-streaming-smoke-002` passed at source `4d7689aa`:
ten optimizer updates and 320 successful forward/backward calls on each rank,
finite metrics and native monitoring at 5/10. Final loss was 14.57432 and
gradient norm 7.6163; maximum native composite memory was 72.57 GiB.
The [separate receipt](g1-smoke-002-receipt.json) verifies unchanged source/config,
clean teardown, no diagnostic outputs and released GPUs. That smoke has no
performance measurement; the G2 A/A TPS baseline is reported below. First
attempt001 remains a failed harness attempt, not a
candidate verdict. Source applicability is recorded in the
[reference evidence](reference-coverage.md); candidate runtime gates remain open.

The smoke's rank0 host boundary intervals were about 13.5–14.8 s for ordinary
updates and 173.5/166.7 s for monitoring updates5/10. These are unsynchronized
smoke cadence observations, not baseline TPS or CUDA kernel time. They reinforce
the required normal-cadence calibration through update50 before choosing W;
the 32-chunk shape and monitoring work cannot be omitted from the controls.

## MDS fixture and real loader contract

The [data receipt](mds-data.json) owns corpus/tokenizer, installed streaming
implementation, row and shard hashes and exact check commands. The fixture is
896 full 4096-token uint32 rows in 29 uncompressed shards, made from the first
3,670,016 BOS/text/EOS tokens of the same local C4/tokenizer as the HF smoke.
Index SHA256: `7cc8d53152c801450fdae44ba84c182eb8dcb9106d8bba9e3c3d69ebdaa8a8c2`.
The [helper](prepare_mds.py) refuses existing output before calling the destructive
packer. The [loader fragment](mds-dataloader.yaml) selects installed streaming0.13,
two persistent workers/rank, shuffle/FIM off, canonical nodes1, infinite iteration
and no pinning. After native smoke passed, the canonical experiment's data
mapping was promoted, with all non-data values preserved. Its affected golden
test passed (1 passed,60 deselected) after regeneration and again in ordinary
check mode. Relative `local` resolves from the normal `gift/` working directory
to the same index and all29 shard hashes. Native002 used the equivalent absolute
external fragment; canonical bytes have decode/path evidence, without another
GPU smoke.

Two fresh actual eight-rank Gloo runs each validated 58 batches/rank, including
the 56-batch epoch wrap. Both passed exact row hashes/IDs, all16 worker partitions,
global coverage, stable worker PIDs, shapes/dtypes, EOS-derived positions, label
shift and final ignored labels. Each microbatch has 8190 valid main targets;
global per-update numerators are 2,097,152 input and 2,096,640 valid main targets.
No MTP token multiplier is applied. The separate native GPU smoke passed; this
does not establish parity with the differently packed historical HF input.

Pass1 exited0 but retained 16 Python multiprocessing finalizer Errno16 warnings
from the provider's NFS TMPDIR. Pass2 used local tmpfs and had none. The runner
will create and remove only its uniquely owned local temp directory; all data,
compiler caches and persistent logs stay in the task NFS area. No shared temp
cleanup was performed. Final helper, packer, data-world, loader-fragment and index
hashes were checked against the receipt after worker handoff.

Native resolved/source correlation identifies an important position distinction:
the fixed SDPA attention selects causal, non-varlen behavior and propagates that
mask type to GDN. `Trainer.prepare_inputs` replaces the collator's EOS-derived
positions with int32 per-row `arange` for this configuration. Thus the data check
proves the collator contract, while actual model attention/scan state spans
documents within each packed row. It is causal within rows, without EOS document
isolation. Keep this existing model behavior; switching to block-causal/varlen
would change the identified baseline. Exact native per-batch partition counters
are unavailable; the prior actual eight-rank Gloo loader check plus native mesh
and source correlation support rank partitioning.

## G2 calibration

`g2-calibration-001` passed60 native updates/1920 microbatches on every rank,
with exactly32 calls/update, finite loss/gradient logs, clean teardown and released
GPUs. Source/config stayed unchanged; no checkpoint/trace/snapshot was emitted.
Raw evidence is under `artifacts/raw/g2-calibration-001/`; resolved config SHA256
is `b88ffc3da42a7507fa47bb33f6f54fb22d023fa5bc0978338fce6fa9d3e9b27a`.
Helper bytes used by calibration are preserved at `02cfb427`.

Rank0 host boundary cadence medians (seconds):

| Updates | Median |
| --- | ---: |
| 2–10 | 13.7441 |
| 11–20 | 13.8981 |
| 21–30 | 15.3566 |
| 31–40 | 15.3071 |
| 41–49 | 14.9610 |
| 50, monitoring | 173.9149 |
| 51–55 | 14.9717 |
| 56–60 | 15.0141 |

Updates51–60 range14.5722–15.5668s without continuing monotonic settling;
maximum same-update host-interval spread across ranks was1.843ms. Recommend W60,
then measure all updates61–110 including monitor100. This is not TPS evidence:
the calibration has no synchronized50-update window. Step1 took259.828s,
including245.228s in its first forward/backward. Monitor50's first microbatch
took8.477s, then median4.8147s; its173.9s cannot all be assigned to compilation.
Diagnostic compiler logs were disabled, so later compilation is unknown.
Monitor100 and surrounding updates must also be checked directly in A1/A2.

### Compilation gate for measured controls

The task-only observer samples existing CPU state at the two synchronized
window boundaries, outside each local timer. It reads unbounded Dynamo compile
timing lists, existing Inductor async-cache counters, the active CuTe compiled
object map and existing Triton JIT device caches. It never indexes a missing
Triton device cache, inserts a defaultdict key, logs per step or synchronizes
the GPU beyond the two existing window boundaries. Boundary scans can still
contribute to rank-start skew in the global elapsed envelope. Runtime-only
Dynamo timing growth is allowed; new compilation/cache entries invalidate the
window. This is scoped evidence, not visibility into every asynchronous worker.

Each rank retains raw snapshots, source pins and the computed comparison;
the parser recomputes it and rejects missing or inconsistent results. Independent
review found and resolved two evidence bugs: identical malformed snapshots
could pass, and rank receipts initially lacked their comparison. Both original
reproductions now fail closed. Final observer36 and existing measurement26 tests pass,
with the task `gift/` explicitly on `PYTHONPATH`; the initial default-image import
test failure is retained as a harness error. Actual CPU snapshots, scoped Ruff,
compilation and whitespace checks pass. `run.py` pins the observer helper hash.

The installed DeepGEMM extension has no exposed compilation counter. Its actual
binary and active warmup code identify `DG_CACHE_FORCE_SKIP=0` as a cache-miss
rejection guard; the binary reads that environment value on each compiler call.
A serial GPU probe passed on13 November02:21 UTC. Both actual `m_grouped` and
`k_grouped` paths compiled A with flag1, reused A with flag0 without new cache
entries, rejected a different absent configuration B with literal `CACHE MISS`
and unchanged cache inventories, then compiled B with flag1. The installed build
was `2.3.0+f093a4c`, extension SHA256
`99b7afad2e2fe7ebabd26c22afc99dc652e2e6935fcbf37324359eb5aeeefe26`.
The frozen [probe](probe_deepgemm_guard.py) used serial GPU0 children and fresh
private caches; both exited0 and all eight GPUs were released. Raw receipt/logs:
`artifacts/raw/g0/deepgemm-guard-probe-001/`. This proves guard/cache behavior,
not numerical parity or speed. Matching full fork source is unavailable, so
upstream behavior is not substituted for installed-binary/runtime evidence.
The thin adapter sets literal0 at W60 after synchronization and before the
timer, preserves it through measurement and records both boundaries on every rank.
It verifies the actual extension build/hash before training and restores the
activation-time environment value on adapter exit, including errors. The profile
run uses the same guard policy with one explicit synchronization at W, before the
target update; this supersedes the earlier capture-only zero-sync cost statement.
Plain smoke/calibration mode does not import or activate the guard. Parser checks
require the manifest policy and complete, matching guard receipts from all ranks.
Independent integration/runtime-cost review accepted this path; its final finding
was missing `previous_value` being accepted as explicit null. The validator now
requires presence; the regression reproduced the false pass and all5 affected
cases pass. Root inspected the small fix. Post-run probe lint found E731 in a
planner lambda; an equivalent local function now passes Ruff and produces the
identical CPU plan. Exact GPU-tested script bytes remain with the raw receipt.
The first staged whitespace check found trailing blank lines in two newly added
files, which the earlier unstaged check had not included. They are removed;
executable statements are unchanged and the final helper hash is in results.
A1/A2 and TPS remain open.

### First measured control: invalid window

`g2-baseline-a1`, source `72a814f2`, completed110 native updates, but its runner
failed required acceptance at03:24:19 UTC. The direct observer found new Triton
cache state on ranks0,2,5 during61–110; all eight DeepGEMM guard receipts were
complete with literal0 at both boundaries. A2 was not launched. This window is
not an accepted baseline or speed claim. Exact changed-key and settling diagnosis
is active; raw failure, summary and rank snapshots remain under
`artifacts/raw/g2-baseline-a1/`. Rank0 monitor100 took169.400s; ordinary intervals
also drifted downward over the window, so source/cache interpretation and actual
warmup selection remain open rather than selecting fast steps after the fact.

Read-only diagnosis resolved the only additions as forward/backward fused FP8
SwiGLU specializations in the padded routed-expert path: the runtime token bucket
`total_tokens // 2048` changed from an unaligned int32 signature to Triton's
divisibility16 attribute `D`. Hidden1536, launch options and all other signature
fields match. Dynamo/Inductor/CuTe state and Triton registry/function/cache
identities stayed stable. The run's shared disk metadata, TTIR, PTX and cubin for
both aligned kernels predate the window; installed Triton returns `CompiledKernel`
on a disk-metadata hit. Disk loading is strongly supported; actual new device
compilation during the window is unproven. No compiler listener was recorded.
The before/after snapshot is inside training, before exit; stop/export
contamination is unsupported. Keep the conservative cache-stability criterion.

All eight ranks hold both bucket-alignment classes by110. Rank0 ordinary medians
were14.7735s at61–70,14.2267 at71–80,13.9576 at81–90,13.7936 at91–99 and13.8516
at101–110. The trend's routing/clock/warmup cause is unresolved, while the final
tail is stable. Reuse this run as additional calibration and select **W110,
updates111–160, monitor150** symmetrically for fresh controls. No kernel/observer
change or feature removal is required; first-control failure stops before A2.
All110 native updates and finite metric pairs completed on all ranks; source and
config hashes match, retries/OOMs were zero, owned temp/group stopped and all GPUs
were released. Its unaccepted diagnostic timing/memory remain in results with
the explicit failure label. No baseline TPS has been accepted.

After that diagnosis, the obsolete calibration001 compiler cache was removed:
9,217 files /602,133,276 logical bytes. Stop/ownership/path/symlink checks passed;
all non-cache artifacts remain unchanged. Its cleanup receipt is
`artifacts/raw/g2-calibration-001/obsolete-cache-cleanup.json`. The failed A1's
cache evidence and the new control's independent fresh caches remain retained.

### Accepted W110 baseline and identical-control variation

`g2-baseline-a1-w110` PASS: all eight ranks completed160 finite updates; raw
snapshots show stable observed compiler/cache state and guard0after110 through160.
All50 updates111–160 include monitor150. Complete-window global inputTPS is
**124,500.985** (15,562.623/GPU), supervisedTPS124,470.589 (15,558.824/GPU), using
104,857,600 physical and104,832,000 valid tokens over842.2231s. Source/config
hashes match; temporary resources and GPUs were released. Root inspected the raw
all-rank comparison/guard/counter records, not just the launch settings.

Ordinary host median13.7997s, range13.4381–14.4012; monitor150 median167.1280s.
Rank0 successive ordinary-bin medians111–120/121–130/131–140/141–149/151–160 are
13.8784/13.8527/13.6500/13.7850/13.7915s: no continuing monotonic settling.
Allocator peak maxima are67.401GiB allocated and68.467GiB reserved, native reset
scope including the101–110 tail, with zero retries/OOMs. These exclude non-Torch
device memory. Start/end rank skew1.698/0.596ms; primary elapsed uses the global
envelope, with only two boundary synchronizations.

The identical `g2-baseline-a2-w110` also PASSed, finishing05:23:46 UTC with all160
finite updates on every rank. Source `87230b5b`, initial helper/config/data hashes,
runtime and fresh-cache policy match A1. Root verified all eight raw window,
counter, observer and guard records. Both runs include all50 updates111–160 and
monitor150; ordinary cadence shows no continuing monotonic settling.
The actual `native/config/resolved_config.yaml` files differ only in each run's
`dump_folder`. Their SHA256 values are
`aa2c792ce61e88717c2171f6fad8e8d4f20f8a5b54675de2887524b5305a7007` (A1) and
`099734af6018ecbe8d7b2b67337a43b82f68628e5c7651de592bfc995768f2e1` (A2), matching
the completion receipts. `source_config.yaml` is a separate artifact.

| Complete-window metric | A1 | A2 |
| --- | ---: | ---: |
| Global input tokens/s | 124,500.985 | 124,408.639 |
| Global valid-target tokens/s | 124,470.589 | 124,378.266 |
| Global elapsed seconds | 842.223058 | 842.848222 |
| Ordinary host median seconds | 13.799678 | 13.789450 |
| Ordinary host range seconds | 13.438092–14.401202 | 13.423635–14.463464 |
| Monitor150 host median seconds | 167.127963 | 167.607337 |
| Maximum allocated/reserved GiB | 67.401 / 68.467 | 67.387 / 68.494 |
| Allocator retries / OOMs | 0 / 0 | 0 / 0 |

The fixed MDS baseline is the run-level median: **124,454.812 global input
tokens/s (15,556.851/GPU)** and **124,424.427 valid-target tokens/s
(15,553.053/GPU)**. A2 versus A1 is−0.074173%; the two-run range divided by
their mean is0.074200%. These are descriptive A/A observations, not a confidence
interval or guaranteed bound on future variation. No candidate gain is accepted.
Memory uses native allocator reset epochs spanning101–160, including the warmup
tail; it is not an exact-window or simultaneous device-wide peak. Validation,
checkpointing, profiling and numerical payload capture are excluded from TPS.

A2 rank0 ordinary-bin medians111–120/121–130/131–140/141–149/151–160 are
13.930675/13.805102/13.638698/13.816832/13.787526s. Start/end rank skew is
4.619/0.774ms. Both runs have stable observed compiler state and tested DeepGEMM
guard coverage; asynchronous compiler-worker completion and native JIT paths
outside the probed grouped APIs remain unverified. Both process groups stopped,
owned temporary directories were removed and GPUs released; private run caches
were subsequently removed with verified receipts; raw run evidence remains.

## Native profiler controls

Implemented generic opt-in `profile_ranks` and `record_shapes` controls in the
native typed profiler. Defaults remain all ranks and shape capture enabled;
this campaign selects ranks 0/1 and disables shape capture. Nonselected ranks
return before profiler or postprocessor construction. Native schedule, exports
and full-update callsite remain unchanged. With profiling disabled, no new
per-step runtime work is introduced. Stacks and profiler-memory capture are
explicitly false; memory-history snapshots remain a separate disabled feature.

Focused checks from `gift/`, using the existing interpreter and task source:

```bash
python -m pytest tests/unit_tests/test_profiler.py tests/unit_tests/test_config_torch_free.py::test_profiler_rank_and_shape_controls_decode_torch_free -q
python -m pytest tests/unit_tests/test_experiment_parity.py tests/unit_tests/test_model_arch_presets.py -q
```

Results: 21 and 112 passed respectively; scoped F821/F811/F841, default Ruff,
formatting and whitespace checks passed. Goldens were regenerated through the
native update mechanism, then checked without update mode. The CPU native
schedule test exported only update 6, with 32 microbatch markers and optimizer
coverage. It does not establish CUDA tracing. A broader torch-free run retained
one pre-existing registry mismatch (`campaign_numerical_probe` is registered but
missing from the expected callback set); no registry files changed.

The [independent review](profiler-review.md) accepted the source handoff and
verified all 56 golden files contain only the two expected default additions.
The selected-rank baseline trace integrity and Perfetto analysis are now accepted
at G3. Numerical equivalence and candidate/lifecycle acceptance remain open at
later gates. Full-tree autofixing hooks were not run across
the flattened task tree because their source-root path filters do not protect
its byte-preserved reference and contrib snapshots; changed-file lint was used.

## Native runner review and prelaunch repairs

No GPU run was started before review. The independent reviewer found two
material integration defects: a missing measured-window completion attribute,
and inherited DeepGEMM startup overriding private cache variables with hardcoded
`/temp/.cache/rank_N/deepgemm` paths. The task adapter now redirects the actual
warmup helper before Trainer construction and fails closed on alternate shared
cache paths, retaining native warmup and source/package bytes. Five focused
tests passed, including actual setup/share-path sentinels for ranks0–7 and patch
restoration, full W..W+50/two-sync timing, incomplete-window rejection and smoke
cadence/microbatch counting. The parser passed a synthetic eight-rank global
envelope/counter check; actual measured-window acceptance remains at G2.
The host CPU core limit was unlimited; task child processes set it to zero
and clear inherited diagnostic/core-dump flags. The
[review](runner-review.md) owns findings and their final resolution. These are
prelaunch harness/inherited-path repairs, not failed candidate measurements.

### First native attempt: census failure

The [runtime receipt](g1-runtime-receipt.json) records exit1 on all eight ranks:
`parameter_census` calls `RaggedShardDTensor.to_local()` with grad enabled,
which this wrapper explicitly rejects. Its same method returns the local shard
under no_grad. Zero native updates or microbatch calls occurred; at that failed
attempt, finite loss, monitoring and execution gates were still open. The scoped repair was to run metadata
census under no_grad, retaining original trainability and restoring caller grad
mode, then test that real method guard before retrying.

The correction is implemented and accepted by targeted independent review.
Two affected tests pass, including the actual ragged-shard method guard and
success/error grad-mode restoration. The exact historical census loaded from
Git into memory reproduces the old exception; the fixed implementation counts
the existing local shard. No model arithmetic or training-source bytes changed.
After owned resources were released, the obsolete failed-run compiler cache was
removed (23,453,549 logical file bytes); logs and compact evidence are retained.

Initialization did verify the resolved fixed contract, all-rank FP8/SAC/compile/
GFSDP/streaming/MTP/EMA builds and native DeepGEMM warmup (39 kernels/node),
with all rank caches under the attempt's private tree and child core limits0.
These are initialization observations, not executed CuTe/model kernels or
training correctness. Source/config/helper hashes stayed unchanged; the owned
group stopped, local temp directory was removed, GPUs were released, and no
checkpoint/trace/snapshot was emitted. Logs and compact failure evidence remain.

### Active compiler-cache scope

TorchInductor, Triton, DeepGEMM rank caches and HybridEP use the run's fresh NFS
cache tree. CUTLASS DSL4.4.1 additionally caches generated MLIR under
`CUTE_DSL_CACHE_DIR`, or by default `$TMPDIR/$USER/cutlass_python_cache`
(`cutlass/base_dsl/cache_helpers.py:66–82,235–306`). Root verified that
`CUTE_DSL_CACHE_DIR` and `TVM_FFI_CACHE_DIR` are unset after the image profile;
thus active CuTe IR cache uses each run's unique private tmpfs directory and is
removed with that directory. Its `COMPILED`/`PLANS` maps are process-local.
The active CuTe JIT exports a function pointer to TVM-FFI rather than invoking
`tvm_ffi.cpp.extension`, so that extension builder's persistent default cache
is not the norm path. This qualifies the cache policy without changing kernels
or introducing another experiment. Shared/personal unrelated caches are intact.

### G3 baseline capture and Perfetto analysis

Run `g3-baseline-update111-001` completed111 native updates on all eight ranks
with finite metrics, all guard values0, source/helpers frozen and clean teardown;
all GPUs were idle afterward. The profiler was constructed only on ranks0/1.
Sidecars on all ranks attest one target train step,32 microbatches and one
optimizer call. No throughput measurement or numerical diagnostic mode was
active. Source revision was `6123a24f`; the runtime training code matches the
G2 control pin `87230b5b`.

The original runner result is still recorded as exit1: its strict parser counted
CPU logical annotations and GPU projections by name alone, reporting2/64/1 on
each trace instead of the sidecar's1/32/1. This parser failure is immutable
history; no retraining or trace recapture was done. The category-aware post-hoc
reparse passed: CPU `user_annotation` contains exactly1 train step,32 microbatches
and1 optimizer step per selected rank; separate `gpu_user_annotation` projections
contain1 train step,32 microbatches and no optimizer projection. Neither trace
has incomplete slices or a microbatch outside its CPU train frame. The original
summary/failure and the corrected [reparse summary](artifacts/raw/g3-baseline-update111-001/summary.reparsed.json)
and [receipt](artifacts/raw/g3-baseline-update111-001/reparse-receipt.json) remain.

Actual Perfetto MCP loaded both traces and completed the accepted analyses. The
[canonical result](g3-baseline-perfetto.json) preserves the original loaded paths
and trace IDs; its current artifact map points to the two gzip files after full
stream roundtrip size/hash checks. The pair was reduced from602,500,095 raw bytes
to45,573,265 compressed bytes, with556,926,830 logical bytes removed. The
compression receipt records identical pre/post hashes for all other raw-run
artifacts and zero immediately observed NFS available-byte/inode change. The
approximately604,978,569-byte,9217-file profile run cache was preserved when
the compression receipt was written. The four minimal compiled CE wrapper
sources are now preserved for the K1a contract. Separate [cache cleanup](artifacts/raw/g3-baseline-update111-001/obsolete-cache-cleanup.json)
completed:9217 files/604,978,569 logical bytes removed; the four exact wrappers
retain73,504 bytes plus their manifest. Raw evidence and canonical traces stayed
unchanged; observed shared NFS capacity/inode deltas were0.

The trace scopes are rank-local. Every kernel, memcpy and memset event falls
inside the exact CPU train-step frame; GPU busy unions were13,044.218/12,996.754ms
on ranks0/1. Optimizer flow analysis links17 fused AdamW kernels per rank.
A607.371/643.908ms interior gap precedes the input H2D copy and
overlaps host loading. Update111 crosses the 56-batch local epoch after8 old
batches; 33 underlying loader calls are consistent with32 deliveries plus the
epoch restart. This makes safe streaming prefetch the first candidate to inspect,
but does not confirm a gain or show that the full gap is removable. Strict
NCCL-exclusive union is714.427/694.566ms after excluding memset overlap; the
historical broad result remains in the JSON. There is no cross-rank clock mapping
or complete stream-dependency DAG, so neither union proves critical-path time.

The compiled CE already avoids full-vocabulary FP32 intermediates. Its ordinary
baseline kernels total about201/198ms per rank in overlapping kernel-duration
sums. Removing the baseline's one-hot-gradient reduction remains an independent
applicable donor algorithm candidate; no port or gain is established. See the
[compiled CE contract](k1a-compiled-contract.md). Candidate order and remaining
checks are recorded in the canonical JSON and reference table. One profiled
update provides mechanism evidence only, not TPS, monitoring cadence cost,
convergence or candidate acceptance.

## Reused validation interfaces and limits

The historical `CampaignNumericalProbe` is not an unchanged validation route for
this campaign: its trainer schema requires `hf_text/c4_test`, zero workers and
synchronous full checkpointing (`gift/torchtitan/config/schema/experiments/cuda_graph_campaign/trainer.py`).
It also persists multiple full tensor families. Do not switch this streaming
campaign to that trainer or generate its full-model payloads merely to reuse
the old probe. Candidate checks must exercise their actual affected loss, state
or lifetime path with matched inputs and elementwise gradients/state, then native
integration; any unresolved full-workload numerical gate stays explicit.

### K1 implementation contract (design only; no candidate enabled)

Keep completed-chunk release and fused CE independently selectable, default off.
The release point is the existing chunk-loop tail after backward/gradient
collection, preserving head calls, argmax, components, probes and flag restoration.
For CE, select a cap-zero callable before `_maybe_compile`, retain current
normalization and weighted MTP, and route DTensor to the current distributed CE.
Do not copy the donor loss class or change Unembedding. The pinned donor's
softcap tests do **not** test fused CE directly, so its CUDA/Inductor autograd
path needs new evidence, including V128256, signed int64 ignored labels,
all-ignored rows, noncontiguous inputs and explicit higher-order rejection.

Release tests must prove output weakrefs expire before the next projection and
compare the actual current chunked A/B paths through gradients and two AdamW
updates (parameters, moments and steps), ordinary/monitoring chunks, MTP,
evaluation/probes, FSDP protocol and exception restoration. For fused CE, first
freeze baseline error/variation against an FP64 CE/analytic-gradient oracle at
the same quantized logits and output cast; compare elementwise downstream
gradients/state through the same projection. Different reduction rounding is
not subject to an invented percentage or unjustified bitwise gate. Any
uncalibrated difference stays unresolved and the feature stays disabled.
Tiny tests do not replace fixed-crop native GPU/SAC/GFSDP integration.

### V3 partial-backward contract (design only)

The isolated donor guard belongs at the start of the group loop in
`GFSDPState._root_post_backward_final_callback`. During nonfinal inner backward,
skip groups outside PRE_BACKWARD/POST_BACKWARD; preserve final cleanup. Current
MTP already folds its hidden states into the chunk loop and sets shared
`is_last_backward=False`, so no loss-class transplant is needed. Prefetch borrows
PRE_BACKWARD temporarily and restores IDLE: an untouched IDLE group can still own
a gathered buffer/pending all-gather. Regression coverage must preserve that
ownership through repeated head chunks until the group's actual backward.

Keep this default-off guard independently selectable from arena/tail retention.
It adds Python state checks per group/inner backward and may remove premature
reshard/regather, while extending existing buffer lifetimes; collision/overlap
safety needs native evidence. Existing explicit EP8 chains, double buffering and
head private gradients are already present. Startup shard slicing is an
applicable initialization optimization with no warmed-runtime effect from the
inspected mechanism; do not credit it as stable-step TPS or silently drop its
initialization-equality gate.

Current loss restores incoming `is_last_backward`/reduce-grads flags, but its
finally hardcodes both reshard policies to true. This inherited limitation is
outside the fixed crop's required policy and must not be described as arbitrary
policy restoration. A guard port must preserve the current policy and verify
normal/exception paths, MTP0.3, chunks2/32, actual owner/event lifetimes and
elementwise losses/gradients/two optimizer updates. Exact callback/state checks
are justified; native FP8/HybridEP floating-point repeatability is not yet
established. Calibrate unchanged numerical A/A before setting native tolerances;
timing A/A and rounded norms do not establish numerical equivalence.

### Native numerical receipt: integration verified, calibration open

#### Observation path and limits

The opt-in receipt observes native `after_dataloader` callback-mutated CPU
inputs and labels; `forward_backward_step` metadata, denominator and loss;
`dist_utils.clip_grad_norm_` before and after clipping; the optimizer after
actual AdamW; and `after_load`/`batch_end` state. Preserve arguments, returns,
hooks and wrapper restoration. Adaptive load balancing is an optimizer-container
pre-hook: capture incoming counts/bias before clipping, then updated bias/reset
counts after the step. Its counts are FP32 and nonpersistent; `ntokens_seen` is
int64. Neither dtype is changed.

Optimizer parameter groups own the sharded tensors consumed by fused AdamW.
Capture each RaggedShardDTensor local shard under `no_grad`, without padding,
gathering or casting; retain identity/FQN/alias, shape, dtype and shard metadata.
One-tensor-at-a-time CPU hashes establish exact equality only when unchanged
native A/A repeats exactly. A mismatch requires bounded temporary payloads and
elementwise error metrics, not a tolerance inferred from hashes. The receipt
covers pre/post-clip gradients, parameters, optimizer moments/steps, losses and
components, normalization, and routing/counter state. The clipping return is the
norm; a separately computed clip coefficient is derived rather than observed.
Post-clip tensors attest the applied result.

The first isolated observer copied each microbatch to CPU and could serialize
the next microbatch, masking asynchronous buffer-lifetime faults. The corrected
path hashes CPU batches after `after_dataloader` (all 32 are fetched before GPU
compute), retains detached native scalar losses and one immutable denominator
reference through forward/backward, then hashes those at preclip. It samples
native loss-component totals and mutable auxiliary totals/current scale once at
preclip, before the next-update reset. It needs no per-microbatch D2H, auxiliary
clones or graph-bearing references. Native loss provides one normalized scalar
per microbatch; component tests own elementwise CE evidence. This preserves
microbatch overlap but still perturbs update boundaries, so hash runs alone do
not validate V3 asynchronous ownership. The uninstrumented native integration
and separate representative trace remain required.

The mode is default-off and separate from TPS runs. With it disabled, the runner
does not import or install the observer, add native flags or hashes, or perform
receipt work. Enabling it bounds captures to selected updates; tensor snapshots
copy local values to CPU and exported references can create substantial I/O.
These are numerical diagnostics, never throughput measurements.

#### Integration and parser status

The original helper review found that `dist_sum` can return a Python-float
denominator while the first observer required a tensor. A CPU probe reproduced
the TypeError; support for the scalar without tensor conversion fixed it, and
the original 11 CPU tests remain valid. The earlier isolated runs (61 affected
tests plus two targeted tests) are historical evidence: their executable and
ambient `PYTHONPATH` were not captured. Main integration now closes that check
with an explicit campaign interpreter and task `PYTHONPATH`.

Main integration is at `f29581c4dcae78928cb940313b13893826f4e10`. It applies
the accepted helper revisions `84b72524fd274830c0bcb8b49afd3c196eabd118`
(hash mode) and `be8d75ad6d90cc3de9175993be638d2f8bdd3fa4` (payload
extension). Independent review accepted the source changes. The combined main
suite passed 110 tests, including CPUAdamW observation-neutrality checks; scoped
Ruff and `git diff --check` passed. The exact command, interpreter and package
versions, outputs and integration details are recorded in
[the main integration evidence](artifacts/raw/native-numerics-integration-evidence/INTEGRATION.md).
The prior worktree test counts do not replace this main-checkout result.

The G3 category-aware parser fix is preserved in the combined main source:
CPU logical `user_annotation` markers and GPU projections are counted
separately, and unsupported categories fail validation. The G3 assertions still
pass. GIFT source and canonical config bytes are unchanged.

The four parser gaps found in independent review are resolved. Receipts bind the
selected metric files, counts and digests to the A/B identities. Only selected
mathematical float fields are whitelisted; learning rate, scheduler state,
optimizer step and counters remain exact, and any nonfinite value invalidates a
receipt. Selected FP32 `tokens_per_expert_E` and `expert_bias_E` are compared
numerically at three router phases; initial routing and integer counters remain
exact. Original post-flush record-count handling was also corrected by wrapping
the actual `MetricEmitter.emit` publication callback.

#### Calibration state and acceptance gates

C1A's one-update export completed on all eight ranks, covering32 microbatches
per rank and approximately47.5 GB per rank of numerical payloads. C1B failed before
update 1: all eight ranks raised
`ValueError('initial parameter, optimizer, scheduler, buffer, RNG, or counter gate differs')`
at `numerics_payload.py` (removed during cleanup) line 580. The runner exited, owned
cleanup completed, and all GPUs are idle. Native A is complete, but the A/B
initial-state gate did not pass because live tuple mesh coordinates differed
from their serialized list representation. The original B live hashes were not
persisted, so fresh exact-state verification is still required. This failed
control supplies no candidate numerical verdict or error-floor measurement. No tensors were
compared elementwise and no tolerance was calibrated. C2 and monitor50 remain
open. The monitor wrapper now observes
actual `MetricEmitter.emit` publications, but native monitor50 inventory/equality is still
pending. EMA is inactive through update 50, so this evidence makes no EMA claim.
The selected online scalar records retain native reduced CPU mappings, order,
multiplicity and metadata before clearing, without another reduction, GPU copy
or per-microbatch drain. The actual inventory must still establish whether any
non-scalar or offline records are uncovered. The campaign does not configure
`MtpAccLenMonitor`; MTP coverage is its actual 0.3-weighted loss/state and the
32-chunk monitoring path.

The normal W110 controls already differ in rounded first-update values:
A1 loss 15.31946/norm 9.5378 versus A2 loss 15.31943/norm 9.5374. These values
do not establish elementwise errors or their cause, but exact native
repeatability cannot be assumed. No tolerance is inferred from them or from
hashes.

The selected extension exports temporary exact local reference bytes and
compares one tensor at a time in bounded CPU blocks. Acceptance uses three fresh
matched prefix pairs, stopping and capturing at updates 1, 2 and 50. That is
106 native updates and six cold starts (four extra starts over the hash-only
route). One update's five large tensor families occupy about 380.122 GB across
eight ranks. Across the three reference runs, writes and comparison reads are
about 1.140 TB each; both sides transfer about 2.281 TB to CPU, plus about
456 GB of initial-parameter traffic across six starts. Check free bytes/inodes,
host RAM and the largest local shard before execution. Copies synchronize and
can change scheduling; these costs are diagnostic, never TPS.

The thin task adapter uses PyTorch
[`assert_close`](https://docs.pytorch.org/docs/main/testing.html) for tolerance
and metadata checks, while recognizing that it does not calibrate an error floor
or manage streamed payload lifetimes. PyTorch
[`frombuffer`](https://docs.pytorch.org/docs/stable/generated/torch.frombuffer)
interprets owned CPU bytes with an explicit dtype. Distributed
[Checkpoint](https://docs.pytorch.org/docs/main/distributed/checkpoint.html)
adds state-dictionary/resharding machinery and possible collectives this
comparison does not require. No dependency or checkpoint path is added.

Before comparison, require matching source/runtime/config/data, initialization,
ownership, optimizer/scheduler/counter state, and hashes for every prefix CPU
input batch. Capture RNG hashes after native `fit_start` without advancing or
changing RNG or determinism settings. Initial-state mismatches fail the gate;
they are not numerical error. Report per-tensor maximum absolute error and
index, RMS error and reference/current RMS, relative L2 with explicit
zero-reference behavior, nonfinite counts and exact agreement. Post-optimizer
parameters are values, not parameter deltas. One A/A pair per capture measures
an observed error floor, not a statistical upper bound. Freeze family-specific
acceptance budgets from valid precision references and A/A evidence before
candidate comparison; unsupported tolerances stay open.

Reference namespaces must be immutable and complete. Verify byte hashes,
identity/shape/dtype/shard metadata and observation coverage; durably write each
compact comparison result before removing its temporary payload. Retain bounded
failure evidence when comparison or evidence writing fails.

Earlier V5b source review corrected one premise: `int(global_valid_tokens)`
at the counter update is a CPU conversion, not a new GPU read. Removing the
producer `.item()` while preserving the baseline clamp, normalization precision
and callback/counter order remains an unmeasured retained-device hypothesis; no
V5 speedup is established. Numerical receipts also do not establish checkpoint
continuation or convergence correctness.

The [C1 diagnosis](native-c1-initial-gate-diagnosis.md) isolates tuple/list mesh
coordinate serialization. The two-line producer fix keeps the exact comparator
and all value/metadata gates intact. Its regression failed on the old producer;
14 observer,12 payload and2 default-path checks pass. Root approved reusing the
immutable A bytes with explicit old/new adapter hashes in `results.json` because
serialized metadata, training mathematics and payload format are unchanged.
Fresh B must still pass every exact initial/input gate and verify unchanged
GIFT/config/runtime/data; the original failed B remains failed.

## Runtime applicability: V4

The native002 parameter census found 243 trainable tensors and 2,375,764,304
local FP32 elements per rank, with zero frozen tensors/elements on every rank.
EMA storage and buffers are not frozen model parameters. V4 frozen BF16 storage
is therefore INAPPLICABLE to this fixed crop; no precision change is made.

## Cleanup

Both native smoke processes have stopped and released the GPUs. Their logs,
resolved configs and compact receipts are retained; no checkpoint, trace or
snapshot was produced. Attempt001's obsolete compiler cache was removed as
recorded above. Attempt002's obsolete compiler cache was removed after smoke
acceptance (604,228,268 logical file bytes); calibration/controls use fresh
per-process caches. Compact native/cache-path evidence remains.

After G2 acceptance, the rejected W60 and both accepted W110 caches were removed:
27,651 files /1,806,310,969 logical bytes. Each run retains an
`obsolete-cache-cleanup.json` receipt; stop/ownership/path/symlink/reference gates
passed, non-cache artifacts matched their pre-cleanup hashes, and active G3 used
an independent cache. Immediate `statvfs` available-byte/inode deltas were zero;
logical deletion is not reported as a measured free-capacity increase.
The rejected run retains six SwiGLU specializations/48 exact artifacts
(2,161,925 bytes) under
`artifacts/raw/g2-baseline-a1/rejected-w60-fused-fp8-swiglu-evidence/`.
Its manifest SHA256 is
`2f29bf1bb64b926bda95e4c0eda464b5a0b7f5785d17a25ec90858727496a2b3`;
original paths, mtimes and content hashes preserve the pre-window cache evidence.
