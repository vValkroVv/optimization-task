# Production-data cumulative A/B/C/D comparison

Status: complete, 15 November 2026. Input acceptance, six component processes, all four timing runs, all four diagnostics and MCP reconciliation passed. The closed native campaign confirms all owned training processes exited and all eight GPUs were idle.

## Local/server synchronization and trace shortcuts

Owner instruction, 15 November 2026: synchronize the completed implementation,
compact evidence and locally maintained documentation, then commit and push the
consolidated task. This supersedes the execution assignment's earlier
no-publication restriction without authorizing another GPU run.

Four byte-identical original rank-0 diagnostics are copied into the ignored
root `artifacts/` directory as `baseline.json`, `4fix.json`, `4fix_qla.json` and
`4fix_qla_embed.json`. The [shortcut manifest](trace-shortcuts.json) records
source paths, exact sizes and SHA256 hashes; the [repository index](../../README.md#latest-rank-0-traces)
links the local copies. All GPU/FSDP annotations are preserved. The original
server run directories and historical local evidence remain intact. Ignored
runtime datasets, compiler caches and other raw campaign payloads remain at
their recorded server paths.

## Scope and finite acceptance

This report owns the rerun requested in the 15 November goal attachment. Keep
all historical fixture results separate. The source starts at task `main`
`2ada87125e433e1f76c4cb2bfad244b641a78a93`, with the contributor edits recorded
in [initial state](initial-state.json) and the
[initial patch manifest](initial-patch-manifest.jsonl). No allocation,
installation, download, commit or push is authorized.
The [preservation check](preservation-check.json) compared 78 initial GIFT
file-byte entries: 77 are unchanged and none are missing. The only changed
initial file is the expected crop golden; the only newly tracked dirty paths
are the canonical crop YAML and its relevant unit test, with no new untracked
GIFT paths. Existing FFN/row-owner and contributor code remains byte-identical.
The final executor record confirms source and configuration stayed unchanged
through diagnostic D.

- [x] Verify the production tokenizer, prepare real diverse data, correct every
  active consumer, and verify actual loaded rank/global batches, bounds,
  packing, uniqueness and embedding-owner skew before GPU measurements. The
  [prepared-data receipt](data-002.json), [actual-loader receipt](data-002-loader-summary.json)
  and [runner-readiness check](input-readiness.json) record this CPU/input
  acceptance; they do not establish GPU behavior.
- [x] Refresh embedding, `layers.0` and MTP component timings on representative
  new-data inputs before native timing. Reuse accepted correctness evidence;
  do not reopen the MTP/FP8 A/A investigation.
- [x] Complete serial A/B/C/D timing, each 10 warmup plus 10 measured complete
  optimizer updates; verify actual routing/CE bindings and all feature settings.
- [x] Complete four separate diagnostics: 20 warmup updates, then three complete
  forward/backward microbatches on update 21, ranks 0/1. Every rank completes
  accumulation 32 and the optimizer update.
- [x] Analyze through MCP Perfetto, retain all original GPU annotations, and
  reconcile coverage/imports with any clearly labeled analysis derivatives.
- [x] Consolidate exact commands and identities, throughput, full update series,
  allocated/reserved memory, retries/OOMs, incremental and cumulative differences,
  component/trace findings and limitations; confirm owned-process exit and GPU release.

## Fixed workload and variants

All runs retain seed 2048, fresh matching initialization, identical data/order
and software, B2/S4096, global batch 512, accumulation 32, EP8, dense FSDP8,
FP8/BF16/FP32, selective SAC, model/loss compilation, MTP, clipping, AdamW,
configured EMA, LR horizon/warmup 65000/2000 and logical epoch 102400.

| Variant | GDN | CPU prefetch | BF16 expert reuse | Routing / CE | MTP FFN / layer-0 FFN / row-owner embedding |
| --- | --- | ---: | --- | --- | --- |
| A | Native FLA | 0 | Off | Original / original | Off / off / off |
| B | Native FLA | 32 | On | Fused indices_to_map / stable cap-zero | Off / off / off |
| C | FlashQLA external_gate | 32 | On | Fused indices_to_map / stable cap-zero | Off / off / off |
| D | FlashQLA external_gate | 32 | On | Fused indices_to_map / stable cap-zero | On / on / on |

GPU input-preparation fusion and fused Linear+CE remain excluded. The existing
shared-buffer FFN and standard-autograd embedding implementation is preserved.
Profiling, detailed monitoring, validation, saves, snapshots and numerical
capture stay off during timing. Each timing run gets a fresh task-owned compiler
cache; its separate diagnostic reuses that matching completed cache. Consistent
boundary cache observations and full cadence series will record settling limits.
The [eight-run command plan](campaign-plan.json) fixes the timing A/B/C/D order,
followed by their separate diagnostics. The timing command round trips and all
producer/consumer checks passed. Exact host and child `torchrun` argv, resolved
configuration hashes, and per-run metadata/summary hashes are retained in the
[performance summary](performance-summary.json); the live executor record also
contains the full ordered campaign state.

## Execution route and input discovery

The current host is the previously owner-confirmed allocation
`lm-mpi-job-23027817-4bfe-4730-926f-5c2de590b32b-mpimaster-0`.
The [existing ownership record](../ffn-row-owner/report.md#execution-authorization-and-environment)
and current objective authorize its reuse. Initial inspection found all eight
H100 80GB GPUs idle, with no compute applications. The task path is the verified
SR008 NFS mount. The existing Python is `/opt/torchtitan-venv/bin/python3.12`;
no environment installation is needed. Existing Perfetto services are preserved.

The exact named production tokenizer directory is available at
`/home/jovyan/shares/SR008.nfs1/models_templates/dclm-7b-128k_digit_tokenizer_FIM`.
Its share mounts from
`pd62-nfs-001.sr008.aicloud.sbercloud.tech:/vol_fg_001/namespace_ai0001071-01058`.
The supplied production donor reference is `ec47fb8e40a9a1e55d3c18dfa704ca104ae773af`,
whose documented asset path is `/mnt/nfs/07-share/models_templates/dclm-7b-128k_digit_tokenizer_FIM`;
that historical mount spelling is absent in this container. The supplied donor
revision is not locally available for direct checkout inspection; verification
uses the named execution-host asset and its actual loaded wrapper behavior. The actual asset's
`tokenizer.json` SHA256 is
`cc4f2dc6801b6dbe8042c09b1f0fb7cf0e156098c7769e217e29ffe7b6dbb587`.
It differs from the initially found checkpoint tokenizer; that copy was rejected.
Base vocabulary is 128000, full vocabulary 128256, UNK/BOS/EOS are 0/1/2,
and FIM prefix/middle/suffix are 128253/128254/128255. The loaded GIFT wrapper
confirms that an empty document with BOS/EOS requested encodes to `[1, 2]`;
preparation therefore uses the wrapper without manually adding special tokens.
The [data-002 receipt](data-002.json) records hashes for all three tokenizer
files and the loaded wrapper identity. The asset README says EOS postprocessing
was removed, but actual JSON includes BOS and EOS; measured loaded behavior
takes precedence over the README.

The completed `data-002` mixture uses existing GigaSets 5.3.1 English reference
text, Russian web, mathematics and Python code under
`/home/jovyan/shares/SR008.fs2/datasets/GigaSets/sources/5.3.1`. Its receipt
records both eligible source files and files actually consumed. Of 12 sorted
FineMath `00000` candidates, 3 were consumed; of 64 sorted Python `starcoder2`
candidates, 27 were consumed; and of 34 Russian-web candidates across
`CC-MAIN-2013-20`, `CC-MAIN-2013-48` and `CC-MAIN-2014-10`, 17 were consumed
(10 from `2013-20`, 7 from `2013-48`, none from `2014-10`). One eligible
Wikipedia shard was consumed. The receipt contains the exact consumed paths and
hashes. The prepared comparison mixture has
102400 distinct full packed rows, 25600 per domain, in deterministic row
round-robin order; it does not claim to reproduce the production mixture.
Duplicate-row checks skipped 870 repeated math rows. Packing and dataset,
index and row-manifest identities are recorded in [data-002.json](data-002.json).
The prepared data is at
`runs/production-data-abcd/artifacts/raw/data-002`, split `train`; its 26 MDS
shards retain the 102400-row logical epoch without oversampling.
Existing PyArrow 24.0.0 successfully
streams all four selected compressed raw formats with `input_stream` and
`TextIOWrapper`; no custom decompressor is needed. The established interfaces are
[PyArrow input streams](https://arrow.apache.org/docs/python/generated/pyarrow.input_stream.html)
and [MDSWriter](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.MDSWriter.html),
with the pinned installed/local source governing actual behavior.

Actual native-loader acceptance passed on CPU/Gloo at prefetch depths 0 and 32.
Each arm completed 672 batches per rank on all eight ranks, covering 1344 rows
per rank and 10752 distinct global rows over 21 updates. Both arms preserved
the same opening order and exact loaded input, label and loader-position order.
Each arm covered 44040192 physical tokens and 44029440 valid main targets; the
loaded token-ID bounds were 0–127998. Unique token IDs ranged from 12–3499 per
microbatch and 66681–69847 per update. The largest raw embedding-owner share
was 74.80–77.56% (5.98–6.20 times the mean); the largest distinct-ID owner share
was 18.94–19.61%. Full per-rank evidence is in the
[loader summary](data-002-loader-summary.json). This is CPU/Gloo loader evidence;
CUDA remained uninitialized and every worker shut down.
The training path uses causal per-row positions without EOS document isolation;
the loader's EOS-reset positions are part of loader acceptance, not the model's
effective position policy.

## Changes and verification

Independent bridge binding switches and bounded native component capture/replay
are implemented. The [binding checks](binding-checks.json) record 22 passing
bridge tests and an actual pinned-module CPU smoke covering all four independent
routing/CE selections, constructed main/chunked/MTP loss callables and restoration.
The [component fixture checks](component-fixture-checks.json) record eager,
compiled/checkpoint, replay, mask, failure/restoration and corrected-path checks.
Both CPU-stage receipts preserve their result/failure history, label their old
GPU limits as historical, and point resolved gates to the closed native
campaign and component evidence.
Scoped Ruff passes. Both native meta tests pass after the canonical
tokenizer-path correction, with CUDA hidden. The
[independent integration/runtime-cost review](integration-review.md)
resolved the binding/capture/replay source and runtime-cost review, including
canonical initialization/RNG preservation. After that review, final input
integration and actual-loader acceptance passed; the runner's actual
`_verify_ready_inputs(TASK_ROOT)` check also passes, recorded in
[input-readiness.json](input-readiness.json). These source and CPU/Gloo checks
establish source and input acceptance; the native GPU evidence is reported below.

Corrected preparation, active consumers, runner readiness and actual-loader
acceptance are complete. The one-update acquisition and six component processes
passed; [component results](component-results.json) retain their series, memory,
fixture identities and limitations. All four native timing runs and bounded
diagnostics passed; MCP trace analysis passed with its parser-coverage limits
documented below. No core algorithm is
being replaced. The launcher creates a fresh child process
session and cleans up only that exact owned process group; the CUDA profile only
exports environment settings and does not perform shared process cleanup.

Expected runtime costs: data preparation and loader census are outside timing;
new binding selection and receipts run at bootstrap; component hooks/copies run
only in explicit acquisition mode. Timing retains the existing native adapter
and agreed feature implementations. The source/runtime-cost review confirms the
disabled path installs no per-microbatch instrumentation.

## Native A/B/C/D timing

All four runs completed in the fixed serial order, with ten warmup and ten
measured complete optimizer updates on all eight H100 ranks. Each measured
update retained accumulation 32, all input loading/transfers, forward/backward,
SAC, communication and the active native path. Lightweight native logging was
on; profiling, detailed monitoring, validation, checkpoint saves, snapshots and
numerical capture were off. Each timing used a fresh task-owned compiler cache.
Source and resolved config checks passed for every run. The exact resolved
configuration hashes, raw metadata/summary paths and hashes, and the complete
host and child `torchrun` argv are in the
[performance summary](performance-summary.json) under `/native/A` through
`/native/D`; the [campaign plan](campaign-plan.json) records the commanded
timing and diagnostic sequence. The performance summary was generated while
the executor record still showed the four timings complete and diagnostics in
progress. Its executor-record reference is a path and JSON pointer that marks
the file mutable until diagnostics complete; no executor-file hash is stored
there. The closed [native campaign results](artifacts/raw/native-abcd-orchestration/native-campaign-results.json)
is 1,758,373 bytes with SHA256
`1b37551f23d95d1b052a57930c18118ee30772219df36a88c58fce5db14306f6`.

The source starts at `main` `2ada87125e433e1f76c4cb2bfad244b641a78a93`; the
[initial state](initial-state.json) and [initial dirty-file manifest](initial-patch-manifest.jsonl)
preserve contributor edits. The timing receipts record the same task source and
configuration after each run. The dataset and tokenizer identities remain those
documented above and in the linked preparation/loader receipts. The executor's
final source snapshot matches the frozen initial identity, and source/config
remained unchanged throughout the campaign. Final trace identities are in the
closed native campaign results.

The common recorded runtime was eight NVIDIA H100 80GB GPUs, Python 3.12.3,
PyTorch `2.15.0.dev20260822+cu130`, CUDA 13.0 and NCCL 2.30.7. The recorded
package versions include TorchTitan 0.2.2, DeepGEMM 2.3.0+f093a4c, DeepEP
1.2.1+1b8f467, Streaming 0.13.0, Tokenizers 0.22.2, Transformers 4.57.1,
Triton 3.8.0+git3f6e4113 and CUTLASS DSL 4.4.1; full runtime paths and hashes
are in the performance summary.

Primary elapsed is the same-host envelope `max(rank end timestamp) - min(rank
start timestamp)` around measured updates 11–20. Throughput is reported both
for 4096 physical input/label positions and for the 4095 valid main targets per
row; auxiliary MTP targets are not added to the TPS denominator. Every run
covered 20,971,520 physical tokens and 20,966,400 main-valid tokens globally.
Memory is the maximum PyTorch peak across ranks during the measured window, in
GiB (`2**30`); active and allocated peaks matched in these runs. Allocator retry
and OOM columns are end-minus-start counter deltas across the measured window.
Partial compiler-boundary observations reported stable covered sources for all
four timings, but do not establish that asynchronous compiler workers or
uncovered DeepGEMM native JIT paths were settled. The timing caches were fresh
per run; each diagnostic reuses its matching timing cache.

| Variant | Run | Envelope (s) | Physical global TPS | Main-valid global TPS | Peak allocated / reserved (GiB) | Retry / OOM deltas | First-five to last-five cadence drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | `proddata-a-timing-001` | 134.560254 | 155,852.262866 | 155,814.212997 | 69.1577 / 69.7520 | 0 / 0 | +0.240116% |
| B | `proddata-b-timing-001` | 128.509044 | 163,191.004760 | 163,151.163206 | 69.1492 / 69.7676 | 0 / 0 | +0.141124% |
| C | `proddata-c-timing-001` | 125.728196 | 166,800.452033 | 166,759.729266 | 67.2964 / 67.6875 | 0 / 0 | +0.163214% |
| D | `proddata-d-timing-001` | 123.337618 | 170,033.444259 | 169,991.932188 | 65.8904 / 68.0879 | 0 / 0 | +0.580241% |

All eight ranks in all four timing runs reported zero allocator-retry and OOM
deltas. These counters and peaks are in the machine-readable summary.

| Comparison | Physical TPS change | Main-valid TPS change | Envelope change |
| --- | ---: | ---: | ---: |
| B/A | +4.708781% | +4.708781% | −6.051210 s (−4.497026%) |
| C/B | +2.211793% | +2.211793% | −2.780847 s (−2.163931%) |
| D/C | +1.938239% | +1.938239% | −2.390578 s (−1.901386%) |
| D/A | +9.099118% | +9.099118% | −11.222635 s (−8.340231%) |

These are observed ratios from one fixed-order serial comparison. No identical
control repeat or order alternation was run, so these measurements do not
establish repeatability, causal attribution, or a general speedup. Cadence drift
is descriptive, not proof that every asynchronous compiler path or the whole
run had reached universal steady state.

The following table shows the maximum host interval across ranks for each
corresponding `Profiler.step` return. These are unsynchronized host cadence
intervals, not separately synchronized GPU update times. Step 1 has no prior
boundary and is null; step 20 includes the final device drain. The full eight-rank
arrays are retained in `performance-summary.json` at
`/native/{A,B,C,D}/cadence/per_rank_interval_s`.

| Step | A (s) | B (s) | C (s) | D (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | — | — | — | — |
| 2 | 14.031 | 13.854 | 13.102 | 12.909 |
| 3 | 13.371 | 12.721 | 12.444 | 12.206 |
| 4 | 13.383 | 12.756 | 12.495 | 12.289 |
| 5 | 13.518 | 12.781 | 12.499 | 12.266 |
| 6 | 13.445 | 12.797 | 12.532 | 12.262 |
| 7 | 13.434 | 12.834 | 12.538 | 12.284 |
| 8 | 13.448 | 12.811 | 12.554 | 12.277 |
| 9 | 13.433 | 12.807 | 12.541 | 12.295 |
| 10 | 13.452 | 12.834 | 12.568 | 12.281 |
| 11 | 13.424 | 12.835 | 12.545 | 12.285 |
| 12 | 13.449 | 12.826 | 12.562 | 12.288 |
| 13 | 13.427 | 12.843 | 12.563 | 12.291 |
| 14 | 13.466 | 12.858 | 12.571 | 12.320 |
| 15 | 13.434 | 12.848 | 12.574 | 12.308 |
| 16 | 13.457 | 12.856 | 12.568 | 12.321 |
| 17 | 13.508 | 12.856 | 12.588 | 12.339 |
| 18 | 13.464 | 12.849 | 12.592 | 12.370 |
| 19 | 13.449 | 12.871 | 12.581 | 12.412 |
| 20 | 13.484 | 12.869 | 12.587 | 12.406 |

Each bootstrap recorded the intended active bindings on all eight ranks: A used
original routing and original cross-entropy; B, C and D used fused
`indices_to_map` routing and stable cap-zero cross-entropy. A/B used native FLA;
C/D used FlashQLA `external_gate`. The three D-only controls were resolved true
in D and false in A/B/C. The D launcher log has SHA256
`a1ee82982337116e90623ca3632882e7bf9457b84ccc15bcff5f462e95d6df58`; it records
the first trunk-layer `layers.0.feed_forward` and MTP FFN activation on every
rank. D retained the shared FFN storage with no added pools or embedding
workspace; the row-owner embedding marker was active on all eight ranks. The
closed native campaign result contains exact per-rank D activation payloads and
log-line references. Per-rank bindings and source module identities are in
`performance-summary.json`.

## New-data component refresh

The six isolated processes all passed with ten warmup and ten measured samples
on all eight ranks, replaying the captured new-data native inputs. Paired
harness initialization is canonical; it does not export or load native-run
weights. Ratios are candidate/control ratios of medians across per-sample
rank-max metrics. The component results and their raw series remain in
[component-results.json](component-results.json).

| Component metric | Control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| Embedding synchronized wall (ms) | 12.886427 | 2.716073 | −78.922993% |
| Embedding joined CUDA-event span (ms) | 12.869648 | 2.695424 | −79.055962% |
| Embedding host-call span (ms) | 0.988989 | 2.352088 | +137.827429% |
| Embedding explicit end-sync wait (ms) | 11.934790 | 0.315548 | −97.356065% |
| `layers.0` complete-block wall (ms) | 23.828990 | 21.950527 | −7.883100% |
| MTP complete-block wall (ms) | 31.890389 | 29.666788 | −6.972636% |

Embedding peak allocated memory fell 35.401696%; peak reserved memory rose
11.032441%. These scopes stay separate: the event span joins participating
GFSDP streams and includes queue/submission gaps and collectives; the host-call
span contains dispatch/submission and internal waits; endpoint synchronization
measures its explicit wait. None is kernel-compute attribution. The FFN block
wall includes zeroing, forward/SAC replay/backward and GFSDP, excludes optimizer,
input loading, H2D and the barrier. Existing per-rank FFN pool budgets remained
1,214,547,072 bytes for `layers.0` and 1,405,627,392 bytes for MTP; the change
adds no pool. Isolated medians cannot be added or extrapolated into full-step
time. The component harness does not expose allocator retry counts; no explicit
OOM/retry log markers were observed, which is not a measured zero count.

## Results and gate closure

Real-data input acceptance, source/runtime-cost review, native acquisition,
six component processes, all four timing runs, all four diagnostics and MCP
reconciliation are complete. Diagnostic A
passed: all eight ranks completed 21 updates, accumulation 32 and the optimizer
step; ranks 0/1 captured exactly three complete forward/backward microbatches
on update 21. Its two original traces are about 27.7 MB each under
`artifacts/raw/proddata-a-diagnostic-001/native/profiling/traces/iteration_21/`.
The transient NFS supervisor wait recovered without restart, retry or extra run.
Diagnostic B passed its all-rank full-workload validation with no retries:
all ranks completed 21 updates, accumulation 32 and the optimizer step; ranks
0/1 captured exactly three update-21 forward/backward microbatches, with one
train-step marker. Its original traces are 26,344,477 and 26,391,220 bytes at
`artifacts/raw/proddata-b-diagnostic-001/native/profiling/traces/iteration_21/`.
The recorded summary SHA256 is
`fc30ddb34bef7ec1bc7e5e19c7c6244c91fe492732a13842c568412318555948`.
Diagnostic C also passed all-rank full-workload validation with no retries: all
ranks completed 21 updates, accumulation 32 and the optimizer step; ranks 0/1
captured exactly three update-21 forward/backward microbatches, with one
train-step marker. Its original traces are 25,870,825 and 25,873,181 bytes at
`artifacts/raw/proddata-c-diagnostic-001/native/profiling/traces/iteration_21/`.
The recorded summary SHA256 is
`cfd299b0955d54efda0f890d699ff5834e9e89f40d59e4517d31e92c17181779`.
Diagnostic D produced both original traces, 27,119,126 and 27,119,516 bytes at
`artifacts/raw/proddata-d-diagnostic-001/native/profiling/traces/iteration_21/`.
Each contains exactly three update-21 microbatch markers, one train-step
marker, and six microbatch GPU projections. The recorded summary SHA256 is
`694649d30eb675e7ae088239caab550bee8424299394e27a77f6efa426416245`. Native
integration assertions pass: all eight run entries and plan rows are complete,
64 rank series are present (20 timing steps and 21 diagnostic steps per rank),
all eight original traces have required markers, and compiled/early-reduction
markers for both FFN groups are present on all eight ranks in D. The frozen initial
and final identities match. All child groups and the supervisor exited; the GPU
idle receipt confirms eight H100s at 0 MiB with no compute applications, and the
lease is closed. The closed native JSON identity is linked above.

MCP reconciliation is complete; detailed trace scopes and the parser limitation
are summarized below. The separately unprofiled timing runs remain the only
evidence for full-run throughput and memory magnitudes.

## MCP Perfetto trace comparison

The MCP worker loaded all eight original trace paths first, then reconciled them
against clearly labeled lane derivatives. The originals remain unchanged. Their
initial imports dropped overlapping kernel events; the derivatives preserve the
complete event census and device flows with zero import errors. A derivative
colors overlapping device/annotation events onto synthetic display tids and
updates matching GPU flow-finish tids only. CUDA stream IDs and all event fields
except those permitted tids remain unchanged. The derivative tids are display
lanes, not CUDA streams. All host and GPU `task.profile` markers and GPU
`FSDP::post_backward` annotations remain present.

| Variant/rank | Kernels omitted by original import | Kernels in derivative | GPU annotations | GPU `post_backward` annotations | Matched device flows |
| --- | ---: | ---: | ---: | ---: | ---: |
| A/0 | 41 | 8,795 | 354 | 98 | 9,930 |
| A/1 | 52 | 8,795 | 354 | 98 | 9,930 |
| B/0 | 35 | 8,391 | 298 | 98 | 9,445 |
| B/1 | 42 | 8,391 | 298 | 98 | 9,445 |
| C/0 | 32 | 8,103 | 298 | 98 | 9,157 |
| C/1 | 45 | 8,103 | 298 | 98 | 9,157 |
| D/0 | 40 | 8,230 | 330 | 103 | 9,317 |
| D/1 | 39 | 8,230 | 330 | 103 | 9,317 |

All derivative imports reported zero errors, every kernel/memcpy/memset launch
has a matching device flow, and all flow endpoints matched their correlations.
GPU microbatch projections are three per trace in A/B/C
and six in D; D's six are range structure within the same three microbatches,
not additional captured microbatches. The original importer’s overlapping-slice
errors are parser coverage limits, not failed trace captures.

The table below reports candidate-minus-control changes in the three captured
forward/backward microbatch spans on ranks 0/1. Host F+B scope sums and device
unions are separate measures. NCCL communication kernels can overlap compute;
these columns must not be added to estimate elapsed time.

| Comparison | Host F+B scope sum Δ (r0/r1 ms) | Device busy union Δ (r0/r1 ms) | Compute union Δ (r0/r1 ms) | NCCL communication union Δ (r0/r1 ms) |
| --- | ---: | ---: | ---: | ---: |
| B/A | −25.465 / −28.617 | −19.525 / −21.832 | −23.337 / −14.656 | +18.422 / +1.821 |
| C/B | −31.578 / −25.715 | −31.484 / −27.955 | −26.972 / −28.831 | −9.089 / −7.839 |
| D/C | +11.168 / +6.687 | −20.760 / −19.539 | −12.219 / −15.719 | −45.938 / −50.375 |
| D/A | −45.875 / −47.645 | −71.769 / −69.326 | −62.528 / −59.206 | −36.605 / −56.392 |

Observed mechanisms match the requested cumulative variants:

- A consumed synchronously from `DataLoader`; B consumed from the ready queue.
  B's per-rank DtoD copies across three F+B microbatches fell from 181 to 100,
  and their volume fell from 34,037,467,596 to 8,660,093,388 bytes. B changes
  four controls together, so the trace does not isolate their individual effect.
- C/D used the externally gated scan path. The selected scan/gate family fell
  from 396 kernels per rank in A/B to 216 in C/D. Native activation records
  confirm `external_gate`; kernel-family sums are not full-step elapsed time.
- D contains six FFN split reduce-scatters per rank across the three captured
  microbatches (two FFN groups × three; 12 cases across ranks 0/1). Each
  has 134,479,872 FP32 input
  elements (537,919,488 bytes; 513 MiB of logical input payload) with group size eight and starts before
  its parent group reduce-scatter. The reported 96.094–98.246% is the interval
  union of intersections between that reduce-scatter and linked parent-scope
  kernels, excluding NCCL, copy and memset, divided by the reduce-scatter
  duration. This compute-only overlap does not identify a wait dependency or
  prove causality.
- Each C trace has nine `tok_embeddings` collectives; D has zero. The six
  `lm_head` collectives remain. D records three `_OwnerLookup` and three
  `_OwnerLookupBackward` scopes per trace, with real uneven owner request counts.
  The local ID requests differ between ranks, so this is not balanced synthetic
  traffic.
- Row-owner embedding adds six host `cudaStreamSynchronize` calls per rank
  across the three forward microbatches, summing to about 0.034/0.033 ms on
  ranks 0/1, plus small DtoH metadata copies. The measured DtoH metadata volume
  is 1,794 bytes per rank versus 42 in C. Host API wait duration is not GPU
  delay, and the host scope may lengthen while device unions shorten.

These traces cover only ranks 0/1 and the first three of 32 microbatches. All
ranks completed accumulation and the optimizer outside the capture, but the
traces do not expose the full update, optimizer tail, cross-rank straggler cause,
complete CUDA event dependencies, or exact wait causality. They do not establish
throughput, a whole-step critical path, or individual causal gains. The full
per-microbatch records, all eight original and derivative hashes, SQL, MCP import
receipts, source hashes and verification log are in the [trace analysis](trace-analysis.md)
and [machine-readable result](trace-analysis.json).

## Interpretation limits

The comparison mixture is globally balanced, but deterministic row-round-robin
input with DP8 fixes each rank to one domain: ranks 0/4 read Wikipedia, 1/5
mathematics, 2/6 Python and 3/7 Russian web. Diagnostic ranks 0/1 therefore
cover Wikipedia and mathematics locally; expert-parallel traffic still spans
all eight ranks. These traces do not represent every local data domain on each
rank.

Configured EMA stayed enabled, but these 20/21-update runs do not establish
long-run EMA update or saved-state behavior. The comparison includes no
convergence study, evaluation, checkpoint save, checkpoint resume or long-run
stability validation. The single-node crop's EP8/dense-FSDP8 results do not
establish production PP4/EP32 or inter-node behavior. These are evidence limits,
not additional acceptance requirements.

## Failed or superseded checks

- An optional final table checker selected the earlier timing B/A row instead
  of the trace B/A row. A corrected check restricted to the MCP section verified
  all four comparison rows and all four metric columns against
  `trace-analysis.json`; no measurement was repeated.
- A metadata probe queried an absent `MLS_JOB_ID` and stopped before its chained
  GPU commands; a subsequent direct all-GPU inspection succeeded.
- Installed `zstd.decompress` rejected a verified raw corpus file with a length
  mismatch. PyArrow streaming decoded all selected source types successfully.
- Early tokenizer candidates had no production provenance; one checkpoint's
  sibling config named a different Qwen tokenizer. They are not used.
- Preparation `data-001` failed after 470 seconds: the selected Russian-web
  crawl exhausted at 17491 of 25600 required domain rows, leaving 69967 total
  partial rows. Its failed status, log and raw output are preserved. It is
  excluded from loader acceptance and every GPU run. The completed `data-002`
  preparation extends only sibling crawls in the same named Russian-web corpus,
  retains the four equal quotas and packing rules, and records 870 skipped
  duplicate math rows in its [receipt](data-002.json).
- The initial command-plan checker expected the removed tokenizer CLI override.
  The corrected checker validates canonical input paths and asserts no such
  override; all eight timing/diagnostic command round trips pass. No native
  process was launched by this check.
- Root's static review found that the initial `--review-only` pass omitted
  freeze execution while referencing a nonexistent MDS-fragment path and
  assuming three loss objects. These issues were corrected before launch; the
  actual freeze and eight-rank captured-schema check then passed, confirming the
  single `MTPLoss` and frozen state. The mistaken review-only pass did not launch
  a timing run.
- A validator initially mishandled a `None` mask and the aggregate checker
  assumed a top-level shape that did not match the nested results. Both were
  corrected before campaign interpretation; no GPU run was repeated.
- During component teardown, a nonfatal NFS `.nfs` busy-file error (errno 16)
  was reported even though the child exited 0. The owned process group stopped
  with no orphan; the warning remains recorded in
  [component results](component-results.json).
- Component command review corrected a relative capture path before launch:
  ranks run from `gift/`, so capture, output and fixture paths must be absolute.
  No GPU attempt or retry resulted from this correction.
- The first goal-create call found the existing file-reading goal active; after
  reading it, that goal was completed and the full campaign objective was started.
