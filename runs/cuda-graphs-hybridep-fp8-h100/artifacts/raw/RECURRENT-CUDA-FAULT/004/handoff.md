# Completion-prefix review and next decision

## Decision

No source fix is justified. Select **one future, bounded, all-rank Compute
Sanitizer memcheck attempt with user-kernel binary patching disabled**, after
storage, compute release and runner gates
are satisfied. Do not launch now, repeat the synchronization sweep, or make a
provenance-only rerun. The result is an unresolved native CUDA fault, not a
negative CUDA Graph verdict.

Finite acceptance completed: review the actual adapter and immediate stream/
ownership helpers; interpret the completed prefix; review diagnostic runtime
cost; verify an installed diagnostic interface and its limitations; select one
next decision and stop. Only this directory was written. No source edit, CUDA
query, import, GPU execution, allocation, installation or process action occurred.
The code-rules and perf-experiments skills informed this review.

## What the prefix establishes

The canonical [extraction](../../COMBINE-BWD-PROBE/attempts/001/postrun-extraction-verified.json)
and [handoff](../../COMBINE-BWD-PROBE/attempts/001/handoff.md) establish identical
completed ordinals 1361–1384 on eight ranks and unmatched enter 1385. There were
20 completed optimizer updates and 81 completed forward/backward microbatches.
Ordinal 1385 is the eighth selected call in update 21 microbatch 1 (zero-based),
not a recorded layer identity. Rank 6 reports CUDA 715 in event.synchronize.

`_call_adapter` calls the original helper once, then records its event on the
current compute stream after the helper's join. `_launch_on_comm_stream`
enqueues comm-stream dependence on prior compute, records tensor input use on
comm, runs the lambda there, then joins comm into compute and records output
use on compute. When comm-stream mode is disabled, the callable runs directly
on current stream. The selected lambda is the actual registered custom-op body
at `hybridep.py:651`; it retains the handle via closure. The adapter frame retains
that closure, produced inputs and returned outputs through its wait.

Consequently, each successful event wait establishes completion of that rank's
compute-stream prefix and work transitively joined into it. The last success is
a completion boundary before the next joined prefix reports a fault. Work
submitted between those events includes intervening backward/recomputation and
the next combine, not just combine 1385. No timestamps delimit this interval.
The 575.82 ms scope is the whole rank-6 failing forward/backward microbatch.

The boundary does not exclude earlier silent corruption that only becomes an
invalid access later, work on unrelated streams without a dependency, or a peer
fault affecting local progress. Enter means the native helper returned on the
host; it does not mean its GPU work completed. Peer unmatched enters can reflect
peer waits or termination; they do not establish eight originating faults.
Neither CUDA 715 nor the asynchronous reporting site identifies quantize,
combine, GEMM, a PC or the same cause as earlier failures.

## Integration and cost findings

| Finding | Disposition |
| --- | --- |
| Duplicate native invocation or premature input/output/handle release caused by adapter | No such violation found on inspected path. Original helper is called once; references survive the wait. |
| Observer causes a demonstrated peer-progress protocol violation | Unproven. Added host waits delay later launches and may change peer skew, reuse and overlap. This is a causal confound, not evidence of an invalid protocol. |
| Original helper's tensor stream lifetime protection | Present for produced tensor inputs and outputs; adapter preserves it. Opaque handle remains retained. This does not certify all native internals. |
| Native wrapper hidden-stream behavior | Python wrapper constructs HandleImpl, passes handle tensors into native combine and returns its outputs. Backend launch helper accepts a stream and launches flag/combine/sync kernels on it. Installed C++ binding source/build identity is unavailable here; Python join cannot certify uninspected hidden native work. |
| Repair selected by this result | None. Keep quantizer/source-origin findings from review 003 open. |

Active calls add a no-timing event allocation/record, host completion wait and
two flushed JSON records each: 68 checkpoints per optimizer update if all four
microbatches finish. Twenty-four waits succeeded per rank in this run. The
adapter also adds selector/counter work before arming and JSON lifecycle logs.
No extra arithmetic or tensor copies appear in the adapter, but live references
persist through waits and submission/communication overlap changes. Event,
compute, host-wait, copy/communication and peak-memory costs were not separately
measured. The diagnostic is unsuitable for throughput claims or production
runtime acceptance. This review itself has no training-runtime impact.

## One next experiment, conditional on release

**Unresolved question:** can memcheck, with user-kernel instrumentation disabled,
associate the recurrent hardware exception with an executed kernel and possibly
a device PC/backtrace in the preserved native configuration, rather than another
later CUDA API reporting site? This tests attribution, not memory correctness.

Use installed `/usr/local/cuda/bin/compute-sanitizer` (resolves to
`/usr/local/cuda-13.2/bin/compute-sanitizer`), version **2026.1.0.0 build 37182542**.
Its locally checked help supports `--tool memcheck`, `--target-processes all`,
`--error-exitcode`, `--show-backtrace`, `--log-file`, `--launch-timeout` and
`--print-limit`. Apply memcheck to the existing torchrun process tree so all
eight interacting ranks are checked. Preserve graph-OFF/mix0, exact model,
FP8, SAC, compilation, EP8/FSDP2, data and seed. Disable only this completion-wait
observer for that diagnostic through its supported configuration; retaining its
waits would combine two scheduling interventions. This needs an explicitly
reviewed diagnostic overlay before release, not a source rewrite.

The shipped manual documents `NV_COMPUTE_SANITIZER_BINARY_PATCHING`, which
controls whether user-kernel code is instrumented (default enabled; documented
integer range includes zero). Set it to `0` before launching the tool for this
debugging attempt. It is explicitly a debugging-only variable. The installed
help has no `--binary-patching` option; do not copy that historical CLI.

Use default nonblocking launches, zero forced-synchronization limit, no added
allocation padding, no coredump and no new profiling tool. Do not guess an
ordinal-to-kernel launch-skip mapping, filter only quantize, or rebuild binaries
with sanitizer compiler flags. Those choices could miss the unknown origin or
change the executable being investigated. Disabling patching avoids the intended per-access user-kernel instrumentation
by the documented control, but actual overhead has not been measured. It does
not remove CUDA API interception, tool tracking, host overhead or peer skew.

The shipped version-matched documentation confirms nonblocking concurrent-kernel
operation and describes imprecise hardware-exception reports with a kernel-launch
host backtrace and potentially a device backtrace (sometimes none). It also
documents Hopper bulk-copy and sm_90a wgmma checking for normal instrumentation;
those precise checks must not be claimed for the patching-disabled proposal.
The manual does not explicitly promise hardware-exception reporting survives
patching disablement. Its suitability for exception attribution is therefore an
inference to test in this one bounded attempt, not a supported hardware-only
mode guarantee or a promise of the faulting PC. This CUDA 13.2 installation is a plausible H100 tool route,
not a verified runtime compatibility result. Neither shipped documentation nor
the existing run establishes complete NCCL/HybridEP peer-allocation coverage,
actual exception-attribution coverage, or sufficient host/device memory. In particular,
initcheck's documented IPC limitation does not prove a memcheck limitation, and
must not be generalized to one. This mode must not be credited with detecting out-of-bounds accesses that do
not raise hardware exceptions, earlier silent writes, wrong values, per-access
alignment/bounds errors ordinarily found by instrumentation, or distributed
races. It cannot establish caching-allocator lifetime correctness. If any module
already contains compile-time sanitizer instrumentation, this variable is not
proof that such embedded instrumentation has been removed; keep that provenance
limit explicit. This installed version also documents a possible
cudaMallocAsync/null-stream race-check false positive; evaluate such a report
against actual allocation APIs before calling it the fault.

NVIDIA documents substantial slowdown, possible timeouts, and large host/device
tracking-memory requirements. Do not solve sanitizer OOM by serializing kernels
or dropping production features. Exact driver/tool compatibility, eight-rank
tracking and actual coverage remain launch/first-attempt gates. A clean or
nonreproducing run cannot clear this fault.

| Outcome | Changed decision |
| --- | --- |
| Useful hardware-exception attribution to kernel, with PC/backtrace if available | Inspect that exact mapped module and consumer/ownership contract; select a fix only after corroboration. |
| Only later API error, clean bounded completion, missing coverage, tool failure, OOM or timeout | No source fix or fault clearance. Preserve evidence and stop; do not auto-retry or expand a sweep. |

**Bounds:** one attempt, at most 60 optimizer updates and the existing 20-minute
outer deadline, no retries. Stop on first fatal error, tool failure, storage
floor or deadline through the reviewed task-owned supervisor. A useful first exception
report is enough; preserve it before peer shutdown. Sixty updates or deadline
without attribution is inconclusive. No automatic full-instrumentation fallback
is selected. No performance claim is permitted.

**Runner gates:** storage was last reported only 12.04 GiB free and falling;
this proposal is not released. Before launch, require a reviewed bounded output
budget and storage floor, fresh ownership/availability checks, and explicit
compute release. Start the existing binary watcher before launch and require a
fail-closed READY receipt confirming watcher liveness, correct attempt directory
and process-discovery readiness. Then launch and capture actual same-run mapped
quantize/GEMM modules while ranks are alive. READY must precede launch, not the
training error. Limit sanitizer logs, preserve the patching setting and record whether all
eight ranks were tracked; do not overwrite the failed attempt. This fixes the prior missed
monitor without authorizing a provenance-only run or requiring a new harness.

## Pins and references

- SOURCE-SNAPSHOT/007 manifest SHA-256:
  `2585e504350276cb4689ae9ea45a33706e9f898bb61969abe5d08f2650c55620`;
  frozen base `446431dfa191ef0300f116257ae918bc85e38c06`.
- `combine_bwd_completion.py` SHA-256:
  `b07e53b3f6c40e89bbdd8e243f8ec061a29e8dd29574691ea375c96274c0ab6a`;
  `hybridep.py` SHA-256:
  `12b1921372f265ede5cca4974f5867f954f84862cc257ff8e02bb7ecafa0814d`.
- Rank-6 runtime Python wrapper SHA-256:
  `63e425916b3c6b0c881ad402af35a2b00c4c84203fcde611a6715744b89210cb`;
  runtime backend pin is preserved in its existing runtime_manifest.json.
- Sanitizer executable SHA-256:
  `5cba659e1cd603aa2370e103425f184368e6ed405da90ce85743c4399c324dbc`.
- Shipped v2026.1.0 manual SHA-256:
  `48c6bead77ae1787c60f88049dbcc14076415064cb8a96d07ca481bfefa0ad8c`;
  shipped release notes SHA-256:
  `7a158d7f755497a5e0c2d2d94bf614ff4eda48e6665aeb52f354929a51a3e316`.
  Both were read under `/usr/local/cuda/compute-sanitizer/docs/`; they govern
  version-specific claims here. Current online
  [manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
  and [release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)
  were cross-checked, but now describe newer releases; their later fixes are
  not attributed to the installed executable.

Open acceptance gates: origin/PC and actual loaded quantize/build identity;
native distributed correctness and reliable completion; then the campaign's
remaining numerical, performance and checkpoint/storage gates. This review
does not reopen baseline bit-equality or nondeterminism work.
