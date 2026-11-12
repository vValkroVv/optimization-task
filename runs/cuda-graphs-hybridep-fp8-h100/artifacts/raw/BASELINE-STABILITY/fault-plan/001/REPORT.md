# First-device-fault localization plan

## Selected next action

Capture a native GPU exception core on every rank, then inspect the faulting
kernel and PC offline with the installed CUDA-GDB. The coordinator's tiny isolated
exception smoke test has now established capture and decoding on this driver
route. Next run the coordinator's bounded 200-update graph-off diagnostic using
the native `train_scan_fix` entry, with no timing, numerical, completion-prefix
or profiling observer. This is a diagnostic proposal; this author executed no
GPU work and changed no training source or installed package.

Finite acceptance for this planning deliverable: reconcile existing failures;
verify installed tool/options; select a way to reach the failing workload and
prove capture coverage; specify result-dependent next actions; check this
document. These items are complete. Actual capture, fault localization and
repair remain open.

Use the exact 3B MoE, HybridEP EP8/FSDP2, FP8, FlexAttention, selective AC,
compiled model and chunked-8 CE, local/global batch 8/256, sequence 4096,
accumulation 4 and seed 42. Preserve math, data and scheduler horizon 30000.
Graphs remain off. No independent bitwise loss/norm gate is introduced.

## Why change the diagnostic method

- [Performance 007](../../../PERFORMANCE-ASSESSMENT/analysis/001/REPORT.md)
  completed 33 optimizer updates on all eight ranks and failed during 34.
  Rank 1's NCCL watchdog reported an illegal instruction; it is a reporting
  site, not kernel attribution. The same graph-off workload previously failed
  at 22, 33 and 48; the successful 41-update control does not clear recurrence.
- [Loss-only control 001](../../../LOSS-ONLY-PERF/attempts/001/failure-diagnosis.md)
  reports compiled backward/SAC recomputation and
  `cudacherry.quantize_blk128x128` after 47 completed updates. The quantizer's
  launch check calls `cudaGetLastError`, which can expose an earlier fault.
- [Completion-prefix probe](../../../COMBINE-BWD-PROBE/attempts/001/handoff.md)
  stopped in update 21. Completed ordinal 1384 and unmatched 1385 bound a
  prefix containing other compute and joined communication; they do not
  isolate combine. Extra waits/references changed scheduling and lifetimes.
- [Sanitizer 001](../../../SANITIZER-ORIGIN/analysis/001/handoff.md) reached
  only 16 complete updates in 20 minutes. Its 50-report limit was filled by
  NCCL initialization API messages; all-rank tool attachment was unproven.
  Repeating that command cannot answer the first-kernel question reliably.

Native exception capture avoids instrumenting every kernel and retains the
asynchronous workload. It can still perturb execution, and the smoke test
cannot predict native capture overhead or ensure reproduction. Native 200
updates is the coordinator's selected reach budget, not a stability pass.

## Installed tools and supported settings

Read-only `--help`, `--version`, debugger help, installed headers and retained
source were inspected on 2026-11-12. No CUDA initialization was requested by
this inspection. Repository base is
`446431dfa191ef0300f116257ae918bc85e38c06`; the coordinator owns current
Source010's 98-file freeze and launch release.

| Tool or interface | Verified installed identity |
| --- | --- |
| Runtime Python | `/opt/torchtitan-venv/bin/python3.12` |
| CUDA-GDB wrapper | `/usr/local/cuda/bin/cuda-gdb`, selecting `/usr/local/cuda/bin/cuda-gdb-python3.12-tui`; CUDA-GDB 13.2, GDB 16.3 |
| CUDA compiler | `/usr/local/cuda/bin/nvcc`, CUDA 13.2 V13.2.51 |
| Disassembly | `/usr/local/cuda/bin/cuobjdump`, `/usr/local/cuda/bin/nvdisasm`, CUDA 13.2 V13.2.51 |
| Compute Sanitizer | `/usr/local/cuda/bin/compute-sanitizer`, 2026.1.0.0 build 37182542 |
| Installed flags | `/usr/local/cuda/include/cuda.h:25204`, `CUCoredumpGenerationFlags` |
| Debugger driver library | `/usr/local/cuda-13.2/compat/lib.real/libcudadebugger.so.595.45.04` exists |

The retained runtime receipt pins Torch `2.15.0.dev20260822+cu130`, git
`8b353065ce260336c5fdfa74d754e8bd9b91b062`, CUDA build 13.0, NCCL 2.30.7,
Triton 3.8.0, DeepEP `1.2.1+1b8f467`, DeepGEMM `2.3.0+f093a4c` and
CudaCherry 0.1.0. Toolkit 13.2 is the debugger/compiler tool identity, not a
claim that Torch was built with it.

The [version-matched CUDA-GDB 13.2 manual](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-gdb/index.html#gpu-core-dump-support)
documents exception cores, memory omission flags, retained relocated ELF,
PID/time filenames and offline `target cudacore`. It warns that another
developer tool or process termination during generation can invalidate capture.
No debug rebuild of production kernels is required. The installed header lists
the six flags below as bits 0–5; their union is 63.

Set before the smoke process or native worker initializes CUDA:

```bash
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_GENERATION_FLAGS=skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory,skip_abort
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_FILE="$ATTEMPT_DIR/cores/core.%p.%t.nvcudmp"
```

Create the unique task-owned `cores/` directory before launch. The launcher must
propagate the same four variables to all eight ranks. PID/time make files unique;
join PID to rank using native identity records. A rank wrapper may add a literal
`r${LOCAL_RANK}` component after torchrun has assigned it. Shell expansion in the
parent alone cannot produce eight rank-specific values.

Omit global/shared/local/constant payloads and redundant nonrelocated ELF;
keep relocated ELF, register/warp state and exception metadata for PC decoding.
This sacrifices tensor contents and stack-memory inspection. Retain the
existing native JIT/cache files and mapped binary identities. `skip_abort`
avoids the core generator's extra CPU abort; native CUDA failure handling and
the runner's failure stop remain active.

Do not use removed `CUDA_ENABLE_LIGHTWEIGHT_COREDUMP`, or newer manual options
`faulted_contexts_only`, `log_only` or `no_errbar_at_exit`: they are absent from
the verified 13.2 header/manual. Do not combine this route with sanitizer,
CUDA-GDB attachment, profiler or launch blocking.

### Storage and lifecycle

Memory payload omission prevents the intended core from containing the model,
optimizer and large activation allocations. Size still depends on loaded ELF
images and resident execution state; no documented hard byte bound exists.
Use a **16 GiB aggregate diagnostic envelope plus 32 GiB free reserve** as an
operating budget, not a promised maximum or a file truncation limit. Record
actual smoke/core sizes, free space and growth. The coordinator reports ample
fresh capacity; recheck it at release. Never truncate a core or send a signal
while progress indicates active generation. Normal torchrun peer shutdown can
also interrupt another rank's dump: inspect its shutdown timeout/cleanup policy
and retain the first fully readable core before considering capture complete.

## Small capture smoke: passed before native 200

The coordinator resolved whether this container's actual compat-driver path
can emit and decode a small exception core. At 10:29:50–10:29:51 UTC,
[smoke 001](../../tool-smoke/001/result.json) returned the intended process
exit 86 after CUDA synchronization error 719. The 252,565-byte core completed
with `All done`; omitted global payload was zero bytes. The
[essential decode](../../tool-smoke/001/decode-essential.log) identified
`opt004_trap_kernel`, PC `0x7495537b9d20` (function offset +32), and
`BPT.TRAP 0x1` at the preceding instruction (+16). This demonstrates
kernel/PC/instruction coverage on one device. Do not repeat this completed smoke
without a relevant tool/driver/capture change or contradictory evidence.

The satisfied smoke contract used a separate task-owned executable on one
authorized idle GPU. Its device work was one named kernel with one thread
executing inline PTX
`asm volatile("trap;");`; the host selects the intended device, checks setup,
launches once, calls `cudaDeviceSynchronize`, prints the returned error and
returns a deliberate nonzero result. This is an intentional exception test,
not training or fault reproduction. Its retained source, build command and
executable use the installed nvcc with `-arch=sm_90 -lineinfo`, without `-G`.
The four capture variables above and the bounded timeout were applied.

The smoke acceptance was an expected failing CUDA call, completed core progress, one
nonempty unique core, and successful offline decoding that names the smoke
kernel, identifies an exception and PC, and shows the relevant instruction.
The trap exception need not have exactly the native 715 error number. Its actual
result is 719, recorded above. A core file's existence alone does not prove PC coverage.
One capture validates this execution route; it does not prove eight-rank
coverage or correctness of any training kernel.

Minimal offline command, with `CORE` set to the selected absolute filename:

```bash
/usr/local/cuda/bin/cuda-gdb --batch \
  -ex 'set pagination off' \
  -ex "target cudacore $CORE" \
  -ex 'info cuda kernels' \
  -ex 'cuda device sm warp lane block thread' \
  -ex 'info cuda lanes' \
  -ex 'p/x $pc' \
  -ex 'info symbol $pc' \
  -ex 'bt' \
  -ex 'set cuda disassemble_from elf_image' \
  -ex 'x/4i $pc-32' \
  -ex 'x/4i $pc'
```

The original [decode](../../tool-smoke/001/decode.log) failed when `/r` tried
to read omitted raw instruction memory. That failure is retained; ELF-based
`x/4i $pc-32` and `x/4i $pc` decoded successfully in the essential rerun.
Use ordinary instruction decoding, without `/r`, for native cores. Remaining
optional commands above were checked through installed help. Preserve stderr
and each command's result; missing symbols/line information is distinct
from missing exception/PC. If auto-focus is unavailable, use the exception's
device/SM/warp/lane coordinates to select focus before decoding.

## Native diagnostic coverage and result decisions

Before release, reuse current source/config/runtime pins and fresh all-GPU
ownership/topology/storage evidence. Verify all eight worker environments
contain the capture settings and bind each PID/start-time to rank and GPU.
Read-only `/proc/<pid>/environ` selection must print only these named variables.
Do not claim coverage from the parent environment or a lone launcher log.
Existing native logs must show actual FP8/HybridEP/Flex/AC/compiler activation;
graphs remain off. No extra completion synchronization is added.

Stop the native attempt on its first fault, completion at 200, or its reviewed
deadline, without automatic retry. Record all-rank completed optimizer and
microbatch counts, the earliest device-core exception, core completion status,
PC/kernel/module, native error sequence, source/binary hashes and exact release.
A core identifies a faulting instruction; it does not automatically identify
the first corrupting operation. Concurrent rank errors need not have a reliable
global first-error order.

| Result | Next justified action |
| --- | --- |
| Readable core attributes quantize/CudaCherry | Inspect that actual binary's SASS and selected specialization around PC; check operands, input alignment and producer lifetime before selecting a repair. Existing asynchronous quantize stack alone remains insufficient. |
| Core attributes HybridEP | Decode dispatch/combine/scan template and failing barrier/TMA instruction; compare stream/handle/flag contracts with the actual source and binary. Reuse the corrected scan proof. |
| Core attributes DeepGEMM | Identify grouped M/K specialization and descriptor stage at PC; test whether the known descriptor/scheduler race fits these bytes before proposing a port. |
| Core attributes Triton/Inductor/Flex/loss | Preserve generated Python/PTX/cubin/cache identity; map PC to the actual forward, backward or recompute kernel and inspect that producer/consumer contract. |
| Fault occurs but no readable core | Capture infrastructure remains incomplete. Diagnose enable/flags, loaded debugger-driver route, exited-warps and peer termination using smoke and progress evidence; do not call this nonreproduction. |
| Native 200 completes | Nonreproduction with capture enabled only; retain it as diagnostic evidence and keep the original asynchronous stability problem open. |
| Timeout/hang before useful fault | Reach or tool compatibility is unresolved; inspect progress and last native boundary before changing the method. No automatic repeat or larger budget follows. |

## Source fit and conditional tools

Current `hybridep.py:_launch_on_comm_stream` orders compute into communication,
records input lifetime, overlaps expert-weight quantization, then joins
communication outputs into compute. `_dispatch_bwd_impl` scatters quantized
gradients; `_combine_bwd_impl` gathers FP32 routing-probability gradients.
`cuda_gemm.py` quantizes weights on both forward/recompute and dgrad paths;
`grouped_gemm.py` selects DeepGEMM M-grouped fwd/dgrad and K-grouped wgrad.
These concurrent paths justify all-rank/all-kernel capture initially.

Reuse the [quantizer review](../../../RECURRENT-CUDA-FAULT/003/handoff.md),
[selected HybridEP review](../../../RECURRENT-CUDA-FAULT/006/handoff.md) and
[pinned upstream research](../../../BASELINE-AA/fault-research/001/handoff.md).
The scan correction is already present. The pinned standalone-permutation host
count defect does not fit fused permutation. DeepGEMM's known scheduler and
descriptor races remain conditional because installed fork/build provenance
does not establish patch equivalence. CudaCherry's exact source-to-binary
revision is unresolved. No new source evidence selects a package upgrade,
kernel patch, or weakened configuration. A core's concrete kernel/PC determines
which maintained upstream issue/fix to revisit.

If a later core selects one family, installed Compute Sanitizer supports
`--kernel-name kns=<mangled-substring>`, repeated additive filters,
`--launch-skip N`, and `--launch-count N` (counts are matching kernel launches,
not optimizer steps). Use a rank wrapper under torchrun `--no-python` to invoke
one sanitizer per selected worker with `--target-processes application-only`,
rank-specific logs and `--print-session-details`; other ranks still execute
the full required workload. Executable-name process filters cannot distinguish
eight Python ranks. Keep normal binary patching enabled for real memory checks.

Installed `--report-api-errors no` disables the NCCL-init API noise class;
`--print-limit 0` prevents a hidden report cap, with a bounded log budget.
`explicit` is a less aggressive alternative, but the prior log does not prove
all initialization reports would disappear. Native stderr must remain intact.
The [NVIDIA release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)
confirm 2026.1 introduced default extended API reports and 2026.1.1 silenced
non-actionable reports. This supports changing the reporting option without
changing installed packages.

The installed sanitizer release notes explicitly warn that hardware exceptions
in filtered/skipped kernels lose imprecise-error detail. Filters therefore
cannot replace initial fault attribution. A selected-kernel sanitizer pass does
not clear corruption by an excluded kernel. `--generate-coredump` is available
for a separately selected sanitizer attempt; native exception cores must not
be enabled concurrently. Patch-disabled sanitizer is not a supported substitute
for bounds/race checking. Full instrumentation previously failed the reach budget.

`CUDA_LAUNCH_BLOCKING=1` is process-wide and debugger `set cuda launch_blocking`
takes effect on the next run; neither is a validated dynamic native-step switch.
After PC localization, a diagnostic synchronization immediately around the
selected operation may narrow causality under source-edit authorization.
Preserve peer progress and stream ordering. Global serialization can mask a
lifetime/communication race and is not an acceptable production repair.

## Verification and runtime-cost boundary

Completed here: retained-artifact reconciliation; installed tool/header/help
inspection; current affected call-site inspection; version-matched official
documentation research; report review, local-link and whitespace checks.
No training source, shared library, runtime setting or process was changed.
The document has no production runtime cost. Expected diagnostic costs are
driver exception tracking and one small-state dump on fault; native latency,
memory impact, capture completion and PC precision remain execution gates.
No correctness, performance, convergence or checkpoint claim is made.
