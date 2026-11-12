# SANITIZER-ORIGIN attempt 001: bounded outcome

## Result

The 20-minute attempt expired before a native CUDA fault reproduced. It completed
16 optimizer updates on all eight ranks, then stopped during update 17. The
runner returned **124** at 2026-11-12 06:39:17.919938 UTC. No originating kernel,
device PC, or CUDA 715 was established. This is an inconclusive attribution
attempt; it does not clear the recurrent fault or establish memory correctness.
No source repair, automatic retry, or instrumentation fallback is selected.

Finite acceptance: inspect preserved configuration and activation evidence;
extract all eight native event sequences and first fatal error; characterize
tool reports and coverage limits; verify same-run binary receipt status; report
cause versus unknown. These analysis items are complete. Origin attribution
remains unresolved because this attempt did not provide it.

## Preserved workload and activation

The [resolved configuration](../../attempts/001/native/config/resolved_config.yaml)
retains the exact 3B model (one dense and 17 MoE blocks), FlexAttention,
HybridEP comm stream/fused permutation, FP8 recipe, FSDP2, EP8, selective AC,
and model/loss compilation. It retains batch 8/256, sequence 4096, seed 42,
accumulation 4, schedule 30000, c4_test, AdamW, and checkpoint disabled.

Rank-zero [supervisor output](../../attempts/001/supervisor.log) explicitly records:

- mesh pp1/dp_replicate1/dp_shard8/cp1/tp1/ep8;
- 72 dense and 34 grouped FP8 projections;
- loss compilation, selective AC, and compilation of all 18 layers;
- local/global batch 8/256, accumulation 4, and sequence 4096.

All eight compiler receipts record torch `2.15.0.dev20260822+cu130`, graphs
false, mix-order reduction false, and graph partition true. All 18 blocks remain
listed as graph excluded in the resolved configuration; exclusion does not
remove their compilation. The public step-boundary callback remains configured.
Profiling, memory snapshots, validation, and completion/timing observers are off.
No new numerical tolerance or loss/norm bitwise A/A gate was introduced.

Frozen source: SOURCE-SNAPSHOT/007, 92 paths, base
`446431dfa191ef0300f116257ae918bc85e38c06`, manifest SHA-256
`2585e504350276cb4689ae9ea45a33706e9f898bb61969abe5d08f2650c55620`, archive
`0fc5ddad83c5f99b4036a47dd7f485ba2f42d1284c42f086d5f051cebad66082`.
Preset SHA-256 is
`b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`.
These reuse the accepted frozen launch pins; this analysis did not re-audit source.

## Native event integrity and termination

[Reproducible extraction](extract.py) parsed all eight rank logs and all 16
rank stdout/stderr files. [Extraction JSON](extraction.json) records input
SHA-256 hashes, complete event counts, identities, terminal events and errors.

Every rank has 1,256 valid JSON records, contiguous seq_id 0–1255, one stable
rank/PID identity, and optimizer starts/ends for steps 1–16. Each has 68
fwd_bwd starts and 67 ends: four completed microbatches in each of steps 1–16,
three completed in step 17, and its fourth microbatch entered without a recorded
end. Last record is step17 `post_dataloading_process_end` on every rank. Thus
all eight completed the same recorded prefix; optimizer update 17 did not finish.
These are native log boundaries, not new per-kernel completion probes.

There are no ERROR/CRITICAL structured records and no native rank-log traceback,
CUDA error, RuntimeError, illegal-instruction/access, launch-failure, or 715
matches. The first supervisor traceback is torchrun PID620952 receiving signal15
at the deadline, followed by a second SignalException while shutting down.
It is a timeout consequence, not an originating native CUDA failure. Existing
quack disk-cache loader warnings fell back to in-process compilation and did
not prevent the 16 updates; no unsupported-stack conclusion follows from them.

The watcher [failure receipt](../../attempts/001/watcher-failure.json) honestly
records FAILED_CLOSED and cleanup `verified_exit=false`: rank0 no longer matched
the pinned entrypoint during exit. Do not relabel that cleanup as passing.
Root separately verified release afterward in
[gpu-release-check-complete.json](../../attempts/001/gpu-release-check-complete.json):
all 638 recorded PID/start-tick identities absent/no-live; same eight GPU UUIDs
idle with zero used memory and no applications. Root sent no signals. This
independent later evidence resolves release; it does not change the watcher's
own result. This analyst made no process or GPU query.

## Sanitizer reports and coverage

The sole tool log is [sanitizer.620943.log](../../attempts/001/sanitizer.620943.log).
It contains exactly **50 CUDA API reports**:

| Report | Count |
| --- | ---: |
| Error handling fatbinary | 18 |
| No available relocatable PTX entries for GPU | 17 |
| No device code available for GPU ISA 90 | 15 |

Every report block contains `ncclInitKernelsForDevice`; the first host stack
starts at `cudaFuncGetAttributes`, then NCCL initialization and the trainer's
initialization barrier. Native work continued beyond these reports. They
establish intercepted CUDA API initialization errors; they do not identify the
recurrent training fault, its cause, or a failing quantize/combine/GEMM kernel.
No device frame, hardware-exception attribution, or kernel-origin PC is present.

The count equals the configured `--print-limit 50`. No suppression/limit message
or ERROR SUMMARY footer is present. The log ends after an initialization host
backtrace. Accordingly report capacity is exhausted; later report visibility
cannot be relied upon. There is no evidence establishing whether any later
report was actually suppressed. The timeout prevents a successful final tool
summary, and exit124 is not sanitizer error-exit86 or a clean tool exit.

`--target-processes all` requests the full torchrun tree, but the single filename
uses launcher PID620943; neither it nor the eight native logs proves per-rank
sanitizer attachment. The reports have no rank/PID binding. CUDA API interception
of native NCCL initialization occurred, but actual eight-rank exception-reporting
coverage remains unverified. The binary receipt below proves mapped files, not
sanitizer attachment. No process-introspection workaround was added.

Installed Compute Sanitizer is 2026.1.0.0 build37182542, executable SHA-256
`5cba659e1cd603aa2370e103425f184368e6ed405da90ce85743c4399c324dbc`.
The launch uses memcheck, default nonblocking behavior, target-processes all,
error86, backtrace yes, launch-timeout180, print-limit50, and the documented
`NV_COMPUTE_SANITIZER_BINARY_PATCHING=0` debug setting. Reused version-matched
manual interpretation is in the [prior bounded review](../../../RECURRENT-CUDA-FAULT/004/handoff.md).
Patching disabled does not establish precise bounds/alignment checking, earlier
silent-write detection, allocator lifetime correctness, race coverage, complete
peer-memory coverage, or guaranteed hardware-exception reporting. The new run
does not remove those limits. API reports and report saturation further limit
its actual attribution evidence.

## Same-run binary capture

The helper ran once after validated all-rank optim_end1 and returned zero.
[Receipt](../../attempts/001/binary-provenance-receipt.json) status is complete;
capture ran 06:22:35.832–06:22:35.949 UTC. All eight rank/PID/start-tick bindings
were stable across maps and hashing. All ranks mapped the same four hashed
CudaCherry family files; device/inode-bound file checks were stable.

| Shared object basename | SHA-256 |
| --- | --- |
| `_C_quantize.cpython-312-x86_64-linux-gnu.so` | `e134ff643cc1e40cdd5443e4cf744b7221fe719ef46b28d69f1866185d44bc87` |
| `_C_requantize.cpython-312-x86_64-linux-gnu.so` | `1515ae9f465a12f422865ce880bc2e713e97af7f477dd8f2488d0f0394a0d435` |
| `_C_swiglu.cpython-312-x86_64-linux-gnu.so` | `af9f1c2ed633e59ee7657920de8d0c59869245029daf0bc52da9d79f04d1ca98` |
| `_C_gemm.cpython-312-x86_64-linux-gnu.so` | `9548e0fdb0be6033a4bbd0e159ee9b04e20fcc12874b0d968e497ea18c53aaa5` |

This closes same-run mapped-file provenance for these conservative families.
It does not attest registered operator body, executed branch, build source or
causality. No live sys.modules snapshot exists, opaque JIT names remain outside
family matching, and adjacent loader/wrapper hashes are static provenance.
Source-to-binary/build attestation therefore remains open.

## Interpretation and next decision

The time budget covered only 16 complete updates, below the earlier native
fault updates22/33/48 and the completion-probe failure in update21. Those failed
runs remain valid unresolved evidence; this shorter nonreproduction does not
contradict them. Fault origin, silent corruption, peer/stream involvement and
source-build identity remain unknown. There is no numerical-quality, memory-
correctness, performance, checkpoint-continuation or production-readiness claim.
Sanitizer/watcher/helper overhead and changed scheduling are not separated.

Stop this finite attempt and retain the evidence. Any future investigation
requires a separately selected and authorized decision that addresses the actual
coverage/report-capacity limitations. This result supports no specific source
fix or automatic fallback. Numerical, quality, performance and checkpoint gates
from the campaign remain open; independent bitwise loss/norm A/A is not made a
prerequisite.

## Analysis checks

The extraction's assertions passed: eight ranks, contiguous identities/sequences,
optim_end1–16 on every rank, 68/67 fwd_bwd starts/ends, and all eight graph-off
compiler receipts. Forty-six input files were hashed. No source, runner or
attempt artifact was edited; only this analysis directory was written. Document
relative links and trailing whitespace were checked. This documentation and CPU
file extraction add no production-runtime cost.
