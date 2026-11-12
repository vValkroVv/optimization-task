# Host-RAM feasibility for the encoded numerical reference

Design only, 2026-11-12. Uses code-rules and perf-experiments. No source edits,
GPU queries/runs, allocation tests, installations, checkpoint creation, servers,
cleanup or publication. This document adds no production runtime work.

## Decision and finite acceptance

**Host RAM has ample observed capacity for the existing encoded gradient archive.
A CPU-only parent with ordinary byte buffers and bounded AF_UNIX streams is a
plausible standard-library transport. The requested gradient-in-RAM/native-S0-on-disk
route is currently blocked by disk space and is not ready to launch.**

At **04:19:10 UTC**, `df -B1` reported **12,928,155,648 B (12.04 GiB)** free
on the task NFS mount. That is below native S0's estimated **36.5045 GiB** even
before metadata or the existing 32-GiB free reserve. This fluctuating observation
supersedes the earlier 130.86-GiB and design 003's 211.66-GiB observations; it
is neither a reservation nor a permanent capacity claim. No checkpoint exists.
Do not move native checkpoints into a new RAM filesystem or invent a storage
framework to bypass this blocker.

Acceptance for this bounded feasibility deliverable:

1. Read actual cgroup limits/current use and host available RAM; distinguish
   capacity, authorization and reservation.
2. Assess an available established mechanism and bound all additional memory.
3. Preserve the coherent lossless reference, consumer/liveness contracts and
   native checkpoint semantics; name the actual missing gates.
4. Check this directory's links and whitespace, then stop.

All four deliverable items are complete; recorder implementation and runtime
acceptance remain open.

## Evidence and resource budget

[Timestamped observation](memory-observation.json), **04:20:03 UTC**, records
read-only procfs/cgroup reads using the authorized Python, without importing
PyTorch. The [campaign execution route](../../../../report.md#authorized-local-preflight)
authorizes the current host and `/opt/torchtitan-venv/bin/python`, Python 3.12.3.
No new allocation or remote connection was made.

| Resource | Observed bytes | GiB |
| --- | ---: | ---: |
| Host MemTotal | 1,621,906,083,840 | 1510.52 |
| Host MemAvailable | 1,574,715,109,376 | 1466.57 |
| cgroup memory.max | 1,563,368,095,744 | 1456.00 |
| cgroup memory.current | 173,188,022,272 | 161.29 |
| cgroup remaining | 1,390,180,073,472 | 1294.71 |

`memory.high=max`, `memory.swap.max=0`, no swap, and all reported memory/OOM
limit-event counters are zero. `/proc/self/cgroup` reports `0::/`; these figures
describe the current execution cgroup, not a task-exclusive reservation.
`vm.overcommit_memory=1` permits overcommit. Neither a successful allocation nor
current MemAvailable guarantees that future shared-host demand leaves pages
available. No cgroup policy was changed or memory allocated to test the estimate.

Use design [003's source-derived inventory](../003/handoff.md#source-and-runtime-inventory),
not an assumed compression ratio:

| Concurrent additional validation RAM | Source-derived estimate |
| --- | ---: |
| Parent's one immutable ten-update preclip archive | 121.6815 GiB |
| Active consumer's decoded reference P/m/v CPU cache | 36.5045 GiB |
| Consumer actual pre-update endpoints for delta comparisons | 12.1682 GiB |
| Subtotal | 170.3542 GiB |
| Proposed total validation RAM cap | 192 GiB |
| Remaining allowance for metadata, inputs, persistent state, staging, socket buffers and comparison scratch | 21.6458 GiB |

This cap is a proposed admission contract, not an allocation or proven peak.
It includes the parent and all eight ranks collectively; never multiply the
entire archive by eight. The archive's nominal share is 15.2102 GiB/rank across
ten updates. The existing largest-local-tensor estimate is about 75 MiB, but
runtime layout/storage inventory must establish the real bound. Count views,
backing stores, transport buffers, FP64 delta/error scratch and checkpoint staging;
count a real copy once and do not count an alias as an extra allocation.

Before a future launch, establish the native workload's maximum additional host
RAM demand `H` above the contemporaneous cgroup baseline, including loader,
compiler, native checkpoint staging and decoder lifetimes. Admit only if
`min(host MemAvailable, memory.max - memory.current) >= 192 GiB + H + 64 GiB`
with byte-exact accounting and an explicit owner budget. The 64 GiB is proposed
free safety headroom, not a guarantee against arbitrary other-user growth.
Recheck admission before each process launch and large allocation; continuously
bound task-owned bytes and stop producing payload on budget loss, MemoryError,
rank failure or missing acknowledgement. Serialize checkpoint staging against
other optional peaks. Unknown `H` and shared-host reservation mean there is
**no verified no-overcommit guarantee today**. Available capacity supports the
design estimate; it does not close this gate. No page-locking or pinned archive.

## Smallest storage/streaming contract

The problem is retaining one reference after native A1 exits so fresh A2 and B
can consume it. Native DCP remains the checkpoint implementation. Its installed
`torch/distributed/checkpoint/filesystem.py` exposes FileSystemWriter and writes
files; its tensor staging and serialization do not provide an existing bounded
cross-process RAM archive. `torch.save` to BytesIO is supported but wrapping the
entire archive risks full serialized copies and retained backing storages.
[PyTorch serialization](https://docs.pytorch.org/docs/2.15/notes/serialization.html)
preserves storage sharing; it does not itself supply this lifetime policy.

Select, conditionally, one CPU-only owner parent that starts before A1 and lives
through the exact decode validation, A2, B and any explicitly named consumer.
It imports no CUDA state and owns only one compact bytearray per actual tensor
record, indexed by run/update/rank/canonical tensor identity and its complete
metadata. Allocate each record to its exact inventoried size once; fill it with
`recv_into(memoryview(...))`, looping over partial reads. `sendall` accepts buffer
views; neither side needs `bytes(full_archive)` or pickle queues. The Python 3.12
[socket APIs](https://docs.python.org/3.12/library/socket.html#socket.socket.recv_into)
and [memoryview](https://docs.python.org/3.12/library/stdtypes.html#memoryview)
provide these established primitives. Socket transfer still copies through the
kernel; no zero-copy or throughput claim is made.

One active logical tensor per rank is staged canonically at a time; its native
P/m/v/gradient and FP64 comparison buffers are counted in the scratch budget.
The existing decoder state cache is retained between updates, without making
whole-family temporary replicas. An exactly sized CPU buffer can be exposed as
a tensor using installed/documented
[torch.frombuffer](https://docs.pytorch.org/docs/2.15/generated/torch.frombuffer.html),
which shares its buffer rather than copying it. Keep the owner alive while any
view exists; never let native decoding mutate archived gradient bytes. Recreate
required native layout on bounded scratch and retain 003's exact hash gate.
The CPU-only owner needs no PyTorch import. The native consumer's own PyTorch
process performs decoding beside its single live training model.

This avoids storing tensor payloads in `/dev/shm` (observed capacity 256 GiB)
and avoids Python multiprocessing SharedMemory/resource-tracker ownership.
The installed Python 3.12 `multiprocessing/shared_memory.py` has signature
`SharedMemory(name=None, create=False, size=0)` and registers attached segments
with the tracker; it has no `track=False`. The documented optional tracker
control appears in Python 3.13, so do not invent it for this interpreter.
See [Python 3.12 SharedMemory](https://docs.python.org/3.12/library/multiprocessing.shared_memory.html).
No multiprocessing manager, tensor queue, shared-memory patch or dependency is
needed for the proposed byte stream.

Use one bounded request at a time per rank, explicit length framing, versioned
identity metadata, checksum and acknowledgement. Enforce inventoried record
sizes and the global cap before accepting a body. Unexpected EOF, short body,
wrong identity/hash, duplicate key, missing rank/update or timeout fails the
reference. Seal the archive only after all ten complete updates and every rank's
completion. Serve read-only copies by tensor; verify hashes again at consumption.
Do not fork a CUDA process or inherit/copy the archive into training ranks.

Before implementation, verify a task-owned local AF_UNIX endpoint route reachable
by the eight actual workers; do not assume NFS socket-file support. Use bounded
socket buffers, a session capability and task-owned endpoint permissions. Shared
`jovyan` identity alone is not isolation. No socket or server was started here.
This route and its tested lifecycle are missing integration evidence.

## Unchanged numerical and lifecycle requirements

Preserve the exact 3B MoE/FlexAttention/SAC/FP8/HybridEP EP8/dense FSDP8 compiled
model/loss workload, local/global batch 8/256, sequence 4096, accumulation four.
Keep [003's encoding and native decoder](../003/handoff.md#recorder-and-reader-contract):
nonzero native S0 P/m/v/steps plus all actual preclip gradient elements for ten
uninterrupted updates from the same run. Keep None/layout/alias/shard/step/group
metadata, actual clip scalars, input elements and all required floating persistent
state, including FP8 plain attributes discovered by the runtime inventory.

Before A/A or A/B comparison, same-run native CUDA clipping and fused AdamW must
reproduce **actual postclip/P/m/v/steps hashes at every boundary**. A RAM transport
cannot waive this gate. No second training model, reduced precision, tensor
stripes, repeated-prefix substitution or scalar replacement. Independent runs
use [002's production A/A variability and precision criteria](../002/acceptance-amendment.md),
frozen before A/B interpretation; they have no new bitwise floating gate.

Keep owner code, source/config/runtime pins, compact progress/error logs,
per-record hashes/completion manifests, frozen criteria and consumer results on
persistent task storage. Persist the consumer roster and deletion eligibility.
No reference tensor bytes need be written to disk. Owner liveness belongs to
a reviewed task runner with precise PID/command identity and bounded deadlines;
training consumers must fail on lost owner/stream rather than silently skip
comparison. Closing a consumer does not release the parent's reference.

After every named consumer finishes, all result/decode manifests are persisted
and unresolved mismatches have their required compact slices, release only the
task-owned byte buffers and endpoint, recording bytes/hashes/consumer completion.
Do not clean unrelated memory segments or kill shared processes. Consumer failure
retains the sealed archive while its explicitly bounded recovery decision remains
open; no indefinite idle server. Parent crash, OOM, container/server restart or
host failure **loses the volatile archive**. Preserve S0 and logs; rerun A1 under
new admission rather than claiming recovery from hashes or partial RAM. There is
no durable archive guarantee and no implied permission to start such a parent.

Native bidirectional 5+5 continuation remains separate, with exact checkpoint
load and frozen floating trajectory criteria. RAM gradients do not solve the
space for native S0/Q5 or native checkpoint staging. Do not claim continuation
feasible from this capacity observation.

## Open gates and stop condition

1. Native checkpoint disk capacity: current 12.04 GiB is insufficient even for
   S0; required reserve and actual serialized state must fit before the probe.
2. Explicit RAM budget/owner lifetime, native host peak `H`, actual tensor bytes,
   dtypes/aliases/layouts and FP8/plain-attribute state inventory.
3. Authorized recorder/transport implementation with corruption, partial-read,
   cap refusal, owner/consumer failure, missing-rank and cleanup tests; verified
   endpoint path and persistent logs/manifests.
4. One production-shaped native inventory/decoder probe, actual host/device peaks,
   observer inertness/replay, exact same-run ten-update decoding and unchanged
   A/A calibration/A/B/continuation acceptance.

No further allocation, implementation or experiment is needed to answer this
feasibility question. Capacity is plausible; present launch feasibility is
blocked, and required runtime verification is unperformed.
