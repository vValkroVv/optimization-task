# Lane A004 preparation and readiness

## Prepared scope

This attempt is the single source003, graph-on allocation-history diagnostic
specified by the pool-design handoff. It uses the pinned eight-rank source003
entrypoint and preset, with an attempt-local Python bootstrap that sets only
torch._inductor.config.triton.cudagraph_trees_history_recording=True before
importing that unchanged entrypoint. The existing CUDA Graph Trees allocator
guard remains enabled.

The diagnostic is not prepared for timing or numerical claims. The standard
failure path should include the bad pointer and its allocator-history frames.
The built-in path is not expected to save a standalone snapshot or block size.
Ownership is established only when retained frames identify the producer for
the failing address. One block or an expected allocation size is corroboration.
If the frames are missing or the failure changes, stop and report that evidence
gap without retrying.

## Gates before launch

The source/archive/preset hashes and shell syntax are checked locally without
importing Torch, initializing CUDA, querying GPUs, or starting training. GPU
state remains unchecked here. Before launch, the coordinator must review this
attempt and explicitly confirm the launch. Then refresh all eight same-UUID
H100s, verify they are idle with no compute applications, and verify every
previous attempt PID is absent. Record those observations in preflight.json
and set it to READY_TO_LAUNCH; record the scoped final confirmation in
release.json. The launch script fails closed until both files pass validation.

After one launch, use one monitor with a 15-minute deadline and roughly
two-minute polling. Do not relaunch or change options. Preserve the complete
supervisor output, all rank error files, cudagraph logs, runtime manifests, and
the memory sampler output. After unwind, verify exact task processes are gone
and all eight GPUs are idle before inspecting the full ownership trace.

## Current status

Prepared only. No GPU query, CUDA initialization, remote operation, source edit,
package operation, or launch has occurred for attempt 004. Awaiting coordinator
review, explicit launch confirmation, and the fresh preflight.
