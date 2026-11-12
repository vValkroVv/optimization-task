# Independent integration and runtime-cost review: descriptor candidate 001

## Decision

**Accept for the bounded candidate fixture004 GPU test. No blocking source or integration defect found.** This is a test-release decision, not production acceptance or proof of the native CTA82 failure's cause. No GPU work, source edit, installation, or process change was performed by this reviewer.

Finite acceptance: review both descriptor publication branches and every reader; reconcile the two-slot scheduler and initialization; inspect actual binary/include activation and fixture admission; assess added runtime work; identify unresolved gates. Reviewed the candidate [patch](../../dependencies/descriptor-fix-001/descriptor-protocol.patch), enclosing header and helpers, saved compiler/SASS evidence, caller review, merged upstream343 patch, and original003/candidate004 fixture and launcher bytes. Applied code-rules and perf-experiments.

## Findings and dispositions

| Severity | Finding and evidence | Disposition |
| --- | --- | --- |
| None blocking | Both GMEM publication pairs in `sm90_fp8_gemm_1d1d.cuh` now drain the issuing lane's descriptor reads before overwrite. The corrective branch additionally publishes through release before the existing acquire. | Suitable for targeted test. |
| Required runtime gate | Candidate has compiled but has never executed on a GPU; the synthetic saved samples select descriptor reuse but do not expose native descriptor bytes. | Keep candidate output and native stability acceptance open. |
| Required activation gate | An unchanged original launcher resets `PYTHONPATH`. Identical generated `kernel.cu` bytes alone cannot prove patched-header activation. | Launcher004 fixes package precedence and uses fresh local/shared caches; inspect loaded-module metadata for candidate include root/library digest and preserve actual cubin identity. |
| Scope limitation | Fixture004 retains the validated cases: four positive groups, trailing empty group, and production-scale K. It does not claim every empty-group arrangement or full training correctness. | Appropriate unchanged regression comparison; affected native path remains a separate required gate. |

### Descriptor ownership and ordering

`sm90_fp8_gemm_1d1d.cuh:164–268` elects one lane of the producer warp once around its persistent scheduling loop. That lane both submits TMA and updates GMEM maps. Unelected lanes never read these GMEM descriptors. Initial shared maps are populated by a separate elected initialization lane, followed by the existing CTA/cluster synchronization. Initial valid group uses grid-constant base maps; later valid groups select slot `current_num_valid_groups & 1`. `Scheduler::get_next_k_group` skips zero-K groups and `get_next_block` increments the valid-group ordinal, so slot parity follows valid groups rather than raw expert index. The patch changes neither scheduler state nor parity. Empty groups do not add TMA reads requiring a new participating lane.

Both corrective-current and prefetch-next branches can reuse a slot whose former descriptor is still being read. Staging-map replacement remains safe before the drain because TMA reads the GMEM map, not that shared staging copy. Each subsequent GMEM store pair is protected by commit, wait-read-zero and the upstream one-lane compiler scheduling barrier. The corrective release is necessary even when no next-map publication follows. Existing acquires remain immediately before current-map consumption.

`common/tma_utils.cuh:tma_copy` submits SM90 multicast only from cluster rank0, using that CTA's own descriptor. The peer's map is not consumed by the leader. Thus the publishing/reading lane's local completion and `.cta` proxy scope are consistent with this ownership; a peer-wide rendezvous is unnecessary. `1U << lane_idx` includes exactly the elected executing lane and imposes no obligation on unelected lanes.

The [merged upstream343 patch](../../cause/001/upstream343.patch), pinned commit `1f6f3f378920ccb5cc036ef43eb3f5972e921713`, uses the same drain and compiler barrier before GMEM overwrite. Adapting it to both fork-specific publication branches is justified. The installed CuTe helpers lower to commit-group and wait-group.read0; NVIDIA explicitly allows bulk-group completion of tensor-map reads for mbarrier tensor loads and defines the read wait accordingly. The proxy publication is a separate dependency. See [PTX tensor completion](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor) and [wait-group](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group).

Independently inspected [SASS](../../dependencies/descriptor-fix-001/compile-kernel.sass): corrective DEPBAR at `0x3020` reaches one-lane WARPSYNC at `0x5cd0` before first GMEM store `0x3120`; next-map DEPBAR `0x3550` reaches WARPSYNC `0x5d30` before first store `0x3650`. Final descriptor stores `0x32d0`/`0x3800` precede release lowering. `MEMBAR.ALL.GPU` in that lowering is a fence, not a global thread rendezvous. This supports this exact compiled specialization, without promising other compiler versions' SASS.

### Integration, fixture and causal boundary

The preserved real extension initializes its compiler root from the imported package directory; [import-check.json](../../dependencies/descriptor-fix-001/import-check.json) records copied package/extension, actual binary include root, digest `a122322508f98be4c07c04c7237e53ab`, mutable-output custom-op schema, and CUDA uninitialized. The unchanged extension hash and 844-file manifests support isolation; only the named header differs. The build string remains the installed version, so the patch/header hash must accompany it.

Original003 metadata has library digest `0015c5c30202d513849b339660066739`; candidate runtime metadata must select its new digest/include root. [Launcher004](../../reproducer/004/launch.py) prepends the copy before GIFT and points both cache variables to004-owned locations. Its changes are limited to activation and bytecode suppression. `cmp` confirms [fixture004](../../reproducer/004/fixture.py) is byte-identical to003. Existing loaded-CUBIN records preserve path, hash and metadata; root must inspect those records before interpreting a clean result, because the fixture's exact-source assertion intentionally compares generated template text, which is identical across header variants.

Fixture packing uses exact representable FP8 integers, unit fp32 scales, zeroed BF16 accumulation outputs, and an independent residue-count oracle. The CPU packing check already validates flattened group boundaries. Thirty-two retained output buffers are zeroed and separately checked after each submission chunk; the first singleton call admits the actual module. Subsequent submissions and checks are unchanged. These controls preserve the cold003 failure mechanism without adding per-call synchronization.

The saved-coordinate derivation agrees with the actual scheduler: width16, effective12, tile75/group1, global159, CTA43; route0→1→3 overwrites slot1 in the corrective branch. The16 samples uniquely fit A3×B1 among this fixture's16 operand pairs. That is specific evidence for premature descriptor reuse, not direct observation of the corrupt native CTA82 map. A clean candidate fixture strengthens this mechanism; native fault causality still needs native evidence.

The [caller review](../../caller-review/001/REPORT.md) establishes current-stream, per-call scratch lifetime and the real compiled custom-op path. This patch changes neither wrapper/schema, scratch ownership, requantization, data/scale layout, multicast schedule nor GEMM arithmetic. FP32 intermediates and accumulation, FP8/BF16 contracts, FSDP2/HybridEP EP8, selective AC, model/loss compilation, optimizer/evaluation/checkpoint behavior remain on the existing paths. Both w2 and w13 grouped wgrad inherit this header and need native affected-path evidence.

## Runtime-cost review

Added device work occurs only at descriptor publication: commit/read-wait plus one-lane WARPSYNC at both sites, and one corrective release. It is not inserted per K block. At most two publication pairs execute per entered valid group; if both execute, there are no intervening new TMA loads before the second drain. The exact four-group fixture has336 logical tiles/116 persistent CTAs and at most3 entered groups per CTA. The waits can reduce producer overlap and delay dependent math; their magnitude is unknown.

No allocation, descriptor count, tensor precision, output pass, host scalar read, transfer, launch, graph break, host/stream synchronization, or collective is added. Existing512 bytes of descriptor scratch per CTA and224472-byte dynamic shared memory remain. Saved ptxas reports168 registers/16 barriers, with reduced static spill counts; these are resource observations, not speed or peak-memory measurements. A new header digest causes a new JIT cache entry and cold compilation. Communication volume/order is unchanged, although altered producer latency can affect integrated overlap.

Device time, host time, waits/idle, transfer/communication time, integrated step latency and peak runtime memory are unmeasured for this candidate. No performance claim follows from static review. Representative numerical/runtime/memory evidence remains required for production acceptance.

## Remaining acceptance

1. Execute the bounded cold-cache004 comparison and verify actual candidate library/include/cache/cubin activation and every fixed-case output. A failure preserves the artifact and redirects diagnosis; a pass supports moving to affected-path/native validation.
2. Verify both grouped-wgrad shapes through the real compiled/autograd/SAC path under the frozen exact preset.
3. Obtain two fresh all-eight-rank native200 successes, with resolved preset and feature activation retained. Native causal attribution and relevant numerical/runtime/memory acceptance remain explicit until supported.

Verification here: read-only protocol/caller/SASS inspection; `cmp` fixture003/004 passed; launcher diff inspected; report local links and whitespace checked. Existing CPU compile/import/oracle evidence was reused, not rerun. No requested GPU gate is represented as passed.
