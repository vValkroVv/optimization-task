# DeepGEMM descriptor protocol candidate 001

## Result and acceptance boundary

Prepared an isolated **candidate**, with one header changed in the exact installed `2.3.0+f093a4c` dependency. No GIFT, shared package, original header, original extension, preset, launcher, or running process was changed. No GPU operation, training, benchmark, installation, or upgrade was performed here. Root owns independent review and release to GPU execution.

Finite acceptance for this deliverable: establish the protocol and the observed mismatch relation; preserve the installed dependency; make the smallest matching header patch in a task copy; prove actual import and binary include-root activation without CUDA; compile the exact specialization; inspect instruction ordering and costs; record remaining GPU and native gates. These source/preparation checks pass. This is not production acceptance or proof that the native TMA-B exception has the same cause.

## Evidence selecting this candidate

Read the existing [cause assessment](../../cause/001/REPORT.md), [caller review](../../caller-review/001/REPORT.md), [native fault analysis](../../analysis/002/REPORT.md), and [warm-cache nonreproduction](../../reproducer/002/REPORT.md). The original standalone fixture failed call2 after33 submissions in cold-cache001, passed2112 calls in warm-cache002, then reproduced call2 in cold-cache003 using the same original cubin. The two positive observations remain distinct from the native CTA82 TMA-B exception.

Cold003 [raw events](../../reproducer/003/events.jsonl) retain8448 wrong elements in group1,16 coordinates, finite outputs and maximum absolute error11. The fixture's saved `tile_cta=111` was an unvalidated row-major estimate and is not used.

[check_mismatch.py](check_mismatch.py), with [results](mismatch-check.json), evaluates the inspected `Scheduler::get_num_1d_blocks_per_group`, `get_swizzled_block_idx`, and `get_next_block` formulas. Width8 costs3304; width16 costs3264, so width16 is selected. With12 M tiles, the effective width is12. Every saved coordinate is in `(m_tile3,n_tile6)`, local tile75, group1 global tile159, persistent CTA43, iteration1. CTA43 schedules groups0→1→3 at global tiles43,159,275. Its transition to group3 corrects the current map into slot1, which previously served group1. This is the corrective publication branch, not the next-map prefetch branch. The earlier informal M8/CTA19 inference was incorrect and is superseded by this executable derivation.

All16 saved actual values match **A from group3 multiplied by B from group1**, uniquely among the16 group-pair possibilities; the independent integer oracle still matches expected A1×B1. This fits an old group1 A descriptor being overwritten with group3 before the old TMA read completes. A is unicast in this specialization. B is multicast from the even cluster leader, so A/B may consume different descriptor instances. Missing release of the newly published group3 map explains no direct ordering that would protect the old group1 read. The observation therefore selects descriptor-read completion before overwrite as a concrete candidate mechanism. It is a partial-output causal fit, not a saved descriptor or first-corruption proof; native CTA82 causality remains open.

## Protocol and chosen patch

The pinned [merged upstream343 patch](../../cause/001/upstream343.patch), commit `1f6f3f378920ccb5cc036ef43eb3f5972e921713`, supplies the established descriptor completion sequence and one-lane compiler scheduling barrier. This fork has two descriptor slots and two publication branches, so the sequence is applied before **both** A/B GMEM store pairs. Shared staging descriptors are not consumed by TMA; preserving their replacement sequence before the drain is appropriate.

NVIDIA's [PTX tensor-copy completion table](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor) allows bulk async-group completion of descriptor reads even when tensor payload completion uses mbarrier. Its [wait-group contract](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group) defines `.read` completion, and its [proxy fence example](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar) pairs generic publication release with tensor-map acquire. The installed CuTe helpers implement `cp.async.bulk.commit_group` and `cp.async.bulk.wait_group.read 0`. The corrective branch lacked release when no next-map branch ran. Acquire alone does not supply that publication dependency.

[descriptor-protocol.patch](descriptor-protocol.patch) changes only [sm90_fp8_gemm_1d1d.cuh](deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh):

- Before each publication pair: `cute::tma_desc_commit_group()`, `cute::tma_desc_wait_group()`, and `__syncwarp(1U << lane_idx)`.
- After corrective A/B publication: `tensor_map_release_cta()`, matching the already present next-map publication release and later acquire.

The drain protects GMEM reuse; release/acquire publishes the new map. Both dependencies are required for a complete protocol even though the captured old-group corruption selects the former. No open375 hypothesis about `tensormap.replace` joining an async group is needed. The two-slot layout, scheduler, multicast, descriptors, data/scales, accumulation arithmetic, dtype, precision, and tensor shapes remain unchanged. No new control or API is introduced.

## Exact copy and activation

[original-manifest.json](original-manifest.json) records844 original files before copying; [candidate-manifest.json](candidate-manifest.json) records the copy. Only `include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh` differs. Header SHA256 changes from `fd8a620440cc31ae7d01010fde1f914aae6998f2dcf6ede2a65e3b5e76001726` to `ed8bfb6a182d6a8a225d36b359287f17dba73e95d5cc50150ca177ea979557d2`. The original and copied extension are separate files with unchanged SHA256 `99b7afad2e2fe7ebabd26c22afc99dc652e2e6935fcbf37324359eb5aeeefe26`; no symlink to a writable original is used. The build label stays `2.3.0+f093a4c`, with the local patch identified by the manifest rather than a fabricated upstream commit.

The installed `deep_gemm/__init__.py` calls its real `_C.init(package_directory, cuda_home)`. The pinned binary exports `Compiler::library_root_path`, `library_include_path`, and `library_version`. [check_import.py](check_import.py) imports the actual copied binary, then reads these diagnostic globals using its pinned libstdc++ string layout. This is inspection, not a production wrapper or API. [import-check.json](import-check.json) proves:

- Actual package and extension resolve inside this copy.
- The binary's include root is this copy's `deep_gemm/include`; its library digest is `a122322508f98be4c07c04c7237e53ab`.
- The actual GIFT grouped custom op imports successfully with its mutable-output schema preserved.
- All844 original files still match their before hashes, all844 copied files match their manifest, and `torch.cuda.is_initialized()` is false.

For an independently reviewed fixture/native launcher, prepend this directory before GIFT **before importing Torch/GIFT/DeepGEMM**. Existing fixture launchers reset `PYTHONPATH`; calling them unchanged would select the original, so root must put these values into the task-owned launcher environment. Use fresh task-owned candidate cache paths; never reuse the original JIT cache as activation evidence.

```bash
TASK_ROOT=/workspace-SR008.nfs2/vkropotin/optimization-task
CANDIDATE="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/dependencies/descriptor-fix-001"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CANDIDATE:$TASK_ROOT/gift${PYTHONPATH:+:$PYTHONPATH}"
export DG_JIT_CACHE_DIR="$ATTEMPT_DIR/deepgemm-cache"
export DG_JIT_SHARED_CACHE_DIR="$ATTEMPT_DIR/deepgemm-shared-cache"
# Preserve the reviewed graph-off/native environment and exact preset.
```

`ATTEMPT_DIR` must be the fresh reviewed attempt directory chosen by root. These exports are a handoff, not an executed GPU launch. The already executed CPU check was:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH="$CANDIDATE:$TASK_ROOT/gift" \
DG_JIT_CACHE_DIR="$CANDIDATE/cpu-cache" \
DG_JIT_SHARED_CACHE_DIR="$CANDIDATE/cpu-shared-cache" \
/opt/torchtitan-venv/bin/python3.12 "$CANDIDATE/check_import.py"
```

## Compilation and runtime-cost review

The exact original fixture [generated specialization](compile-kernel.cu) was compiled on CPU with NVCC13.2 for `sm_90a`, preserving the recorded original flags except the candidate include directory and output path. [compile-command.json](compile-command.json) is the exact command; [compile.log](compile.log) records exit0. No cubin was loaded on a GPU. Existing unroll warnings remain. The compiled [SASS](compile-kernel.sass) was reviewed for both publication paths:

- Corrective path: flush at `0x2ec0`, `DEPBAR.LE SB0,0` at `0x3020`, branch through one-lane `WARPSYNC` at `0x5cd0`, return to `0x3050`, first GMEM descriptor store at `0x3120`. Release lowering follows the final store at `0x32d0`.
- Prefetch path: flush at `0x3390`, wait at `0x3550`, one-lane `WARPSYNC` at `0x5d30`, return to `0x3580`, first descriptor store at `0x3650`. Existing release follows final store at `0x3800`.

The wait is on **this issuing thread's prior tensor-map reads**, not tensor payload completion or every GPU thread. `__syncwarp` names only the elected producer lane. No CTA-wide, device-wide, rank-wide, host, stream, or distributed barrier is added. The release has PTX `.cta` scope; ptxas lowers the proxy operation to instructions including `MEMBAR.ALL.GPU`, which is a fence, not a barrier requiring all GPU threads to rendezvous.

Before coding, expected costs were extra producer-side descriptor-read waits and publication fences at group transitions, with no new data computation/storage/launch/communication. The final code retains that scope:

| Cost | Final effect and open evidence |
| --- | --- |
| Device computation | No GEMM arithmetic or output operation changes. Two branch-local commit/wait/one-lane sync sites; one additional release site. |
| Frequency/scale | Once per executed publication pair, not per K block. At most one corrective pair and one next-group pair per entered group; a second drain in the same transition has no intervening TMA loads. With4 groups and116 CTAs, a CTA enters at most3 groups here. Wait magnitude is unmeasured. |
| Memory | No allocation/layout/descriptor-count change: four128-byte maps per CTA remain512 bytes,59392 bytes for116 CTAs. Shared memory remains224472 bytes. Ptxas keeps168 registers/16 barriers; stack8/spill stores16/spill loads16 bytes versus original24/40/36. These static compiler reports are not peak-memory or speed measurements. |
| Host/device | Same Python/custom-op/binding/launch/current-stream path and scratch lifetime. No scalar read, host sync, transfer, added launch or graph break. |
| Communication | No NCCL/HybridEP call, group, ordering, volume or buffer changes. Existing multicast remains. Reduced TMA descriptor overlap may change integrated step time; GPU measurement remains required. |
| Compilation/startup | New header digest requires candidate JIT compilation/cache entry; the original extension is reused. Existing canonical import initialization and include hashing run normally. |
| Training lifecycle | Only the selected SM90 FP8 KGroupedContiguous descriptor branch changes. Forward/dgrad, grouped wgrad arithmetic, SAC, compiled model/loss, clipping/optimizer, evaluation and persisted state formats remain on their inspected existing paths. The kernel's numerical result still needs affected-path verification; no loss/gradient/native-pass equivalence is claimed. |

The cold003 evidence concerns the exact w2 wgrad specialization; the inherited generic header also serves w13 grouped wgrad. The two publication branches are both repaired to maintain the shared protocol. Production preset stays3B MoE/HybridEP EP8/FSDP2, FP8, Flex, SAC, model+loss compile, chunked CE, batch8/global256/S4096/accumulation4, data, optimizer and schedule. No throughput or numerical-quality claim is made.

## Checks and remaining gates

Executed: source/protocol/caller review; integer saved-coordinate/oracle check; before/after original and copied package hashes; actual copied import/custom-op/binary-root check without CUDA; exact-template compilation; generated SASS ordering/resource review; Ruff F821/F811/F841 on the two diagnostic Python files; report link/whitespace checks. Import emits the inherited `pynvml` deprecation warning; no package change was made.

Root must independently review this patch, mapping and costs before GPU release. Required next evidence is a matched original-fail/candidate-pass fixture under the same cold-cache inputs/budgets, actual candidate library/include/cache/cubin activation, and exact output checks covering both four-positive and empty-group transitions. A clean candidate alone cannot prove native CTA82 causality or a general speedup. Then verify both affected wgrad shapes through the real compiled/autograd/SAC path and obtain **two fresh all-eight-rank native200 successes** under the frozen exact preset. Relevant numerical and representative runtime/memory evidence remain required before production acceptance; this source/import/compile work closes none of those GPU gates.
