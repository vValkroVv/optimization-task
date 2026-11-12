# Baseline CUDA fault: bounded primary-source research

## Result

A concrete, version-matched **HybridEP fused-scan source defect** is present in the installed backend header and has an upstream correction. It is a candidate for investigation, not an established cause of the rank-2 illegal instruction at native update 22. The watchdog queried an event after asynchronous work; its report alone does not identify the failing kernel.

Acceptance completed: inspect relevant maintained issue/fix contracts; reconcile installed versions and actual driver-library loading; identify one practical next diagnostic without changing production settings. No GPU tests, training-source changes, installs, library modifications, or retries occurred. Artifacts only were written. This research does not pass the failed strict numerical or baseline timing gates.

## 1. Concrete installed HybridEP fused-scan bound

Installed `deep_ep/backend/hybrid_ep_backend.cuh` is byte-identical to [upstream at `1b8f467`](https://github.com/deepseek-ai/DeepEP/blob/1b8f467/csrc/hybrid_ep/backend/hybrid_ep_backend.cuh#L5128). Its fused scan allocates a local integer array of length `NUM_OF_EXPERTS_PER_RANK - 1` at line 5128, then writes indices below `NUM_OF_EXPERTS_PER_RANK` at lines 5132–5133 and reads the current expert at lines 5153–5154. For the target's four experts/rank, index 3 is outside a three-element array. The resolved native smoke config records `fuse_permute: true` at line 124, and the task wrapper passes that setting through dispatch/combine. This is relevant to the configured fused path.

Upstream [commit `17cfb817bccec3a9c247013360cc550c2bac441e`](https://github.com/deepseek-ai/DeepEP/commit/17cfb817bccec3a9c247013360cc550c2bac441e), PR #625, changes the array length to `NUM_OF_EXPERTS_PER_RANK`. Its full patch is archived here. The commit also substantially rewrites standalone permutation; a wholesale upgrade is not equivalent to correcting only the bound.

Verified: defective installed header bytes and matching upstream correction. Not verified: whether the failing process compiled/loaded this exact header into its scan kernel, emitted behavior after compiler optimization, or whether this defect caused either observed failure. Source undefined behavior does not establish its runtime manifestation.

## 2. Known pinned-count race: distinguish the inactive path

[DeepEP issue #756](https://github.com/deepseek-ai/DeepEP/issues/756) names installed revision `1b8f467`, blocking-mode pinned host count lifetime, delayed CUDA faults and a later watchdog report. However, its faulty count dereferences are in **standalone** permute/unpermute. The target enables fused permutation. The issue's later retest credits #625 with removing that standalone host-pointer dependency. Its GB300/64-rank results do not establish this H100/8-rank cause. Retaining all pinned handles or changing dispatcher mode is not justified from this report alone.

## 3. DeepGEMM tensor-map races: related, unresolved fit

[DeepGEMM issue #237](https://github.com/deepseek-ai/DeepGEMM/issues/237) describes H100 K-grouped illegal instructions when small per-group grids cause scheduling to skip the anticipated next group. The installed SM90 header already contains a group-mismatch correction and scheduler-derived current K offset at lines 191–210. That observation does not establish binary behavior.

[Issue #405](https://github.com/deepseek-ai/DeepGEMM/issues/405) links [fix `1f6f3f378920ccb5cc036ef43eb3f5972e921713`](https://github.com/deepseek-ai/DeepGEMM/commit/1f6f3f378920ccb5cc036ef43eb3f5972e921713), which waits for outstanding tensor-map use before overwriting global descriptors. The installed version declares `2.3.0+f093a4c`; that short commit URL returned 404 on public upstream. Its shipped SM90 header uses two descriptor stages and differs structurally from the patch's one-descriptor path. Therefore neither missing textual wait calls nor the upstream issue establishes that this binary has the same race. Do not transplant the patch without inspecting the actual fork and generated code. Patch archived; no ancestry claim made.

## 4. CUDA 13 on driver 570: actual compatibility route exists

A CPU-only `ctypes.CDLL('libcuda.so.1')` and `/proc/self/maps` inspection loaded `/usr/local/cuda-13.2/compat/lib.real/libcuda.so.595.45.04`. No CUDA initialization API was called. The attempt launcher does not override loader paths. This observation is for a fresh process in the current shell; failed ranks have exited, so their loaded libraries are not retrospectively proven.

NVIDIA's [forward-compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html#deployment-considerations-for-forward-compatibility) supports `cuda-compat-13-2` over driver branch 570 on data-center GPUs. H100 qualifies for that GPU class. Consequently, host driver 570 plus CUDA 13 is not sufficient evidence of an unsupported stack. Package presence alone would be weaker than this observed loader route. This does not certify every installed library or rule out a driver defect.

## 5. cuDACHerry and NVJET limits

Installed cuDACHerry metadata points to its GigaLLMs GitLab repository and `direct_url.json` records only a local `/opt/cudacherry` build directory, with no commit. Docker's `585bd50a...` declaration is not proof of installed binary provenance. The bounded public search produced no version-matched maintained cuDACHerry fault fix; this is **unknown**, not evidence that no fix exists. Earlier local row2col full-tile inspection remains applicable.

NVIDIA identifies [NVJET as an internal kernel backend](https://forums.developer.nvidia.com/t/purpose-of-these-kernels/328273/2). Its name does not imply Jetson-only execution or prove an illegal instruction. The public SM120/TF32 NVJET fault reports do not match this SM90 FP8 workload; no matching maintained fix was established.

## One next diagnostic and decision

**Check the actual HybridEP scan JIT artifact's source-to-binary provenance for the bound above**, reusing the failed attempt's cache without launching work or changing libraries. Locate its generated scan source/compile manifest, confirm four experts/rank and fused compilation, and reconcile the included installed header with its cache hash and any available line/disassembly records.

- If the failing artifact demonstrably includes this defective bound, prioritize a narrowly scoped correction and controlled validation proposal under the existing source/compute authorization boundaries.
- If its generated source already contains the correction, deprioritize this defect for the observed fault and give the execution owner's first-failure localization evidence precedence.
- Missing/inconclusive artifact provenance leaves causality unknown; it is not a pass or a reason to silently upgrade the stack.

No extra GPU diagnostic is proposed here. [CUDA launch blocking](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html) serializes host submission, and #756 reports that it masked its lifetime race. NVSHMEM's [launch/progress contract](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html) depends on collective/cooperative behavior; a safe global launch-blocking route for the exact HybridEP stack was not established. NVIDIA [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) defaults to nonblocking launches, but instrumentation still changes timing and memory. Its general capability does not prove compatibility or sufficiency for this exact multi-rank stack.

## Reproduction and artifacts

`receipts.json` records retrieval time, primary URLs, SHA-256 hashes of archived upstream source/patches and inspected local files, and loaded `libcuda` path. Local inspections used the existing `/opt/torchtitan-venv/bin/python`, standard-library file/HTTP operations, `rg`, and text reads. No package imports requiring CUDA, CUDA initialization, compilation, or device work were performed.

## Appendix: actual failed-run JIT path and isolated correction route

### Actual artifacts and freshness

The failed rank 2 has PID 459018 in `attempts/001/supervisor.log`. Its actual scan artifact exists at `/home/user/.deepep/hybrid_ep/jit/proc-459018/1536-32768-4-8-1-128-64-256-108-1-0.so`. Matching scan artifacts exist for all eight failed worker PIDs 459016–459023; their modification times are 2026-11-11 21:26:49 UTC, within this attempt. `jit-route-receipts.json` records paths, times, sizes and SHA-256 hashes. Only these task-owned PID paths were inspected beyond listing cache-directory names.

The rank-2 artifact contains symbol `hybrid_ep::scan<256,108,128,64,8,1,4>` (mangled symbol receipt in `failed-scan-symbols.txt`). `cuobjdump --list-elf` reports SM90 cubins. This establishes the expected four-expert, eight-rank, one-node specialization. It does not reconstruct the local-array bound from machine code.

Installed wrapper line 106 provides the installed package directory as base path. The shipped header has timestamp 2026-10-31 and exactly matches defective upstream bytes. The per-PID artifact creation times strongly support compilation during the failed attempt using that installed header; neither a preprocessed translation unit nor a captured exact compiler command was retained, so this is a source/loader inference, not direct binary-to-header proof. Native source currently asks for `load_cached_kernels=True` (`gift/torchtitan/distributed/deepep/hybridep.py:932`). That flag loads only the current process's cache directory, not prior PIDs.

### Existing API and source contracts

Pinned [JIT compiler source](https://github.com/deepseek-ai/DeepEP/blob/1b8f467/csrc/hybrid_ep/jit/compiler.cu), archived as `deepep-jit-compiler.cu`, establishes:

- `get_jit_dir()` reads `HYBRID_EP_CACHE_DIR`, falling back to the passwd home directory, then `/tmp`; it appends `.deepep/hybrid_ep/jit/proc-<pid>`. This explains `/home/user` here independently of shell HOME.
- `NVCCCompiler(base_path, comm_id)` supplies `-I<base_path>/backend` and CUDA includes to nvcc. There is no separate header-path environment variable in this pinned implementation.
- Generated preprocessing code includes `hybrid_ep_backend.cuh` and instantiates the requested configuration. Fused compilation adds `HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE`.
- Generated `.cu` files are deleted after successful compilation. The resulting `.so` is loaded with `RTLD_NOW | RTLD_LOCAL` and renamed to its configuration key.
- Keys concatenate configuration values, including expert/rank counts, fusion and nonblocking flags. They contain **no header content hash**. `load_cached_kernels=True` loads existing `.so` files in the current per-PID directory; an existing buffer also keeps function pointers in memory.

The [pinned pybind binding](https://github.com/deepseek-ai/DeepEP/blob/1b8f467/csrc/hybrid_ep/pybind_hybrid_ep.cu#L174), archived as `deepep-pybind-1b8f467.cu`, explicitly exposes:

```python
hybrid_ep_cpp.HybridEPBuffer(
    process_group, buffer_config, local_rank, node_rank, group_size,
    base_path, load_cached_kernels=False,
    use_shared_buffer=True, enable_custom_allgather=True,
)
```

This is an existing internal/native constructor route. The installed high-level `deep_ep.HybridEPBuffer.__init__` exposes `load_cached_kernels`, but **does not expose base_path or an include-directory argument**. It passes `os.path.dirname(os.path.abspath(__file__))` to the native constructor at lines 100–110. Installed extension strings/symbols confirm the native constructor, base_path/cache argument names, per-PID path, and fusion macro exist; no constructor was invoked during this read-only pass.

### Feasible task-contained route, not implemented

An existing public high-level include override was not found. A scoped GIFT experiment can use the established base_path contract without modifying the installed package or rebuilding its native extension:

1. A GIFT-owned preparer generates an isolated runtime directory under this campaign's raw artifacts. Copy the installed `hybrid_ep_buffer.py` and its five-file `backend/` directory there. Preserve wrapper/support-header bytes and license notices; correct exactly the single backend array bound from the upstream patch. Record source/destination hashes and an exact one-line diff. This is a generated runtime copy, not an installed-package edit.
2. A task-owned GIFT adapter loads that unchanged wrapper from its generated path under a unique module name using Python's standard `importlib.util.spec_from_file_location`/`module_from_spec`/`exec_module`. The inspected wrapper has absolute imports (`torch`, `os`, `shutil`, `hybrid_ep_cpp`, `warnings`) and no package-relative imports, so relocation preserves its import contracts. Its existing `__file__` calculation then supplies the isolated directory as the native base_path. Select this class explicitly in the experiment integration; do not mutate installed module globals, reassign their `__file__`, or replace `sys.modules['deep_ep']`.
3. Set the existing `HYBRID_EP_CACHE_DIR` to a distinct task-owned directory for each corrected/control variant and launch fresh processes. This redirects generated files only; it does not select headers. Both sides must use identical relocation/cache policy, differing only in that header bound, to avoid confounding the causal comparison.

The route uses existing native base_path and Python file-loading contracts. The new GIFT adapter/preparer would be task code requiring normal integration review; it is not an upstream-supported configuration feature. There is no need to change fusion, FP8, streams, SAC, compilation, topology, data, or native extension ABI. The copied header must remain the installed revision except for the one bound: do not substitute the full newer backend/PR.

### Cache and runtime costs

Use a new runtime/cache root and new processes, with no live HybridEP buffer. Header-only edits do not invalidate an in-memory cache or a populated configuration-key cache. Never call the wrapper's `empty_jit_cache()` for this task: it targets an installed-package path and is unnecessary. Do not delete shared caches. The isolated route permits retaining every original artifact.

Copy preparation is a one-time host filesystem cost (roughly 0.5 MB). Header-bound correction increases one local integer array from three to four entries for this specialization; register allocation, spills, generated code and steady-state effects require actual compiler/runtime evidence. Fresh isolated caches add native nvcc compile startup; exclude this explicitly from stable-step timing while recording startup. Relocation itself creates no intended per-token/device work. A comparison must preserve all existing checks and cannot be declared successful merely because one run survives past update 22.

**Remaining gates:** no adapter/preparer or correction was implemented, no import/runtime integration was exercised, no compiler was invoked, and no CUDA work ran. Exact installed-extension/source correspondence beyond the inspected symbols and wrapper/header provenance remains limited. The native-binding base_path route is concrete enough to review a task-contained implementation; failure causality, numerical parity and sustained GPU acceptance remain open.
