# Direct FP32 expert gradient component

The standalone extension targets the Hopper EP8 expert-gradient shapes
`w13=(16,3072,4096)` and `w2=(16,4096,1536)`. It writes the legacy rounded
BF16 result into an existing FP32 slot view. DeepGEMM's normal accumulation
entry in the installed package is used as the reference.

## Build and verify

Use the baseline Torch environment, its CUDA toolkit, and a task-owned build
directory. The donor is the flattened `sources/donors/deepgemm-baseline`
directory from the hybrid-ep-overlap checkout. Its DeepGEMM revision is
`f093a4cd643bc06a7894a90dee852ec70e989a53`, CUTLASS is
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`, and fmt is
`553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`. The build checks the staged
donor hash and the five corresponding headers from installed DeepGEMM
`2.3.0+f093a4c`. It compiles only the focused host binding. The source slice
can stage without Torch or CUDA by adding `--stage-only`.

```bash
python direct_dw.py --donor "$direct_dw_donor" --build-root "$direct_dw_build"
DG_CACHE_FORCE_SKIP=1 DG_JIT_CACHE_DIR="$direct_dw_legacy_jit_cache" \
  python verify_direct_dw.py --build-root "$direct_dw_build" \
  --evidence-dir "$direct_dw_evidence" --num-sms 116 --samples 12
```

The value 116 comes from the resolved YAMLs of the passing fixed timing control
(`runs/kernel-fusion/artifacts/raw/efsdp-control-ep8-001-timing/native/config/resolved_config.yaml`)
and update-21 trace control
(`runs/kernel-fusion/artifacts/raw/efsdp-control-ep8-002-trace/native/config/resolved_config.yaml`).
The runner verifies each control's completion receipt and resolved-config
checksum before using that value for direct dW. The standalone check applies
the explicit `--num-sms` value to both the installed DeepGEMM reference and the
candidate, then records it in `component.json`.

For training ranks, the runner passes the verified budget as
`GIFT_DIRECT_EXPERT_DW_NUM_SMS` only when direct dW is enabled. The rank requires
that variable and has no fallback to the device's full SM count.

The build writes `manifest.json` and the extension under `extension/`.
Make the host build serial before importing it in workers. The candidate
compiler uses `$direct_dw_build/jit` and ignores the shared JIT cache.
`DG_JIT_CACHE_DIR` in the command selects the task-owned cache for the
installed reference. The candidate uses the same NVCC/NVRTC environment
selectors as the pinned donor.

## Fixed-recipe training arm

After the native component checks pass, use the existing production runner
with the full committed revision and the verified build directory. Run the
10+10 timing and update-21 profile separately on `sr008-gpu1`.

```bash
python runs/kernel-fusion/run.py production timing <run-id> \
  --source-revision <revision> --gfsdp-optimization direct-expert-dw \
  --direct-dw-build "$direct_dw_build"
```

The launcher verifies the generated experiment and resolved configuration
against the matching fixed EP8 control. Inspect the target compiler's
backward for a direct write to the real FP32 slot and check the trace for
the rounded-FP32 kernel and the removed expert gradient-packing work.

## Native contract

Import `load_direct_dw` from `direct_dw.py` once per process after selecting
the rank's GPU. The returned extension exposes
`k_grouped_fp8_gemm_nt_contiguous_rounded_f32(a, b, d, ks, ks_tensor, compiled_dims="mn")`.
It returns `None` and overwrites `d` on the selected CUDA device's current
stream. The caller owns slot acquisition and publication, keeps inputs alive
on that stream and provides disjoint inputs and output.

- `a` and `b` are contiguous E4M3FN data packed consecutively by expert
  with each expert's features as rows. The matching FP32 scale views have
  width `total_k/128` and row stride 1. The data and scales are TMA aligned.
- `d` is a contiguous, TMA-aligned FP32 view with one of the two target
  shapes. Its base address may come from a nonzero slot offset.
- `ks` contains 16 nonnegative multiples of 128. The contiguous CUDA
  int32 `ks_tensor` on the same device must contain the same values. The
  caller supplies both from the same routing result. The extension checks
  tensor metadata without reading counts back from the GPU.
- Each nonempty expert performs one FP32-accumulated WGMMA result per output
  element. The epilogue uses the original RN-even BF16 conversion, explicitly
  adds BF16 positive zero, widens the individual bits, and uses a FP32 TMA
  overwrite. The host enqueues one zero fill per contiguous run of empty
  experts, or one full-view zero fill when all experts are empty.

[Pinned source provenance](https://gitlab.ai.cloud.ru/gigallms/deepgemm/-/tree/f093a4cd643bc06a7894a90dee852ec70e989a53) · [DeepGEMM build reference](https://github.com/deepseek-ai/DeepGEMM/blob/main/setup.py) · [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor)

## Acceptance and remaining checks

The component harness compares whole-tensor FP32 bits for both slot views,
including zero-token experts, and checks guard regions around the views. The
reference allocates the two zeroed BF16 gradients, runs the installed native
GEMMs, and widens both with one `torch._foreach_copy_`. The Hopper probe
runs the original BF16 TMA reduce-add and the new FP32 TMA overwrite on
paired FP32 inputs containing rounding ties, signed zeros, subnormals,
infinities, and quiet and signaling NaNs. It records the resulting bits.

Profiler traces check the distinct legacy and candidate GEMM kernel names,
pack call count, and capture device work duration. Allocation peaks bound
candidate materialization relative to a full BF16 gradient. Optional warm
CUDA Graph replay records raw single-stream CUDA-event spans and the median
of each variant and the identical legacy control. These replay spans exclude
the original Python and dispatcher execution. Synthetic K counts exercise
full, mixed, and all-empty groups. The harness leaves full training,
distributed memory, host/wait attribution, and steady-state throughput open.

The BF16-selected WGMMA tile, K reduction order, and routing scheduler are
unchanged. The candidate uses more shared memory for the per-CTA output tile
and may require fewer pipeline stages. It adds a BF16 pair instruction and
retains the small TMA descriptor workspace. Nonempty experts require no
destination prefill, global BF16 intermediate, or packing pass. Empty groups
add only their destination zero writes. Benchmark the tradeoff on Hopper
before interpreting the bytes saved as latency saved.

Local validation applied the patch to a verified donor, resolved 173 pinned
source includes, and compiled the JIT source formatter with pinned fmt to
verify the 19 candidate template arguments. Python syntax and Ruff checks
passed. The local macOS
environment has no CUDA toolkit or H100. Native compilation, exact hardware
special-value behavior, GPU activation, allocation and timing remain open.
