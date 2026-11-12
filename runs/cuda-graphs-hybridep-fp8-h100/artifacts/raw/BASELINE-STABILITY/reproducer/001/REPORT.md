# Exact-template KGroupedContiguous fixture

## Result

The unchanged DeepGEMM implementation returned a wrong elementwise result in the positive-group fixture. Case1's first admission call passed. The following asynchronous chunk submitted calls2–33; boundary verification identified **call2 alone** as unequal to the exact BF16 oracle. Execution stopped after33 GEMMs. No CUDA exception occurred and no exception core was produced. Cases2/3 were not run, as required by the stop rule.

This is a numerical failure at the matching machine code, not a reproduction of the original TMA illegal instruction. No causal repair or production patch is selected. Independent review of fixture admission, oracle, and the reused-output reset remains required. The process did not save mismatching coordinates, actual values, expected values or per-group mismatch counts before exiting; those cannot be reconstructed from its logs. No additional execution was performed to recover them.

Finite acceptance was: resolve actual call/layout; validate packing/oracle on CPU; admit the exact specialization; execute at most2112 calls/120 seconds with every output checked; stop on first observed exception/mismatch; preserve identities and report gaps. The first three items and bounded stopping are complete. Full case coverage is intentionally stopped by the numerical failure. No training, dependency/source edit, installation, profiler, sanitizer, launch blocking or candidate comparison occurred.

## Executed interface and oracle

[fixture.py](fixture.py) calls `_DeepGemmHandle.k_grouped_fp8_gemm_contiguous((a,sa),(b,sb),d,ks,kc,kd)`. `ks` is a Python list, `kc` matching CPU int32 counts and `kd` matching CUDA int32 counts. The GIFT custom op therefore avoids device `.tolist()` and calls installed DeepGEMM with `c=d`, preserving accumulation=true. BF16 D is `[4,1536,1024]`. A/B are FP8 e4m3fn group-major packed flattened `[1536,k_g]` / `[1024,k_g]`; FP32 unit scales are transposes of contiguous `[sum(ks)/128,M/N]` buffers. The actual signature/schema/docstring are in [cpu-check.json](cpu-check.json).

Each group/channel uses different support and signs over128 active K positions. Row support is `(j+3*g) mod (5+g) == row mod (5+g)`; column support is `(2*j+g) mod (7+g) == col mod (7+g)`. Independent integer residue-intersection counts, multiplied by row/column signs, form the oracle. Values are small exactly representable BF16 integers; inputs are0/±1 and scales1, so no tolerance is used. Active positions spread across the full K range in the large case. A zero-size group has a zero oracle and no packed elements.

CPU behavior checks unpacked the actual FP8 buffers and compared integer matrix products against the independent oracle for all-positive, empty-final and multi-block cases, at M13/N11. All passed without CUDA initialization. This verifies packing and the oracle, not GPU execution.

## Admission and provenance

Executed fixture SHA256: `302862ff7a5500023b70ea81c86f8f237cd2e78a860423225d3147ac54aa3fa0`. Repository base and Source010 remain those in the [causal assessment](../../cause/001/REPORT.md); all eight affected installed/GIFT source receipts were rechecked unchanged in [source-and-sass-checks.json](source-and-sass-checks.json).

`DG_JIT_DEBUG=1` exposed actual selection and loaded-CUBIN paths. It also adds `--ptxas-options=--verbose,--warn-on-local-memory-usage`, changing the cache key. This caused the first process to reject a cache miss before launching any GEMM; the complete failed harness record is retained under [harness-failure-001](harness-failure-001/launch.json). After explicit coordinator authorization, the one adaptive execution allowed unchanged-source runtime compilation into this task's cache. There was no automatic retry.

The actual loaded specialization has M1536/N1024/Kdynamic, groups4, block128/152/128, stages5, grid116, multicast2 on B, BF16 CD and accumulation=true. Generated `kernel.cu` is byte-identical to the fault-associated original. Installed library version remains `0015c5c30202d513849b339660066739`, compiler NVCC13.2. The original kernel key is `500d411276c82c1ae1681a9672269982`; actual loaded key is `04254286e5f40c1e389b04ec1a89cd8d`.

- Original cubin SHA256: `f4bbbae842de6bf676e66bf846713dbe626d9c5a08434c8e993aca15ad0a1bdf`.
- Actual cubin SHA256: `dafcb551bb26cc90fb1e37c647262ca1dfaf1fd539ca9440e34326843b813f4c`.
- `cuobjdump --dump-sass` output is **exactly equal** for original and actual cubins: SHA256 `608315a473b77f9e4e8bcadd4e4916f542fe17b788b4c13f6f08e82e0523f04e`. See [actual-sass.txt](actual-sass.txt).

Admission occurred after synchronizing the first actual call; the loaded binary's generated source and metadata were checked before proceeding. The exact first output also passed the numerical check.

## Output checking, ownership and limits

The fixture retains all operands/scales/counts and a32-output ring. Each GEMM's D is zeroed immediately before its call on the same native current stream. Each chunk synchronizes once, then computes32 device mismatch flags and copies those flags to CPU together. Every output in the chunk is checked. No CPU equality or synchronization runs between GEMM launches. Case1 call2 reuses ring slot0 from the admission call after zeroing it; calls3–33 use the other31 slots. Only call2 was flagged. The logs preserve the submitted range and exact failing-call list, not the individual output tensors.

The admitted first call is included in the case limit: 1024/1024/64, total2112, with no extra warmup GEMM. The internal120-second workload deadline is checked before each chunk; the subprocess owner has a180-second wait and preserves an active child/core rather than killing core generation. Neither cap was reached. This is a bounded causal fixture, not a timing experiment.

Fresh launch admission verified all eight UUIDs against reconciliation002, no compute applications, each H100 with more than75000MiB free, and more than20GiB filesystem capacity. Only GPU0 `GPU-c5efeb4a-3298-d3d3-d740-9aeaa1cdd60c` was visible. Exact command/environment and fresh exception-core path are in [launch.json](launch.json); returncode1 is in [completion.json](completion.json). Native graph/mix-order flags were0. Exception settings matched tool-smoke002, omitting global/shared/local/constant memory; no debugger/profiler was attached.

## Runtime-cost review and checks

This diagnostic changes no production path. Existing GIFT layout/validation and installed scratch allocation, current-stream launch and accumulation remain active. Diagnostic perturbations are explicit:32 BF16 outputs consume384MiB rather than one12MiB output; each launch includes same-stream D zeroing; each boundary adds synchronization and equality kernels; DG debug prints host metadata. The retained references and output ring change allocator/cache history. There are no collectives or host device-value reads per GEMM. Magnitudes are not benchmarked; no performance claim follows from this fixture.

Commands:

```sh
PYTHONPATH="$PWD/gift" /opt/torchtitan-venv/bin/python3.12 runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/reproducer/001/fixture.py
/opt/torchtitan-venv/bin/python3.12 runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/reproducer/001/launch.py
/opt/torchtitan-venv/bin/python3.12 -m ruff check --select F821,F811,F841 --no-cache runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/reproducer/001/fixture.py runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/reproducer/001/launch.py
```

CPU oracle/interface PASS; exact-template/source/SASS admission PASS; lint PASS; GPU numerical check FAIL at call2; cases2/3 SKIPPED due failure. Original-fault causality, mismatch details, candidate fail/pass comparison, integration, and required native stability passes remain open. The coordinator owns the independent fixture review and next decision.
