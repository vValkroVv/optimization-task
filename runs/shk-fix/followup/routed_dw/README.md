# Hopper routed BF16 weight-gradient overwrite probe

The installed Hopper DeepGEMM grouped FP8 weight-gradient API requires the BF16 accumulator. It rounds the FP32 WGMMA result to BF16, then adds that packed result to the zero-initialized output through TMA. The candidate repeats the same BF16 addition with an explicit positive zero in registers and stores the BF16 bits through TMA. It retains the selected BF16 launch configuration, WGMMA accumulation order, FP32 scale normalization, descriptor workspace, and output allocation size. The K-group scheduler skips K=0 experts, so the binding zeroes only contiguous runs of their output slices. The all-empty case writes the full zero output and skips GEMM.

The variant builds from the pinned donor `f093a4cd643bc06a7894a90dee852ec70e989a53`. It checks the installed distribution `2.3.0+f093a4c`, its extension hash and six CUDA header hashes through the existing direct-dW driver. The old EP8 FP32 slot receipt verified the packed BF16 add-to-zero helper on H100. This probe uses a distinct BF16 pybind module, JIT kernel name and private cache. It does not modify GIFT or the installed DeepGEMM package. [DeepGEMM pin](https://gitlab.ai.cloud.ru/gigallms/deepgemm/-/tree/f093a4cd643bc06a7894a90dee852ec70e989a53) · [NVIDIA PTX tensor reduction](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor)

## Cases and measurements

- **Native EP8** uses rank 0's captured `w2` and `w13` rowwise operands and routing from `backward-inputs-001`. It runs the installed CudaCherry converter once outside the measurement. The fixture predates this probe and comes from the recorded native direct-FP32 run.
- **EP4-shaped derived** splits each captured K segment into two 128-aligned groups and replays the converter with 32 expert IDs at the same aggregate K. It uses the captured operand values and is not a native EP4 capture.
- **Synthetic edges** use seeded E4M3 operands with FP32 scales, 16 or 32 experts, and mixed or all-zero K counts. Mixed cases include leading, interior, consecutive and trailing empty groups.

Each pair measures BF16 allocation, any zero writes, native scale normalization and both GEMMs. The outputs have identical shapes and lifetimes. Correctness compares both full BF16 outputs bitwise using a dirty offset destination, checks guard regions and positive zero for K=0, and probes the original and candidate TMA epilogues on special FP32 inputs. The profiler checks launch names and geometry. Allocation peaks use warmed execution. Interleaved eager timings and CUDA Graph replays each include an identical legacy control. Eager host-to-completion timings include allocation and submission. Graph replay measures the device work after capture.

## Build and run on the authorized host

Run from the verified personal shell on `sr008-gpu1` after the probe files are staged there. Choose the idle GPU index allocated for this component. The launcher binds its UUID and checks that it has no active compute process.

```bash
source /etc/profile.d/wm.sh
task=/workspace-SR008.nfs2/vkropotin
checkout="$task/shk-fix"
reference="$task/optimization-task"
donor="$task/hybrid-ep-overlap/sources/donors/deepgemm-baseline"
fixture="$reference/runs/kernel-fusion/artifacts/raw/backward-inputs-001/component-inputs/rank0.pt"
followup="$checkout/runs/shk-fix/followup"
probe="$followup/routed_dw/probe.py"
python=/opt/torchtitan-venv/bin/python3.12
attempt="$checkout/runs/shk-fix/artifacts/raw/routed-dw-bf16-001"
gpu=0

mkdir -p "$(dirname "$attempt")"
mkdir "$attempt"
mkdir "$attempt/tmp" "$attempt/launcher"
export TMPDIR="$attempt/tmp" TMP="$attempt/tmp" TEMP="$attempt/tmp"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/opt/torchtitan-venv/bin:$PATH"
env -u DG_JIT_SHARED_CACHE_DIR DG_JIT_CACHE_DIR="$attempt/legacy-jit" MAX_JOBS=1 \
  "$python" -B "$probe" build --reference-root "$reference" --donor "$donor" \
  --build-root "$attempt/build" > "$attempt/build.log" 2>&1

bash "$followup/run_component.sh" "$checkout" "$attempt/launcher" "$gpu" \
  "$probe" verify --reference-root "$reference" --build-root "$attempt/build" \
  --fixture "$fixture" --evidence-dir "$attempt/component" --samples 24 \
  > "$attempt/component.log" 2>&1
```

The build receipt and extension live under `build/`. Installed DeepGEMM warms under `legacy-jit/` and the candidate warms under `build/jit/`. The probe disables shared caches and rejects new JIT misses after each case warms. Results are in `component/component.json` beside both traces for each case.

The pinned build compiled and ran on `sr008-gpu1`. Both projection shapes matched bytewise on the captured EP8 case and the derived EP4 case. The four synthetic cases passed the zero-slice and destination-canary checks. CUDA Graph medians were 3.0669 → 2.9279 ms for the EP8 pair and 3.3248 → 2.9416 ms for the derived EP4 pair. Allocator peaks matched in both comparisons. [Component receipt](../../artifacts/raw/routed-dw-bf16-001/component.json)

## Compiled GIFT operator

The second component calls the actual grouped backward allocation and GIFT
operator in a full Inductor graph with CUDA Graphs disabled. It also tests
fixed-capacity tails and skips GEMM when all counts are zero. The original
grouped forward save set is unchanged. The native comparison uses
`GIFT_ROUTED_DW_BF16_OVERWRITE=1` to enable the candidate and
`GIFT_ROUTED_DW_BF16_OVERWRITE_BUILD` to select the private build.

```bash
compiled="$checkout/runs/shk-fix/artifacts/raw/routed-dw-bf16-compiled-002"
mkdir "$compiled"
bash "$followup/run_component.sh" "$checkout" "$compiled" 7 \
  "$followup/routed_dw/compiled_probe.py" \
  --reference-root "$reference" --build-root "$attempt/build" --fixture "$fixture" \
  --output "$compiled/compiled" --warmup 100 --samples 256 --pairs 8
```

All eight matched pairs produced identical BF16 outputs on every case and
the same GEMM geometry. Both zero fills disappeared in candidate traces.
The candidate cut synchronized latency by 1.99% [1.69%, 2.30%] on the
EP8 fixture and 8.54% [7.67%, 9.41%] on the derived EP4 case. Peak allocated
bytes matched at both shapes. [Compiled receipt](../../artifacts/raw/routed-dw-bf16-compiled-002/compiled.json)
