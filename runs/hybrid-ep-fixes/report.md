# HybridEP routing metadata

The current kernel reduced isolated GPU replay time by **47%** versus the
original helper compiled directly with Inductor. Grouping eight tokens per
CUDA block reduced it by another **36%** on both native route samples. With
Python, allocation and completion included, the wrapped prototype improved
isolated call latency by **4.5–4.8%** versus the current wrapper.

## Compared boundary

Time the conversion of post-capacity top-k expert IDs and scores into contiguous
bool routes and FP32 probabilities. The two outputs have shape `[8192, 128]`
and occupy 5 MiB plus one sink element each. The timed boundary starts after
capacity selection and ends before FP8 input quantization or dispatch. HybridEP's
backward gathers gradients separately. [Dispatch call site](../../gift/torchtitan/distributed/deepep/hybridep.py#L289) ·
[Gradient gather](../../gift/torchtitan/distributed/deepep/hybridep.py#L728)

The original helper creates masks and flat indices, zero-fills two buffers, and
scatters routes and probabilities. It ran eagerly inside the opaque dispatch
CUDA custom op even when the model was compiled. The isolated benchmark also
compiles the original helper directly with `torch.compile(fullgraph=True,
dynamic=False)` and the default Inductor backend. That gives Inductor visibility
into the helper. The current Triton implementation scans eight IDs for each
expert position, with one four-warp CUDA block per token. The Gluon prototype
uses four warps to process eight tokens per CUDA block and writes the same dense
outputs. It reduces the native-shape grid from 8,192 to 1,024 blocks. The wrapped
prototype exercises a CUDA custom op and checks the layout before launch.
[Original helper](../../gift/torchtitan/distributed/deepep/routing.py) ·
[Current kernel](../../gift/torchtitan/experiments/four_fusion/routing_metadata.py) ·
[PyTorch custom ops](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/programming_model.custom_ops.html) ·
[Gluon layouts](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)

## Input and execution evidence

The saved rank-0 resolved config for `followup-native-001-cute-replay-stats`
has B2, S4096, EP8/eFSDP1, dense FSDP8, TP/PP/CP1, top-8, 128 experts,
and fixed capacity 50,000 with group gating. Its SHA-256 is
`c39efcb45f6355be56ab5a3ee8f849b5c939eb79c591d8cdb3088b3998543217`.
The native production-data trace `4fix_qla_embed_selector_gpu_5fusions.json`
shows all 39 rank-0 metadata launches using grid 8,192 and block size 128.
Its SHA-256 is `61acd78b08ee443a451d44885fde50a1802bbe96ebbb723bdcddb2d12c6c5597`.
[10+10 receipt](../../../../runs/fusion-followup/report.md#native-1010-results)

Replay uses the saved router operands from `capacity-selector-001-capture`,
a separate native S update on production-tokenizer data. All eight rank files
confirm contiguous `[8192, 8]` int64 IDs and FP32 scores. The benchmark
reconstructs the post-capacity inputs using the saved rank counts. Each
destination group is below the native group budget of 800,000, so this sample
retains every route. It times rank 0 and the other rank with the highest
single-expert count. [Original fixture acquisition](../../../../runs/capacity-selector/report.md#isolated-ep8-result) ·
[Group gate](../../gift/torchtitan/models/common/moe/dispatch/capacity.py#L132)

Run `metadata-h100-005-native` used physical GPU 0 on the `sr008-gpu` H100
80 GB node `lm-mpi-job-84fd3579-7bd2-4969-ae47-b7af17b73ca2-mpimaster-0`.
The default `python` after the image's `/etc/profile.d/wm.sh` initialization
is `/opt/torchtitan-venv/bin/python`, with PyTorch
`2.15.0.dev20260822+cu130` at `8b353065ce260336c5fdfa74d754e8bd9b91b062`
and Triton `3.8.0`. The first attempt stopped during CUDA initialization before
any timing. The image's default initialization resolved the driver library.
The local branch starts at `529d8f27d`; the remote checkout is at `2c67c182e`.
The three files on the measured path have matching SHA-256 hashes across
both trees. No package installation or training was needed.

## Timing protocol

All variants warm up and check output shape, stride, dtype, values and trailing
sink bitwise before timing. Eight trials alternate forward and reverse arm
order. An identical eager baseline arm measures same-session variation. CUDA
event times cover replay of 64 calls and exclude Python submission per call.
A second graph uses 24 distinct live output pairs, about 120 MiB of output
storage, to test reuse sensitivity. Wall-clock call batches include tensor
allocation, Python/dispatcher submission and a final device synchronization.
An eight-call CUDA activity capture reports kernel and copy durations separately
from the unprofiled timing. Timing covers the metadata helper on one GPU.

The replay script is [bench_routing.py](bench_routing.py). Raw logs, the
preflight receipt and the exact executed script are under
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/hybrid-ep-fixes/artifacts/raw/metadata-h100-005-native/`.

The launch used the image's default Python and task-owned compiler caches.
From the SR008 task root, the command body was:

```bash
source /etc/profile.d/wm.sh
out="$PWD/runs/hybrid-ep-fixes/artifacts/raw/metadata-h100-005-native"
fixtures="$PWD/runs/capacity-selector/artifacts/raw/capacity-selector-001-capture/routing-fixtures"
PYTHONPATH="$PWD/gift" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  TORCHINDUCTOR_CACHE_DIR="$out/inductor" TRITON_CACHE_DIR="$out/triton" \
  /usr/bin/timeout --signal=TERM --kill-after=15s 480s \
  python -uB "$out/bench_routing.py" --fixture-dir "$fixtures" --output "$out/results.json"
```

## Native replay results

Each timing cell lists the rank 0 and rank 6 fixture medians in µs per call.
Rank 6 has the highest single-expert count among the seven peers. The largest
observed destination-group count was 67,189 of the 800,000 budget. All output
and layout checks passed, including the two initialized sink elements. A small
separate case also passed bitwise checks for negative int64 IDs, valid zero,
signed zero and NaN scores.

| Implementation | GPU replay | GPU replay, distinct outputs | Host enqueue | Integrated latency | Profiled kernel sum | Launches per call |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original, eager | 22.76 / 23.50 | — | 77.26 / 106.67 | 77.44 / 106.92 | 22.38 / 22.46 | 11 kernels, 1 DtoD copy |
| Original, identical control | 22.79 / 23.52 | — | 77.81 / 107.31 | 78.02 / 107.59 | 22.39 / 22.40 | 11 kernels, 1 DtoD copy |
| Original, compiled directly | 11.08 / 11.29 | 11.57 / 11.50 | 55.65 / 62.85 | 55.92 / 63.15 | 11.65 / 11.68 | 4 kernels |
| Current four-warp custom op | 5.90 / 5.96 | 6.04 / 6.04 | 42.95 / 47.29 | 43.17 / 47.54 | 6.24 / 6.24 | 1 kernel |
| Current kernel, one warp, direct call | 5.93 / 5.98 | — | 22.94 / 26.51 | 23.12 / 26.69 | 6.27 / 6.27 | 1 kernel |
| Four tokens per block, direct call | 3.73 / 3.78 | — | 23.71 / 27.30 | 23.87 / 27.48 | 4.00 / 4.00 | 1 kernel |
| Eight tokens per block, direct call | 3.75 / 3.81 | — | 23.71 / 27.21 | 23.87 / 27.39 | 3.84 / 3.84 | 1 kernel |
| Eight tokens per block, custom op | 3.75 / 3.80 | 3.88 / 3.87 | 41.01 / 45.04 | 41.21 / 45.25 | 3.83 / 3.84 | 1 kernel |

The eager control medians differ by at most 0.12% in graph replay. The current
and grouped kernel trial ranges are disjoint on both fixtures. The final batch
synchronization takes 0.19–0.31 µs per call across the main arms when amortized
over 64 calls. The eager DtoD copy takes 1.19–1.20 µs per call
in its separate CUDA capture. The component has no EP or other communication.
Host enqueue and GPU execution can overlap. [Full samples and timing scopes](results.json)

The four-token variant replayed marginally faster in the direct-call test.
The eight-token variant had a slightly lower kernel sum in the separate profile
and was also timed through a custom op. This first run used contiguous top-8,
128-expert inputs. The wrapped benchmark includes custom-op
dispatch overhead and writes the same 5 MiB of dense metadata. Promotion would
require validation inside the real HybridEP caller, including its SAC and
backward path. This experiment establishes a component latency improvement.

## Shape sweep

The sweep uses the eager reference, the directly compiled reference, the
current Triton custom op and the eight-token Gluon custom op. The Gluon block
still uses four warps with two token rows per warp. Each lane owns 2, 4 or 8
expert columns at 64, 128 or 256 experts, respectively. Top-k stays at eight.

| Tokens per rank | Experts | Input construction | Dense output per call |
| ---: | ---: | --- | ---: |
| 8,192 | 128 | Original rank 0 and rank 6 routes | 5 MiB |
| 16,384 | 128 | Concatenate the two rank captures in each order | 10 MiB |
| 8,192 | 64 | Fold IDs modulo 64 and advance duplicates to the next free ID | 2.5 MiB |
| 8,192 | 256 | Split alternating top-k slots across two 128-expert banks | 10 MiB |

The requested `8,192 *` is interpreted as `8,192 × 2`. The three added
shapes preserve the captured FP32 scores and distinct IDs within each row.
They represent component inputs derived from native routes. Shape-specific
training and capacity selection are outside this replay.

Run `metadata-size-sweep-001` used `--size-sweep` with the same default
Python, libraries and physical GPU 0 on `sr008-gpu`. The launcher saved
the eight-GPU availability and process snapshot before starting. Each shape
uses two input sets, eight alternating trials per set and an identical eager
control. Separate graphs replay 24 distinct output pairs per implementation
to cover more output storage than the H100 L2 cache, including at 64 experts.
The raw run with the executed script, preflight and log is under
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/hybrid-ep-fixes/artifacts/raw/metadata-size-sweep-001/`.

### Sweep results

Each cell spans the two input-set medians in µs per call. The GPU reduction
column compares the wrapped prototype with the current Triton custom op.

**CUDA Graph replay**

| Tokens | Experts | Original eager | Original compiled | Current Triton | Eight-token prototype | Lower GPU time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 128 | 22.91–23.54 | 11.11–11.35 | 5.91–5.96 | 3.78–3.83 | 35.7–36.1% |
| 16,384 | 128 | 25.83–25.87 | 13.91–13.92 | 10.89 | 6.61 | 39.3% |
| 8,192 | 64 | 22.84–22.88 | 9.79–9.98 | 5.96 | 3.17–3.18 | 46.7–46.9% |
| 8,192 | 256 | 24.99–25.02 | 13.94 | 7.31–7.32 | 5.62 | 23.1–23.2% |

**CUDA Graph replay with 24 distinct output pairs**

| Tokens | Experts | Original eager | Original compiled | Current Triton | Eight-token prototype |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 128 | 23.54–23.58 | 11.51 | 6.04–6.05 | 3.90 |
| 16,384 | 128 | 26.10–26.14 | 14.18 | 10.97 | 6.98–6.99 |
| 8,192 | 64 | 22.83–22.94 | 9.41–10.03 | 6.04 | 3.23 |
| 8,192 | 256 | 25.09 | 14.36–14.37 | 7.52–7.53 | 5.70–5.71 |

**Host to completion with output allocations and wrapper overhead**

| Tokens | Experts | Original eager | Original compiled | Current Triton | Eight-token prototype |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 128 | 77.30–107.69 | 57.45–63.68 | 43.79–49.02 | 42.64–46.56 |
| 16,384 | 128 | 107.25–107.34 | 64.88–65.56 | 48.57–49.21 | 47.34–47.70 |
| 8,192 | 64 | 107.51–108.17 | 64.15–64.61 | 48.27–48.97 | 47.07–47.24 |
| 8,192 | 256 | 107.16–107.24 | 64.71–64.96 | 49.26–49.55 | 47.50–47.93 |

Every timed arm passed bitwise map, probability, layout and trailing sink
checks in all eight shape/input combinations. The current and prototype also
passed negative-ID, zero, signed-zero and NaN tests at each expert width.
CUDA activity confirms 11 kernels and one DtoD copy for the eager helper,
four kernels for the compiled helper, and one kernel for each custom op. The
prototype's `grouped_rows` kernel was active at all widths. Each case profiles
kernels and copies separately from the unprofiled batches. The host samples
record enqueue and final synchronization separately. [Trial data and profile summaries](size-sweep.json)

The prototype's slowest GPU replay was faster than the current kernel's
fastest replay in every case, including with distinct output allocations.
The identical eager-control replay medians differed by at most 0.48%.
Host-to-completion medians were 2.5–5.0% lower for the prototype. Individual
host trials favored it in 62 of 64 paired observations.
These results cover the metadata helper on this H100 and the four tested shapes.

## Top-k sweep

The Gluon wrapper now reads `tokens, top_k = ids.shape` and passes `top_k`
as a `gl.constexpr`. The kernel uses it for both `gl.static_range(top_k)`
and the contiguous input row offset `token * top_k + slot`. Eight token rows
per CUDA block is a separate launch choice. Contiguous layouts and the 64,
128 and 256 expert widths keep the same eligibility checks. [Gluon benchmark kernel](bench_routing.py#L28)

This sweep fixes the production 10+10 dimensions at 8,192 tokens per rank
and 128 experts. Top‑8 uses the unmodified rank 0 and rank 6 router captures.
Top‑4 takes the four highest-scoring slots from each capture. Top‑16 appends
eight slots from the other capture and advances colliding expert IDs to the
next free ID in the row. Top‑4 and top‑16 are derived inputs using saved
production data scores. The measured block copies scores into a dense map.
Capacity selection and router scoring are outside the timed boundary.

Run `metadata-top-k-sweep-001` used `--top-k-sweep` on physical H100 GPU 0
on `sr008-gpu` with the default Python. The eight-GPU preflight snapshot,
logs and exact executed source are under
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/hybrid-ep-fixes/artifacts/raw/metadata-top-k-sweep-001/`.
The protocol uses two input sets per width and eight alternating trials per
input set, with an identical eager control and a distinct-output repeat.

### Top-k results

Each value is the range of the two input-set medians in µs per call.

**CUDA Graph replay**

| Top-k | Original eager | Original compiled | Current Triton | Eight-token Gluon | Lower time vs. current |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 22.10–22.80 | 10.60–10.83 | 5.79–5.85 | 2.73–2.75 | 52.9% |
| 8 | 23.49–23.52 | 11.06–11.07 | 5.95 | 3.81 | 35.9–36.0% |
| 16 | 24.34–24.38 | 11.84 | 8.27 | 7.62 | 7.9% |

**CUDA Graph replay with distinct output pairs**

| Top-k | Original eager | Original compiled | Current Triton | Eight-token Gluon | Lower time vs. current |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 22.79–22.81 | 11.02–11.03 | 5.92–5.93 | 2.85–2.87 | 51.6–51.8% |
| 8 | 23.46 | 11.24–11.25 | 6.03 | 3.87–3.88 | 35.7–35.8% |
| 16 | 24.34 | 12.00–12.02 | 8.37 | 7.77–7.79 | 7.0–7.2% |

**Host to completion, including allocations and the CUDA custom-op wrapper**

| Top-k | Original eager | Original compiled | Current Triton | Eight-token Gluon | Lower time vs. current |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 77.79–107.19 | 55.99–63.41 | 43.64–48.57 | 41.45–46.23 | 4.8–5.0% |
| 8 | 106.63–107.11 | 62.91–63.09 | 48.26–48.50 | 46.11–46.20 | 4.4–4.7% |
| 16 | 107.17–108.24 | 63.38–64.10 | 48.20–48.55 | 46.16–46.82 | 2.9–4.9% |

All four arms matched the eager reference bitwise for the dense map, scores
and trailing sinks at all widths. Negative int64 IDs, zero, signed zero and
NaN scores also passed on the two Triton kernels for top‑4, top‑8 and top‑16.
CUDA activity confirms the Gluon kernel ran on all six cases. Eager emitted
11 kernels and one DtoD copy, Inductor emitted four kernels, and each custom
op emitted one kernel. Per-call kernel/copy time, host enqueue and amortized
synchronization are in the [raw samples](top-k-sweep.json).

Gluon won all 48 paired GPU trials against the current Triton kernel
and 47 of 48 host-to-completion trials. The identical
eager-control replay medians differed by at most 0.41%. The measured
GPU reduction narrows at top‑16. Each kernel has twice the route slots to
scan at top‑16 as at top‑8, while the dense output remains 5 MiB plus sinks.
The results apply to the isolated metadata operation.

A separate GPU check with the generalized kernel passed bitwise reference
parity on both fixtures at all four earlier top‑8 shapes. Its output is in
`size-compatibility.log` beside the top-k sweep logs.

## Gluon launch layout sweep

The sweep varies token rows per CUDA block over 1, 2, 4, 8, 16, 32 and 64
and warps per block over 1, 2, 4 and 8. It tests the 22 combinations with at
least as many rows as warps. Each configuration uses the same Gluon kernel,
output allocation and CUDA custom-op wrapper. Eight rows with four warps is
the original layout control. A second instance of that control measures
repeat variation, and the current GIFT Triton kernel provides a reference.

The 10+10 crop dimensions are 8,192 tokens per rank and 128 experts. The
two inputs for each of top‑4, top‑8 and top‑16 are the same native and derived
fixtures used in the [top-k sweep](#top-k-sweep). The benchmark checks routing
flags, FP32 probabilities, and trailing sinks bitwise for every configuration
before timing. It measures eight alternating trials on each input. Each CUDA
Graph captures either 64 calls with output reuse or 24 calls with distinct
output pairs. Each ordinary call batch records Python enqueue and device
completion separately. CUDA activity for the finalists verifies kernel
activation outside the timed batches. [Layout tuner](tune_routing.py)

Run `metadata-layout-sweep-001` uses H100 GPU 0 on `sr008-gpu` and the
image's default Python. The resolved preflight snapshot, full log and exact
executed source are under
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/hybrid-ep-fixes/artifacts/raw/metadata-layout-sweep-001/`.

The table gives the lowest GPU replay time at each row count with reused
outputs. Each cell averages the two fixture medians and shows the warp count
that achieved it. Times are in µs/call.

| Token rows per block | Top‑4 | Top‑8 | Top‑16 |
| ---: | ---: | ---: | ---: |
| 1 | 5.774 (1 warp) | 5.927 (1 warp) | **6.685 (1 warp)** |
| 2 | 3.345 (2 warps) | 3.834 (1 warp) | 7.230 (2 warps) |
| 4 | **2.631 (4 warps)** | **3.811 (4 warps)** | 7.029 (4 warps) |
| 8 | 2.703 (8 warps) | 3.832 (4 warps) | 7.247 (8 warps) |
| 16 | 2.771 (8 warps) | 4.041 (8 warps) | 8.096 (8 warps) |
| 32 | 3.010 (8 warps) | 4.358 (8 warps) | 9.384 (8 warps) |
| 64 | 4.141 (8 warps) | 5.154 (8 warps) | 10.589 (8 warps) |

[Full sweep samples](layout-sweep.json)

Run `metadata-layout-confirm-001` repeated six layouts on the same GPU in a
fresh process with a changed candidate order and 12 trials per input. The
three sweep winners led on both inputs in both replay modes. Each won all
24 paired trials per replay mode against the original eight-row layout.
The identical controls differed by at most 0.35% across the three top-k
widths and two replay modes, using the averages of the fixture medians.
Confirmed latencies below are in µs/call.

| Top-k | Fastest rows × warps | Current Triton, reused | Original 8 × 4, reused | Fastest, reused | Lower GPU time, reused / distinct | Host to completion, original → fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 × 4 | 5.814 | 2.724 | 2.608 | 4.28% / 3.04% | 43.58 → 43.59 |
| 8 | 4 × 4 | 5.955 | 3.816 | 3.786 | 0.78% / 1.29% | 46.18 → 46.39 |
| 16 | 1 × 1 | 8.275 | 7.627 | 6.639 | 12.95% / 12.73% | 45.91 → 45.82 |

Four rows with four warps had the smallest worst-case GPU latency penalty
among fixed layouts across all three widths in the full sweep, at 5.15% with
reused outputs and 5.03% with distinct outputs. The GPU advantage at native
top‑8 is 0.03–0.05 µs per call in the confirmation. Host-to-completion timing
showed no repeatable improvement for the GPU winners over eight rows. The
sweep measures the isolated metadata operation and leaves training TPS and
total memory unmeasured.

Both runs passed bitwise parity on every timed input. The confirmation also
passed sentinel cases for four rows at all three widths and one row at
top‑16, including negative IDs, signed zeros, NaNs, and a partial block for
the four-row layout. Both runs use the same source hashes, runtime versions,
and fixture receipts. The confirmation log and preflight are in
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/hybrid-ep-fixes/artifacts/raw/metadata-layout-confirm-001/`.
[Confirmation launcher](confirm_layouts.py) · [Confirmation samples](layout-confirm.json)

### Four-row layout in the HybridEP custom op

The `hybrid-ep-fixes` worktree uses the four-row, four-warp layout inside
`four_fusion::routing_metadata_cuda` for contiguous CUDA inputs with 64, 128
or 256 experts. The original Triton kernel still handles strided inputs and
other expert counts within the existing CUDA bounds. The custom op keeps
the same output buffers, fake implementation and gradient registration.
Layout selection reads tensor metadata on the host. The selected path launches
one kernel with no input copy, device-value read, or added synchronization or
collective. The kernel checks the full-width ID range before casting to int32.
[Gluon candidate](candidate/routing_metadata.py) · [Integration patch](gluon-routing.patch)

Run `metadata-gluon-integration-001` tested the worktree module and tests on
SR008 H100 GPU 0 with the image's default Python. The test process prepended
the staged worktree module to the experiment package path. The existing SR008
HybridEP dispatcher and eager reference matched the local file hashes. The
observed native `8192 × 128`, top‑8 call launched one Gluon kernel with grid
`[2048, 1, 1]` and block `[128, 1, 1]`. The integrated path took 3.790 µs per
call versus 3.795 µs for the standalone 4 × 4 prototype under reused-output
CUDA Graph replay, with eight alternating trials. This comparison checks the
device cost of integrating the prototype. The EP8 and EP4/eFSDP2 training
TPS and total memory gates remain open.

The integrated kernel passed bitwise output checks, including both sink
elements, on all 12 native and derived fixtures. The native top‑8 custom op
also passed fullgraph output and weight-gradient parity. Additional checks
covered partial blocks, BF16 weights, wide signed IDs, NaNs and empty top-k.
The focused suite passed 21 cases, including the HybridEP dispatch custom op
with a recording buffer for the strided and grouped paths. The CPU arm of
the native-shape CUDA-only test skipped. The exact source, launch command,
log, profile and timing samples are under
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/hybrid-ep-fixes/artifacts/raw/metadata-gluon-integration-001/`.

## 30+30 acceptance on original A

The `hybrid-ep-fixes` branch points to the original A commit
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`. Its GIFT tree is clean. The
comparison covers baseline, Gluon routing and GPU capacity selection on
EP8/eFSDP1 and EP4/eFSDP2. The established FlashQLA bridge selects the
original eager routing helper for baseline and selector. The routing arms
load the verified Gluon candidate inside the existing custom-op boundary.
The selector arms enable `branchless_capacity_drop`
for the GDN and MLA MoE blocks. The original A tree already implements
this flag with a default of false. Every arm keeps the 128-counter reduction,
fixed capacity 50,000 and the existing group gate.

The reference is the verified A 40+100 run `original-a-vs-cute-140-001-a`
in the SR008 checkout `/workspace-SR008.nfs2/vkropotin/a10-cute-001`.
The launcher checks the GIFT source manifest, the generated YAML, the native
CLI and the complete resolved config against this reference with the specified
changes. All arms use the original FLA and cross-entropy, production streaming
data and tokenizer, prefetch 0, batch 2, sequence length 4,096, global batch
512, compile, selective activation checkpointing, FP8, and seed 2048. Each
rank records the routing implementation it used.
[Launcher](acceptance.sh) · [Run and result validation](acceptance.py)

Each arm runs 30 warmup and 30 measured optimizer updates. Report global
physical input tokens per second and the maximum native GIFT total GPU memory
from all eight ranks over updates 31–60. The launcher parses the existing
logs offline. The profiler and detailed monitoring are disabled. All arms
reuse A's compiler cache at
`runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2/cache`,
including its per-rank Triton directories. New specializations can compile
during warmup. The observer records compiler activity at the window boundaries.

Campaign `acceptance-a-30x30-001` started native training on the existing
eight-H100 allocation at 15:06 UTC on 2026-11-20. EP8 baseline and EP4 with
both candidates passed dry preflight. The campaign driver and status live
under `runs/hybrid-ep-fixes/artifacts/raw/acceptance-a-30x30-001/` in
the SR008 original A checkout. Native logs and receipts use the
`acceptance-a-30x30-001-ep<degree>-<arm>` names under
`runs/production-data-abcd/artifacts/raw/`.

| EP / eFSDP | Arm | Global TPS | Change vs. baseline | Total GPU memory, GiB | Change, GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 / 1 | Baseline | 144,480.35 | — | 72.27 | — |
| 8 / 1 | Gluon routing | 144,364.14 | −0.08% | 72.33 | +0.06 |
| 8 / 1 | GPU capacity selector | 145,565.37 | +0.75% | 72.58 | +0.31 |
| 4 / 2 | Baseline | 144,813.13 | — | 73.67 | — |
| 4 / 2 | Gluon routing | 144,948.49 | +0.09% | 73.77 | +0.10 |
| 4 / 2 | GPU capacity selector | 145,369.61 | +0.38% | 73.86 | +0.19 |

All six arms passed their source, configuration and routing activation
checks on the same eight GPUs and shared compiler cache. Every measured
window covered 62,914,560 physical input tokens. The compiler observer
reported `observed_stable` in its covered scope for each arm. The two
combined arms were skipped as requested. The sixth arm passed at 16:51 UTC
and all GPU compute processes had exited. The campaign's `matrix.tsv`,
`stop-after-six.log` and `six-arm-exit-code.txt` record the stop. The original
scheduler's exit code 143 records its intentional termination after EP8
selector finished training.

The GPU selector increased window TPS by 0.75% on EP8 and 0.38% on EP4,
with total memory increases of 0.31 and 0.19 GiB. Gluon routing changed
TPS by −0.08% on EP8 and +0.09% on EP4, with total memory increases of
0.06 and 0.10 GiB. Each row is one run. The size and repeatability of
these deltas remain uncertain.

Across all 60 updates, the largest absolute candidate-to-baseline
differences in rank-0 console metrics were 0.00057 in loss and 0.0612 in
gradient norm. The historical A reference and rerun EP8 baseline differed
by up to 0.00052 in loss and 0.0245 in gradient norm over the overlapping
updates. Full-training deterministic parity remains unverified.
[Routing metadata component parity](#four-row-layout-in-the-hybridep-custom-op)

### Superseded promoted-source campaign

Campaign `acceptance-30x30-001` used the promoted production recipe at
`2c67c182e`. Its EP8 baseline completed at 166,182.79 TPS and 69.14 GiB
total GPU memory. EP4 stopped at startup because that recipe enabled the
EP8-only `gfsdp_direct_expert_dw` option. The EP8 routing arm was stopped
when the user requested the original A comparison. Its evidence remains at
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/hybrid-ep-fixes/artifacts/raw/acceptance-30x30-001/`.
