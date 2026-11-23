# Bounded EP4 expert trace

The trace records the first three complete microbatches of update 21 on global
ranks 0 and 4. This includes the first use of updated expert weights and two
subsequent uses. The native adapter validates all 32 training microbatches and
the optimizer step on all eight ranks. Only the selected ranks collect and
export the trace. The run preserves the original A recipe, including the
learning-rate schedule, accumulation, precision, compile, SAC, and EMA.

The sparse mesh orders the eFSDP dimension before the four EP lanes. The
runtime checks every expert GFSDP group's actual membership before training and
verifies the resolved profiler ranks. A mismatch aborts the diagnostic.

## Plan

From the candidate task root, use Python 3.12 or later with the existing
PyYAML installation. This command applies the proposed patch in a temporary
tree and calls the live runner parsers without starting training.

```sh
python -B runs/efsdp-optimizations/trace/verify_plan.py
```

The test compares the traced and ordinary EP4 plans. Native training overrides
differ only in the planned update count and profiler settings. The ordinary
benchmark still runs 10 warmup and 10 measured updates. The diagnostic uses
20 warmup updates and disables timing. The CPU regression uses the actual
expert rank entry point, FlashQLA main and native measurement adapter with a
stub runtime. It constructs the inherited `WindowMeasurement`, completes
32 synthetic training microbatches and one optimizer call on each tested
profiled rank, and checks that only ranks 0 and 4 collect three microbatches.
Rank 1 also writes a native receipt, and the unprofiled control has no capture
receipt. The profiler rank list is adapter metadata and stays out of the
measurement constructor. A separate CPU fixture passed native receipt
validation for ranks 0 and 4, with rejection of both a wrong-rank trace and a
fourth microbatch. The native EP4 capture then completed update 21 on eight
H100s and recorded three microbatches on ranks 0 and 4.
[Native trace extraction](../artifacts/raw/efg-n22-04-ep4-expert-trace2/extracted.json)

On the bounded trace path, each selected rank enables Kineto CUDA sync
activities immediately before entering native training. The setting stays
active through the profiler's scheduled warmup and collection, then resets
when the call returns or raises. Ordinary measurement and the other six ranks
keep the usual profiler state. The CPU regression verifies this order with a
fake setter through the rank entry point. The pinned Torch nightly supports the
setter and uses it to select Kineto `CUDA_SYNC` activities.
[Torch setter test](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/test/profiler/test_profiler.py#L351)
· [Torch Kineto configuration](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/csrc/profiler/kineto_shim.cpp#L403)

## Native capture

Apply the proposed hook only on the intended idle training tree. The runner
hashes the modified native scripts and the bounded trace helper.

```sh
git apply --check runs/efsdp-optimizations/trace/native-profile.patch
git apply runs/efsdp-optimizations/trace/native-profile.patch

python -B runs/efsdp-optimizations/run.py run "$trace_id" \
  --ep 4 --trace-expert-pair --reference "$reference" \
  --optimization bf16-cache --optimization fp8-cache \
  --optimization direct-dw --optimization tail-retention \
  --direct-dw-build "$verified_build"
```

Use the approved eight-H100 allocation, the verified image interpreter, a fresh
attempt ID, and the completed original A reference. `plan` with the same
arguments prints the native argv without training. The runner creates its
inherited raw output root before launching the native bridge. After capture,
read `summary.json` under `runs/overnight-streaming-perf/artifacts/raw/<trace-id>`
for native validation, trace hashes and rank receipts. The trace files are
`native/profiling/traces/iteration_21/rank0_trace.json` and
`native/profiling/traces/iteration_21/rank4_trace.json` under that attempt root.
Compare the `GFSDP::fp8_expert_refresh` events inside each complete microbatch
scope and correlate gather, reduce-scatter, release waits and consumer work on
both ranks. Measure throughput and total GPU memory in a separate unprofiled
10+10 run.

## Extract the dependency evidence

Run the extractor on the host with the raw artifacts. It reads one rank trace
at a time, validates its size and SHA256 against the native receipt, and prints
JSON. Three-microbatch original A traces were about 27.7 MB per rank. A full
32-microbatch trace historically reached 301.2 MB per rank. Keep the originals
in the run directory and use the compact extraction on the review workstation.

```sh
python -B runs/efsdp-optimizations/trace/analyze_ep4.py "$attempt"
python -B runs/efsdp-optimizations/trace/test_analyze_ep4.py
```

- `expert_collective_pairs` matches exact expert group membership, sequence,
  dtype, operation and element counts across the two ranks. CUDA correlations
  are rank-local. Kernel attribution uses `ac2g` or a unique compatible
  correlation. CPU microbatch markers classify the launch, so device work can
  finish outside its host scope. The output carries unpaired operations and
  missing launch or metadata coverage.
- `gfsdp_stream_waits` supplies matching device sync and host event-record
  evidence when the runtime exports it. Compare the slot wait with the previous
  FP32 reduce-scatter and the shard accumulation that follows it. Then check
  both weight-gather planes, compute-stream predecessors and first dgrad or
  row-to-column work. An advance is plausible only if the slot release is the
  last prerequisite and independent work is ready. The backward windows list
  nearby launches and do not assign those launches to a layer.
- `queued_behind_expert_rs` lists cases where a same-stream expert reduce-scatter
  outlasts the next gather's host launch. To establish a useful overlap
  opportunity, also check the gather's receive-buffer and cache readiness,
  consumer wait, and competing dense or token-exchange work. Count an exposed
  interval once. CPU enqueue duration and kernel adjacency cannot establish it.
- `direct_dw.owner_windows` includes both exact writer custom-op calls, the
  CUDA waits inside those calls, and matched device work. It lists compiled
  weight transpose and row-to-column kernels, and identifies dgrad candidates.
  The GFSDP host bracket groups calls by owner. The stream and event fields
  carry the separate device evidence and any missing producer records.

The original A runtime exported CUDA wait and record calls in the host API
trace but no `cuda_sync` or `cuda_event` rows. The sync opt-in can recover the
record-to-wait link. Device event timestamps depend on the actual Kineto,
CUPTI and driver support. When absent, retain the record link and bound timing
with same-stream kernels and accumulation. Keep the event timing unresolved
when those bounds overlap competing prerequisites. If a second diagnostic is
needed, add CPU scopes around the existing direct dW slot release record and
slot-reuse wait, plus the prepared-weight consumer wait if that edge remains
ambiguous. Add no waits or synchronizations for instrumentation.
[Pinned Kineto linkage](https://github.com/pytorch/kineto/blob/4f8caaae3f109b63767e75bbb7fae7857e1d3e13/libkineto/src/CuptiActivityProfiler.cpp#L436)
