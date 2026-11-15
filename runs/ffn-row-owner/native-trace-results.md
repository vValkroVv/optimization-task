# Native combined FFN and row-owner trace

The combined mechanism passes on the two profiled ranks: both early FFN reductions overlap parent backward, and embedding table transports are replaced by routed owner exchanges. Exact event-wait causality and a full-update critical path remain unavailable.

## Capture and import

The eight-rank diagnostic completed 21 updates with 32 microbatches and optimizer work per update. Only update 21, three complete consecutive F+B microbatches on ranks 0/1, was captured. FP8, selective SAC, model/loss compilation, MTP1 and the agreed crop remain enabled. No new GPU or numerical work was performed.

Resolved flags are supported by the same-source all-eight-rank [native inventory](native-memory-results.json), actual child GFSDP collectives and `_OwnerLookup` execution. The final owner source SHA256 is `f3a01c6cf99b981db7b4ef66866fe46259af84e9d2c21e2044e9ae18ff1696fd`. Missing setup INFO text is not used as activation proof.

Canonical traces were loaded first through MCP. Raw imports omitted 47/37 overlapping kernels; the pinned display-lane normalizer recovered all **8,230/8,173 kernels**, **202 copies** and **885 memsets** per rank. All **9,317/9,260** device endpoints have exactly one matching launch/correlation, with zero importer errors. Exact names, times, durations and arguments match canonical events; only display tids and corresponding flow endpoints change, and GPU annotation ranges are omitted. Canonical files remain unchanged.

The [JSON](native-trace-results.json) records canonical/derivative paths and hashes, trace handles, SQL, source/config identities, six rank/microbatch cases and actual slices. Ignored MCP exports and task-specific reconstruction scripts are under `artifacts/analysis/native-perfetto/`.

## Early FFN reduction

Every captured case has actual dInput and all three weight-gradient endpoints before FP32 foreach-copy packing, followed by exactly one **513 MiB FP32** FFN ReduceScatter before the corresponding parent ReduceScatter. CPU/FQN ancestors → runtime/driver launch → actual flow endpoint establish ownership. The three TNT GEMMs follow the accepted generated-code mapping w2/w3/w1; `triton_poi_fused_add_3` completes dInput.

| Rank | MB | layer0 RS / compute overlap, ms | MTP RS / compute overlap, ms |
|---:|---:|---:|---:|
| 0 | 0 | 1.571044 / 1.514531 | 1.761189 / 1.719587 |
| 0 | 1 | 1.646179 / 1.589604 | 1.624355 / 1.590817 |
| 0 | 2 | 1.568388 / 1.511237 | 1.530052 / 1.504421 |
| 1 | 0 | 1.681736 / 1.625640 | 1.877481 / 1.832872 |
| 1 | 1 | 1.687496 / 1.625897 | 1.840329 / 1.800711 |
| 1 | 2 | 1.643080 / 1.580615 | 2.279340 / 2.231915 |

Clipped interval unions cover **96.20–98.32%** of FFN RS with actual remaining-parent compute across all 12 cases. Copies, memsets and communication are excluded from this compute intersection. Device compute/packing uses stream 7 and NCCL uses stream 24. Parent FP32 payloads remain 259.188232 MiB for layer0 and 380.673828 MiB for MTP; FFN input/output are 537,919,488 / 67,239,936 bytes per rank. NCCL spans can contain peer waits; these overlaps establish mechanism, not causal savings.

Each selected parent/FFN group executes six BF16 all-gathers per rank: forward and backward/recompute for each of three microbatches, with recorded forward/backward prefetch scopes. Child reduction precedes parent reduction on the shared communication stream. The [native inventory](native-memory-results.json) confirms aligned adjacent nonoverlapping regions, original alternating slots and preserved private head storage; the trace introduces no buffer-allocation claim.

## Owner path and retained costs

There is exactly one `_OwnerLookup` and `_OwnerLookupBackward` per microbatch. The [decoder](../../gift/torchtitan/models/common/decoder_blocks/decoder.py) performs one lookup, carries the same embeddings through trunk and MTP, and invokes ordinary autograd once after shared-consumer accumulation. The profile shows the once-only path and relevant trunk/MTP scopes; tensor operands and separate derivative contributions are not recorded. Existing equivalence tests own exact shared-consumer derivatives.

| Rank | MB | Unique requests / owned incoming rows | BF16 value send / receive bytes |
|---:|---:|---:|---:|
| 0 | 0 | 75 / 587 | 4,808,704 / 614,400 |
| 0 | 1 | 72 / 589 | 4,825,088 / 589,824 |
| 0 | 2 | 78 / 584 | 4,784,128 / 638,976 |
| 1 | 0 | 78 / 0 | 0 / 638,976 |
| 1 | 1 | 77 / 0 | 0 / 630,784 |
| 1 | 2 | 76 / 0 | 0 / 622,592 |

Each case has one signed-int64 count AllGather (72-byte input / 576-byte output) and exactly three SendRecv kernels, attributed to request IDs, BF16 values and BF16 source gradients. Value traffic reverses for gradients. These are logical payloads, not physical wire bytes. Traffic is concentrated on rank 0; rank 1 owns no incoming rows in these captures. Existing component cases cover broader distributions.

Native `compute_grad_weight<c10::BFloat16, long>` and `sum_and_scatter<c10::BFloat16, long>` remain in backward, followed by unique-row gather, gradient exchange and per-source BF16→FP32 copies / FP32 indexed SUM on owner rank 0. Rank 1 has no indexed owner SUM because it owns no requests. Standard autograd receives the owned 250.5 MiB FP32 local gradient; source confirms that returned gradient is not borrowed scratch.

Native unique/sort, one pinned DtoH from dynamic-output unique, count-matrix pageable DtoH and two `cudaStreamSynchronize` calls remain per forward. Three memcpy API calls include these two DtoH operations and one DtoD copy. Their profiled host spans are retained separately in JSON and are not unprofiled CPU-cost or bandwidth estimates. Native BF16 dense source-coalescing scratch remains.

No device collective has a `tok_embeddings` GFSDP ancestor. The [current control](control-trace.json) had six embedding table gathers and three reductions per rank over three microbatches. The candidate retains **three full-table BF16 gathers and three FP32 reductions per rank**, all attributed to unchanged untied `lm_head`: 125.25→1002 MiB gathers and 2004→250.5 MiB reductions. Large collectives were not indiscriminately counted as removed.

## Natural head borrowing and reuse

These native microbatches have no component-harness per-MB device synchronization. The observable scratch lifecycle is consistent with the source borrow/release contract:

| Rank | Boundary | Prior head RS end → next owner first scratch write, ms | Prior owner last scratch consumer → next head first pack, ms |
|---:|---|---:|---:|
| 0 | 0->1 | 233.422575 | 119.266942 |
| 0 | 1->2 | 236.354039 | 119.070685 |
| 1 | 0->1 | 233.413399 | 118.283219 |
| 1 | 1->2 | 236.342822 | 119.140216 |

On rank 0 the forward first scratch write is owner `index_select`; rank 1 has no selected owned rows, so its first observed scratch write is the value-exchange receive. Forward reconstruction finishes before same-MB head packing in all six cases. Backward last scratch consumers are the final FP32 indexed SUM on rank 0 and gradient SendRecv on rank 1; all four observed next-head reuses occur later. Slice IDs and endpoints are retained in JSON.

`GFSDPBufferContext.borrow` waits overlapping release events and records its release; `_scratch` selects existing private head storage, and `post_backward` waits before packing. Event-record/wait API scopes exist, but handles are absent (only External id, correlation and cbid fields are available). Temporal separation does not prove which event each wait consumes. [Existing delayed-collective tests](buffer-regions.md) and [owner storage checks](owner-storage.md) remain the lifecycle-safety basis; no stronger causal claim is made.

## Limits

The profile covers two ranks and three of 32 microbatches, excluding optimizer device execution and the full update critical path. Eight-rank readiness evidence remains owned by the component checks. This diagnostic establishes native combined scheduling and transport mechanisms; it does not establish throughput magnitude, contention-free communication, convergence or long-run behavior. No further numerical requirement is introduced. Unprofiled timing and memory conclusions belong to their result records.
