# Dense GFSDP component verification

The original A control uses `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`, native FLA, original routing and CE, and no CPU materialized-batch prefetch. The two FFN splits and standard-autograd row-owner embedding already exist at this revision as independent default-off controls. Enable each feature in its own pair. Compare the transition with MTP FFN split active in both arms. The later original-A receipt is `runs/fusion-followup/results/original-a-vs-production.json` at revision `529d8f27d`.

## Inputs and execution scope

Reuse the [native `data-002` capture](../production-data-abcd/component-results.json) if its eight per-rank manifests and tensor hashes are available. The capture contains signed int64 token IDs and the actual hidden states, output gradients, positions and masks for `layers.0` and MTP, with the unshifted shared input embedding for MTP. It was acquired from C at its first microbatch. Replay therefore uses the same real operands in both arms under A's selected block implementation. Component weights are initialized identically by the harness with seed 2048.

If the comparison requires activations and gradients produced by A, the existing native bridge can capture them in a separate A diagnostic. Supply `ABCD_COMPONENT_FIXTURE_DIR` inside `runs/production-data-abcd` in the checkout and run exactly one update with warmup and timing window zero, profiling disabled and accumulation 32. The hook writes the first forward/backward after hashing its input provenance and removes itself before the remaining microbatches. Its output passes directly to `--input-fixtures`. [Capture implementation](../production-data-abcd/component_fixtures.py)

Run the commands from this worktree on the authorized eight-H100 allocation. Put the worktree `gift` and pinned FLA dependencies on `PYTHONPATH`. Set `CAP` to the verified capture directory and use a fresh personal `OUT` path per arm. Reuse the task's reviewed launch and logging wrapper.

### Layer-0 FFN and MTP FFN

Run this command for each of the four `(CMP, VARIANT)` pairs: `layer0 control`, `layer0 split`, `mtp control`, and `mtp split`.

```sh
/opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run --standalone --nproc_per_node=8 \
  runs/dense-gfsdp-fix/ffn_component.py --component "$CMP" \
  --mode benchmark --variant "$VARIANT" --input-fixtures "$CAP" \
  --warmup 10 --samples 10 --output "$OUT"
```

The adapter selects the native FLA implementation used by A for `layers.0`, keeps BF16 expert reuse and loader prefetch disabled, and delegates to the existing harness. The historical `external_gate` receipt key contains the actual backend identity, with `variant` set to `fla`. The crop MTP block uses MLA. Both pairs retain the original per-block SAC policy, fullgraph compilation, native FP8 linear modules and eight-rank dense GFSDP. Model loss, data loading and optimizer are outside the benchmark.

Verify the resolved toggles, compiled FFN FQNs, same fixture hashes, and existing allocator storage. Compute the median of ten per-sample maxima across eight ranks for synchronized complete block forward/SAC replay/backward and GFSDP including gradient zeroing. Use the existing `--mode check --accumulation 32` for a focused A/A/B deterministic-input state replay if needed. It uses seeded tensors to cover both input gradients for MTP, elementwise block outputs and gradients, clipped AdamW updates and moment state. Treat split versus control errors against the matched A/A errors and report unresolved differences. [FFN harness](../ffn-row-owner/ffn-component-harness.md)

### Row-owner input embedding

Run this command for `VARIANT=baseline` and `VARIANT=owner`.

```sh
/opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run --standalone --nproc_per_node=8 \
  runs/ffn-row-owner/components.py --execute --variant "$VARIANT" \
  --scope integrated --cases fixture --input-fixtures "$CAP" \
  --warmup 10 --samples 10 --output "$OUT"
```

Both arms keep the existing private `lm_head` gradient bucket resident. The replay covers lookup and the completed dense local FP32 gradient. It uses actual token IDs with the same deterministic upstream in both arms. Report synchronized wall, joined CUDA-event span, host-call span and end synchronization separately. Confirm the borrow aliases that existing bucket in the owner arm and interpret the payload bytes as logical tensor lengths.

For exact full-shape output and first/accumulated/reset gradient comparisons, use the existing `runs/ffn-row-owner/embedding_numeric.py --execute --fixture "$MDS_TRAIN" --seed 2048 --output "$OUT/embedding-numeric.json"` with the same eight-rank launch. `MDS_TRAIN` is the `data-002/train` directory. The replay selects raw dataset rows `2 × rank` and `2 × rank + 1` per rank in both arms. The historical exact-output/gradient check used an older fixture and the same owner source SHA-256. [Input format](../production-data-abcd/prepare_data.py) · [Numerical check](../ffn-row-owner/embedding-numeric-results.md)

### MTP transition

The isolated MTP component removes all trunk units, the final norm and `lm_head`, which are required by the transition. Use matched native `mtp-only` and `mtp+transition` diagnostic captures, with MTP split on and layer-0 split and row-owner off in both. Keep the same compiled/SAC recipe, input, source patch, update and capture ranks. Use the existing three-microbatch bounded profile. The isolated timing comparison of the transition is the same toggle pair in the native unprofiled campaign.

Identify collectives by FQN, launch flow, NCCL family, group, dtype and element counts. Compare head pre-backward launches and MTP parent/FFN gather readiness relative to head reduce-scatter, FFN reduce-scatter completion relative to entry into last-trunk dense gather, norm packing relative to MTP-parent reduce-scatter, and actual last-trunk compiled kernels relative to the parent reduce-scatter. Verify the two region remaps and unchanged deduplicated communication pool bytes in the initialized-group inventory and focused buffer tests. Use [the diagnostic reader](analyze_diagnostic.py) with the matching capture receipts. The prior experiment enabled both FFN splits and row-owner embeddings in both arms. It found earlier gathers and norm packing, with zero overlap between MTP-parent reduce-scatter and last-trunk compute. Its method and findings are in `runs/schedule-embedding/trace-analysis/comparison-004.md` at revision `529d8f27d`.

## Memory and evidence

Component allocator peaks cover only the selected block or embedding factory and its resident buffers. Use deduplicated storage bytes for the pool invariant. The native campaign reports GIFT's total GPU memory estimate, taking the largest logged value across all measurement updates and eight ranks. Use the component results to explain storage behavior and the native result for training memory and throughput.
