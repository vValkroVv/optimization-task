# FlashQLA for GatedDeltaNet

**Status:** The bounded task-copy block, native, and trace comparisons are complete. On 14 November 2026 the owner selected `external_gate` as the explicit backend in the current final task baseline; this does not change a generic backend default or establish a general throughput gain or production validation. The [baseline decision](../baseline.md#current-final-task-baseline) owns the selected recipe. The [integrated report](../../runs/flash-qla/report.md) summarizes the comparison and limits; detailed source, block, native, and trace records follow. The pinned public H200 results remain planning context, not local evidence.

## Local path and workload

The current measured call path is `GatedDeltaNet.forward` → `torchtitan::gdn_chunk` → FLA `chunk_gated_delta_rule`, using FLA's Triton implementation for this input contract. The model constructs Q/K/V, computes sigmoid beta, and passes the raw decay projection plus `A_log` and `dt_bias` to the opaque custom op. That op calls FLA with `use_gate_in_kernel=True`, `use_qk_l2norm_in_kernel=False`, and `output_final_state=False`; its custom autograd path replays the FLA call during VJP. See [`gdn.py`](../../gift/torchtitan/models/common/attention/linear/gdn.py#L76) and [`fla_custom_ops.py`](../../gift/torchtitan/models/common/attention/linear/fla_custom_ops.py#L177).

An FLA FlashQLA backend adapter is present in the inspected snapshot, but whether its separate package is installed in the captured environment is not established. The adapter's verifier rejects `use_gate_in_kernel=True`, so the existing call does not select FlashQLA. Swapping the backend name or enabling its registry entry cannot remove this input-contract mismatch. The source marks the FLA snapshot as version 0.5.2 and records the vendored commit as `1d2281ff1b7988d8070f00dba5ffa9747ae3a60e` in [provenance](../../references/provenance.json#L20); see also [the version declaration](../../gift/contrib/flash-linear-attention/fla/__init__.py#L8). The rejecting adapter is at [`flash_qla.py`](../../gift/contrib/flash-linear-attention/fla/ops/gated_delta_rule/backends/flash_qla.py#L59).

The task checkout inspected for this note is at `6c6cd9c0e41a0a5d3d00e1e252b3c120052d3507`. The V2 capture records execution from `3b6b8928ec4b15a529d07ee99e672f2fc75ef73a` plus a separate dirty-snapshot digest. A no-diff comparison of the task-copy GDN wrapper, custom op, FLA chunk/gate files, and FlashQLA adapter was made between current HEAD and the preserved V2 revision `21c1891dda531a1222e6c350802f2765758794f9`. Those file-level matches do not establish the identity of the full dirty execution snapshot. The capture identity is recorded in the [screening JSON](../../runs/v2-prefetch-baseline/v2-candidate-screening-perfetto.json#L5) and the preserved V2 revision in the [source manifest](../../runs/v2-prefetch-baseline/source-manifest.json#L1).

The crop uses batch 2 and sequence length 4096. Its GDN dimensions are Q/K `[2, 4096, 16, 128]`, V `[2, 4096, 32, 128]`, and decay/beta `[2, 4096, 32]`, giving GVA 2. The [task config](../../gift/tests/unit_tests/golden_configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml#L4) sets 16 key heads, 32 value heads, and 128-wide K/V heads; the capture records batch and sequence dimensions. Q/K normalization is upstream of the scan, so the in-kernel normalization flag stays false. CP is 1; the custom op supplies no initial state and does not request a final state.

## Completed task-copy comparisons

The bounded experiment evaluated two opt-in task-copy variants against FLA at the actual `B=2, S=4096, Hq=16, Hv=32, K/V=128` shape. `external_gate` uses the existing FLA gate with pinned FlashQLA; `fused_gate` uses FLA gate/cumsum plus pinned FlashQLA orchestration. The [block report](../../runs/flash-qla/block-results.md) and [machine results](../../runs/flash-qla/block-results.json) record six captured VJP fixtures, eager/compiled checks, full-sequence numerical comparisons, timing scopes, and retained attempt history.

Across the six fixtures, output relative RMS versus FLA was 0.5095% median (0.4408–0.5184%). The largest parameter-gradient relative RMS was about 0.757% for `A_log` and 0.727% for `dt_bias`. The FP64 recurrence diagnostic covers only a zero-state 64-token prefix; it is not a full-sequence oracle. Synthetic CUDA-graph `3F+B` ratios were 1.903× for `external_gate` and 1.915× for `fused_gate`; this is component evidence, not production step speed or evidence that B is superior.

The [native report](../../runs/flash-qla/native-results.md) records all three 140-update eight-rank runs with matching resolved configurations and zero allocator retry/OOM deltas. In fixed order FLA → external → fused, global physical-input throughput was 156,452.36, 160,090.72, and 159,735.21 tokens/s. Each single window also shortened about 10.4% from its first to last 20 cadence intervals. Without repeated or alternating controls, that drift prevents attributing the roughly 2% between-run differences to the backend; there is no established general TPS gain or fused-gate advantage. The FLA native control includes TileLang `common.chunk_bwd_dqkwg`, so it is not an all-Triton baseline.

The new [trace analysis](../../runs/flash-qla/trace-results.md) compares two new rank-0/1 captures with the accepted control. All four new MCP imports passed with exact event/flow parity and zero import issues; [capture provenance and retained harness history](../../runs/flash-qla/trace-capture-results.md) records runtime and raw/gzip validation. Each rank records 18 calls in each of original forward, SAC forward, VJP recompute forward, and native backward, for 288 matched calls across the four new captures. The whole-GDN 3F+B work is 59.2–59.7 ms for control and 30.9–31.4 ms for the variants, a 1.90–1.92× work ratio. That removes 2.25–2.28% of the matched three-microbatch device span; it is not an optimizer critical-path or TPS estimate. The traces show separate FLA gate work in `external_gate` and gate/cumsum fusion before QLA solve in `fused_gate`, while preserving QLA's fused GDR work.

## What the trace measures

Across the first three microbatches, each of six GDN layers has three forward scan calls and one backward scan call: the original forward, SAC replay, VJP forward reconstruction, and native backward. This gives 54 forward calls and 18 backward calls in the trace. The VJP reconstruction contributes 8.868 ms already included in the forward total; FLA backward also recomputes `w/u` and `h` for 4.063 ms. The [V2 screening record](../../runs/v2-prefetch-baseline/v2-candidate-screening-perfetto.json#L52) retains these values and the call disposition.

| Trace quantity | Value | Scope |
| --- | ---: | --- |
| FLA scan forward kernels | 30.245 ms | Includes the 8.868 ms VJP-forward reconstruction |
| FLA scan backward kernels | 28.734 ms | Includes internal `w/u` and `h` reconstruction |
| Forward plus backward kernel sum | 58.979 ms | Opportunity anchor, not proven removable latency |
| Integrated three-microbatch device span | 1293.713 ms | Rank 0, update 21, accumulation 32 |

The capture is a rank-0 prefix, not a full optimizer update, whole-step optimizer tail, all-rank view, or post-GFSDP profile. Its kernel sums can overlap work on other streams. They do not establish candidate gain or a directly removable step-time budget; the [capture limits](../../runs/v2-prefetch-baseline/v2-candidate-screening-perfetto.json#L120) spell out the missing scope. A FlashQLA backend swap also does not, by itself, remove either SAC replay or the separate `fla_vjp` forward reconstruction.

## Pinned implementation and public benchmark

The inspected [QwenLM/FlashQLA revision](https://github.com/QwenLM/FlashQLA/tree/a14dbf3474514e1674526431e960325f72081e9f) provides SM90/SM100 forward and backward paths for BF16/FP16 inputs, FP32 state accumulation, and K/V head dimensions of 128. Its GVA layout supports distinct Q/K and V head counts. The [pinned README](https://github.com/QwenLM/FlashQLA/blob/a14dbf3474514e1674526431e960325f72081e9f/README.md#L13) describes the TileLang/warpgroup-specialized implementation. The pinned chunk implementation is [`flash_qla/ops/gated_delta_rule/chunk/__init__.py`](https://github.com/QwenLM/FlashQLA/blob/a14dbf3474514e1674526431e960325f72081e9f/flash_qla/ops/gated_delta_rule/chunk/__init__.py); its dense/packed handling and automatic context-parallel routing are in [lines 39–57](https://github.com/QwenLM/FlashQLA/blob/a14dbf3474514e1674526431e960325f72081e9f/flash_qla/ops/gated_delta_rule/chunk/__init__.py#L39-L57). The forward path performs gate scan, KKT construction/solve, and fused state/output work; backward reconstructs state, computes fused gradients, reduces Q/K gradients across GVA heads, and reverses the gate scan.

The public [H200 result table](https://github.com/QwenLM/FlashQLA/blob/a14dbf3474514e1674526431e960325f72081e9f/benchmark/benchmark_results_H200.txt#L1) is a useful Hopper planning proxy, as accepted for this analysis. It reports CUDA Graph timings with 10 warmups and 100 repetitions, PyTorch 2.8 cu128, FLA 0.5.1, and TileLang 0.1.12. This differs from the local FLA 0.5.2 snapshot; the README's stated 0.5.0 version is stale relative to the benchmark header.

| Public case | FLA | FlashQLA | Reported ratio | Comparison note |
| --- | ---: | ---: | ---: | --- |
| H200 forward, Q/K heads 16, V heads 32, K/V 128, T=8192 | 0.524 ms | 0.206 ms | 2.55× | Two packed sequences of length 4096 |
| H200 forward, same heads and dimensions, T=8192 | 0.520 ms | 0.289 ms | 1.80× | One sequence of length 8192 |
| H200 forward, same heads and dimensions, T=4096 | 0.269 ms | 0.151 ms | 1.79× | One sequence of length 4096 |
| H200 backward, nearest T=8192, Q/K and V heads 32, K/V 128 | 1.505 ms | 0.655 ms | 2.30× | Equal-head case, unlike local GVA 16/32 |

Forward rows are in the [pinned H200 table](https://github.com/QwenLM/FlashQLA/blob/a14dbf3474514e1674526431e960325f72081e9f/benchmark/benchmark_results_H200.txt#L93); the closest backward row is in [the same table](https://github.com/QwenLM/FlashQLA/blob/a14dbf3474514e1674526431e960325f72081e9f/benchmark/benchmark_results_H200.txt#L196). These are complete low-level multi-kernel calls captured under CUDA Graphs, not standalone-kernel or host-to-step measurements. The public benchmark's [call setup](https://github.com/QwenLM/FlashQLA/blob/a14dbf3474514e1674526431e960325f72081e9f/benchmark/bench_gated_delta_rule.py#L185) measures forward with `is_train=False`. Input preparation, backward's preparatory forward, and context/cache creation are outside the timed region; internal forward/backward work remains part of each measured call.

The two-sequence public fixture is packed B=1 with boundaries `[0, 4096, 8192]`, not the local dense B=2 batch. The benchmark allows automatic CP, but its selected degree is not recorded; the local dense B=2 input does not trigger that automatic CP path. Public inputs use BF16 Q/K/V but FP32 log-gate/beta, a random initial state, and final-state outputs/gradients. Normalization, gate construction, and beta sigmoid are outside the timed call. Locally the gate is learned and beta is the model's sigmoid output, with no initial state and no final-state output. These differences make H200 a useful scenario, not a parity result for the local FLA 0.5.2 path.

## Conditional savings arithmetic

The following arithmetic assumes every saved scan-kernel millisecond shortens the measured 1293.713 ms elapsed window and that the rest of that window stays unchanged. These are conditional elapsed-window scenarios, not optimizer-step estimates or measured speedups. Throughput change uses `T / (T - saved_ms) - 1`, while elapsed-window reduction uses `saved_ms / T`.

| Scenario | Saved scan time | Conditional throughput change | Conditional elapsed-window reduction |
| --- | ---: | ---: | ---: |
| 1.5× faster forward + backward scans | 19.66 ms | +1.54% | 1.52% |
| 2× faster forward + backward scans | 29.49 ms | +2.33% | 2.28% |
| Public ratios: 2.55× forward, 2.30× backward | ≈34.6 ms | +2.75% | 2.67% |
| Remove the complete measured scan-kernel sum | 58.979 ms | +4.78% | 4.56% |

The public-ratio case weights the measured forward and backward sums separately; it does not assume an unweighted one-forward/one-backward workload. Real step gains can be lower when scan work is overlapped with communication or other compute, when gate preparation adds work, or when a different dependency becomes critical.

## Gate adaptation and integration boundary

The FlashQLA adapter rejects `use_gate_in_kernel=True`, while the current GDN wrapper provides raw projection `a`, `A_log`, and `dt_bias`. The current FLA gate is `log_gate = -exp(A_log) * softplus(a + dt_bias)`. A high-level starting point already exists as `fused_gdn_gate(a, A_log, dt_bias, output_dtype=torch.float32)`, which returns ordinary, unscanned log-gate values and has autograd support. It uses the same FLA `gdn_gate_bwd` helper as the current chunk path.

Do not precompute the cumulative gate or multiply by `RCP_LN2` at this high-level adapter. FlashQLA owns the scan representation it consumes; its natural fusion point is the pinned [FlashQLA cumsum utility](https://github.com/QwenLM/FlashQLA/blob/a14dbf3474514e1674526431e960325f72081e9f/flash_qla/ops/utils/cumsum.py#L46).

| Gate path | Change size | Cost and use |
| --- | --- | --- |
| PyTorch formula | Small prototype | Quick shape/value check; adds ordinary elementwise ops and uses PyTorch's own rounding/softplus behavior. |
| Existing `fused_gdn_gate` helper | Smallest integration candidate | One extra gate kernel and a roughly 1 MiB FP32 `[2, 4096, 32]` output intermediate, with write/read traffic; reuses existing FLA gate backward. |
| Gate plus cumsum preprocessing | Moderate kernel/API change | Can remove the standalone gate pass and intermediate if it directly produces the representation FlashQLA expects; the main QLA matrix/state scan remains. |

The first integration boundary should be inside both the opaque `gdn_chunk_op` forward and its VJP recompute callable. Keep the custom-op input schema, output, gate gradients, SAC behavior, and compile boundary intact. Calling `fused_gdn_gate` from the outer model path risks a graph break because the helper is marked `torch.compiler.disable`; keeping it inside the custom op preserves the existing boundary. The source locations are the [op implementation and replay](../../gift/torchtitan/models/common/attention/linear/fla_custom_ops.py#L177) and the [helper/autograd wrapper](../../gift/contrib/flash-linear-attention/fla/ops/gated_delta_rule/gate.py#L285).

A staged path would first use the existing helper for forward gate production and reuse FLA's gate backward after FlashQLA returns the natural-log gate gradient. Only if the measured gate/cumsum preparation cost warrants it should a fused preprocessing kernel be considered. Fusing the full backward would also need to combine reverse cumsum with gate derivatives and the `A_log`/`dt_bias` partial reductions; that is a larger change. This backend work does not imply removing SAC replay or the separate VJP-forward call.

## Numerical contract and remaining evidence

The helper computes gate values in FP32 and returns the requested output dtype. Its Triton softplus uses an approximate PTX implementation and a linear branch above 20; a PyTorch `F.softplus` expression is not guaranteed to be bitwise identical. FLA's gate backward computes `da` in FP32, casts it to the input dtype, sums `dA_log` from FP32 block partials, and forms `ddt_bias` from the already-cast `da`. Reusing this helper retains those existing semantics; FlashQLA-wide parity is not established. See the [softplus implementation](../../gift/contrib/flash-linear-attention/fla/ops/utils/softplus.py#L17) and [gate backward reductions/casts](../../gift/contrib/flash-linear-attention/fla/ops/gated_delta_rule/gate.py#L118).

Any future integration must preserve the current false in-kernel Q/K normalization flag, default scale `128^-0.5`, model beta behavior, GVA head reductions, absent initial/final-state contract, BF16 gathered `A_log`/`dt_bias` leaves with their FP32 masters and intermediate arithmetic, and custom-op/VJP boundaries. The gate-leaf dtype is confirmed by the captured block fixtures and native resolved run; FP32 masters do not make the gathered leaves FP32. The available trace does not justify dropping recomputation or changing precision.

The authorized task-copy implementation passed the six-fixture output/seven-gradient comparison and eager/compiled parity checks, and the native sequence completed with all requested backends active. CP remains 1, so CP warmup is untested. The short run does not establish convergence, checkpoint/resume, EMA updates, production PP4/EP32 behavior, whole-training numerical parity, or a stationary throughput effect. See the [execution plan](../../runs/flash-qla/plan.md) for scope and retained failures. Any further integration or measurements require separate authorization.
