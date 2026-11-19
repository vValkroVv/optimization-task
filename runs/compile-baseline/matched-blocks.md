# Matched PyTorch blocks with and without compile

The most substantial fresh local-kernel candidate is the dense copy that
joins the main-head and MTP cross-entropy gradients. It still costs **8.994 ms
per three microbatches** in the selected five-fusion trace. The next distinct
candidate is the final hidden-state z-loss, with **4.281 ms** of forward and
backward work in the same trace. Both are outside the part of the compiled
graph that removed the most eager overhead. The original A/B's strongest
layout fixes, including MLA rotary packing, are already in the selected
recipe.

| What to test against the selected recipe | Trace evidence | Priority |
| --- | --- | --- |
| Emit one folded `dLogits` directly from main/MTP CE backward | Six full gradient copies still cost 8.994 ms in the five-fusion trace | First new kernel test |
| Row-wise FP32 hidden-state z-loss with a single BF16 gradient store | 4.281 ms in the five-fusion trace, including required arithmetic | Second new kernel test |
| Overwrite expert `dW` in the eligible DeepGEMM epilogue | 3.921 ms for standalone BF16 destination zeroing in the five-fusion trace at EP8 | Concrete part of the existing scale-out gradient project |
| Assemble the incoming CuTe norm gradient in an upstream producer epilogue | 4.557 ms in the five-fusion trace at two norm boundaries per MoE block | Worth inspecting the producer, since norm backward reads the merged result twice |
| Statistics-only GDN output norm replay | 1.262 ms of replay and 0.723 ms of separate gradient scaling in the selected variant's scoped capture | Small follow-up on the existing kernel |
| Unbiased router auxiliary-loss top-k | 1.748 ms for the logical loss chain in the five-fusion trace, 1.229 ms in the native top-k | Small per-layer candidate |
| Reuse dense weight scales for transposed dgrad quantization | 1.078 ms of transposed quantizers in the scoped tail-aware capture, plus 0.370 ms of separate trim copies | Test given-scale quantization before custom code |
| CuTe replay output suppression, MLA shared-K packing, MTP copies | Sub-millisecond standalone targets in the corresponding scoped captures | After the larger targets |
| Post-top-k statistics and capacity histogram | 0.711 ms of elapsed pre-all-reduce forward time with compile | Low priority |

## Comparison and timing

[The experiment report](report.md) defines the two arms. Both start from S
plus the GPU capacity selector, before the five subsequent fusions. The
only intended change is compilation. The A/B uses rank 0, three complete
microbatches of update 21, local batch 2, sequence length 4096, and EP8 on
H100. The third trace used for selected spot checks is
[`4fix_qla_embed_selector_gpu_5fusions.json`](../../artifacts/4fix_qla_embed_selector_gpu_5fusions.json).
It establishes that the named work remains after the five fusions.

| Whole profiled GPU interval | Eager | Compile | Difference |
| --- | ---: | ---: | ---: |
| Wall time | 1,338.882 ms | 1,149.269 ms | −189.614 ms |
| Union of busy intervals on all streams | 1,198.314 ms | 1,118.266 ms | −80.048 ms |
| All CUDA streams idle | 140.568 ms | 31.003 ms | −109.566 ms |

One hundred ten milliseconds of improvement came from shorter gaps with every
CUDA stream idle. This is distinct from work a replacement GPU kernel can
remove from the compiled path. In the largest eager example, all streams
are idle for 2.149 ms in layer 7 backward. The API call launching the next
CuTe `GnPrep` starts 2.134 ms after the previous GPU activity ends. Both
arms use that CuTe kernel. The eager host interval includes an
`aten::add` scope with Python dispatch. The trace shows the late submission
and the enclosing scopes. It does not resolve time inside the Python scope
or distinguish profiler and host scheduling costs there.

Device times below sum matched GPU operations across three microbatches.
Elapsed ranges include the intervening gaps. Scope-specific times exclude
unrelated operations where listed. Some table rows describe portions of
larger rows. The device budgets overlap and may contain real work the next
implementation still needs to perform. None is a measured throughput gain.

## How the traces were paired

Microbatch and GFSDP CPU annotations define 23 sequential regions per
microbatch. The `lm_head`/loss region includes its own backward calls.
The backward embedding region begins at `_OwnerLookupBackward`. Kineto
launch correlations assign device activity to those CPU regions. Kernels
are then compared in sequence, per stream, with category, name, launch
geometry, transfer size and collective metadata as the signature.
There are 5,915 ordered same-signature device pairs. Semantic matches
across changed kernel names use producer/consumer anchors
and source or generated graph code. An ordered exact-signature match
alone establishes recorded kernel identity and geometry.

| Accounting | Eager | Compile |
| --- | ---: | ---: |
| Kernels | 11,818 | 8,272 |
| Copies | 244 | 118 |
| Memsets | 1,017 | 885 |
| Linked inside the 69 paired regions | 13,064 | 9,260 |
| Linked outside the three microbatch annotations | 15 | 15 |
| Unlinked | 0 | 0 |

Reproduce the per-launch alignment and retrieve raw indices, Kineto external
IDs, host launch timestamps and call scopes from the
[alignment ledger](../../artifacts/compile-layer-alignment.json).

```sh
python3 -B runs/compile-baseline/align.py \
  artifacts/baseline_no_compile.json artifacts/baseline_compile.json \
  --output artifacts/compile-layer-alignment.json
```

### Full layer coverage

These are sums of each region's first-to-last activity on stream 7.
The idle columns intersect those same ranges with idle time across all
streams. The embedding exchange on stream 24 lies partly outside these
stream-7 bounds. The table excludes inter-region gaps. Each column uses
the earlier definition of CPU region ownership, including replay in the
main backward rows.

| Region | Eager span, ms | Compile span, ms | Eager idle, ms | Compile idle, ms |
| --- | ---: | ---: | ---: | ---: |
| Forward input and embedding | 1.475 | 1.269 | 1.417 | 1.209 |
| Forward layer 0 | 24.533 | 18.890 | 8.205 | 2.484 |
| Forward layer 1 | 37.463 | 25.562 | 12.932 | 1.659 |
| Forward layer 2 | 35.152 | 24.313 | 10.747 | 0.611 |
| Forward layer 3 | 40.458 | 26.744 | 7.657 | 0.673 |
| Forward layer 4 | 35.302 | 25.672 | 9.613 | 0.645 |
| Forward layer 5 | 37.073 | 27.842 | 10.048 | 2.392 |
| Forward layer 6 | 36.220 | 25.326 | 8.554 | 0.706 |
| Forward layer 7 | 41.852 | 28.171 | 9.398 | 0.636 |
| Final norm and MTP handoff | 4.222 | 3.629 | 1.006 | 0.000 |
| MTP forward | 28.454 | 22.983 | 2.001 | 0.269 |
| Chunked LM head and CE, including local backward | 240.024 | 241.118 | 0.303 | 0.275 |
| MTP backward | 82.107 | 72.238 | 0.810 | 0.593 |
| Final norm backward | 3.374 | 3.347 | 0.000 | 0.000 |
| Backward layer 7 | 109.863 | 81.467 | 16.578 | 3.175 |
| Backward layer 6 | 86.750 | 81.036 | 8.190 | 1.497 |
| Backward layer 5 | 72.610 | 65.860 | 1.746 | 1.172 |
| Backward layer 4 | 84.827 | 75.215 | 5.781 | 1.511 |
| Backward layer 3 | 92.614 | 78.011 | 6.035 | 1.397 |
| Backward layer 2 | 80.953 | 72.805 | 6.843 | 1.479 |
| Backward layer 1 | 79.748 | 75.519 | 3.160 | 1.552 |
| Backward layer 0 | 52.530 | 50.101 | 1.049 | 0.923 |
| Backward embedding tail on stream 7 | 0.511 | 0.510 | 0.000 | 0.000 |

## MoE router and capacity before the all-reduce

The PyTorch statements below combine the active branches from the router,
MoE wrapper and capacity selector. `topk_ids` is the output of the earlier
top-k kernel. Inductor also eliminates work whose value the active
HybridEP consumer does not read.

```python
scores = torch.sigmoid(logits)
weights = scores.gather(-1, topk_ids)
weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * route_scale

routing_map = torch.zeros_like(scores, dtype=torch.bool)
routing_map.scatter_(-1, topk_ids, True)
with torch.no_grad():
    counted = routing_map & token_mask.unsqueeze(-1)
    tokens_per_expert.add_(counted.sum(dim=(0, 1)))

ids = topk_ids.reshape(-1).long()
flat_weights = weights.detach().reshape(-1)
valid = (ids >= 0) & (flat_weights != 0)
buckets = torch.where(valid, ids, num_experts)
counts = torch.zeros(num_experts + 1, dtype=torch.int64, device=ids.device)
counts.scatter_add_(0, buckets, torch.ones_like(buckets))
needs_drop = (counts[:num_experts] > capacity).any()
```

[Router](../../gift/torchtitan/models/common/moe/router.py#L139) ·
[MoE wrapper](../../gift/torchtitan/models/common/moe/__init__.py#L125) ·
[Historical capacity module](../capacity-group-allreduce/capacity.original.py#L124)

| Same range from first post-top-k operation through local predicate | Eager | Compile |
| --- | ---: | ---: |
| One matched forward, elapsed | 1,062.726 µs | 31.296 µs |
| Same forward, device work | 89.633 µs, 25 events | 21.536 µs, 8 events |
| Same forward, all-stream idle | 973.093 µs | 9.760 µs |
| All 21 original forwards, elapsed | 22.045 ms | 0.711 ms |
| All 21 original forwards, device work | 1.971 ms | 0.445 ms |
| All 21 original forwards, all-stream idle | 20.073 ms | 0.266 ms |

The example is microbatch 2, layer 4. Perfetto slice ranges are
`46859–47042` in eager and `24931–24938` with compile, using the paired
boundaries. Inductor's cached graph
`f3f4dhbkolgow63ymj2b6x2wrx3uanuo7g6autm4plvb674cqat2` shows
`_8` zeroing the bool array, `_9` fusing score calculation, routing scatter
and bucket preparation, `_10` and `_11` reducing masked statistics into
the FP32 expert-bias counter, `_12` and `_13` initializing scatter operands,
and `_14` emitting the local predicate and the 128 entries for the
collective. The native scatter-add still takes 10.880 → 10.336 µs at the
matched site. The all-reduce and the GPU selector each remain a separate
operation. The selector still combines the local expert predicate with
the destination group predicate.

One custom approach would histogram the 65,536 selected routes directly.
The expert-bias histogram applies `token_mask` and capacity applies
`valid`. The two semantics must be preserved, including the sentinel
bucket and final 64-bit capacity counters. The **0.711 ms compiled
forward interval** is the ceiling for this region before accounting for
the necessary work. This is roughly 0.062% of the profiled window at the
same shape. The eager overhead makes this a strong explanation of what
compile buys and a low-priority manual kernel. The scope of this estimate
ends before the collective and dispatch.

### MoE end-to-end forward brackets and routed backward

```python
totals = funcol.all_reduce(counts[:num_experts], "sum", ep_group)
group_loads = totals.view(ep_size, num_experts // ep_size).sum(dim=1)
needs_drop = needs_drop & (group_loads > capacity * num_experts).any()
keep = capacity_mask(flat_weights, buckets, valid, counts, needs_drop, capacity)
weights = weights * keep.view_as(topk_ids)
topk_ids = topk_ids.masked_fill(~keep.view_as(topk_ids), -1)

routed, metadata = dispatch(x, topk_ids, weights)
gate_up = grouped_gemm(routed, w13, metadata)
activated = fused_swiglu_quantized(gate_up, metadata.permuted_probs)
expert_out = grouped_gemm(activated, w2, metadata)
out = combine(expert_out, metadata)
```

The final five lines show the active dataflow with wrappers and FP8
payload/gradient-edge bookkeeping elided. [Capacity logic](../capacity-group-allreduce/capacity.original.py#L132) ·
[Dispatcher](../../gift/torchtitan/models/common/moe/dispatch/hybridep.py#L113) ·
[Experts](../../gift/torchtitan/models/common/moe/experts/gmm.py#L273)

| MoE regions, 21 original forward sites | Eager | Compile |
| --- | ---: | ---: |
| Top-k end through capacity selection | 37.657 ms | 4.540 ms |
| All-stream idle within that bracket | 30.338 ms | 1.130 ms |
| Capacity selection end to dispatch | 21.820 ms | 13.320 ms |
| Top-k end to combine end | 152.958 ms | 110.674 ms |

Representative eager external IDs are **1424 → 1630 → 1805** and
compiled IDs are **1047 → 1068 → 1143** for top-k, dispatch, and combine.
The first row includes the pre-collective range listed earlier. The
broader range also includes communication and submission time. It
provides no extra 42 ms budget for a kernel on the compiled path.

| Expert work | Eager | Compile |
| --- | ---: | ---: |
| Routed SwiGLU forward, 42 launches including replay | 8.237 ms | 8.217 ms |
| Routed SwiGLU backward, 21 launches | 6.786 ms | 6.656 ms |
| Shared/dense SwiGLU forward, 54 launches of shared symbols | 4.911 ms | 4.669 ms |
| Shared/dense SwiGLU backward, 27 launches of shared symbols | 2.304 ms | 2.419 ms |
| Routed 1d1d GEMMs, 42 matched launches | 55.950 ms | 55.973 ms |

With compile the interval between backward HybridEP dispatch and combine
contains 14 kernels per site instead of 37 and shrinks **190.540 →
86.355 ms** across 21 sites. The grouped requantization and 1d1d wgrad
GEMMs move after combine. The 42 wgrad launches remain, so the bracketing
delta measures changed schedule as well as earlier combine. In the
later component test, fusing both FP8 gradient layouts into SwiGLU
backward made the chain **492.00 → 615.74 µs** despite eliminating
layout work. Keep that approach deferred. A narrower physical
scale-stride fix could remove the 21 standalone DeepGEMM scale
transposes (0.159 ms in the fusion audit) without moving either FP8
layout earlier. [Prior MoE fusion measurements](../kernel-fusion/report.md#implementation-audit)

### Routed expert dW initialization

```python
wgrad = torch.zeros(ctx.weight_shape, dtype=torch.bfloat16, device=ctx.device)
handle.k_grouped_fp8_gemm_contiguous(
    a=grad_output_quantized_col,
    b=tensor_groups_quantized_col,
    d=wgrad,
    ks=ctx.list_groups_sizes,
    ks_cpu=ctx.tensor_groups_sizes_cpu,
    ks_tensor=ctx.tensor_groups_sizes,
)
```

[Grouped expert backward](../../gift/torchtitan/ops/float8/grouped_gemm.py#L244) ·
[Mutating DeepGEMM wrapper](../../gift/torchtitan/ops/float8/deep_gemm_handle.py#L98)

| GPU initialization, 21 calls per weight | Eager | Compile | Selected five-fusion trace |
| --- | ---: | ---: | ---: |
| `w2`, `[16, 4096, 1536]` BF16 | 1.292 ms | 1.308 ms | 1.303 ms |
| `w13`, `[16, 3072, 4096]` BF16 | 2.573 ms | 2.622 ms | 2.618 ms |
| Total | 3.865 ms | 3.931 ms | 3.921 ms |

Per layer and microbatch the two zero fills write 192 and 384 MiB.
Each is matched by its destination shape/launch grid and the corresponding
K-grouped GEMM on the same stream. In layer 1 backward, eager external IDs
26942 and 27041 are the native fills. The compiled IDs 11811 and 11825
match the two graph allocations `buf226` and `buf256`. Despite the
second kernel's generated `to_copy_transpose` name, it has only the
destination as a pointer argument and initializes it to zero.

The wrapper passes `d` as both destination and accumulator. An overwrite
epilogue for groups where all contributions are accounted for would avoid
the full zero fill and could also be the first independent step of the
existing FP32 reduce-scatter buffer project. It must emit zeros for experts
with zero tokens, handle the all-empty return, and preserve the current
BF16 rounding and accumulation semantics. The current accumulate API
requires the zeroed input. This trace uses EP8/eFSDP1 and measures
initialization only. The separate eFSDP>1 direct-store design also aims
to remove the BF16-to-FP32 packing pass. [Scale-out design](../../docs/profile-my-ideas.md#direct-expert-weight-gradients-into-the-fp32-reduce-scatter-buffer)

### Sequence auxiliary router loss

```python
scores_f = scores.float()
_, aux_ids = torch.topk(scores_f, k=top_k, dim=-1, sorted=False)
probs = scores_f / (scores_f.sum(dim=-1, keepdim=True) + 1e-20)
mask = token_mask.float().unsqueeze(-1)
weights = mask.expand(B, L, top_k).reshape(B, L * top_k)
counts = torch.zeros(B, E, dtype=torch.float32, device=scores.device)
counts.scatter_add_(1, aux_ids.reshape(B, L * top_k), weights)
counts = counts / (L * top_k / E)
probs_mean = (probs * mask).mean(dim=1)
scale = L / mask.sum(dim=1).clamp(min=1.0)
aux = coef * ((counts * scale * (probs_mean * scale)).sum(dim=1)).mean()
```

The auxiliary loss uses **unbiased** top-k. The earlier dispatch choice
includes the expert bias. The auxiliary criterion therefore has its own
top-k and scatter. Eager launches this PyTorch chain once in original
forward and once in replay. The compiled forward uses the saved dispatch
top-k and leaves the auxiliary loss computation to its backward graph.
The cached compiled graph
`f5m62alrbwqgdbmv4bmay4sfhgrhm5477zb2dktc5m4e3bxl64em` shows a
recomputed FP32 sigmoid and native `aten.topk.default` on those unbiased
scores. Inductor combines the mask, scatter, normalization and reductions.
It also fuses part of the derivative with the router gradient. External
IDs 1884 and 26469 anchor the first eager forward/replay aux top-k. In
the compiled trace the first layer's aux top-k is 11321.
[Auxiliary loss source](../../gift/torchtitan/models/common/moe/losses.py#L24) ·
[SAC policy](../../gift/torchtitan/distributed/activation_checkpoint.py#L105)

| Device time for the logical auxiliary loss, excluding the merged derivative | Eager | Compile | Selected five-fusion trace |
| --- | ---: | ---: | ---: |
| Original forward, 21 sites | 2.505 ms | Deferred to backward graph | Deferred to backward graph |
| Replay/backward graph, 21 sites | 2.618 ms | 1.761 ms | 1.748 ms |
| Unbiased top-k included in the rows above | 2.407 ms, 42 calls | 1.226 ms, 21 calls | 1.229 ms, 21 calls |

The compiled row sums the sigmoid, top-k, scatter initialization/update,
probability reductions and aux coefficient kernel across their scheduled
positions. In eager it sums the contiguous top-k through coefficient
chain, using the sigmoid already produced by the router. A specialized
128-to-8 selector with tile-local counts is a small residual candidate.
Preserve score precision, tie selection, token-mask weights, and the
unbiased top-k index set. The device budget is under 0.16% of the A/B's
compiled interval and the selector would still have useful work to do.

## MLA RoPE, Q/K packing, and output gate

The corresponding statements from the historical MLA forward are below.
The attention backend and the projection GEMMs expose useful unchanged
boundaries to match the surrounding work.

```python
q_nope, q_pe = q.split([nope_dim, rope_dim], dim=-1)
(q_pe,) = apply_rotary_emb_complex(q_pe, freqs_cis=freqs_cis, positions=positions)
q = torch.cat([q_nope, q_pe], dim=-1)

kv, k_pe = wkv_a(x).split([kv_lora_rank, rope_dim], dim=-1)
(k_pe,) = apply_rotary_emb_complex(
    k_pe.unsqueeze(2), freqs_cis=freqs_cis, positions=positions
)
kv = wkv_b(alpha_kv * kv_norm(kv)).view(batch, seq, num_heads, nope_dim + v_dim)
k_nope, v = kv.split([nope_dim, v_dim], dim=-1)
k = torch.cat([k_nope, k_pe.expand(-1, -1, num_heads, -1)], dim=-1)
out = inner_attention(q, k, v, attention_masks=attention_masks, scale=softmax_scale)
out = out.contiguous().view(batch, seq, -1)
gate = torch.sigmoid(gate_proj(x).float())
out = (out.float() * gate).to(out.dtype)
out = wo(out)  # Includes the separate rowwise FP8 input quantizer.
```

[MLA forward](../../gift/torchtitan/models/common/attention/softmax/mla.py#L101) ·
[MLA gate](../../gift/torchtitan/models/common/attention/softmax/gate.py#L29) ·
[FP8 projection](../../gift/torchtitan/ops/float8/cuda_gemm.py)

| Matched region | Eager | Compile |
| --- | ---: | ---: |
| Gate/layout after gate projection, 9 forward + 9 SAC replay sites | 9.257 ms, 90 launches | 1.192 ms, 18 launches |
| Separate rowwise quantizer at the same 18 sites | 0.689 ms | 0.738 ms |
| Full rotary and Q/K preparation, 9 forward + 9 SAC sites | 14.448 ms, 306 launches | 4.729 ms, 126 launches |
| Original forward portion of that preparation | 6.169 ms | 2.276 ms |
| SAC portion of that preparation | 8.280 ms | 2.453 ms |
| KV projection end to cuDNN start, 9 original forwards, elapsed | 2.352 ms | 1.398 ms |
| Q/K packing within that elapsed range, 9 original forwards | 1.902 ms | 1.317 ms |
| Gate derivative preparation feeding projection backward, 9 sites | 5.759 ms, 54 launches | 0.959 ms, 9 launches |
| K reduction and first gradient concat immediately after cuDNN backward, 9 sites | 2.817 ms, 18 launches | 0.932 ms, 18 launches |
| cuDNN MLA, forward and backward with its helpers | 31.456 ms | 32.321 ms |

In the first matched site, eager external IDs 3268 and 3303 anchor KV
projection and cuDNN. The corresponding compiled IDs are 1635 and 1639.
For the gate, eager issues separate casts, sigmoid and multiplication.
Inductor fuses those into a single pointwise/layout kernel, followed by a
CudaCherry quantizer despite the word `quantize` in the Inductor name.

The full rotary cohort selects the preparation after the Q and KV-low-rank
projections up to the packed inputs feeding each cuDNN forward. It includes
eager position checks and excludes GEMMs and KV norm. The six MTP sites
can reverse K and Q launch order. The eager gate derivative contains six
elementwise cast/multiply/sigmoid-backward operations per site. Inductor
emits one layout/derivative kernel. Immediately after cuDNN backward,
eager `ExpandBackward0 → sum` and `SplitWithSizesBackward0 → cat` match
compiled `triton_per_fused__to_copy_slice_sum…` and
`triton_poi_fused_cat_slice_transpose…`. Example eager IDs 10875/10878
match compile IDs 4949/4950. The latter concat alone still costs **0.774 ms**
in the five-fusion trace. A direct K/V assembly would need to preserve
the shared-K reduction, BF16 rounding and the layout accepted by
projection backward.

The compiled A/B has **4.729 ms** of forward and SAC rotary preparation and
Q/K packing at the 18 sites. The already-selected manual kernel rotates
and writes final Q/K buffers directly. In the later fusion experiment its
matched device work was **4.157 → 3.570 ms**. Remaining work includes
sharing K rotation across query-head tiles and preserving its gradient
reduction. The candidate that also fused the MLA gate into FP8 had
**1.930 ms** of pre-existing compiled work to attack, but the compiled
gate-through-projection forward/backward component regressed **1.183 →
1.250 ms** and the native run fell between its controls. The gate stays
deferred. [Fusion measurements and audit](../kernel-fusion/report.md)

## GDN short convolution, scan, and output norm

```python
q, k, v = qkv(x, cu_seqlens=cu_seqlens)
ba = ba_proj(x).view(batch, seq, num_k_heads, 2 * num_v_heads // num_k_heads)
b, a = ba.split(num_v_heads // num_k_heads, dim=-1)
beta = b.reshape(batch, seq, num_v_heads).sigmoid()
if allow_neg_eigval:
    beta = beta * 2.0
a = a.reshape(batch, seq, num_v_heads).contiguous()
z = z_proj(x).view(batch, seq, num_v_heads, head_v_dim)
o = gdn_chunk_op(q, k, v, a, beta, A_log, dt_bias, cu_seqlens)
o = o_norm(o, z) * o_norm_output_multiplier
o = o_proj(o.reshape(batch, seq, value_dim))
```

[GDN forward](../../gift/torchtitan/models/common/attention/linear/gdn.py#L76) ·
[FlashQLA autograd boundary](../../gift/torchtitan/models/common/attention/linear/fla_custom_ops.py#L75)

| Matched GDN work | Eager | Compile |
| --- | ---: | ---: |
| FlashQLA forward and backward scan, 72 native launches | 20.800 ms | 22.317 ms |
| Short-conv Q/K/V unpack, 54 launches | 10.033 ms | 9.804 ms |
| Short-conv backward, 18 launches | 6.319 ms | 6.389 ms |
| Short-conv recompute, 18 launches | 1.701 ms | 1.704 ms |
| Short-conv backward pack, 18 launches | 2.003 ms | 2.004 ms |
| FLA norm → output multiplier → FP8 quantizer, 36 forward/SAC chains | 5.337 ms | 5.838 ms |
| Additional FLA norm forward inside backward, 18 VJP sites | 1.252 ms | 1.258 ms |

Inductor handles the ordinary pointwise preparation around these custom
ops. The native scan, convolution, and norm-to-quantize boundary stay.
This is the case where a custom kernel is useful even though compile
does little for the target chain. The selected GDN output norm-to-FP8
kernel saves an intermediate write/reread and fuses the output multiplier.
Its later scoped native device result is **5.826 → 2.514 ms**. The selected
short-conv backward kernel combines recompute, Q/K/V packing and convolution
gradient work, with a later complete-chain result of **13.390 →
12.110 ms**.

At 18 backward sites the selected norm fusion still replays the FLA
forward. The linked fusion audit records **1.262 ms** for that replay and
**0.723 ms** for the separate gradient multiplier. Test a statistics-only
replay with FP32 inverse RMS and the existing saved tensors. Keeping all
inverse RMS tensors would extend the lifetime of 1 MiB per site. For the
convolution, narrower V tiles could lower the register pressure of the
selected kernel while retaining the BF16 casts and FP32 addition order.
[Selected GDN kernels and follow-up](../kernel-fusion/report.md#implementation-audit)

## Decoder residual adds and dense FP8 weight preparation

```python
attn_out = post_attn_norm(attention(pre_attn_norm(x), ...))
x = x + attn_out
ffn_out = post_ffn_norm(feed_forward(pre_ffn_norm(x)))
output = x + ffn_out

# Inside the dense FFN and its FP8 projections:
gate, up = w1(ffn_input), w3(ffn_input)
act_fp8 = swiglu_quantize(gate, up, ...)
ffn_out = w2(act_fp8)

# Inside FP8Gemm before the weight quantizer:
weight_for_fp8 = torch.nn.functional.pad(weight, pad_to_block128)
weight_fp8 = quantize_blk128x128(weight_for_fp8)
```

[Decoder forward](../../gift/torchtitan/models/common/decoder_blocks/transformer_block.py#L121) ·
[Dense FFN](../../gift/torchtitan/models/common/ffn/__init__.py#L87) ·
[Dense FP8 forward and backward](../../gift/torchtitan/ops/float8/cuda_gemm.py#L67)

The native norm runs the same 1,281 CuTe launches in both arms,
**49.144 → 51.112 ms**. The 36 large residual adds immediately after
`GnFwdOut` take **2.487 → 2.441 ms**, with 27 original-forward and
nine replay sites. The first matching producer/add IDs are 862→886 in
eager and 735→757 with compile. Compile can fuse nearby
pointwise operations and preparation without crossing the custom norm
call. The subsequent manual residual merge puts the add directly in
the CuTe output epilogue after the BF16 norm rounding. At 24 replay sites
the next kernel is the same norm backward and the output stage alone
takes **1.468 → 1.939 ms** with the manual fusion. Suppressing the activation
write at sites where only saved stats are consumed would give a more
targeted follow-up. This requires proving the residual output is dead
under the active SAC graph.

The **36 BF16 weight pads take 4.622 → 2.589 ms** before the same
CudaCherry weight quantizer. Eager uses 36 fills and 36 copies, with
exactly one kernel per pad in the compiled path. These sites are
layer 0 and MTP, each with nine original-forward and nine replay pads.
The MoE shared FFN has a different width and no matching pad chain.
The first example is eager 955→959 (fill plus DtoD) and compile 803.
The later selected tail-aware quantizer
cuts the targeted pad-and-quantization work **4.956 → 2.564 ms**
while preserving the padded logical K extent. The measured transposed
dgrad quantizers still spend **1.078 ms** recomputing block scales
across 18 calls. CudaCherry's given-scale transposed quantizer is the
direct candidate to reuse the forward FP32 scale plane. Check FP8 bytes
and the restored GFSDP BF16 weight view through SAC. The existing
compiled SwiGLU already quantizes in its epilogue. The observed layer-0
forward device sum is **19.994 → 20.025 ms**, with **56.361 →
54.689 ms** in backward including replay. These whole-layer sums include
other work and do not isolate any dense GEMM instruction-level change.
[CuTe and dense FP8 measurements](../kernel-fusion/report.md#implementation-audit)

The five selected fusions together provide the direct native comparison
against the pre-five compiled recipe. A single 40-warmup/100-measurement
pair moved from **164,537 to 166,519 tokens/s (+1.20%)**, with peak allocated
lower by 12.46 MiB and peak reserved lower by 100 MiB. Both arms were
compiled. This is separate from the eager/compiled A/B trace deltas.
[Combined training result](../kernel-fusion/report.md)

### CuTe backward input assembly in MoE layers

For the traced GDN-MoE layer-1 backward graph, the two large kernels
immediately before CuTe backward are equivalent to the operations
below. Each operand spans `[2, 4096, 4096]` elements. This code describes
the additions emitted by Inductor and their final store.

```python
# First gradient merge in the generated graph
dy.copy_(((dy.float() + g1.float()) + g2.float()) + g3.float())

# Second gradient merge, later in the same block
dy_other.copy_((dy_other.float() + h1.float()) + h2.float())
```

| Large input-preparation launches at two CuTe boundaries | Eager adjacent work | Compile | Selected five-fusion trace |
| --- | ---: | ---: | ---: |
| Second full-width CuTe backward call in each MoE layer | 2.792 ms, 21 casts and 21 adds | 2.815 ms, 21 merges | 2.815 ms |
| Last full-width CuTe backward call | 1.280 ms, 21 adds | 1.740 ms, 21 merges | 1.742 ms |

The eager values sum only the adjacent cast/adds before the same CuTe
consumers. Other eager gradient contributions are produced earlier.
The compiled values include multiple branch contributions within each
kernel. The columns therefore describe the observed local cost at the
same consumer boundaries and do not establish a regression for the
complete gradient branch. Example eager 27229/27230 → 27243 corresponds
to compile 11427 → 11428. The second merge is compile 11730 → 11731.
The two compiled merges remain at **4.557 ms** together in the selected
five-fusion trace.

A normalizer kernel that reads each branch directly would have to
repeat those extra inputs in both backward passes. CuTe currently reads
the already merged `dy` in each pass. A better first component to
inspect is a final upstream gradient producer that can add the available
contributions and write the existing `dy` destination in its epilogue.
Check operand readiness, BF16 rounding and FP32 addition order. The
4.557 ms is the current assembly cost, with a smaller removable portion
because the merges still read all contributions. [CuTe backward passes](../../gift/torchtitan/models/common/norms/cute_gated_norm/backward.py#L1) ·
[Norm autograd](../../gift/torchtitan/models/common/norms/cute_gated_norm/op.py#L257)

## Remaining cross-entropy gradient copy

```python
logits = lm_head(h_chunk)
logits_by_head = torch.chunk(logits, num_heads, dim=0)
labels_by_head = torch.chunk(label_chunk, num_heads, dim=0)

chunk_loss = None
for weight, head_logits, head_labels in zip(
    head_weights, logits_by_head, labels_by_head
):
    head_loss = existing_ce(head_logits, head_labels, global_valid_tokens) * weight
    chunk_loss = head_loss if chunk_loss is None else chunk_loss + head_loss

chunk_loss.backward()
# SplitBackward0 runs torch.cat on the per-head dLogits.
```

[MTP per-head CE](../../gift/torchtitan/components/loss.py#L1017) ·
[LM-head chunk loop](../../gift/torchtitan/components/loss.py#L734)

| Device work | Eager | Compile | Selected five-fusion trace |
| --- | ---: | ---: | ---: |
| LM-head projection, dX and dW, 18 GEMMs | 204.053 ms | 205.294 ms | — |
| Stable CE, 12 forward and 12 backward kernels | 14.032 ms | 14.011 ms | — |
| `SplitBackward0 → aten::cat`, six calls | 8.961 ms | 8.921 ms | 8.994 ms |

The paired first `cat` launches have Kineto external IDs 9265 and 4123.
Their recorded launch grid is `[4224, 2, 1]`. The two CE backward launches
feed one `[4, 2048, 128256]` BF16 gradient per chunk. The standalone join
reads and writes about 1.96 GiB each per call. It runs immediately before
the projection backward consumers.

A head-aware stable CE backward can write one folded `dLogits` buffer
directly. The existing experimental
[`stable_ce.backward_kernel`](../../gift/torchtitan/experiments/four_fusion/stable_ce.py#L47)
shows the row-to-head mapping and per-head scale. Inspect and preserve the
currently selected CE implementation's arithmetic before borrowing it,
including FP32 max and log-denominator, global valid-token denominator,
ignore-index, head weights, BF16 output rounding, and per-head metrics.
Keep the existing chunk/head GFSDP and GEMM sequence for the first test.
This removes the second full-matrix write and read. The compiled standalone
copy gives an **8.9 ms gross ceiling**, about **0.78% of the A/B's compiled
window**. Separate fused linear-CE experiments and their regressions do
not test this narrower modification. [Earlier loss experiment](../four-fusion/report.md)

## Final hidden-state z-loss outside the compiled trunk

```python
def hidden_z_loss(h, coef):
    return (coef * torch.logsumexp(h.float().abs(), dim=-1).square()).mean()
```

[Hidden z-loss and its hook](../../gift/torchtitan/models/common/regularization.py#L226)

| Exact device work | Eager | Compile | Selected five-fusion trace |
| --- | ---: | ---: | ---: |
| Cast, abs, stable logsumexp, square, multiply and mean | 2.171 ms | 2.310 ms | 2.255 ms |
| Mean backward through logsumexp/abs and the final dtype cast | 2.048 ms | 2.077 ms | 2.026 ms |
| Total | 4.219 ms | 4.387 ms | 4.281 ms |

The compiled trace still launches standalone ATen kernels for these
operations. In microbatch 1, the first forward and backward launches
selected here have external IDs 3220 and 5346. The latter range starts
with `MeanBackward0` and stops after the BF16 cast, before adding this
gradient to the main path. A row-wise kernel can load BF16, reduce
`abs(h)` in FP32 with a stable logsumexp, and keep FP32 row statistics.
The derivative is proportional to
`z * exp(abs(h) - z) * sign(h)`, where `z = logsumexp(abs(h))`.
Forward can write a row scalar and reduce the final mean separately.
Backward can read the original hidden state and write a BF16 gradient
once. Verify gradient and loss at the current coefficient, including
zero and non-finite inputs, and preserve accumulation into the main
gradient and the auxiliary-loss hook. The compiled **4.39 ms** includes
necessary reduction and derivative work, so realizable savings are lower.

## MTP input, hidden-state packing, and embeddings

```python
shifted = torch.cat([input_embeds[:, 1:], torch.zeros_like(input_embeds[:, :1])], dim=1)
proj_in = torch.cat([hidden_norm(h), token_norm(shifted)], dim=-1)
mtp_hidden = input_proj(proj_in)
# After the MTP block and its final norm:
folded_hidden = torch.cat([main.unsqueeze(0), torch.stack([mtp_out], dim=0)], dim=0)
```

[MTP input projection and shift](../../gift/torchtitan/models/common/decoder_blocks/mtp.py#L70) ·
[Main/MTP hidden-state assembly](../../gift/torchtitan/models/common/decoder_blocks/decoder.py#L160)

| Device work | Eager | Compile | Selected five-fusion trace |
| --- | ---: | ---: | ---: |
| BF16 MTP input concat directly before input projection | 0.281 ms | 0.345 ms | 0.336 ms |
| Shift with a zero final token | 0.530 ms | 0.670 ms | 0.590 ms |
| Large copy in shift backward | 0.359 ms | 0.356 ms | 0.363 ms |
| Single-MTP `stack` and outer hidden-state `cat` before the loss | 0.412 ms | 0.416 ms | 0.416 ms |

At the input projection, the first eager `aten::cat` has external ID
8248 and the compiled `triton_poi_fused_cat_1` has ID 3359. They both
materialize a concat for the same BF16 GEMM. At a single MTP head,
`stack([mtp_out])` can be replaced by `mtp_out.unsqueeze(0)`, which
could avoid the standalone stack materialization. The three stack copies
cost 0.139 ms in the five-fusion trace. A combined norm output writer
could place two halves directly in the existing projection input if its
autograd and buffer lifetimes allow it. A shift-aware producer/consumer
could avoid copying the shifted tensor. The measured sizes and the need
to preserve MTP and normalizing semantics make both lower priority than
the CE gradient copy.

The row-owner embedding backward plus linked tail is **4.016 → 4.019 ms**
of summed activity. The FP32-to-BF16 forward cast is **0.174 →
0.174 ms** and the BF16 gradient contribution casts in backward total
**0.321 → 0.323 ms**. The split exchange, owner lookup, and BF16
gradient construction are custom-autograd and communication boundaries.
The comparable SE run already tested lookup and add simplifications at
**+0.005%** throughput with lower reserved memory. This A/B does not
establish a new embedding fusion win. [Embedding implementation](../schedule-embedding/embedding-implementation.md) ·
[Measured S/SE result](../schedule-embedding/report.md)

## GFSDP and native kernels used as controls

```python
torch._foreach_copy_(fresh_fp32_accumulated_grad_views, unsharded_grads)
torch._foreach_add_(existing_fp32_accumulated_grad_views, next_unsharded_grads)
```

Under `FSDP::post_backward_accumulate`, 111 copy kernels take
**36.404 → 33.310 ms** and the three recorded add kernels take
**6.903 → 6.910 ms**. The selected five-fusion trace still shows **32.126 ms**
of copy and **6.861 ms** of add. These are the same families in the same
hooks. A producer writing directly into the existing FP32 reduce-scatter
input would need to reproduce the current BF16 rounding and FP32
accumulation. Separate `lm_head` accumulation and expert/dense paths
have different producers, so their traffic budgets cannot be claimed by
one proposed expert GEMM epilogue. [GFSDP hooks](../../gift/torchtitan/distributed/gfsdp/_param_group.py#L1002) ·
[Existing scale-out ideas](../../docs/profile-my-ideas.md)

| Native matched kernels | Launches per arm | Eager work | Compile work |
| --- | ---: | ---: | ---: |
| CuTe gated norm | 1,281 | 49.144 ms | 51.112 ms |
| cuDNN MLA flash attention and backward helpers | 45 | 31.456 ms | 32.321 ms |
| FlashQLA TileLang scans | 72 | 20.800 ms | 22.317 ms |
| DeepGEMM routed 1d1d | 42 | 55.950 ms | 55.973 ms |
| DeepGEMM routed 1d2d, matched signatures | 66 | 67.825 ms | 67.874 ms |
| GPU capacity-mask and routing metadata | 81 | 0.331 ms | 0.329 ms |

HybridEP combine (60 launches) sums **110.439 → 91.589 ms** in the
two traces. The 39 128-element capacity all-reduces sum
**7.116 → 13.617 ms**. The matching common GFSDP all-gathers (78 launches)
sum **91.220 → 91.735 ms**. Changes in peer arrival and overlap can change
observed collective duration with the same kernel and payload. Changes to
those collectives are distinct from a hand-written replacement for
compiled PyTorch. [Native-family matching](report.md#what-compile-improves)

The full routed 1d2d census is 108 eager launches and 105 compiled.
The 66-pair row above uses only matching signatures. The traces establish
the active path and timing, while numerical parity or changed routing
decisions require tensor-level comparison. This report uses the retained
captures and proposes follow-up components without a new training run.
