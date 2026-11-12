# Softcap and FFN memory

The L8 recipe enables the memory paths below while keeping microbatch size 12,
block activation checkpointing disabled, and CPU offload disabled.

```yaml
loss:
  type: mtp
  loss_weights: [0.1]
  num_chunks: 32
  fused_softcap_ce: true

model:
  config:
    ffn:
      recompute: true
    moe:
      experts:
        recompute: true
      shared_experts:
        recompute: true
```

These fields default to false outside the L8 preset. The dense FFN setting also
reaches the dense MTP block through the model's existing materialization path.
The shared expert setting applies to the ordinary shared FFN. The routed
expert setting replays only local BF16 grouped GEMMs and SwiGLU; dispatch,
combine and their collectives stay outside the checkpoint. Routed FP8 GEMMs
reject this option explicitly. Model parameters, checkpoint keys, MTP weights
and loss normalization do not change.

## Fused loss

With `fused_softcap_ce`, the chunk loop calls `Unembedding` with softcap disabled
for that call. The normal head forward, redistribution and FSDP hooks still run.
Triton row kernels then combine softcap, cross-entropy and their derivatives,
without full FP32 logits/log-softmax intermediates. They retain raw low-precision
logits and one FP32 log-sum-exp scalar per row. Backward writes a low-precision
logits gradient; the existing linear backward computes hidden/weight gradients.
The GEMM and FSDP gradient accumulation are unchanged.

The kernel preserves the existing softcap cast to logits dtype before FP32 CE,
including the corresponding gradient cast. It supports FP16/BF16/FP32 local CUDA
logits, sum reduction, ignored target -100, and first-order gradients. CPU uses the
original eager formula. DTensor logits use the existing distributed CE path with
softcap reapplied and a warning. An Unembedding head is required.
Head-local hidden z-loss falls back to the
original head/loss path with a warning, preserving hook inputs and gradient
ordering. Branch and expert z-losses do not disable fused CE. Evaluation and
acceptance counters retain softcapped predictions; ordinary model/inference calls
still apply the head's configured softcap by default.

When fused CE is disabled, softcap alone uses the earlier pointwise fusion for
local CUDA tensors. CPU/DTensor retain eager softcap. Completed chunk references
are released before the next head call in either mode.

## FFN-local recomputation

`FeedForward.recompute` uses non-reentrant PyTorch checkpointing around that FFN's
forward. It saves the FFN input and replays projections/SwiGLU during backward,
independently of global block checkpointing. Both BF16 and FP8 paths use the same
boundary. Clip counts are accumulated once outside the replay boundary; hidden
z-loss still records in backward. No-grad and evaluation calls run normally.

## Validation and integration impact

CPU tests require exact output, parameter/input gradients, MTP components,
acceptance counters, clip counts and hidden z-loss accounting. AOT-compiled
comparisons use the same compilation settings on both sides. Saved-storage tests
check the FFN retention reduction. CUDA tests cover loss/gradient tolerances,
peak allocation and FP8 recomputation, but require the target GPU stack to run.

CUDA reduction/transcendental rounding can differ from PyTorch CE. Representative
GPU loss/convergence validation is still required; CPU parity is not CUDA bitwise
proof. Warm compilation and measure at least ten training steps before reporting
memory or throughput. The new kernels need no additional package dependency.

The loss chunk loop is upstream-derived code: future cherry-picks must preserve
the raw-logit keyword, FSDP lifecycle, MTP normalization and acceptance-counter
behavior. FFN defaults remain off so other registered model families keep their
existing execution unless explicitly configured.

## L8 measurement and review

The eight-GPU L8 run at local batch size 12 completed 20 steps without block
checkpointing or offload. At 128 loss chunks it reported 69.09 GiB and 50,878
tokens/s; at 32 chunks it reported 70.81 GiB and 55,669 tokens/s. These are
short-run observations, not a controlled convergence comparison or a guarantee
against later routing-dependent peaks. The recipe uses the measured 32 chunks.

Regression tests compare eager and recomputed expert w1/w3 z-loss gradients and
weighted totals over repeated microbatches, including empty experts and frozen
inputs. Router-gradient tests cover both routing-score placements and assert
that dispatch runs once. Head input/output z-loss tests pin the unfused fallback;
FFN tests cover clip-counter replay and hidden z-loss accounting at w1/w2/w3
and whole-FFN input/output hooks. Compiled expert CPU tests use a capture-safe
grouped-MM reference to check replay, z-loss and counters; they do not validate
the CUDA grouped GEMM kernel. Independent adversarial review found no L8 blocker;
its invalid-cap and non-boolean recompute findings are covered by validation
and regression tests. Parameters and
checkpoint keys are unchanged. Exact distributed GPU parity remains a target-stack
validation requirement.
