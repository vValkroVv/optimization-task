# K1a compiled CE contract reconciliation

Read-only judgment, 13 November 2026; task HEAD
`6123a24ff74df7f47efad677070d47ed5a238705`. Donor K is pinned by
`references/kindulov/manifest.json` to
`b983803337dfcc34fdd571b38aaf5e01df583bbe`. No source edits, GPU work,
new tests, benchmarks, or Perfetto queries were performed for this judgment.
A later targeted CPU FP32 arithmetic proof below establishes a donor stability
defect; it is not execution of the CUDA kernel.
Applied `code-rules` and `perf-experiments` by source inspection.

Acceptance: inspect actual compiled forward/backward allocations and saved
values; reconcile donor precision/autograd/normalization; decide whether an
independent mechanism remains; identify sufficient cache preservation. All four
read-only acceptance items are complete. Runtime acceptance of a port is open.

## Decision

**Independently applicable algorithm candidate, with its original large FP32
buffer-removal rationale already present in the compiled baseline.** Do not mark
the entire donor algorithm ALREADY_PRESENT, but correct the K1a coverage claim:
the actual ordinary compiled CE never allocates or saves a full-vocabulary FP32
logits, log-softmax, or gradient tensor. Its virtual FP32 ATen graph tensors
are fused into Triton. Reading `pred.float()` alone gives the wrong memory model.

The remaining concrete difference is the backward algorithm: Inductor reduces a
generated one-hot NLL gradient across V before a second loop reconstructs
probabilities and writes BF16 gradients. The donor directly evaluates
`(exp(logit-LSE)-one_hot)*scale`, removing that vocabulary-wide arithmetic
reduction. Its speed and integration cost remain unmeasured. Both ordinary paths
still require three kernels per logical CE: two forward kernels (row statistics
then loss SUM) and one backward kernel. The donor does not remove a forward
launch: its row kernel writes losses and LSE, and `losses.sum()` is another op.
The unchanged donor has a confirmed large-common-offset numerical defect at
cap0, described below. A stable adaptation is required before port acceptance;
the independent algorithm candidate is not an acceptable unchanged port.

## Actual compiler evidence

All paths below are relative to
`artifacts/raw/g3-baseline-update111-001/k1a-compiled-ce-evidence/`.
These are the canonical byte-identical preserved copies after cache cleanup;
the original provenance prefix was
`artifacts/raw/g3-baseline-update111-001/cache/torchinductor/`.

| File | Inspected contract | SHA256 |
| --- | --- | --- |
| `ye/cye4fjcll6xzwl2msivvwsfdj422r2geczhvsh6pgsbkjaf4wut3.py` | Rank0 AOT6 forward; :70–108 FP32 online row reduction from BF16; :174–193 selected logit/loss SUM; :210–229 allocations and saved outputs | `8634e41215a309a991db995031892dabd81653ddea5b4413def914870307e481` |
| `n4/cn42pgozlpucrwzw6k4knodkpjjyu6eylxud7z56zowl4slt3o4x.py` | Rank0 AOT6 backward; :119–144 one-hot reduction, :149–175 gradient; :192–210 allocations and saved-value release | `d5c4918362b81003d37b87301e982a7f5c8b87429c2dcce74d4e025d7a27853f` |
| `r6/cr62md4usvw5j6jz7jiyah4lfhawuquyzo63la3llzqdyfdicfnn.py` | Rank0 AOT7 dynamic forward; :271–302 split-V row reduction with three partials, merge, then loss SUM; example S128 at :311–314 | `252129041d355d7279771d05d5286d65166a348d7ba725459d19a017dcba746c` |
| `vm/cvmjnq7r5hlj7kd7w5wtsxy5myillrupwavptu22iim5gl2nqqvg.py` | Rank0 AOT7 dynamic backward; same BF16 input/output and two saved FP32 row statistics | `cd1b649abaacb32f22d8277d1f71c08bdbe73d2fd4868846a2a97a25a6d7f3d2` |

Ordinary forward inputs are BF16 `[2,2048,128256]` and int64 `[2,2048]`.
Saved values are original logits, labels, row maximum and log(sum(exp(x-max))).
Only two FP32 `[4096,1]` buffers and a scalar loss are allocated by CE forward.
Backward allocates one BF16 `[4096,128256]` gradient. Statistics and saved
inputs are released after the gradient kernel; outer chunk variables can retain
logits separately (K1b). Each statistic is16 KiB, versus1002 MiB for BF16 logits.
The unchanged donor saves one FP32 row LSE (16 KiB) instead of two stats (32 KiB),
while also allocating a transient16 KiB losses vector. This is no GiB-scale CE
saving, and the stable two-stat adaptation below removes that saved-stat saving.

The dynamic S128 specialization has small split-reduction scratch and a third
forward kernel. Its existence supports the monitoring contract inspection; it
does not prove that specialization executed in update111. The existing actual
Perfetto evidence `g3-baseline-perfetto.json` q31/q38 reports only the ordinary
three kernel names,128 calls each/rank/update, approximately201/198 ms summed
kernel durations. Those are overlapping-scope kernel sums, not elapsed CE cost,
host overhead, or an achievable speedup.

## Donor and integration contracts

K `ops/softcap_cross_entropy.py:12–55` saves logits, labels and FP32 row LSE,
returns FP32 SUM, and accepts first-order gradients only. Its wrapper :58–93
requires int64 labels, matching shapes/devices, CUDA FP16/BF16/FP32, and uses
contiguous flattened inputs; non-CUDA falls back to PyTorch. K
`ops/_softcap_cross_entropy.py:9–15,22–65` skips softcap at cap0, uses FP32 online
LSE over4096-column tiles and signed int64 row offsets, zeros ignored-label
loss/gradient, and stores the final gradient in logits dtype. It asserts valid
nonignored target ranges in forward (`debug=True`). Neither path requires a
full-vocabulary FP32 global buffer. Donor computes `x-(max+logsum)` whereas
baseline reconstructs `(x-max)-logsum`; reduction trees and FP fusion settings
also differ. Equal FP32 arithmetic intent does not establish elementwise parity.

### Confirmed finite-common-offset stability defect

The pinned donor forward literally forms `lse = maximum + log(denominator)`
and `loss = lse - chosen`; backward reads that saved FP32 LSE and uses
`exp(values-lse)`. For cap0, a row of equal BF16 logits `2**28`, vocabulary128256
and any valid target is a concrete failure. The offset is exactly representable
in BF16. The FP32 spacing above it is32; `log(128256)` is approximately11.761784,
so storing their sum rounds back to `2**28`. Donor loss becomes0 and every
reconstructed probability becomes1. With incoming scale1, the target gradient
becomes0 and every nontarget gradient becomes1, rather than approximately-1 and
`1/128256`. This is a material stability failure, not a tolerance choice.

Executed a tiny CPU proof using Python `math` and `struct.pack/unpack('f', ...)`
to round each named operation to FP32, without importing training code:

```python
f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
m = f32(2**28)
logsum = f32(math.log(f32(128256)))
lse = f32(m + logsum)
donor_loss = f32(lse - m)
stable_loss = -f32(f32(m - m) - logsum)
donor_p = f32(math.exp(f32(m - lse)))
stable_p = f32(math.exp(f32(f32(m - m) - logsum)))
```

Observed `logsum=11.761783599853516`, `lse=268435456.0`, donor loss0,
stable loss11.761783599853516, donor probability1 and stable
probability7.79690617491724e-06. The preserved baseline forward :184–186 and
backward :169–171 subtract maximum before logsum, retaining this normalization.
This arithmetic proof establishes the formula defect; it does not claim CUDA
kernel execution or bitwise transcendental parity.

Inspected preserved donor tests: `tests/unit_tests/test_softcap_memory.py`
imports `torchtitan.ops.softcap`, exercises pointwise softcap and chunk lifetime,
and does not test fused CE. The cap0 head comparison in
`tests/unit_tests/test_loss.py:392–418` constructs `ChunkedCELoss.Config` without
`fused_softcap_ce=True`; that field defaults false. Neither covers this fused
CUDA cap0 failure. No large-common-offset fused-CE regression was found there.

Required adaptation: retain FP32 maximum and FP32 log-denominator separately,
compute loss as `-((chosen-maximum)-logdenominator)`, and reconstruct backward
probabilities as `exp((value-maximum)-logdenominator)`. This preserves the
baseline stable centering while allowing the direct softmax-minus-one-hot
gradient algorithm. It needs two saved row statistics, so the original donor's
one-stat lifetime claim cannot be credited to that adaptation. A required
regression must execute the actual adapted CUDA forward/backward, including its
compiled supported path, with the equal BF16 `2**28` row and V128256, compare
loss and elementwise gradients to the stable baseline and include ignored rows.
It must fail on the unchanged donor. That regression and adaptation are open;
this assignment did not modify kernels or invent a new numerical tolerance.

G `components/loss.py:560–561,738–762,790–793,1030–1058` owns compiled CE,
per-chunk inner backward, SUM/global-valid-token division and MTP1 weight0.3.
Its FP32 `GradAccumulator` at :708–712 must remain. CE replacement alone need
not alter head GEMM, hooks, argmax/monitoring, GFSDP gradient buffering or SAC.
The head has its own GFSDP gradient buffer in G
`models/gift_moe/_base/parallelize.py:195–223`; the SAC path wraps model blocks,
while CE executes later on detached head chunks. Preserve that boundary.

K donor integration is not a drop-in compiled-loss replacement: K
`components/loss.py:776–804` directly calls the fused function outside `self.fn`,
therefore bypassing the existing compiled CE callable. It also invokes the head
with `apply_logit_softcap=False` (:712–715), unsupported by current G head
signature (`models/common/embeddings/token_debeddings.py:41–42`). Cap0 does not
need that head API change. Donor disables its fused route for head-local hidden
z-loss (:642–653); current crop selects norm-input hidden z-loss, which must
remain at its existing hook location. Donor's DTensor fallback must be retained
for unsupported paths rather than silently using a local-vocab CE. Actual cached
rank0 baseline CE input is local BF16, compatible with the donor kernel domain.

Remaining runtime gates: preserve model/loss compile requirement with an
explicit supported compiled integration (do not silently take donor's eager
bypass); eager and compiled elementwise loss/gradient checks including ignored
and all-ignored rows, nonunit incoming gradient, main/MTP weighting, ordinary2
and monitoring32 chunks, and actual head/hook/GFSDP integration; inspect activated
compiled donor kernels and saved lifetimes; repeated fixed-workload A/A and
alternating A/B with phase-matched memory before any benefit claim. No new
communication is implied by local CE, but changed head-backward duration can
shift overlap and buffer lifetime. Current evidence establishes neither a speed
gain nor production acceptance.

## Preserved evidence after cache cleanup

The four wrapper files and hashes in the table were preserved: each embeds its full
Triton source, graph comments, shape guards and allocation/return code. Separate
kernel cache copies, binaries and the remaining cache are unnecessary for these
source-contract claims. The canonical preservation manifest is
`artifacts/raw/g3-baseline-update111-001/k1a-compiled-ce-evidence/manifest.json`,
SHA256 `df0ef838c59ed84e199cba5defb279e7d6ff050aa099d1c0e20c79fe8af9465f`;
it records original paths, copied hashes and G3 source/config/trace provenance.
Keep this note with the existing compact Perfetto JSON/SQL. This assignment did
not itself copy or clean cache.
