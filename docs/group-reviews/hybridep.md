# HybridEP group review

The owner’s comment refers to the code in
[`optimization-task/gift/torchtitan/experiments/four_fusion/`](../../gift/torchtitan/experiments/four_fusion/).

> Complete fucking slop

## 2. Routing metadata

Keep the existing routing implementation. Gluon changed training TPS by
−0.08% on EP8/eFSDP1 and +0.09% on EP4/eFSDP2 against original A.
Across seven MoE layers and 32 microbatches, the kernel timings imply
about 4.3 ms of GPU work saved per rank per update, or 0.03% of a
14.5-second update before accounting for overlap.
[30+30 results](../../.worktrees/hybrid-ep-fixes/runs/hybrid-ep-fixes/report.md#3030-acceptance-on-original-a)

The tested Gluon candidate uses four token rows and four warps
per CUDA block. Each warp handles one row, so 8,192 tokens launch 2,048
blocks. The candidate uses this layout inside the existing custom op for
contiguous CUDA inputs with 64, 128 or 256 experts. It derives `top_k`
from the ID tensor width
and writes contiguous routing flags and FP32 probabilities, including the
trailing sink elements. Strided inputs and other expert counts within the
CUDA kernel bounds use the one-token Triton kernel.
[Routing candidate](../../.worktrees/hybrid-ep-fixes/runs/hybrid-ep-fixes/candidate/routing_metadata.py)

The earlier eight-row prototype sweep covered top‑8 with 8,192 tokens and 64,
128 or 256 experts, plus 16,384 tokens with 128 experts. Outputs matched bitwise.
At the native 8,192-token, 128-expert shape, GPU time fell from 5.91–5.96 to
3.78–3.83 µs per call. Across the sweep, GPU time fell 23.1–46.9% and
host-to-completion median latency fell 2.5–5.0% with the custom-op wrapper.
[Sweep results](../../.worktrees/hybrid-ep-fixes/runs/hybrid-ep-fixes/report.md#sweep-results)

The layout sweep selected 4 × 4 as the best single layout across top‑4,
top‑8 and top‑16. At native top‑8, it saved 0.03–0.05 µs of GPU time per call
against 8 × 4, with no repeatable host-to-completion gain.
[Layout sweep](../../.worktrees/hybrid-ep-fixes/runs/hybrid-ep-fixes/report.md#gluon-launch-layout-sweep)

## GPU capacity selector

GIFT already implements the selector behind `token_drop.branchless_capacity_drop`.
It defaults to `false`. The selector arms set it to `true` in both the GDN and
MLA MoE block configs. This is a config-only switch.
[Config default](../../gift/torchtitan/config/moe.py#L24) ·
[Selector implementation](../../gift/torchtitan/models/common/moe/dispatch/capacity.py#L143)

The selector increased TPS by 0.75% on EP8/eFSDP1 and 0.38% on EP4/eFSDP2.
Native GIFT total GPU memory increased by 0.31 GiB and 0.19 GiB, respectively.
Each arm ran once. All six arms used original A (`e9b7cdbb4`) and a shared
compiler cache. The combined arms were skipped.
[30+30 matrix](../../.worktrees/hybrid-ep-fixes/runs/hybrid-ep-fixes/report.md#3030-acceptance-on-original-a) ·
[Final approval criteria](../../planning/validation.md#memory-and-final-approval)
