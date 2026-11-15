# Full-shape embedding numerical check

**Verdict:** `pass`. Fixture lookup outputs are compared elementwise for exact equality. Gradient comparisons use exact equality as the only pass condition; finite nonzero differences remain unresolved, with no invented tolerance.

## Protocol

- Shape: V=128256, D=4096, B=2, S=4096; eight H100/NCCL ranks.
- Seed: `2048`. Fixture index SHA256: `7cc8d53152c801450fdae44ba84c182eb8dcb9106d8bba9e3c3d69ebdaa8a8c2`.
- Two matched, rank-local seeded BF16 consumer derivatives are scaled as native `/32` and MTP `*0.3`, then combined in BF16 before ordinary `output.backward(upstream)`. The exact combined tensor is reused across both variants and all backward states.
- Each variant checks the first backward, a second backward accumulated onto it, and a fresh backward after `weight.grad = None`. One baseline local FP32 gradient per rank is retained on CPU; no model or full table is exported.
- Memory counters cover setup, the identical synchronize/GC/empty-cache point before first forward, first complete forward/backward, warmed accumulated forward/backward, and the `set_none` reset call. These are memory checks, not timing results.

## Results

The first launch completed the all-rank measurements and wrote the JSON, then exited 1 in the rank-zero Markdown formatter; its log is retained at `embedding-numeric-launch.log`. The path bug is fixed, and this Markdown was regenerated offline from the saved rank data without repeating GPU work.

- Output exactly equal across all ranks and passes: **True**.
- Gradient checks exactly equal across all ranks: **True**.
- Non-finite elements counted across comparisons: **0**.
- Baseline post-setup cache: allocated 7,684,554,752–7,684,554,752 B; reserved 16,357,785,600–16,357,785,600 B.
- Owner post-setup cache: allocated 4,138,532,864–4,138,532,864 B; reserved 16,357,785,600–16,357,785,600 B.
- Baseline after equal cold-cache policy: allocated 7,684,554,752–7,684,554,752 B; reserved 7,952,400,384–7,952,400,384 B.
- Owner after equal cold-cache policy: allocated 4,138,532,864–4,138,532,864 B; reserved 5,851,054,080–5,851,054,080 B.
- Owner post-cleanup reserved at or below the baseline budget on every matching rank: **True**; matched rank deltas are in JSON.
- Baseline first-call peak: allocated 8,950,373,888–8,950,373,888 B; reserved 9,003,073,536–9,003,073,536 B.
- Owner first-call peak: allocated 5,519,048,704–5,519,053,824 B; reserved 7,384,072,192–7,384,072,192 B.
- Baseline warmed accumulated peak: allocated 8,950,373,888–8,950,373,888 B; reserved 9,003,073,536–9,003,073,536 B.
- Owner warmed accumulated peak: allocated 5,781,716,992–5,781,722,112 B; reserved 7,648,313,344–7,648,313,344 B.
- Baseline `grad=None` reset peak: allocated 8,950,373,888–8,950,373,888 B; reserved 9,003,073,536–9,003,073,536 B.
- Owner `grad=None` reset peak: allocated 5,519,048,704–5,519,053,824 B; reserved 7,648,313,344–7,648,313,344 B.

The earlier fixture pair reported allocated peaks of 8,950,373,888 B (baseline) and 5,667,090,944 B (owner), while reserved memory was 16,290,676,736 B and 18,610,126,848 B. After equal cleanup in this check, owner reserved memory is 2,101,346,304 B lower than baseline on every rank. The earlier owner reserve premium was retained allocator-cache state and does not persist after the matched cold-cache policy.

No gradient difference was observed in this check. A finite nonzero difference on a rerun would remain unresolved because the baseline distributed FP32 reduction and owner source-order FP32 sum can add contributions in different orders. The JSON has per-rank, per-state max absolute error, RMS, relative L2, unequal and non-finite counts.

## Reproduction

```text
PYTHONPATH=gift /opt/torchtitan-venv/bin/torchrun --standalone --nproc-per-node=8 runs/ffn-row-owner/embedding_numeric.py --execute --fixture /workspace-SR008.nfs2/vkropotin/optimization-task/runs/overnight-streaming-perf/artifacts/raw/mds4096/train --seed 2048 --output /workspace-SR008.nfs2/vkropotin/optimization-task/runs/ffn-row-owner/embedding-numeric-results.json
```

Repository revision: `2ada87125e433e1f76c4cb2bfad244b641a78a93`. Factory SHA256: `5a1a129b3c1d4fe239f66d93f664436aefade87e2fc09e98df70f5904f43b07b`. Owner source SHA256: `f3a01c6cf99b981db7b4ef66866fe46259af84e9d2c21e2044e9ae18ff1696fd`. Numeric script SHA256: `6f15ea7b230c0e794d0b3a15414cc99824a0bb4ccc95dcf0f61225caaa07afc0`.
