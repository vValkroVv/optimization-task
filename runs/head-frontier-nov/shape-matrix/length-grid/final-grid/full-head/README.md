# Final isolated eight-H100 head grid

The B1 grid covers 100B C2, 100B C8 and 700B C8 at 4,096 through 262,144 positions in powers of two. Each cell runs two fresh eight-rank pairs of Baseline A + compile and the consuming CUDA CE plus cuBLASLt. The second pair reverses the first pair’s order. The starting order also alternates across cases and lengths. Both arms use the same fixture tiling and timing boundaries as the original [length grid](../../report.md).

`run_final_grid.py --plan-only` checks all final paths and routes without using CUDA. The [pinned policy](final-policy.json) uses ATen projection for 100B C8 at 131,072 positions. It keeps the calibrated projection for the same folded rows at 100B C2/32k. dW dispatch uses chunk count before folded shape. Projection and dHidden use the exact layout, shape, stride and alignment descriptor. A selected plan must run on every rank. An untuned shape uses its native path. Successful attempts retain per-rank activated algorithms and absolute paths with SHA-256. The replay also saves the settings that reached each OOM rank.

On the assigned GPU host, run the checked queue from this directory with a fresh control tag. The detached runner loads the image settings in `/etc/profile.d/wm.sh`, including the CUDA 13 compatibility library.

```bash
setsid nohup bash run_final_detached.sh final-policy-01 \
    --series final-policy \
    </dev/null >/dev/null 2>&1 &
```

Use `--cases 100b-c8 --lengths 131072` to reproduce the corrected cell alone. `run_grid.py` and `run_detached.sh` reproduce the initial matrix, with explicit `--dw-c2`, `--dw-c8`, and `--linear` path and SHA-256 arguments. `compare_projection.py` compares the prior and initial tuned paths at 100B C8/128k. It saves per-rank UTC measurements and three-second device telemetry alongside the results. That diagnostic uses a single extra recording hook after each timed window.

The scripts save `control/<tag>.{pid,started,log,finished,exit}` and `results/<series>/schedule.json`. Each arm at `results/<series>/<case>-b1-s<sequence>/pair-<repeat>/<variant>/attempt-<number>/` retains the command, GPU preflight, resolved settings, status, rank measurements, route receipts, OOM details and logs. A new control tag can resume the same series after a recoverable failure. Successful and CUDA OOM attempts are terminal. A mismatched schedule or changed source stops the resume.

`<series>-summary.json` lists each cell and repeat, with a median of the slowest rank’s synchronized wall time for the ten measured steps after ten warmups and allocator priming. Cell wall time is the median of the two repeat medians. The native GIFT peak is the highest measured total across ranks, steps and repeats. The integrated timer covers the ordinary BF16 weight all-gather, projection, weighted main and MTP CE, both gradient GEMMs, and FP32 SUM reduce-scatter. The 100B case also runs hidden z-loss. Host submit and host wait are wall-clock intervals on the controlling rank for each sample. The replay does not attribute device compute, wait or communication separately.
