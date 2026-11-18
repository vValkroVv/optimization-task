# GPU capacity selector on S

Compare S with `branchless_capacity_drop` disabled and enabled. S includes the
four fixes, FlashQLA `external_gate`, the MTP and layer-0 FFN splits, row-owner
embeddings, and the GFSDP schedule fixes. Use the S settings for both MoE block
templates, with the original 128-counter capacity all-reduce. The only training
configuration difference is the selector flag.

## Reference and expected cost

The SR008 timing and diagnostic receipts are `schedule-embedding-s-timing-140-004`
and `schedule-embedding-s-diagnostic-140-004`. They identify the reusable cache
as `runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache`, with
rank-specific Triton directories. The original 128-counter `capacity.py` used
by the S launch adapter has SHA-256
`2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120`.

The existing selector carries the predicate as a CUDA tensor. In the common
no-drop branch, it writes a 65,536-byte output in place of the old `valid.clone()`
allocation. The affected path includes the same EP8 collective and the same
consumers of the mask. The component check must account for the selector's
launch and device time, possible copies from `contiguous()`, the compiled
65,536-route kernel's resource use, and any change in peak memory or SAC
behavior. The native comparison keeps the 50,000 fixed expert quota, SAC,
Inductor, FP8, and the S data and tokenizer settings.

The S rank-0 trace at update 21
([local shortcut](../../artifacts/4fix_qla_embed.json)) has 39 windows from
128-counter all-reduce completion to the start of FP8 input row quantization.
MCP Perfetto measured 14.302518 ms total, 0.550977 ms of recorded GPU activity,
and 13.751541 ms without recorded GPU activity across streams. At slice 24303,
the interval was 324.064 µs, including 310.144 µs without recorded GPU activity.
There are 39 one-byte predicate transfers inside the matched windows and 42
across the whole capture. These are control observations. The 13.586 ms estimate
from retaining GPU activity plus 30% overhead is a scenario to compare with the
candidate trace.

## Reproduction

From the task checkout on the authorized, idle eight-H100 master, run
`bash runs/capacity-selector/campaign.sh <new-run-id>` in a detached session.
The campaign records the real router inputs in a single S update, checks and
times the isolated section on EP8, runs the two native 10+10 timing windows,
and profiles the three-microbatch candidate window at update 21. All modes
retain the original LR schedule, seed, production data, and accumulation.
The S launcher validates the historical source, runtime, and warm cache.

Read `artifacts/raw/<new-run-id>-campaign/campaign.log` and `exit-code.txt`
under this run directory. Use the existing native timing summaries and
the component summary for measurements. Run [capacity window SQL](capacity_windows.sql)
and [predicate sync SQL](predicate_sync.sql) in MCP Perfetto on matched traces.
Use [trace_analysis.py](trace_analysis.py) to cross-check the original events
and compare collective and launch signatures.

## Execution status

On 2026-11-18, `sr008-gpu` reconnected after an initial gateway reset. The
owned master `lm-mpi-job-c316d31f-c997-4f8e-81ce-0d2c5fe4aea5-mpimaster-0`
had eight idle H100 80 GB GPUs on one node. The cache producer reported `PASS`
and its per-rank Triton directories were present. The original capacity overlay
matched its recorded hash.

Local checks: seven existing fixed-capacity group tests passed. Command
generation and override parsing verified the capture, timing pair, and trace
recipes against the historical S bridge on CPU. Candidate hardware results
will be recorded after execution.
