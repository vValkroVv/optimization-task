# Repaired native200 attempt004

## Result

PASS for the second fresh native stability run. All eight ranks completed exactly200 successful optimizer/step spans and800 successful microbatch spans. Controller exit0 at16:47:50.596127UTC; owner exit0/natural_exit with all731 recorded task process identities absent; all eight live-recorded worker identities were absent.

[Completion evidence](completion-evidence.json) confirms fully parsed, newline-terminated JSONL, unique strictly increasing sequence IDs, correct span ordering/counts, no errors or incomplete spans, and native process-group teardown on every rank. Each rank has21 finite native loss/norm metrics at1,10,...,200 and42 finite token metrics with exact local131072/global1048576 counts. Final rank0 loss2.58007/norm1.6842 is a native displayed observation, without a bitwise comparison claim.

## Frozen configuration and activation

[Configuration checks](configuration-checks.json) prove the resolved configuration equals003 except dump_folder. Every rank retains FP8 projections, model and loss compilation, selective AC, fully_shard, HybridEP, accumulation4, graph0/mixorder0 and the pinned corrected scan header. The native launcher bypasses campaign observers and disables core capture and launch blocking, as reviewed before release.

[Live maps and selected environment](live-dependency-activation.json), observed16:38:59UTC, prove all eight workers loaded only the approved candidate extension. [Actual kernel activation](kernel-activation.json) verifies both grouped-wgrad shapes and the large-K specialization in every actual rank cache with candidate digest/include root and exact metadata/cubin hashes matching003.

Canonical native warmup reuses `/temp/.cache/rank_<rank>/deepgemm`; its summary reports compiled0/shared0/cached45. The fresh per-attempt shared cache is therefore empty: publication only occurs for newly compiled entries. Actual candidate rank-cache metadata and binaries establish activation. No cache policy or running configuration was changed.

[Runtime message review](runtime-message-review.json) finds no novel warning/error/fallback form relative to003; [selected messages](runtime-messages.json) preserve inherited warnings. No CUDA fault, OOM, hang, skip, required-feature loss or new runtime fallback was observed.

## Scope

Together with [003](../003/REPORT.md), this closes the requested pair of post-fix native200 stability runs. It establishes neither a performance improvement nor numerical parity, convergence, checkpoint continuation, RNG behavior or CUDA Graph acceptance. The exact corrupt native descriptor was not directly observed; the established fixture mechanism and native stability results retain that causal limitation.

Reproduction from repository root: `python runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/analysis/003/extract_completion.py 004`. It reads terminal artifacts and writes task evidence only. Report and extraction have no production-runtime impact; no additional GPU execution or source change was performed by this review.

The coordinator's [post-pass release receipt](../../attempts/004/root-post-pass-release.json),16:50:33UTC, confirms all eight GPUs idle with no applications, controller/owner/worker identities absent, and unchanged Source01098-file and original/candidate844-file manifests after both runs. Branch main and HEAD446431d are unchanged. The one terminal watcher exited0 after the controller receipt; this reviewer has no remaining monitor or GPU process.
