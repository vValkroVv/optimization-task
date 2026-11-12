# Repaired native200 attempt003

## Result

PASS for this native stability run. All eight ranks completed exactly200 successful optimizer/step spans and800 successful microbatch spans. The controller exited0 naturally at16:34:57.191445UTC; the owner recorded all1031 process identities absent and all eight live-recorded worker identities were absent. No training source, launch configuration or running process was changed by this analysis.

[Completion evidence](completion-evidence.json) parses every native rank JSONL fully, requires a final newline and strictly increasing unique sequence IDs, verifies span counts/order and rejects any errored span. No error event, unclosed span, skipped step or incomplete rank was found. Every rank emitted native process-group teardown. Native observed loss/norm metrics occur at1,10,...,200 and are finite on every rank; the final rank0 values are loss2.58007/norm1.6842. Every rank's42 token observations are finite and exact (local131072/global1048576).

## Preset and actual dependency

[Resolved configuration comparison](configuration-checks.json) proves equality with native002 except dump_folder: exact imported3B preset, FSDP2/EP8, FlexAttention, model/loss compile, selective AC, original FP8 recipe, local8/global256/sequence4096/accumulation4, seed42, optimizer/data, scheduler30000 and stop200. Checkpoint, validation and profiling remain disabled. All eight native stdout logs prove FP8 projections, both compilation components, selective AC, fully_shard, accumulation4 and HybridEP buffer activation. All live manifest header hashes retain the approved bound correction.

[Live dependency receipt](live-dependency-activation.json), observed16:25:48UTC, records precise worker PIDs/start ticks and maps showing only the candidate extension path, with unchanged binary hash99b7afad2e2fe7ebabd26c22afc99dc652e2e6935fcbf37324359eb5aeeefe26. [Kernel activation](kernel-activation.json) verifies48 per-attempt shared-cache metadata records select candidate library digest a122322508f98be4c07c04c7237e53ab and its include root. The M1536/N1024 and M2048/N1536 grouped-wgrad specializations, plus large-K specialization, have preserved metadata/cubin hashes and match all eight actual rank-cache cubins.

The canonical native warmup overrides the launch-requested DG_JIT_CACHE_DIR with `/temp/.cache/rank_<rank>/deepgemm`; these local caches are reused. The per-attempt shared cache is fresh. Candidate library-digest selection distinguishes patched entries from original entries. This correction changes the evidence description, without altering native source or running policy.

## Runtime messages and limits

[Message review](runtime-message-review.json) found no novel warning/error/fallback form relative to native002. [Raw selected messages](runtime-messages.json) preserve inherited Triton default launch-parameter warnings, quack disk-cache reload compilation, Inductor complex RoPE lowering, library deprecations, NCCL async policy override, HF-index warning and dataset looping. No required compilation component was disabled. No CUDA fault, OOM, hang or runtime skip was observed.

This establishes one post-fix native200 stability success. The second fresh run remains required. No benchmark, speedup, bitwise parity, convergence, RNG, checkpoint-continuation or CUDA Graph claim follows. The exact corrupt native descriptor remains unobserved; the causal qualification in the descriptor review is retained. Evidence extraction and report generation have no production-runtime impact.

Reproduction from repository root: `python runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/analysis/003/extract_completion.py 003`. Existing executable source/dependency gates and prior independent descriptor/fixture review were reused.
