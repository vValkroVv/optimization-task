# Independent repaired native200 integration review

## Prelaunch decision

PASS for the bounded native200 attempts 003 and 004. This permits the root to release the two runs; it does not establish their acceptance. No GPU execution or source change was performed by this reviewer. Applied code-rules and perf-experiments.

Finite checklist: preserve the exact native payload and scan correction; bypass campaign observers; activate the candidate dependency with fresh DeepGEMM caches and source gates; preserve native logs and bounded cleanup; then inspect actual loaded dependency and both terminal results.

## Reviewed integration

- The two [launchers](../../attempts/003/launch.sh) are byte-identical, SHA-256 `1b11a59590cd167129619204a361358d0b0c6f29b9954c1c1ff0da5e780fc91c`. Their output roots derive from their own directories. The [controller](../../preparation/003/controller.py), SHA-256 `bab461d40e48d6bdef5f2163ce73a25935a3a0116bfd46bbc347fc20fd13aa50`, only accepts 003/004 with explicit repaired-native release status and refuses existing outputs. It retains the previously accepted owner hash and 1800-second deadline, and torchrun permits zero restarts.
- `train_scan_fix` installs only the existing pinned HybridEP bound correction, then calls native `torchtitan.train.main`. Native config/build/train/close and distributed teardown remain active. No campaign observer, phase marker, custom timer, memory sampler, core capture or launch blocking is enabled.
- Exact preset, 200 updates, 30000 scheduler horizon, seed42, all-eight UUID mapping, checkpoint/evaluation/profiling disabled, graph0/mixorder0, native rank logs and torchrun redirects match the frozen diagnostic contract. The existing source manifest gate verifies all98 pinned files and preset hash. No FP8, FlexAttention, FSDP2/EP8, SAC, compilation, optimizer, data or batch setting is overridden.
- Candidate package precedes GIFT and inherited Python paths before imports. Both original and candidate844-file trees are checked against hash-pinned manifests; their sole difference must be the approved descriptor header. Bytecode writes are disabled. The launch requests per-attempt local/shared paths; actual native warmup overrides the local path as recorded below. Candidate digest selection, rather than local-cache freshness, prevents old-header activation. The loader's actual activation must still be established from live extension maps and kernel metadata, not identical generated kernel.cu text.
- DG debug/cache policy variables are unset before import. Inspected `deep_gemm_fast_warmup.py:340–378`: its canonical policy keeps `DG_CACHE_FORCE_SKIP=1` after warmup; the launcher adds no alternate compile/fallback policy.

## Runtime cost and evidence boundary

The launch changes add pretraining file hashing and cold DeepGEMM compilation/cache publication. They add no per-step host observer, transfer, collective or synchronization. The dependency patch's device waits and publication fences retain the separately reviewed costs in [descriptor review](../descriptor-fix-001/REPORT.md); their latency and overlap effect remain unmeasured. No performance claim is made.

Native002's inherited warnings (Inductor complex RoPE lowering, checkpoint context, ignored record_function, quack disk-cache reload compilation) must be preserved and distinguished from a new fallback or lost model/loss compilation. Native descriptor corruption was not directly observed; passing these runs cannot establish that specific native causal claim.

## Required result review

For each run, verify all eight live workers map the candidate extension and the produced kernel metadata selects candidate include root/library digest `a122322508f98be4c07c04c7237e53ab`. Inspect resolved native config and activation markers. Parse every rank JSONL completely with unique increasing sequence IDs; require successful step/optimizer spans exactly1–200 and four successful fwd_bwd spans per step (800 total), rejecting any span with an error even if an end record exists. Check finite observed native loss/norm/token metrics, faults/OOM/hang/skips/new fallback, clean exit and precise process cleanup. Reuse native002's completion model without a new bitwise training gate.

Prelaunch review found no blocking defect. Preparation REPORT's draft manifest-hash transcription and pending asset hashes were reported to the root for evidence correction; executable gates contain the correct candidate manifest hash. Live activation and both run outcomes remain open.

## Native003 activation update

[Live maps](../../analysis/003/live-dependency-activation.json) prove all eight workers loaded only the candidate extension. [Configuration comparison](../../analysis/003/configuration-checks.json) proves the native002 resolved configuration is preserved except dump_folder. [Kernel activation](../../analysis/003/kernel-activation.json) records48 per-attempt shared metadata entries with candidate digest/include root and three K-grouped specializations including both production wgrad shapes; their cubins match all eight actual rank caches.

Resolved integration finding: `deep_gemm_fast_warmup.py:35–37,415–416` canonically overrides the requested local path with `/temp/.cache/rank_<rank>/deepgemm`. Thus local caches are reused; only the per-attempt shared cache is fresh. Candidate library digest separates the patched binaries from original entries. Preserve this native policy unchanged in004; no running configuration was altered.

## Final combined result

PASS for the bounded two-run native200 stability acceptance. [003](../../analysis/003/REPORT.md) and [004](../../analysis/004/REPORT.md) each independently establish all eight ranks'200 optimizer/step completions and800 microbatch completions, intact strictly ordered native JSONL without error spans, finite native loss/norm/token observations, complete required-feature activation, clean natural exit and task process cleanup. Both live maps select only the candidate extension. Both actual kernel-cache checks pin candidate library digest/include root and both production wgrad shapes. Resolved configurations are identical apart from output folders; launch bytes are identical. No new warning/error/fallback form appeared relative to native002/003.

The local-cache integration finding is resolved by evidence: native warmup owns `/temp/.cache/rank_<rank>/deepgemm`, reused by both runs.003 published48 candidate entries to its fresh shared cache;004 reused45 planned cached entries and published none.004's empty fresh shared cache is expected and its actual candidate rank-cache metadata/cubins were independently checked. No launch or source change was required.

Final checks reuse the prior protocol/caller/fixture review; inspect each run's live extension/environment and actual binary metadata; parse each terminal run once; compare resolved configs and runtime message forms; and check report links/whitespace. No material finding remains unresolved for this bounded native stability acceptance. The exact corrupt native descriptor remains unobserved, and numerical/runtime-cost/quality/checkpoint/CUDA Graph claims remain outside this verdict. No benchmark or speedup is inferred from native elapsed time.

The coordinator's [final post-pass release](../../attempts/004/root-post-pass-release.json) additionally records all GPUs idle, all precise task process identities absent and source/dependency hashes unchanged after execution. The reviewer's003 and004 terminal watchers both exited0 after their controller receipts; no extra monitor remains.
