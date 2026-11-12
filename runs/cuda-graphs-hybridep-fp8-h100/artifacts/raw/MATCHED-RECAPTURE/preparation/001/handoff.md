# Matched recapture pair: reviewed preparation

Status: statically reviewed; GPU preflight and per-attempt coordinator release pending.

The interrupted worker prepared identical observer overlays and byte-identical A007 bootstrap copies, but its launcher copies still pointed to A007. Root adapted the two copies before execution. The unfinished, overbroad handoff is preserved in `handoff.interrupted-partial.md`; its proposed extra gates and check claims are not the actual launcher contract.

## Finite question and controls

Measure the integrated graph-policy cost without A007 profiler/verbose logging. Both runs use source006, the same 17 MoE graph exclusions, full-update marker, mixed-order reduction off, all required FlexAttention/SAC/FP8/HybridEP/model+loss compilation, seed42/data and schedule. Only the global graph flag and output path differ. Native26 updates; observer21–25, closure26. Counts are already global: 1,048,576 useful tokens/update, 5,242,880/window.

A007 has real dense/loss forward/backward launches on every rank but no settled representative reuse. This pair includes recurring recording costs. It is a descriptive diagnostic pair, not stable throughput, A/A noise, numerical acceptance or a finalist. Material cost/growth favors discarding A007 for the finite loss-only fallback; unexpectedly modest cost warrants targeted reuse investigation; a fault stops the pair for diagnosis. It replaces none of the required50-update/500-update or numerical/continuation gates for an actual finalist.

## Launch contract

Launch serially, one time each, using a detached isolated session with an external30-minute timeout (TERM, then30seconds before KILL). The existing runner records PID, cleans its own500ms memory sampler, exits on native torchrun failure with zero restarts, and retains source/preset/archive guards. Added guards pin the copied bootstrap and overlay. Root reviews the exact current eight-H100 identity, idle/app-empty state and prior task PID absence before each launch. Root creates each scope-specific release.json only then. The runner itself checks release status/scope/source hash; preflight validity, deadline and control-success gating are coordinator/monitor responsibilities, not additional shell claims.

Attempt001 is graph-off. Attempt002 is graph-on and remains unreleased until001 exit0, all-rank valid5-update window, optimizer26/forward-backward104 completion, intended config/activation and clean GPU release. No automatic retry or chained second launch. Source stays frozen throughout. Keep HOME and shared caches unchanged. No training checkpoints are produced.

## Review evidence

`root-review.json` records hashes and checks: bash syntax, exact off/on differences, same bootstrap/overlay, original A007 unchanged, source guards and existing cleanup. No CPU config probe was repeated because the A007 shared-bootstrap0/1 evidence applies unchanged. Runtime source guards and all-rank option/config records remain required. Fresh preflight/release and actual timing/memory are pending. Raw failure history is preserved.
