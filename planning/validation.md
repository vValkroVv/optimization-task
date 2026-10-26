# Validation protocol

This protocol is for an authorized, selected production workload. It does not authorize compute, source edits, data access, or dependency installation.

## Before an experiment

1. Verify the source revision and recursive dependency pins using [baseline provenance](../additional/baseline/README.md). Record the working diff.
2. Freeze the workload: model/preset, data and split, tokenizer, sequence/packing rules, optimizer, global batch, schedule, seed, evaluation and checkpoint cadence, precision, AC/compile settings, and agreed quality thresholds.
3. Record hardware class, rank placement, mesh dimensions, exact process-group membership, network topology, software/build versions, launcher, and timer scopes. Check all allocated nodes before an authorized run.
4. Inspect the rank-zero resolved configuration and confirm active markers for the features under study. Do not infer activation from a requested setting.
5. Record the expected correctness and runtime costs before implementation.

## Correctness gates

For a computation-preserving change, use matched data and states. Compare loss, elementwise preclip gradients, norm and clip scale, postclip gradients, optimizer moments, and parameter updates. Check boundary cases for masks, packed sequences, short/long lengths, unequal valid-target counts, and stateful paths affected by the change. Compare eager and compiled execution when both are supported.

Use exact checks for structural invariants such as forbidden cross-document attention or causal zero gradients. Use tolerances only for explicitly identified floating-point outputs, with dtype, comparison method, and nondeterminism stated. A norm-only or weight-only comparison is insufficient.

Test checkpoint continuation from the same saved point, including optimizer, schedule, loader cursor, relevant RNG, and persistent model state. Report weights-only loading separately. If the candidate changes the mathematical function, use the accepted quality/convergence plan rather than claiming parity.

## Performance gates

- Run an identical A/A control pair to estimate noise.
- Alternate baseline and candidate in unprofiled runs with identical workload and topology.
- Separate startup, compilation, autotuning, warmup, stable steps, validation, checkpointing, and shutdown. Compare stable-step series and report medians and spread.
- Use traces to explain a bottleneck or overlap mechanism; estimate magnitude from repeated unprofiled runs.
- Define elapsed scopes for host submission, device work, synchronization/waits, transfers, collectives, and integrated training. Mark overlapping intervals; never add them as if they were disjoint.
- Include data preparation, metric reads, evaluation, checkpoint I/O, and other lifecycle costs at their production cadence.
- Pair memory results with the same phases and windows; identify allocator and device-level counters and their limits.
- Report failed/skipped checks, exclusions, noise, uncertainty, job and source identities, and remaining hardware or quality gates.

One trace, one run, an isolated kernel, or an unrepresentative fixture does not establish an end-to-end speedup.
