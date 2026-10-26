# Optimization contract

## Goal

Increase throughput on the selected representative end-to-end workload while preserving the objective, data, precision, distributed behavior, checkpoint continuation, and agreed quality. “End-to-end” includes recurring input, evaluation, checkpoint, and other job phases at their production cadence.

The [initial task boundary](task-boundary.md) specifies GFSDP, model and loss compilation, FP8, and SAC/full-checkpointing directions, with PP deprioritized for now. These are task requirements, not a tested or resolved production configuration. Record the concrete model, preset, data, topology, precision recipe, thresholds, and authorized compute before using this contract to accept a candidate.

## Behavior to preserve

- **Data and objective:** Preserve source examples, ordering and sampling, tokenizer, truncation, target shifting, masks, packed sequence boundaries, attention context, and convolution/scan state resets. Preserve each loss component's weighting, denominator, and eligible target population.
- **Training path:** Preserve optimizer, schedule, global batch, gradient accumulation, activation checkpointing, compilation, and model execution behavior unless an explicit scope change is accepted.
- **Distributed path:** Preserve intended rank groups, reductions, scaling, and ownership. Record physical rank placement and topology. Changing a group or reduction can change both communication and floating-point grouping.
- **State:** Preserve loader cursor and relevant RNG streams, validation state, routing state, optimizer moments and step, and checkpoint compatibility. Test exact continuation separately from weights-only loading.
- **Precision:** FP8 is part of the intended target baseline. Resolve and record its recipe and accumulation behavior, then preserve them in optimization candidates. Changes beyond the selected recipe need explicit agreement and appropriate quality validation.

## Numerical acceptance

For a computation-preserving change, compare baseline and candidate under matched inputs, seeds, configuration, and deterministic settings where available. Compare loss and elementwise preclip gradients, then norm/scale, postclip gradients, optimizer moments, and parameter updates. Equal global norms or equal rounded weights alone are insufficient.

State the tested boundaries, exact structural invariants, dtype-specific tolerances, nondeterminism, and any unsupported path. When the mathematical computation changes, parity is not the acceptance criterion; use an explicitly accepted quality/convergence plan and thresholds.

## Performance acceptance

Confirm the rank-zero resolved configuration and active feature markers. Measure with the production parallel groups and topology. Establish A/A noise, then alternate baseline and candidate in unprofiled runs after separating startup, compilation, autotuning, and warmup. Use traces for mechanism and repeated unprofiled series for magnitude.

Report non-overlapping elapsed scopes and identify overlap among host work, device kernels, copies, waits, and collectives. Include job lifecycle costs at their real cadence and paired memory measurements. State exclusions and uncertainty. A kernel result, trace duration, or single run cannot establish integrated speedup.
