# Initial task boundary

The [26 October 2026 exchange](../additional/task-inputs/2026-10-26-initial-scope.md) sets the current research direction. It gives task requirements, not a tested or resolved production configuration.

## Current direction

- Target GFSDP. The source spells this `Gfsdp`; preserve that spelling in the evidence. No canonical config enum or implementation is inferred.
- Compile both the model and loss.
- Keep `gift 4` as supplied. Its exact meaning, model, and preset are unresolved; this does not assert a version, GPU count, or model size.
- Include FP8 in the intended target baseline. The recipe, backend, scaling, accumulation behavior, and quality gates remain unresolved. After those choices are recorded, an optimization candidate must preserve the selected recipe; changes beyond it need explicit agreement and quality validation.
- Treat SAC and full activation checkpointing as relevant target modes to resolve. The exact policy is open and the exchange does not establish that both run simultaneously.
- Deprioritize PP for now, following the stated rationale that it has little value on small models across one or two nodes. This is planning context, not a production topology, restriction, or support statement.
- Optimize throughput without a stability tradeoff.

The exchange does not select a specific optimization or state whether parallelism-independent opportunities should take priority.

## Unresolved before baseline acceptance

Record the concrete model and meaning of `gift 4`; dataset, sequence length, packing, batch, optimizer and schedule; GFSDP implementation, degrees, and process groups; GPU and node topology; FP8 recipe and numerical behavior; SAC/full-checkpointing policy; compile settings; evaluation and checkpoint cadence; numerical tolerances and quality gates; performance threshold; and authorized compute. Keep source revision pins unchanged unless separately authorized.

The initial scope is a task boundary only. It is not evidence that the settings are implemented, active, or measured.
