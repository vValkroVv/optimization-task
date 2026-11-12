# Experiment workflow

1. Read [current state](../docs/status.md) and [baseline](../docs/baseline.md).
   Reuse completed evidence; do not repeat a smoke to answer a different property.
2. Choose one question from [the plan](README.md). Define finite acceptance,
   constraints and what each outcome changes. Inspect existing implementations
   and their fit before creating a new workaround.
3. Freeze config/data/source and establish the measurement controls described in
   [validation](validation.md). Inspect actual bottleneck evidence before choosing
   a substantial optimization. Configuration requests alone are not activation proof.
4. Work only within authorized source, dependency and compute scope. Review
   correctness and runtime costs of the affected path before implementation and
   once against the final result. Preserve other contributors' work.
5. Verify the affected behavior, then measure performance only after the required
   correctness gates pass. If a control fails, record it and do not treat the
   remaining candidate timing as a valid comparison.
6. Keep one report and compact result per experiment. Record source identities,
   commands and evidence needed to reproduce conclusions. Put large raw traces,
   checkpoints and logs outside Git with explicit locations and hashes.
7. Promote a candidate only after its stated gates pass. Update the baseline
   only by an explicit decision; update [current state](../docs/status.md) and
   [the plan](README.md) without duplicating the run history.

Stop when the bounded question is answered. Before an extra test, name the
remaining uncertainty and decision it changes. CPU checks do not close GPU/NCCL
or performance gates. Keep failures and unavailable checks explicit.
