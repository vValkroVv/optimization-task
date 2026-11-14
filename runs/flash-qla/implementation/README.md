# FlashQLA implementation artifacts

These files preserve the task-local implementation and pinned FlashQLA source snapshot.

## Reproduction order

1. Start from task source revision `0f32623d5e0e7544b20c1f047d94236cb6697758`.
2. From the task root, apply `runs/flash-qla/implementation/implementation.patch`.
3. From the task root, extract the pinned package with `tar -xzf runs/flash-qla/implementation/flashqla-source.tar.gz -C gift/contrib`.
4. Reuse the shared experiment dependencies already available; these artifacts add no installation or shared-environment change.
5. Follow the canonical runtime recipe and evidence in [the plan](../plan.md) and [the pilot record](../pilot.json).

The package tarball is a deterministic, source-only archive. Its SHA-256 identifies this task tarball and is distinct from the full upstream archive SHA-256 in [the raw source receipt](../artifacts/raw/flashqla-source.json). The seven generated benchmark result dumps and Python bytecode/cache files are excluded.

The patch retargets the MDS fragment path in `runs/overnight-streaming-perf/mds-dataloader.yaml` to the personal task setup. That path is task-local setup, not a generic production configuration. Reproduction does not install dependencies, promote defaults, or claim production acceptance.
