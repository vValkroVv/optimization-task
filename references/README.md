# Source references and recovery

[The candidate overview](../docs/candidates.md) maps approaches to local source.
Each selected snapshot preserves upstream paths under `source/`, its license,
and a manifest of revision, Git blob IDs and SHA-256 hashes:

- [Kindulov](kindulov/manifest.json): softcap/CE and local recomputation.
- [Hidden z-loss](hidden-z-loss/manifest.json): GIFT integration and external kernel pin.
- [Vision runtime](vision-runtime/manifest.json): compact loss, prefetch, GFSDP and metrics.

These 109 files are inert reference material, not installable checkouts. Upstream
README links and test imports may refer to files outside the selected snapshot.
The maintained code map is in `docs/candidates.md`. External CudaCherry kernels
are not included; the vision collection is not confirmed as the promised list.

[provenance.json](provenance.json) records GIFT/component source identities,
training integration and smoke revisions, the pre-cleanup GIFT tree ID and the
publication sanitization exception. Git records the exact task source bytes.

## Recovering full historical material

All removed tracked documents, LLM Foundry source, source-import manifests and
intermediate CUDA Graph campaign artifacts are preserved at:

```text
9d56011df976017972d8f2993b910bde03487e2a
```

Read a historical file without changing the active checkout:

```bash
git show 9d56011df976017972d8f2993b910bde03487e2a:docs/gift-4-700b-training-flow.md
git show 9d56011df976017972d8f2993b910bde03487e2a:runs/cuda-graphs-hybridep-fp8-h100/report.md
```

List an archived subtree with `git ls-tree -r --name-only <revision> -- <path>`.
Export selected files into a separate destination with `git archive`; do not
reset the current branch to restore an old experiment. A shallow clone may need
to fetch the historical revision using its authorized remote.

Useful archived paths: `llm-foundry/`, `additional/baseline/`, `backlog/`,
`docs/gift-4-ready-optimizations.md`, `docs/gift-4-700b-training-flow.md`, and
`runs/cuda-graphs-hybridep-fp8-h100/`. Complete history includes failed runs and
qualifications intentionally omitted from current high-level navigation.

Large ignored raw logs/traces/caches were never guaranteed by Git. Their locations
and hashes remain in the retained receipts or historical reports. This cleanup
did not remove those untracked payloads or alter shared installed dependencies.
