# GIFT training optimization

The current final task baseline is a production-shaped eight-layer 100B-dev crop
on eight H100s, with accepted V2 prefetch, the combined-four foundation, and
explicit FlashQLA `external_gate` selection. The measurements are provisional;
they do not establish a general speedup or production validation. See the
[baseline recipe](docs/baseline.md#current-final-task-baseline).

| Read | Purpose |
| --- | --- |
| [Current state](docs/status.md) | What is implemented, verified, unresolved and concluded |
| [Baseline](docs/baseline.md) | Canonical model/config and current final task recipe |
| [Latest A/B/C/D results](runs/production-data-abcd/report.md) | Production-tokenizer comparison of baseline, four fixes, FlashQLA and FFN/embedding changes |
| [Candidates](docs/candidates.md) | Potential improvements, local source code and compatibility limits |
| [CUDA Graphs](docs/cuda-graphs.md) | Historical findings and why graphs are not enabled by default |
| [Work plan](planning/README.md) | Editable priorities, experiment acceptance and next actions |
| [Validation](planning/validation.md) | Exact smoke procedure, correctness and performance evaluation |

`gift/` is the active, flattened training-source tree. `references/` contains
pinned candidate source snapshots and [provenance](references/provenance.json).
`runs/` holds compact evidence: the [crop smoke](runs/100b-crop8-ep8-smoke/report.md)
and [graph analysis](runs/cuda-graphs-evidence/README.md).

## Latest rank-0 traces

The four 15 November 2026 production-data diagnostics are available locally as
original Chrome trace JSON files, with GPU and FSDP annotations preserved:

| Variant | Local trace |
| --- | --- |
| Baseline | [baseline.json](artifacts/baseline.json) |
| Baseline + four fixes | [4fix.json](artifacts/4fix.json) |
| Four fixes + FlashQLA external_gate | [4fix_qla.json](artifacts/4fix_qla.json) |
| Four fixes + FlashQLA + MTP/layer-0 FFN splits and row-owner embeddings | [4fix_qla_embed.json](artifacts/4fix_qla_embed.json) |

`artifacts/` is ignored by Git. The [trace manifest](runs/production-data-abcd/trace-shortcuts.json)
records their original server paths, sizes and SHA256 hashes. Each trace contains
the first three complete forward/backward microbatches of update 21 after 20
warmup updates. These files are originals, not Perfetto lane derivatives.

Follow [AGENTS.md](AGENTS.md) for repository operation and source boundaries.
The full earlier documentation, LLM Foundry snapshot and campaign history remain
recoverable from Git revision `9d56011df976017972d8f2993b910bde03487e2a`;
see [recovery instructions](references/README.md). There is no active root CI.
