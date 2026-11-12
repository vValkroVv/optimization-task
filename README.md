# GIFT training optimization

The default is a production-shaped eight-layer 100B-dev crop on eight H100s.
Its training smoke passed. Performance improvements have not yet been measured.

| Read | Purpose |
| --- | --- |
| [Current state](docs/status.md) | What is implemented, verified, unresolved and concluded |
| [Baseline](docs/baseline.md) | Canonical model/config, SAC, microbatch, precision and runtime |
| [Candidates](docs/candidates.md) | Potential improvements, local source code and compatibility limits |
| [CUDA Graphs](docs/cuda-graphs.md) | Historical findings and why graphs are not enabled by default |
| [Work plan](planning/README.md) | Editable priorities, experiment acceptance and next actions |
| [Validation](planning/validation.md) | Exact smoke procedure, correctness and performance evaluation |

`gift/` is the active, flattened training-source tree. `references/` contains
pinned candidate source snapshots and [provenance](references/provenance.json).
`runs/` holds compact evidence: the [crop smoke](runs/100b-crop8-ep8-smoke/report.md)
and [graph analysis](runs/cuda-graphs-evidence/README.md).

Follow [AGENTS.md](AGENTS.md) for repository operation and source boundaries.
The full earlier documentation, LLM Foundry snapshot and campaign history remain
recoverable from Git revision `9d56011df976017972d8f2993b910bde03487e2a`;
see [recovery instructions](references/README.md). There is no active root CI.
