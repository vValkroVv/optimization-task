# GIFT training optimization

The selected production stack for the eight-layer 100B-dev crop on eight H100s
includes S + the GPU capacity selector + five Hopper fusions, the three expert
GFSDP flags, folded CE, four backward optimizations, and CuTe statistics-only
SAC replay. Its one native 10+10 follow-up measured 180,393.28 global physical
tokens/s, with 60.3078 GiB peak allocated and 62.2051 GiB peak reserved. The
four-fix control measured 179,761.44 tokens/s before CuTe stats-only SAC was
promoted. See the [launch recipe](docs/baseline.md#launch-the-selected-recipe)
and [current state](docs/status.md#production-baseline-and-measured-throughput).

| Read | Purpose |
| --- | --- |
| [Current state](docs/status.md) | What is implemented, verified, unresolved and concluded |
| [Baseline](docs/baseline.md) | Canonical model/config and current final task recipe |
| [Throughput and memory matrix](docs/status.md#production-baseline-and-measured-throughput) | Production-tokenizer measurements from the original baseline through the final stack |
| [GPU capacity selector](runs/capacity-selector/report.md) | S versus S with GPU capacity-drop selection, including the native 10+10 comparison and matched Perfetto traces |
| [Candidates](docs/candidates.md) | Potential improvements, local source code and compatibility limits |
| [CUDA Graphs](docs/cuda-graphs.md) | Historical findings and why graphs are not enabled by default |
| [Work plan](planning/README.md) | Editable priorities, experiment acceptance and next actions |
| [Validation](planning/validation.md) | Exact smoke procedure, correctness and performance evaluation |

`gift/` is the active, flattened training-source tree. `references/` contains
pinned candidate source snapshots and [provenance](references/provenance.json).
`runs/` holds compact evidence: the [crop smoke](runs/100b-crop8-ep8-smoke/report.md)
and [graph analysis](runs/cuda-graphs-evidence/README.md).

## Latest rank-0 traces

The latest available production-data diagnostics are available locally as
original Chrome trace JSON files, with GPU and FSDP annotations preserved.
A/B/C retain their 15 November captures. The two S traces show the stack
before and after enabling the GPU capacity selector.

| Variant | Local trace |
| --- | --- |
| Baseline | [baseline.json](artifacts/baseline.json) |
| Baseline + four fixes | [4fix.json](artifacts/4fix.json) |
| Four fixes + FlashQLA external_gate | [4fix_qla.json](artifacts/4fix_qla.json) |
| S: four fixes + FlashQLA + MTP/layer-0 FFN splits + row-owner embeddings + GFSDP schedule repairs | [4fix_qla_embed.json](artifacts/4fix_qla_embed.json) |
| S + GPU capacity selector (pre-fusion control) | [4fix_qla_embed_selector_gpu.json](artifacts/4fix_qla_embed_selector_gpu.json) |

`artifacts/` is ignored by Git. The [trace manifest](runs/production-data-abcd/trace-shortcuts.json)
records their original server paths, sizes and SHA256 hashes. Each trace contains
the first three complete forward/backward microbatches of update 21 after 20
warmup updates. These files are originals, not Perfetto lane derivatives.

Follow [AGENTS.md](AGENTS.md) for repository operation and source boundaries.
The full earlier documentation, LLM Foundry snapshot and campaign history remain
recoverable from Git revision `9d56011df976017972d8f2993b910bde03487e2a`;
see [recovery instructions](references/README.md). There is no active root CI.
