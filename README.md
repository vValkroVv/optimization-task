# Training optimization task

This repository tracks evidence-driven throughput work and contains flattened, ordinary tracked source snapshots in `gift/` and `llm-foundry/`. The source import records are in [additional/baseline](additional/baseline/README.md). They pin source identity; they do not select a representative workload or establish a measured training baseline.

The goal is to improve representative end-to-end training throughput while preserving the objective, data, precision, distributed behavior, checkpoint continuation, and agreed quality.

## Start here

1. Read the [optimization contract](docs/optimization-contract.md) and [operating rules](AGENTS.md).
2. Check the [tasks](backlog/README.md), [workflow](planning/workflow.md), and [validation protocol](planning/validation.md).
3. Use [research lessons](docs/research-lessons.md) as hypotheses and context, not as evidence for a new workload.
4. Record experiments, including failed and inconclusive results, in [runs/](runs/README.md).

The selected research campaign is [OPT-004: CUDA Graphs for 3B HybridEP FP8 on eight H100 GPUs](backlog/tasks/opt-004-cuda-graphs-hybridep-fp8.md). It contains the full test plan, completion criteria and execution goal. Planning and source refresh do not mean that the GPU campaign has run.

## Repository map

| Directory | Purpose |
| --- | --- |
| [gift/](gift/) | Flattened GIFT training-source snapshot. |
| [llm-foundry/](llm-foundry/) | Flattened LLM Foundry source/reference snapshot. |
| [ci-cd/](ci-cd/README.md) | Reserved CI/CD documentation; no active pipeline is included. |
| [backlog/](backlog/README.md) | Current task scope, dependencies, and acceptance. |
| [docs/](docs/README.md) | Task boundary, optimization contract, and research lessons. |
| [planning/](planning/README.md) | Reusable workflow and validation protocol. |
| [runs/](runs/README.md) | Campaign reports and evidence references. |
| [additional/](additional/README.md) | Source baseline and import provenance. |

## Current scope

GIFT is refreshed from upstream `main` at `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`; see the [11 November refresh receipt](additional/baseline/gift-main-2026-11-11.md). LLM Foundry remains at `1ec1360304a182844372f6c3989c3f10288fcba8`. The historical import records preserve the earlier GIFT pin and recursively flattened dependency provenance. Credential-literal sanitization exceptions remain documented in [additional/baseline](additional/baseline/README.md); those lines are not byte-identical to upstream.

The initial task boundary specifies GFSDP, model and loss compilation, FP8, and SAC/full-checkpointing directions, with PP deprioritized for now. See the [task boundary](docs/task-boundary.md) for that historical exchange. The 11 November request selects the exact `gift_moe_3b_hybridep_fp8_compile.yaml` workload and one server with eight NVLink-connected H100 GPUs for OPT-004. Its inherited FSDP2, EP8, FP8, selective checkpointing, compiled model/loss and chunked cross entropy take precedence over the earlier general directions. Server access, runtime facts and execution evidence remain open in that task. This repository contains no measured result for this campaign. No CI/CD pipeline is active; [ci-cd/](ci-cd/README.md) is reserved for future, scoped work.

Source snapshots are tracked files in this repository. Do not initialize nested repositories or submodules inside them. Source edits, compute, dependency installation, and publication require their own authorization.
