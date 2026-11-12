# Training optimization task

This repository tracks evidence-driven throughput work and contains flattened, ordinary tracked source snapshots in `gift/` and `llm-foundry/`. The source import records are in [additional/baseline](additional/baseline/README.md). They pin source identity; they do not select a representative workload or establish a measured training baseline.

The goal is to improve representative end-to-end training throughput while preserving the objective, data, precision, distributed behavior, checkpoint continuation, and agreed quality.

## Start here

1. Read the [optimization contract](docs/optimization-contract.md) and [operating rules](AGENTS.md).
2. Read the [agreed 100B-dev crop baseline](docs/baseline-100b-crop.md) and check the [tasks](backlog/README.md), [workflow](planning/workflow.md), and [validation protocol](planning/validation.md).
3. Use [research lessons](docs/research-lessons.md) as hypotheses and context, not as evidence for a new workload.
4. Record experiments, including failed and inconclusive results, in [runs/](runs/README.md).

The current default is the agreed eight-layer 100B-dev crop in [the baseline contract](docs/baseline-100b-crop.md). Required runtime ports, configuration checks and source review are complete. The [eight-H100 smoke passed](runs/100b-crop8-ep8-smoke/report.md): ten optimizer updates and 320 microbatches per rank, including monitoring steps, without checkpoints or profiler traces. This establishes basic execution, not a performance baseline or convergence result.

Browse the [meeting optimization candidates and local code map](docs/optimization-candidates.md) for Kindulov fusions, fused hidden z-loss and vision-related runtime approaches.

## Repository map

| Directory | Purpose |
| --- | --- |
| [gift/](gift/) | Flattened GIFT training-source snapshot. |
| [llm-foundry/](llm-foundry/) | Flattened LLM Foundry source/reference snapshot. |
| [ci-cd/](ci-cd/README.md) | Reserved CI/CD documentation; no active pipeline is included. |
| [backlog/](backlog/README.md) | Current task scope, dependencies, and acceptance. |
| [docs/](docs/README.md) | Current crop baseline, task boundary, optimization contract, and research lessons. |
| [planning/](planning/README.md) | Reusable workflow and validation protocol. |
| [runs/](runs/README.md) | Campaign reports and evidence references. |
| [additional/](additional/README.md) | Source baseline and import provenance. |

## Current scope

The task-local GIFT source base was refreshed from upstream `main` at `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`; see the [11 November refresh receipt](additional/baseline/gift-main-2026-11-11.md). The selected 100B-dev mechanisms are ported into that base. LLM Foundry remains at `1ec1360304a182844372f6c3989c3f10288fcba8`. The historical import records preserve the earlier GIFT pin and recursively flattened dependency provenance. Credential-literal sanitization exceptions remain documented in [additional/baseline](additional/baseline/README.md); those lines are not byte-identical to upstream.

The 26 October task boundary is retained as historical context. The current baseline selects GIFT 3.5, FP8, model/loss compilation, SAC, and GFSDP2, with EP8 and PP/TP/CP at 1; details and limits are in the [baseline contract](docs/baseline-100b-crop.md). [OPT-004](backlog/tasks/opt-004-cuda-graphs-hybridep-fp8.md) preserves the separate 3B HybridEP/FP8 stability campaign; it is not the current baseline, and its CUDA Graph/performance comparison remains paused. No performance result is claimed for the new crop. No CI/CD pipeline is active; [ci-cd/](ci-cd/README.md) is reserved for future, scoped work.

Source snapshots are tracked files in this repository. Do not initialize nested repositories or submodules inside them. Source edits, compute, dependency installation, and publication require their own authorization.
