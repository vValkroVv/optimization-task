# Training optimization task

This repository tracks evidence-driven throughput work and contains flattened, ordinary tracked source snapshots in `gift/` and `llm-foundry/`. The source import records are in [additional/baseline](additional/baseline/README.md). They pin source identity; they do not select a representative workload or establish a measured training baseline.

The goal is to improve representative end-to-end training throughput while preserving the objective, data, precision, distributed behavior, checkpoint continuation, and agreed quality.

## Start here

1. Read the [optimization contract](docs/optimization-contract.md) and [operating rules](AGENTS.md).
2. Check the [tasks](backlog/README.md), [workflow](planning/workflow.md), and [validation protocol](planning/validation.md).
3. Use [research lessons](docs/research-lessons.md) as hypotheses and context, not as evidence for a new workload.
4. Record experiments, including failed and inconclusive results, in [runs/](runs/README.md).

## Repository map

| Directory | Purpose |
| --- | --- |
| [gift/](gift/) | Flattened GIFT training-source snapshot. |
| [llm-foundry/](llm-foundry/) | Flattened LLM Foundry source/reference snapshot. |
| [ci-cd/](ci-cd/README.md) | Reserved CI/CD documentation; no active pipeline is included. |
| [backlog/](backlog/README.md) | Current task scope, dependencies, and acceptance. |
| [docs/](docs/README.md) | Optimization contract and research lessons. |
| [planning/](planning/README.md) | Reusable workflow and validation protocol. |
| [runs/](runs/README.md) | Campaign reports and evidence references. |
| [additional/](additional/README.md) | Source baseline and import provenance. |

## Current scope

The selected upstream source revisions are GIFT `f840cf25e4294f93a5c91a20707c30b97e3c1dbf` and LLM Foundry `1ec1360304a182844372f6c3989c3f10288fcba8`; the import flattens recursively pinned dependency trees. Credential literals in three dependency helper files were sanitized, with paths and exceptions recorded in the import report. See [additional/baseline](additional/baseline/README.md) for provenance; sanitized snapshots are not byte-identical at those lines.

The production model, workload, preset, dataset, topology, precision, numerical thresholds, and compute allocation have not been selected here. This repository contains no measured optimization result. No CI/CD pipeline is active; [ci-cd/](ci-cd/README.md) is reserved for future, scoped work.

Source snapshots are tracked files in this repository. Do not initialize nested repositories or submodules inside them. Source edits, compute, dependency installation, and publication require their own authorization.
