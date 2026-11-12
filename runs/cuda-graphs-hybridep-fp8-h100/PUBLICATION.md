# Campaign publication

The source changes, campaign report, task record, and compact text evidence are versioned. [publication-manifest.json](publication-manifest.json) identifies the preserved reports, review results, commands, configurations, patches, and verification receipts under `artifacts/raw/`. Their original paths and bytes are retained.

Large profiler exports, native raw logs, CUDA core dumps, model/tensor payloads, build caches, copied installed packages, and transient test directories remain outside Git on the task server. Links to these runtime artifacts therefore require that server or a separate download. No run evidence was deleted during publication.

The isolated DeepGEMM repair is preserved as [descriptor-protocol.patch](artifacts/raw/BASELINE-STABILITY/dependencies/descriptor-fix-001/descriptor-protocol.patch), with its [source and activation instructions](artifacts/raw/BASELINE-STABILITY/dependencies/descriptor-fix-001/REPORT.md) and original/candidate manifests. Reconstruct the isolated dependency from the recorded installed version and patch; the binary package itself is not vendored.

The current two-rank trace capture is optimizer update 111 in `artifacts/raw/BASELINE-TRACE/attempts/003/native/profiling/traces/iteration_111/`. Both trace JSON files are explicitly ignored and transferred separately.

Publication preserves the complete research history, including stopped experiments. Passing native baseline stability and trace capture do not imply acceptance of every historical experimental feature. See [report.md](report.md) for the current result and boundaries.
