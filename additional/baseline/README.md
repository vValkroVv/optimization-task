# Source baseline

The selected upstream revisions are GIFT `f840cf25e4294f93a5c91a20707c30b97e3c1dbf` and LLM Foundry `1ec1360304a182844372f6c3989c3f10288fcba8`. Recursive dependency pins and import details are recorded in [source-baseline.json](source-baseline.json) and [import-verification.md](import-verification.md). The flattened snapshot includes credential-literal sanitization in three dependency helper files; the import report identifies the paths and exact exceptions. Do not describe those lines as byte-identical to upstream.

These records identify upstream revisions, dependency pins, and any documented sanitization exceptions. They do not choose a model, preset, dataset, production topology, precision, numerical threshold, compute allocation, or measured training baseline.

The `gift/` and `llm-foundry/` trees are ordinary tracked files in this repository. Preserve source metadata, but do not initialize nested Git repositories or submodules, add remotes inside the trees, or infer authorization for later source edits from the import.
