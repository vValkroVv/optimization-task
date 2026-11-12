# Staged phase-memory adapter

This artifact targets source snapshot 007 in the task repository at commit
`446431dfa191ef0300f116257ae918bc85e38c06`. Source 007 is frozen for the
active diagnostic, so this patch has only been checked and has not been applied.

The patch adds `phase_memory.py`, `train_phase_memory.py`, and the focused CPU
test, and appends one section to the campaign README. The mirrored files are
under `staged/gift/`. `SHA256SUMS` records their individual hashes and the patch
hash.

After the source freeze is lifted, run these commands from the repository root:

```bash
git apply --check --whitespace=error-all \
  /workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PHASE-MEMORY-IMPLEMENTATION/001/phase-memory.patch
git apply \
  /workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PHASE-MEMORY-IMPLEMENTATION/001/phase-memory.patch
```

The checked patch has four target paths: two new adapter/entrypoint files, one
new CPU test, and the campaign README. It does not include the ordinary native
entrypoint or any other training-source file.

After applying, the focused verification commands are:

```bash
python -m pytest -q gift/tests/unit_tests/test_campaign_phase_memory.py
ruff check --select F821,F811,F841 --no-cache \
  gift/torchtitan/experiments/cuda_graph_campaign/phase_memory.py \
  gift/torchtitan/experiments/cuda_graph_campaign/train_phase_memory.py \
  gift/tests/unit_tests/test_campaign_phase_memory.py
```
