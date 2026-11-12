"""Extract bounded evidence without opening checkpoint tensor payloads or CUDA."""

import collections
import csv
import hashlib
import json
import pickle
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[2]
ATTEMPT = RAW / "NUMERICAL-NATIVE-PROBE/attempts/001"
CREATE = ATTEMPT / "create_s0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def inventory(root):
    return [
        {"path": str(p.relative_to(root)), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*")) if p.is_file()
    ]


result = {"scope": "Native attempt 001; metadata/log analysis only; no CUDA queries"}
manifest = RAW / "SOURCE-SNAPSHOT/009/manifest.json"
source = load(manifest)
assert sha(manifest) == "cf95f726518062714913b8155dad878b73442fc5a0d68d993b7abdf6a55cf3f8"
result["source"] = {"manifest_sha256": sha(manifest), "base_commit": source["base_commit"],
                    "archive_sha256_recorded": source["archive_sha256"]}
with zipfile.ZipFile(manifest.parent / "changed-files.zip") as archive:
    name = "gift/torchtitan/experiments/cuda_graph_campaign/numerical_probe.py"
    result["source"]["numerical_probe_sha256"] = hashlib.sha256(archive.read(name)).hexdigest()

event_names = ["step_start", "fetching_batch_end", "fwd_bwd_end", "optim_end",
               "checkpoint_save_end", "step_error", "step_end"]
ranks = []
for path in sorted((CREATE / "structured_logs").glob("*.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    steps = [r for r in rows if r.get("step") == 1]
    ranks.append({"rank": rows[0]["global_rank"], "log": str(path.relative_to(ATTEMPT)),
                  "counts": {name: sum(r["log_type_name"] == name for r in steps)
                             for name in event_names},
                  "events": [{k: r[k] for k in ["seq_id", "time_us", "log_type_name", "value"] if k in r}
                             for r in steps if r["log_type_name"] in event_names],
                  "tokens": {r["event_name"]: r["value"] for r in steps
                             if r.get("event_name") in ["local_valid_tokens", "global_valid_tokens"]}})
result["ranks"] = sorted(ranks, key=lambda r: r["rank"])
result["compiler_receipts"] = [load(p) for p in sorted((CREATE / "compiler-options").glob("*.json"))]
probe = CREATE / "checkpoint/numerical_probe"
result["probe_manifest"] = load(probe / "manifest.json")
result["unique_errors"] = dict(collections.Counter(
    load(p)["error"] for p in probe.glob("rank-*.s0.failure.json")))
result["probe_files"] = inventory(probe)
result["observe_directory_exists"] = (ATTEMPT / "observe").exists()
result["exit"] = load(ATTEMPT / "exit-status.json")
result["owner_status"] = load(CREATE / "owner-status.json")
release = load(ATTEMPT / "root-release-check.json")
result["release"] = {k: v for k, v in release.items() if k != "checks"}
result["release"]["identity_status_counts"] = dict(collections.Counter(c["status"] for c in release["checks"]))

checkpoint = CREATE / "checkpoint/step-1"
result["checkpoint_inventory"] = inventory(checkpoint)
result["checkpoint_total_bytes"] = sum(f["bytes"] for f in result["checkpoint_inventory"])
metadata = checkpoint / ".metadata"
result["checkpoint_metadata_sha256"] = sha(metadata)
# This is the trusted task-produced, small DCP metadata pickle, not tensor data.
native = pickle.loads(metadata.read_bytes())
keys = list(native.state_dict_metadata)
result["checkpoint_keys"] = keys
result["checkpoint_root_counts"] = dict(collections.Counter(k.split(".")[0] for k in keys))
moments = {}
for field in ["exp_avg", "exp_avg_sq", "step"]:
    prefix = "optimizer.state."
    selected = {k[len(prefix): -len(field)-1]: native.state_dict_metadata[k]
                for k in keys if k.startswith(prefix) and k.endswith("." + field)}
    moments[field] = {"count": len(selected),
                      "missing_model_keys": [k for k in selected if k not in native.state_dict_metadata],
                      "shape_mismatches": [k for k, v in selected.items()
                                           if field != "step" and v.size != native.state_dict_metadata[k].size]}
result["optimizer_metadata_coverage"] = moments
result["checkpoint_storage_entries"] = len(native.storage_data)
result["checkpoint_storage_files"] = sorted({v.relative_path for v in native.storage_data.values()})
result["checkpoint_storage_bounds_valid"] = all(
    v.offset >= 0 and v.length >= 0 and v.offset + v.length <= (checkpoint / v.relative_path).stat().st_size
    for v in native.storage_data.values())

with (ATTEMPT / "resource-samples.tsv").open() as handle:
    samples = list(csv.DictReader(handle, delimiter="\t"))
gpu_peaks = collections.defaultdict(int)
for row in samples:
    for item in row["gpu_memory_mib"].split(";"):
        if item.strip():
            uuid, value = item.split(",")
            gpu_peaks[uuid.strip()] = max(gpu_peaks[uuid.strip()], int(value))
result["sampled_resources"] = {
    "sample_count": len(samples), "first_utc": samples[0]["timestamp_utc"],
    "last_utc": samples[-1]["timestamp_utc"],
    "max_owned_rss_bytes": max(int(r["owned_host_rss_bytes"]) for r in samples),
    "max_owned_process_count": max(int(r["owned_process_count"]) for r in samples),
    "gpu_max_mib_by_uuid": dict(gpu_peaks),
    "directory_samples": [{k: r[k] for k in ["timestamp_utc", "phase_bytes", "attempt_bytes", "fs_available_bytes"]}
                          for r in samples if r["phase_bytes"] != "not_sampled"],
    "definitions": "Owned-process RSS sum can double-count shared pages; GPU memory is sampled nvidia-smi used MiB, not PyTorch peaks. Directory bytes are logical sizes. No instantaneous peak claim."}
result["small_input_sha256"] = {
    str(p.relative_to(RAW)): sha(p) for p in [CREATE / "config/resolved_config.yaml",
                                          ATTEMPT / "resource-samples.tsv",
                                          probe / "manifest.json", ATTEMPT / "exit-status.json"]}
(HERE / "extraction.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"ranks": len(ranks), "checkpoint_bytes": result["checkpoint_total_bytes"],
                  "unique_errors": result["unique_errors"], "resources": result["sampled_resources"]}, indent=2))
