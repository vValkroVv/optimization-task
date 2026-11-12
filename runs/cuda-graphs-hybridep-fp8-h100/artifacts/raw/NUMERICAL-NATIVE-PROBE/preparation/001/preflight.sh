#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=/workspace-SR008.nfs2/vkropotin/optimization-task
PREP_DIR="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/001"
ATTEMPT_ROOT="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/attempts/001"
GIFT_DIR="$TASK_ROOT/gift"
PYTHON=/opt/torchtitan-venv/bin/python3.12
RELEASE_FILE="$PREP_DIR/root-release.json"
STAMP=$(date -u +%Y%m%dT%H%M%S%N)
PREFLIGHT_DIR="$PREP_DIR/preflight"
GPU_CSV="$PREFLIGHT_DIR/gpu-$STAMP.csv"
APPS_CSV="$PREFLIGHT_DIR/apps-$STAMP.csv"
TOPOLOGY_TXT="$PREFLIGHT_DIR/topology-$STAMP.txt"
MOUNT_TXT="$PREFLIGHT_DIR/mount-$STAMP.txt"
RUNTIME_JSON="$PREFLIGHT_DIR/runtime-$STAMP.json"
RECEIPT="$PREFLIGHT_DIR/receipt-$STAMP.json"

umask 077
test -x "$PYTHON"
test -f "$RELEASE_FILE" || { echo "missing root source pin: $RELEASE_FILE" >&2; exit 2; }
test ! -e "$ATTEMPT_ROOT" || { echo "numerical probe attempt output already exists" >&2; exit 2; }
"$PYTHON" "$PREP_DIR/verify_release.py" --source-preflight "$RELEASE_FILE"
mkdir -p -m 0700 "$PREFLIGHT_DIR"
cd "$GIFT_DIR"
export PYTHONPATH="$GIFT_DIR${PYTHONPATH:+:$PYTHONPATH}"

timeout --signal=TERM --kill-after=2s 10s nvidia-smi \
  --query-gpu=index,uuid,name,compute_cap,driver_version,memory.used,utilization.gpu \
  --format=csv,noheader,nounits > "$GPU_CSV"
timeout --signal=TERM --kill-after=2s 10s nvidia-smi \
  --query-compute-apps=pid,used_memory --format=csv,noheader,nounits > "$APPS_CSV"
timeout --signal=TERM --kill-after=2s 10s nvidia-smi topo -m > "$TOPOLOGY_TXT"
findmnt -n -o TARGET,SOURCE,FSTYPE --target "$PREP_DIR" > "$MOUNT_TXT"
"$PYTHON" "$PREP_DIR/preflight_software.py" "$RUNTIME_JSON"

"$PYTHON" - "$PREP_DIR" "$GPU_CSV" "$APPS_CSV" "$TOPOLOGY_TXT" "$MOUNT_TXT" \
  "$RUNTIME_JSON" "$RELEASE_FILE" "$RECEIPT" "$STAMP" <<'PREFLIGHT'
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

prep, gpu_csv, apps_csv, topology_file, mount_file, runtime_file, release_file, receipt, stamp = map(Path, sys.argv[1:])
sys.path.insert(0, str(prep))
from preflight_topology import is_exact_nv18, parse_topology_matrix

expected_uuids = [
    "GPU-519db4ab-52c1-7730-5f03-bb8ecba718c3",
    "GPU-f0ff1e03-5cda-3f99-6b5b-22016cc776bf",
    "GPU-35c5887c-b022-4a95-bdc4-f6c6e2175d52",
    "GPU-384c07d4-8c7d-6c6f-c4a8-66a7c18316eb",
    "GPU-d52fef45-0944-a850-6d82-86acee633ce8",
    "GPU-9a1e0c80-7b57-12a4-6220-6a2b9a267466",
    "GPU-5f286803-df1a-a5db-5868-d31d1dde3201",
    "GPU-d03bb98d-079d-b834-e5a0-558188e55667",
]
admission_bytes = 732 * 1024**3
before = os.statvfs(prep)
free_before = before.f_bavail * before.f_frsize
mount_fields = mount_file.read_text(encoding="utf-8").strip().split(maxsplit=2)
mount = {
    "target": mount_fields[0] if len(mount_fields) == 3 else "",
    "source": mount_fields[1] if len(mount_fields) == 3 else "",
    "filesystem": mount_fields[2] if len(mount_fields) == 3 else "",
}
with gpu_csv.open(newline="", encoding="utf-8") as stream:
    gpu_rows = [[item.strip() for item in row] for row in csv.reader(stream)]
apps_output = apps_csv.read_text(encoding="utf-8").strip()
runtime = json.loads(runtime_file.read_text(encoding="utf-8"))
release = json.loads(release_file.read_text(encoding="utf-8"))
source_digest = release.get("source_manifest_sha256")

matrix = parse_topology_matrix(topology_file.read_text(encoding="utf-8"))
topology_ready = is_exact_nv18(matrix)

gpu_ready = (
    len(gpu_rows) == 8
    and all(len(row) == 7 for row in gpu_rows)
    and [row[0] for row in gpu_rows] == [str(index) for index in range(8)]
    and [row[1] for row in gpu_rows] == expected_uuids
    and all("H100" in row[2] and row[3] == "9.0" for row in gpu_rows)
    and all(int(row[5]) == 0 and int(row[6]) == 0 for row in gpu_rows)
    and not apps_output
)

probe = prep / "preflight" / f"storage-fsync-probe-{stamp}.bin"
probe_bytes = b"numerical-native-probe-preflight\n" * 128
probe_sha256 = None
write_error = None
try:
    if free_before < admission_bytes:
        raise OSError("free space is below the 732-GiB native S0 admission gate")
    with probe.open("xb") as stream:
        stream.write(probe_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    if probe.read_bytes() != probe_bytes:
        raise OSError("storage write/fsync readback mismatch")
    directory_fd = os.open(probe.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    probe_sha256 = hashlib.sha256(probe_bytes).hexdigest()
except Exception as error:
    write_error = f"{type(error).__name__}: {error}"

after = os.statvfs(prep)
free_after = after.f_bavail * after.f_frsize
inode_count = after.f_favail
storage_ready = (
    mount["filesystem"] in {"nfs", "nfs4"}
    and mount["target"] == "/workspace-SR008.nfs2"
    and free_before >= admission_bytes
    and free_after >= admission_bytes
    and inode_count > 256
    and write_error is None
)
software_ready = (
    runtime.get("task_cwd") == str(prep.parents[6] / "gift")
    and runtime.get("python_path_gift_first") is True
    and runtime.get("torch", {}).get("version")
    and runtime.get("torch", {}).get("git_version")
    and runtime.get("torch", {}).get("cuda_build")
    and set(runtime.get("task_imports", {})) == {
        "torchtitan",
        "config_pre_submit",
        "campaign_train_scan_fix",
        "campaign_numerical_probe",
    }
)
report = {
    "status": "READY_TO_LAUNCH" if gpu_ready and topology_ready and storage_ready and software_ready else "REFUSED",
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "NUMERICAL-NATIVE-PROBE/preparation/001",
    "source_manifest_sha256": source_digest,
    "gpu_rows": gpu_rows,
    "expected_gpu_uuids": expected_uuids,
    "compute_apps_output": apps_output,
    "all_eight_expected_h100s_idle": gpu_ready,
    "compute_capability": "9.0" if gpu_ready else None,
    "topology_ready": topology_ready,
    "topology_matrix": matrix,
    "software_ready": software_ready,
    "runtime": runtime,
    "mount": mount,
    "free_bytes_before": free_before,
    "free_bytes_after": free_after,
    "available_inodes": inode_count,
    "launch_admission_bytes": admission_bytes,
    "storage_probe": {
        "path": str(probe),
        "bytes": len(probe_bytes) if probe_sha256 else 0,
        "sha256": probe_sha256,
        "error": write_error,
    },
    "storage_ready": storage_ready,
    "attempt_output_root_absent": not (prep.parents[1] / "attempts" / "001").exists(),
}
receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"preflight_receipt={receipt}")
print(f"preflight_status={report['status']}")
print(f"preflight_topology_ready={topology_ready}")
print(f"preflight_software_ready={software_ready}")
if report["status"] != "READY_TO_LAUNCH":
    raise SystemExit(1)
PREFLIGHT
