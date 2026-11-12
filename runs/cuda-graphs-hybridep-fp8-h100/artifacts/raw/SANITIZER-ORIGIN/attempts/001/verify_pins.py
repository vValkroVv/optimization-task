#!/usr/bin/env python3
"""Read-only checks for the frozen source and one-shot attempt pins."""

import hashlib
import json
import subprocess
from pathlib import Path

TASK_ROOT = Path("/workspace-SR008.nfs2/vkropotin/optimization-task")
ATTEMPT = TASK_ROOT / "runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/SANITIZER-ORIGIN/attempts/001"
RAW = TASK_ROOT / "runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    pins = json.loads((ATTEMPT / "pins.json").read_text())
    manifest_path = RAW / "SOURCE-SNAPSHOT/007/manifest.json"
    archive_path = RAW / "SOURCE-SNAPSHOT/007/changed-files.zip"
    preset_path = TASK_ROOT / "gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = {
        "source_manifest_sha256": "2585e504350276cb4689ae9ea45a33706e9f898bb61969abe5d08f2650c55620",
        "source_archive_sha256": "0fc5ddad83c5f99b4036a47dd7f485ba2f42d1284c42f086d5f051cebad66082",
        "sanitizer_sha256": "5cba659e1cd603aa2370e103425f184368e6ed405da90ce85743c4399c324dbc",
    }
    actual = {
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_archive_sha256": digest(archive_path),
        "sanitizer_sha256": digest(Path("/usr/local/cuda/bin/compute-sanitizer")),
    }
    if actual != expected or any(pins.get(key) != value for key, value in expected.items()):
        raise SystemExit("pinned source/archive/sanitizer identity mismatch")
    if manifest.get("archive_sha256") != actual["source_archive_sha256"]:
        raise SystemExit("SOURCE-SNAPSHOT/007 manifest archive hash mismatch")
    if manifest.get("base_commit") != pins.get("source_base_revision"):
        raise SystemExit("SOURCE-SNAPSHOT/007 base revision mismatch")
    if digest(preset_path) != pins.get("preset_sha256") or pins.get("preset_sha256") != manifest.get("preset_sha256"):
        raise SystemExit("production preset SHA-256 mismatch")
    for relative, expected_hash in pins["source_files"].items():
        path = TASK_ROOT / relative
        metadata = manifest["files"].get(relative)
        data = path.read_bytes()
        if (
            metadata is None
            or expected_hash != metadata["sha256"]
            or len(data) != metadata["bytes"]
            or hashlib.sha256(data).hexdigest() != expected_hash
        ):
            raise SystemExit(f"SOURCE-SNAPSHOT/007 file pin mismatch: {relative}")
    if set(pins["source_files"]) != set(manifest["files"]):
        raise SystemExit("pin list differs from SOURCE-SNAPSHOT/007 manifest files")
    if len(manifest["files"]) != 92:
        raise SystemExit("SOURCE-SNAPSHOT/007 manifest does not contain 92 files")
    for name, expected_hash in pins["attempt_files"].items():
        if digest(ATTEMPT / name) != expected_hash:
            raise SystemExit(f"attempt file SHA-256 mismatch: {name}")
    for path, pin_key in (
        (TASK_ROOT / "gift/torchtitan/experiments/cuda_graph_campaign/train_scan_fix.py", "train_scan_fix_sha256"),
        (TASK_ROOT / "gift/torchtitan/experiments/cuda_graph_campaign/backend_scan_fix.py", "backend_scan_fix_sha256"),
    ):
        if digest(path) != pins[pin_key]:
            raise SystemExit(f"scan-fix source pin mismatch: {path.name}")
    tracked = set(subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--", "gift"], cwd=TASK_ROOT, text=True
    ).splitlines())
    untracked = set(subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "gift"], cwd=TASK_ROOT, text=True
    ).splitlines())
    if tracked | untracked != set(manifest["files"]):
        raise SystemExit("current gift changed-file set differs from SOURCE-SNAPSHOT/007 manifest")
    print("verified_source_files=92")
    print("verified_source_changed_set=92")
    print("sanitizer_sha256=" + actual["sanitizer_sha256"])


if __name__ == "__main__":
    main()
