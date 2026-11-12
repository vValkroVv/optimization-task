"""Fail-closed verification for the one-update native numerical probe."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[7]
PREP_REL = Path(
    "runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/"
    "NUMERICAL-NATIVE-PROBE/preparation/002"
)
PREP = ROOT / PREP_REL
ATTEMPT = ROOT / "runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/attempts/002"
PRESET_REL = Path("gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml")
CONTINUATION_REL = Path(
    "gift/configs/experiments/cuda_graph_campaign_continuation.yaml"
)
CANONICAL_ENTRY_REL = Path(
    "runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/"
    "LOSS-ONLY-PERF/attempts/001/compiler_entry.py"
)
EXPECTED_UUIDS = [
    "GPU-519db4ab-52c1-7730-5f03-bb8ecba718c3",
    "GPU-f0ff1e03-5cda-3f99-6b5b-22016cc776bf",
    "GPU-35c5887c-b022-4a95-bdc4-f6c6e2175d52",
    "GPU-384c07d4-8c7d-6c6f-c4a8-66a7c18316eb",
    "GPU-d52fef45-0944-a850-6d82-86acee633ce8",
    "GPU-9a1e0c80-7b57-12a4-6220-6a2b9a267466",
    "GPU-5f286803-df1a-a5db-5868-d31d1dde3201",
    "GPU-d03bb98d-079d-b834-e5a0-558188e55667",
]
REQUIRED_ARTIFACTS = {
    PREP_REL / "config/create_s0.yaml",
    PREP_REL / "config/observe.yaml",
    PREP_REL / "compiler_entry_create_s0.py",
    PREP_REL / "compiler_entry_observe.py",
    PREP_REL / "verify_release.py",
    PREP_REL / "phase_process_owner.py",
    PREP_REL / "test_phase_process_owner.py",
    PREP_REL / "preflight.sh",
    PREP_REL / "preflight_topology.py",
    PREP_REL / "preflight_software.py",
    PREP_REL / "launch.sh",
    CANONICAL_ENTRY_REL,
}
ADMISSION_BYTES = 732 * 1024**3
MAX_PREFLIGHT_AGE_SECONDS = 900


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"expected a repository-relative path, got {value!r}")
    path = (ROOT / relative).resolve(strict=True)
    if ROOT not in path.parents:
        raise ValueError(f"path escapes repository: {value!r}")
    return path


def require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def verify_source_pin(release: dict) -> tuple[Path, dict, str]:
    if release.get("scope") != "NUMERICAL-NATIVE-PROBE/preparation/002":
        raise ValueError("root release scope does not match this preparation")
    manifest_rel = release.get("source_manifest_path", "")
    if not manifest_rel.startswith(
        "runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/SOURCE-SNAPSHOT/"
    ) or not manifest_rel.endswith("/manifest.json"):
        raise ValueError("root release must name a SOURCE-SNAPSHOT manifest")
    manifest_path = repo_path(manifest_rel)
    manifest_digest = require_hash(
        release.get("source_manifest_sha256"), "source_manifest_sha256"
    )
    if sha256(manifest_path) != manifest_digest:
        raise ValueError("source manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("source manifest has no files map")
    archive = manifest_path.parent / "changed-files.zip"
    if sha256(archive) != manifest.get("archive_sha256"):
        raise ValueError("source snapshot archive SHA-256 mismatch")
    for relative, expected in files.items():
        path = repo_path(relative)
        actual = sha256(path)
        if path.stat().st_size != expected.get("bytes") or actual != expected.get(
            "sha256"
        ):
            raise ValueError(f"source snapshot mismatch: {relative}")

    preset_rel = release.get("preset_path")
    if preset_rel != str(PRESET_REL) or manifest.get("preset") != preset_rel:
        raise ValueError("source manifest is not for the exact 3B FP8 preset")
    preset_digest = require_hash(release.get("preset_sha256"), "preset_sha256")
    if (
        manifest.get("preset_sha256") != preset_digest
        or sha256(repo_path(preset_rel)) != preset_digest
    ):
        raise ValueError("the preset hash does not match the source manifest")
    continuation_rel = release.get("continuation_config_path")
    if continuation_rel != str(CONTINUATION_REL):
        raise ValueError("root release names the wrong continuation config")
    continuation_digest = require_hash(
        release.get("continuation_config_sha256"), "continuation_config_sha256"
    )
    if files.get(continuation_rel, {}).get("sha256") != continuation_digest:
        raise ValueError("continuation config hash does not match the source manifest")

    artifacts = release.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        str(path) for path in REQUIRED_ARTIFACTS
    }:
        raise ValueError("root release artifact hash set is incomplete or unexpected")
    for relative, expected in artifacts.items():
        expected = require_hash(expected, f"artifact_sha256[{relative}]")
        if sha256(repo_path(relative)) != expected:
            raise ValueError(f"preparation artifact hash mismatch: {relative}")
    if sha256(repo_path(str(PREP_REL / "compiler_entry_create_s0.py"))) != sha256(
        repo_path(str(CANONICAL_ENTRY_REL))
    ):
        raise ValueError("create_s0 compiler adapter differs from the canonical entry")
    if sha256(repo_path(str(PREP_REL / "compiler_entry_observe.py"))) != sha256(
        repo_path(str(CANONICAL_ENTRY_REL))
    ):
        raise ValueError("observe compiler adapter differs from the canonical entry")
    return manifest_path, manifest, manifest_digest


def main() -> int:
    arguments = sys.argv[1:]
    if len(arguments) == 2 and arguments[0] == "--source-preflight":
        release_path = Path(arguments[1]).resolve(strict=True)
        release = json.loads(release_path.read_text(encoding="utf-8"))
        if release.get("status") != "ROOT_PINNED_FOR_PREFLIGHT":
            raise ValueError("root source pin is not released for fresh preflight")
        manifest_path, manifest, manifest_digest = verify_source_pin(release)
        print(f"verified_source_manifest={manifest_path}")
        print(f"verified_source_manifest_sha256={manifest_digest}")
        print(f"source_files_verified={len(manifest['files'])}")
        return 0
    if len(arguments) != 1:
        raise SystemExit("usage: verify_release.py ROOT_RELEASE.json | --source-preflight ROOT_RELEASE.json")
    release_path = Path(arguments[0]).resolve(strict=True)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("status") != "ROOT_CONFIRMED_FOR_LAUNCH":
        raise ValueError("root release is not confirmed for launch")
    confirmations = release.get("confirmations", {})
    for field in (
        "source_and_integration_review_complete",
        "runtime_safety_review_complete",
        "no_active_gpu_confirmed",
    ):
        if confirmations.get(field) is not True:
            raise ValueError(f"root release lacks confirmation: {field}")

    manifest_path, manifest, manifest_digest = verify_source_pin(release)
    manifest_rel = release["source_manifest_path"]
    preset_digest = release["preset_sha256"]

    preflight_rel = release.get("preflight_path", "")
    if not preflight_rel.startswith(str(PREP_REL / "preflight/") + "/"):
        raise ValueError("fresh preflight receipt must be inside preparation/002")
    preflight_path = repo_path(preflight_rel)
    preflight_digest = require_hash(
        release.get("preflight_sha256"), "preflight_sha256"
    )
    if sha256(preflight_path) != preflight_digest:
        raise ValueError("preflight receipt SHA-256 mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "READY_TO_LAUNCH":
        raise ValueError("fresh preflight is not READY_TO_LAUNCH")
    if preflight.get("scope") != release.get("scope"):
        raise ValueError("preflight receipt scope does not match root release")
    checked_at = datetime.fromisoformat(preflight["checked_at_utc"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - checked_at).total_seconds()
    if age < 0 or age > MAX_PREFLIGHT_AGE_SECONDS:
        raise ValueError("preflight receipt is future-dated or older than 15 minutes")
    if preflight.get("expected_gpu_uuids") != EXPECTED_UUIDS:
        raise ValueError("preflight GPU UUID list does not match the authorized host")
    if not preflight.get("all_eight_expected_h100s_idle"):
        raise ValueError("preflight does not prove all eight expected H100s idle")
    if preflight.get("source_manifest_sha256") != manifest_digest:
        raise ValueError("preflight source manifest identity differs from the root release")
    if preflight.get("topology_ready") is not True:
        raise ValueError("preflight does not prove the exact eight-GPU NV18 topology")
    if preflight.get("compute_capability") != "9.0":
        raise ValueError("preflight does not prove SM90 capability")
    if preflight.get("software_ready") is not True:
        raise ValueError("preflight software/task-import identity was not recorded")
    if release.get("approved_runtime") != preflight.get("runtime"):
        raise ValueError("root has not approved the exact preflight software/build identity")
    if not preflight.get("storage_ready"):
        raise ValueError("preflight storage gate did not pass")
    if preflight.get("free_bytes_after", 0) < ADMISSION_BYTES:
        raise ValueError("preflight is below the 732-GiB admission requirement")
    if not preflight.get("attempt_output_root_absent"):
        raise ValueError("numerical probe attempt output already exists")
    probe = preflight.get("storage_probe", {})
    probe_path = Path(probe.get("path", ""))
    if PREP not in probe_path.parents or not probe_path.is_file():
        raise ValueError("preflight storage probe file is missing or outside preparation")
    if probe_path.stat().st_size != probe.get("bytes") or sha256(probe_path) != probe.get(
        "sha256"
    ):
        raise ValueError("preflight storage probe file does not match its receipt")
    if ATTEMPT.exists():
        raise ValueError("numerical probe attempt output already exists")

    print(f"verified_source_manifest={manifest_rel}")
    print(f"verified_source_manifest_sha256={manifest_digest}")
    print(f"verified_preset_sha256={preset_digest}")
    print(f"verified_preflight={preflight_rel}")
    print(f"preflight_age_seconds={age:.1f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"launch gate refused: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
