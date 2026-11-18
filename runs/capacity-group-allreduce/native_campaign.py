#!/usr/bin/env python3
"""Guard and run the native D control/candidate pair plus candidate diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
RAW = EVIDENCE / "artifacts/raw"
PLAN = ROOT / "runs/production-data-abcd/campaign-plan.json"
SAVED_D_CACHE = ROOT / "runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache"
PYTHON = "/opt/torchtitan-venv/bin/python3.12"
CAPACITY_REL = "gift/torchtitan/models/common/moe/dispatch/capacity.py"
TEST_REL = "gift/tests/unit_tests/test_token_drop_fixed_capacity_group.py"
CAPACITY = ROOT / CAPACITY_REL
ORIGINAL = EVIDENCE / "capacity.original.py"
RUNNER = ROOT / "runs/flash-qla/run.py"
FREEZE = RAW / "native-source-freeze-002.json"
STATE = RAW / "native-campaign-state-002.json"

ORIGINAL_SHA256 = "2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120"
CANDIDATE_SHA256 = "164af05bd72ceb0675f0d9f756ecf04926ced17430a58dd2e38bc160c514dfdc"
TEST_SHA256 = "2abffc21929ed2dd1c4731f67a8602d051866ea496aecc20abdeaf4c476260b6"
RUNS = (
    ("control", "capacity-control-timing-002"),
    ("candidate", "capacity-candidate-timing-002"),
    ("diagnostic", "capacity-candidate-diagnostic-002"),
)

_active: subprocess.Popen[bytes] | None = None
_requested_signal: int | None = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, stdout=subprocess.PIPE
    ).stdout


def _snapshot() -> dict[str, Any]:
    tracked = set(filter(None, _git("ls-files", "-z", "--cached", "--", "gift/").decode().split("\0")))
    untracked = set(filter(None, _git("ls-files", "-z", "--others", "--exclude-standard", "--", "gift/").decode().split("\0")))
    all_hash = hashlib.sha256()
    other_hash = hashlib.sha256()
    for relative in sorted(tracked | untracked):
        path = ROOT / relative
        if path.is_symlink():
            content = b"symlink\0" + os.readlink(path).encode()
        elif path.is_file():
            content = f"file:{stat.S_IMODE(path.stat().st_mode):o}:".encode() + _sha(path).encode()
        else:
            content = b"missing"
        entry = relative.encode() + b"\0" + (b"T" if relative in tracked else b"U") + content + b"\0"
        all_hash.update(entry)
        if relative != CAPACITY_REL:
            other_hash.update(entry)
    return {
        "head": _git("rev-parse", "HEAD").decode().strip(),
        "tracked_count": len(tracked),
        "untracked_count": len(untracked),
        "all_sha256": all_hash.hexdigest(),
        "other_sha256": other_hash.hexdigest(),
        "capacity_sha256": _sha(CAPACITY),
        "test_sha256": _sha(ROOT / TEST_REL),
    }


def _assert_source(current: dict[str, Any], frozen: dict[str, Any], control: bool) -> None:
    for key in ("head", "tracked_count", "untracked_count", "other_sha256"):
        if current[key] != frozen[key]:
            raise RuntimeError(f"GIFT source changed outside the capacity allowlist: {key}")
    expected = ORIGINAL_SHA256 if control else CANDIDATE_SHA256
    if current["capacity_sha256"] != expected:
        raise RuntimeError(f"capacity.py hash mismatch: expected {expected}, found {current['capacity_sha256']}")
    if current["test_sha256"] != TEST_SHA256:
        raise RuntimeError(f"affected test hash mismatch: {current['test_sha256']}")
    if not control and current["all_sha256"] != frozen["all_sha256"]:
        raise RuntimeError("candidate GIFT tree differs from its frozen snapshot")


def _replace_capacity(data: bytes) -> None:
    temp_root = ROOT / ".tmp"
    temp_root.mkdir(exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="capacity-group-allreduce-", dir=temp_root)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(stat.S_IMODE(CAPACITY.stat().st_mode))
        temporary.replace(CAPACITY)
    finally:
        temporary.unlink(missing_ok=True)


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _preflight() -> None:
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader,nounits"],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if len(rows) != 8 or any("H100" not in row or not row.rstrip().endswith(", 0") for row in rows):
        raise RuntimeError(f"expected eight idle H100s; GPU inventory was {rows}")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    if apps and "No running processes found" not in apps:
        raise RuntimeError(f"GPU compute applications are present: {apps}")

    needles = (str(RUNNER), str(ROOT / "runs/flash-qla/measure.py"), str(EVIDENCE / "component_benchmark.py"))
    for proc in Path("/proc").glob("[0-9]*"):
        if int(proc.name) == os.getpid():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if any(item in cmdline for item in needles):
            raise RuntimeError(f"task process is still live: pid {proc.name} {cmdline[:300]}")
    for path in RAW.glob("capacity-*/process.json"):
        receipt = json.loads(path.read_text())
        pgid = receipt.get("process_group_id")
        if isinstance(pgid, int) and _group_exists(pgid):
            raise RuntimeError(f"previous native process group is still live: {pgid}")
        if receipt.get("process_group_stopped") is not True:
            raise RuntimeError(f"previous runner did not confirm process exit: {path}")
    for path in RAW.glob("capacity-*/failure.json"):
        pid = json.loads(path.read_text()).get("pid")
        if isinstance(pid, int) and _group_exists(pid):
            raise RuntimeError(f"interrupted native process group is still live: {pid}")
    for path in RAW.glob("capacity-*/status.txt"):
        if path.read_text().strip() == "RUNNING":
            raise RuntimeError(f"previous native run is still marked RUNNING: {path}")


def _d_command(role: str, run_id: str) -> list[str]:
    plan = json.loads(PLAN.read_text())
    row = next(r for r in plan["runs"] if r.get("variant") == "D" and r.get("role") == role)
    argv = list(row["argv"])
    if argv[:2] != [PYTHON, str(RUNNER)]:
        raise RuntimeError("D plan does not use the pinned interpreter and FlashQLA runner")

    def replace(option: str, value: str) -> None:
        index = argv.index(option)
        argv[index + 1] = value

    replace("--run-id", run_id)
    replace("--output-root", str(RAW))
    if role == "diagnostic":
        replace("--compiler-cache-root", str(SAVED_D_CACHE))
        if "--observe-compiler-cache" not in argv:
            argv.append("--observe-compiler-cache")
    else:
        argv.extend(("--compiler-cache-root", str(SAVED_D_CACHE)))
    return argv


def _signal_handler(signum: int, _frame: Any) -> None:
    global _requested_signal
    _requested_signal = signum
    if _active is not None and _active.poll() is None:
        try:
            os.killpg(_active.pid, signal.SIGINT)
        except ProcessLookupError:
            pass


def _run_one(state: dict[str, Any], role: str, run_id: str, frozen: dict[str, Any]) -> None:
    global _active
    if _requested_signal is not None:
        raise InterruptedError(f"campaign received signal {_requested_signal}")
    _preflight()
    before = _snapshot()
    _assert_source(before, frozen, control=(role == "control"))
    command = _d_command("diagnostic" if role == "diagnostic" else "timing", run_id)
    record = {"role": role, "run_id": run_id, "argv": command, "source_before": before}
    state["current_run"] = record
    _save_state(state)
    print(f"BEGIN {role} {run_id}", flush=True)
    _active = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    record["runner_pid"] = _active.pid
    _save_state(state)
    try:
        code = _active.wait()
    finally:
        _active = None
    if _group_exists(record["runner_pid"]):
        os.killpg(record["runner_pid"], signal.SIGTERM)
        deadline = time.monotonic() + 10
        while _group_exists(record["runner_pid"]) and time.monotonic() < deadline:
            time.sleep(0.25)
        if _group_exists(record["runner_pid"]):
            os.killpg(record["runner_pid"], signal.SIGKILL)
        raise RuntimeError("runner left its owned process group alive; exact group was stopped")
    if code:
        raise RuntimeError(f"existing native runner exited with {code}: {run_id}")

    attempt = RAW / run_id
    receipts = _verify_receipts(attempt, role, run_id)
    bridge = receipts["bridge"]

    after = _snapshot()
    _assert_source(after, frozen, control=(role == "control"))
    record.update({
        "source_after": after,
        "summary_path": str(attempt / "summary.json"),
        "metadata_path": str(attempt / "metadata.json"),
        "resolved_config_path": str(attempt / "native/config/resolved_config.yaml"),
        "bridge_receipt_path": str(attempt / "flash-qla-bridge.json"),
        "bridge_source_hashes": bridge.get("source_identity", {}).get("bridge_sources"),
        "bridge_source_identity": bridge.get("source_identity"),
        "status": "PASS",
    })
    prior_identities = [r["bridge_source_identity"] for r in state["runs"]]
    if prior_identities and any(identity != record["bridge_source_identity"] for identity in prior_identities):
        raise RuntimeError("FlashQLA bridge/backend/staged-source hashes changed between native runs")
    state["runs"].append(record)
    state.pop("current_run", None)
    _save_state(state)
    print(f"PASS {role} {run_id}", flush=True)


def _arg(argv: list[str], option: str) -> str:
    if argv.count(option) != 1:
        raise RuntimeError(f"expected one {option} in actual child argv")
    return argv[argv.index(option) + 1]


def _verify_receipts(attempt: Path, role: str, run_id: str) -> dict[str, Any]:
    completion = json.loads((attempt / "completion.json").read_text())
    process = json.loads((attempt / "process.json").read_text())
    metadata = json.loads((attempt / "metadata.json").read_text())
    bridge = json.loads((attempt / "flash-qla-bridge.json").read_text())
    summary = json.loads((attempt / "summary.json").read_text())
    if completion.get("status") != "PASS" or not completion.get("source_unchanged") or not completion.get("config_unchanged"):
        raise RuntimeError(f"base runner source/config completion receipt failed: {run_id}")
    if process.get("process_group_stopped") is not True or process.get("orphaned_process_group"):
        raise RuntimeError(f"base runner process-group receipt failed: {run_id}")
    if not (attempt / "native/config/resolved_config.yaml").is_file():
        raise RuntimeError(f"resolved rank-0 config is missing: {run_id}")
    if (
        metadata.get("num_prefetch_batches") != 32
        or metadata.get("requested_epoch_size") != 102400
        or metadata.get("seed") != 2048
        or metadata.get("scheduler_total_steps") != 65000
        or metadata.get("scheduler_warmup_steps") != 2000
        or metadata.get("gfsdp_reuse_bf16_expert_weights") is not True
        or metadata.get("requested_parallelism_features")
        != {
            "gfsdp_row_owner_embeddings": True,
            "gfsdp_split_layer0_ffn": True,
            "gfsdp_split_mtp_ffn": True,
        }
    ):
        raise RuntimeError(f"native D configuration receipt differs: {run_id}")
    if (
        bridge.get("backend_variant") != "external_gate"
        or bridge.get("routing_metadata") != "fused"
        or bridge.get("cross_entropy") != "stable"
        or bridge.get("capture_mode") != "disabled"
    ):
        raise RuntimeError(f"FlashQLA activation receipt differs from D: {run_id}")
    if summary.get("status") != "complete" or summary.get("errors"):
        raise RuntimeError(f"native measurement summary is incomplete: {run_id}")
    expected_steps = 21 if role == "diagnostic" else 20
    if summary.get("training_steps") != expected_steps:
        raise RuntimeError(f"native step count differs: {run_id}")

    actual = metadata.get("argv", [])
    if "--nproc_per_node=8" not in actual or str(ROOT / "runs/flash-qla/measure.py") not in actual:
        raise RuntimeError(f"actual child argv does not match the eight-rank D path: {run_id}")
    steps, warmup, window = ("21", "20", "0") if role == "diagnostic" else ("20", "10", "10")
    expected_args = {
        "--steps": steps,
        "--warmup": warmup,
        "--window": window,
        "--dataloader.num_prefetch_batches": "32",
        "--training.steps": steps,
        "--lr_scheduler.total_steps": "65000",
        "--lr_scheduler.warmup_steps": "2000",
        "--debug.seed": "2048",
        "--metrics.log_freq": "1",
        "--checkpoint.enable": "false",
        "--validator.enable": "false",
        "--profiler.enable_profiling": "true" if role == "diagnostic" else "false",
        "--profiler.enable_memory_snapshot": "false",
        "--dataloader.dataset.epoch_size": "102400",
        "--parallelism.gfsdp_reuse_bf16_expert_weights": "true",
        "--parallelism.gfsdp_split_mtp_ffn": "true",
        "--parallelism.gfsdp_split_layer0_ffn": "true",
        "--parallelism.gfsdp_row_owner_embeddings": "true",
    }
    for option, value in expected_args.items():
        if _arg(actual, option) != value:
            raise RuntimeError(f"actual child argv {option} differs for {run_id}")
    if "--observe-compiler-cache" not in actual:
        raise RuntimeError(f"compiler-cache observation is missing from child argv: {run_id}")
    if role != "diagnostic":
        timing_window = summary.get("window", {})
        if timing_window.get("updates") != 10 or timing_window.get("warmup_updates") != 10:
            raise RuntimeError(f"native run is not a 10+10 timing window: {run_id}")
        if summary.get("world_size") != 8:
            raise RuntimeError(f"timing world size differs from D: {run_id}")
        if (
            metadata.get("compiler_cache_policy") != "reuse"
            or Path(metadata.get("compiler_cache_root", "")) != SAVED_D_CACHE
            or metadata.get("compiler_cache_producer_run_id") != "proddata-d-timing-001"
        ):
            raise RuntimeError(f"timing did not reuse the verified saved D cache: {run_id}")
    else:
        profile = metadata.get("profile_capture", {})
        if (
            summary.get("mode") != "profile_update"
            or profile.get("target_update") != 21
            or profile.get("profile_ranks") != [0, 1]
            or profile.get("profile_microbatches") != 3
            or profile.get("warmup_updates") != 20
            or metadata.get("compiler_cache_policy") != "reuse"
            or Path(metadata.get("compiler_cache_root", "")) != SAVED_D_CACHE
            or metadata.get("compiler_cache_producer_run_id")
            != "proddata-d-timing-001"
        ):
            raise RuntimeError(f"candidate diagnostic receipt differs: {run_id}")
    rank0 = json.loads((attempt / "measurement/rank-00.json").read_text())
    cache = rank0.get("bf16_expert_cache", {})
    if cache.get("feature_configured_enabled") is not True:
        raise RuntimeError(f"BF16 expert-weight reuse activation is missing: {run_id}")
    identity = bridge.get("source_identity", {})
    if not identity.get("bridge_sources") or not identity.get("backend_module", {}).get("sha256"):
        raise RuntimeError(f"FlashQLA source hashes are missing: {run_id}")
    return {"completion": completion, "metadata": metadata, "bridge": bridge, "summary": summary}


def _save_state(data: dict[str, Any]) -> None:
    temporary = STATE.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(STATE)


def _resume_after_control() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = json.loads(FREEZE.read_text())
    if (
        freeze.get("original_capacity_sha256") != ORIGINAL_SHA256
        or freeze.get("candidate_capacity_sha256") != CANDIDATE_SHA256
        or freeze.get("affected_test_sha256") != TEST_SHA256
        or freeze.get("only_source_allowlist") != CAPACITY_REL
    ):
        raise RuntimeError("existing source-freeze receipt does not match this campaign")
    frozen = freeze["candidate_source_snapshot"]
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError("original capacity snapshot changed before resume")
    if hashlib.sha256(Path(freeze["candidate_backup"]).read_bytes()).hexdigest() != CANDIDATE_SHA256:
        raise RuntimeError("candidate capacity backup changed before resume")
    state = json.loads(STATE.read_text())
    current = state.get("current_run", {})
    if (
        state.get("status") != "FAIL"
        or state.get("runs") != []
        or current.get("role") != "control"
        or current.get("run_id") != RUNS[0][1]
        or not state.get("candidate_restored_after_control")
    ):
        raise RuntimeError("existing campaign state is not a completed control-only failure")
    source_now = _snapshot()
    _assert_source(source_now, frozen, control=False)
    if source_now != state["candidate_restored_after_control"]:
        raise RuntimeError("candidate source changed after the control-only failure")
    attempt = RAW / RUNS[0][1]
    receipts = _verify_receipts(attempt, "control", RUNS[0][1])
    bridge = receipts["bridge"]
    record = dict(current)
    record.update({
        "source_after": source_now,
        "summary_path": str(attempt / "summary.json"),
        "metadata_path": str(attempt / "metadata.json"),
        "resolved_config_path": str(attempt / "native/config/resolved_config.yaml"),
        "bridge_receipt_path": str(attempt / "flash-qla-bridge.json"),
        "bridge_source_hashes": bridge.get("source_identity", {}).get("bridge_sources"),
        "bridge_source_identity": bridge.get("source_identity"),
        "status": "PASS",
    })
    state["runs"] = [record]
    state.pop("current_run", None)
    state.pop("error", None)
    state["status"] = "RUNNING"
    state["resumed_after_control_only_wrapper_failure"] = True
    _save_state(state)
    return state, frozen


def _campaign(resume_after_control: bool = False) -> int:
    if sys.executable != PYTHON:
        raise RuntimeError(f"use {PYTHON}; found {sys.executable}")
    if os.environ.get("ABCD_COMPONENT_FIXTURE_DIR"):
        raise RuntimeError("ABCD_COMPONENT_FIXTURE_DIR must be unset")
    if resume_after_control:
        if any((RAW / run_id).exists() for _, run_id in RUNS[1:]):
            raise RuntimeError("candidate or diagnostic output exists; refusing to overwrite or repeat")
        state, frozen = _resume_after_control()
        original = ORIGINAL.read_bytes()
        candidate = CAPACITY.read_bytes()
        installed_original = False
    else:
        if FREEZE.exists() or STATE.exists() or any((RAW / run_id).exists() for _, run_id in RUNS):
            raise RuntimeError("native output already exists; refusing to overwrite or repeat the campaign")
        original = ORIGINAL.read_bytes()
        candidate = CAPACITY.read_bytes()
        if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
            raise RuntimeError("original capacity snapshot hash mismatch")
        if hashlib.sha256(candidate).hexdigest() != CANDIDATE_SHA256:
            raise RuntimeError("candidate capacity hash mismatch")
        RAW.mkdir(parents=True, exist_ok=True)
        frozen = _snapshot()
        if frozen["test_sha256"] != TEST_SHA256:
            raise RuntimeError("affected test hash mismatch")
        backup = RAW / "candidate-capacity-002.py"
        backup.write_bytes(candidate)
        freeze = {
            "original_capacity_sha256": ORIGINAL_SHA256,
            "candidate_capacity_sha256": CANDIDATE_SHA256,
            "affected_test_sha256": TEST_SHA256,
            "only_source_allowlist": CAPACITY_REL,
            "candidate_source_snapshot": frozen,
            "candidate_backup": str(backup),
        }
        FREEZE.write_text(json.dumps(freeze, sort_keys=True, indent=2) + "\n")
        state = {"status": "RUNNING", "runs": [], "freeze": str(FREEZE)}
        _save_state(state)

    if not resume_after_control:
        installed_original = False
        try:
            _preflight()
            _replace_capacity(original)
            installed_original = True
            try:
                _run_one(state, "control", RUNS[0][1], frozen)
            finally:
                if installed_original:
                    if hashlib.sha256(CAPACITY.read_bytes()).hexdigest() == ORIGINAL_SHA256:
                        _replace_capacity(candidate)
                    installed_original = False
                    restored = _snapshot()
                    _assert_source(restored, frozen, control=False)
                    state["candidate_restored_after_control"] = restored
                    _save_state(state)

        except BaseException as exc:
            state["status"] = "FAIL"
            state["error"] = f"{type(exc).__name__}: {exc}"
            _save_state(state)
            print(state["error"], file=sys.stderr, flush=True)
            return 1
        finally:
            if installed_original:
                if _active is not None and _active.poll() is None:
                    os.killpg(_active.pid, signal.SIGINT)
                    _active.wait()
                if hashlib.sha256(CAPACITY.read_bytes()).hexdigest() == ORIGINAL_SHA256:
                    _replace_capacity(candidate)
                restored = _snapshot()
                _assert_source(restored, frozen, control=False)
                state["candidate_restored_after_control"] = restored
                _save_state(state)

    try:
        _preflight()
        _run_one(state, "candidate", RUNS[1][1], frozen)
        _save_state(state)
        _run_one(state, "diagnostic", RUNS[2][1], frozen)
        state["status"] = "PASS"
        state["final_source"] = _snapshot()
        _assert_source(state["final_source"], frozen, control=False)
        _save_state(state)
        return 0
    except BaseException as exc:
        state["status"] = "FAIL"
        state["error"] = f"{type(exc).__name__}: {exc}"
        _save_state(state)
        print(state["error"], file=sys.stderr, flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-plan", action="store_true", help="print exact planned D commands; no GPU/source changes")
    parser.add_argument("--resume-after-control", action="store_true", help="resume after validating a completed control-only wrapper failure")
    args = parser.parse_args()
    if args.show_plan:
        for role, run_id in RUNS:
            plan_role = "diagnostic" if role == "diagnostic" else "timing"
            print(" ".join(_d_command(plan_role, run_id)))
        return 0
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    return _campaign(resume_after_control=args.resume_after_control)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"native_campaign: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
