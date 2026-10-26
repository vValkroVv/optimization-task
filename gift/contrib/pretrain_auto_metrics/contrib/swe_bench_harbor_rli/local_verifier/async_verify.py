#!/usr/bin/env python3
"""async_verify.py — HOST-side async gen-parallel-verify driver for the unified SWE-bench.

Implements the "verify pool" half of the async gen||verify design
(SWE_UNIFIED_RESEARCH.md §5): PAM jobs generate trials (mini-swe-agent inside
harbor), each trial drops agent/prediction.patch as a first-class artifact;
this daemon watches the harbor_jobs trees on NFS, picks up new non-empty
patches as they appear, grades them incrementally with the proven host
grader (swebench_nodocker via run_model_eval.sh, official swebench parser),
and maintains:

  <out>/results_async.jsonl   accumulated grader verdicts, dedup by iid (latest patch wins)
  <out>/parity_live.json      per-iid comparison ours vs in-job harbor verifier/reward.txt
  <out>/preds.json            incremental {iid: {"model_patch": ...}} fed to the grader
  <out>/state.json            crash-safe scan/grade state (atomic writes)
  <out>/driver.log            log
  <out>/runs/                 grader scratch (testbeds are cleaned by the grader itself)

Run dirs may appear LATER (e.g. ckpt_gigar_n* score-axis runs): nonexistent
--watch paths are tolerated and re-checked every poll. Trials appear
incrementally inside a run; patches younger than --min-age-sec are skipped
until the next poll (partial writes).

Grading backends (--grader):
  <path to script>   host nodocker grader via run_model_eval.sh (default, proven)
  rli                verify inside RLI sessions (https://rli.sberdevices.ru) —
                     reproduces the in-job harbor verifier byte-for-byte:
                     session env=swe-bench-<short_id> v1.0.0, apply patch in
                     /testbed, upload dataset tests/ -> /tests, bash /tests/test.sh,
                     resolved = report.json verdict. --par = parallel sessions.
                     NOTE: RLI is reachable only from mlspace jobs, not from the
                     notebook host.
  rli-mock           same pipeline against an in-process mock transport
                     (no network; for testing the driver logic).

Usage:
  /home/user/conda/bin/python3 async_verify.py \
      --watch /workspace-SR008.nfs2/mvnikonov/swe_unified/ckpt_mx_n4/artifacts/metrics/swe_bench \
      --out   /workspace-SR008.nfs2/mvnikonov/swe_unified/async/out_mx_n4 \
      --par 6 --poll-sec 60 [--once] [--grader rli]

--watch accepts any of: an artifacts .../metrics/swe_bench dir, a harbor_jobs
dir, a single run dir (harbor_jobs/<ts>) or any ancestor — the scanner just
looks for */agent/prediction.patch below it (depth-limited walk).

READ-ONLY on the watched trees. Never modifies swebench_nodocker. Never raises
out of the main loop.
"""
import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

GRADER_DEFAULT = "/home/jovyan/mvnikonov/run_model_eval.sh"
WALK_MAX_DEPTH = 8          # relative depth under a watch root
PRUNE_DIRS = {"agent", "verifier", "artifacts", "__pycache__", "logs", "testbed"}
MAX_GRADE_ATTEMPTS = 3      # per (iid, patch-sha); reset when the patch changes

log = logging.getLogger("async_verify")


# ---------------------------------------------------------------- utilities

def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True))


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def iid_from_trial(trial_dir: Path) -> str:
    """harbor trial dir = <instance_id>__<hash>; instance ids contain '__' themselves
    (astropy__astropy-13453__UP2oV3M) so strip exactly one '__<suffix>' from the right."""
    name = trial_dir.name
    if "__" in name:
        return name.rsplit("__", 1)[0]
    # fallback: harbor result.json task_name = "swe-bench/swebench-verified__<iid>"
    rj = trial_dir / "result.json"
    try:
        task_name = json.loads(rj.read_text())["task_name"]
        return task_name.split("swebench-verified__", 1)[-1]
    except Exception:
        return name


# ---------------------------------------------------------------- RLI backend
#
# Second grading backend (--grader rli): instead of the host nodocker grader,
# each pending (instance_id, patch) is verified inside a fresh RLI session,
# reproducing exactly what the in-job harbor verifier does
# (swe_bench_harbor_rli/src/environment.py):
#   create session env=swe-bench-<short_id> v1.0.0 (start(), _build_env_name)
#   -> wait ready (RUNNING)                        (_wait_until_session_ready)
#   -> mkdir/chmod /logs /logs/verifier /tests     (_prepare_remote_layout)
#   -> install uv (test.sh runs `uv run parser.py`) (_install_uv)
#   -> upload patch via chunked base64 heredoc      (upload_file)
#   -> cd /testbed && git apply (3way / patch --fuzz=5 fallbacks)
#   -> upload dataset <iid>/tests/ as tar.gz -> /tests (upload_dir)
#   -> bash /tests/test.sh  (writes /logs/verifier/report.json + reward.txt)
#   -> resolved = report.json[iid]["resolved"] (official swebench parser verdict)
#   -> close session ALWAYS.
#
# All HTTP goes through one seam (RliTransport) so --grader rli-mock (or
# ASYNC_VERIFY_RLI_MOCK=1) can swap in MockRliTransport for offline testing:
# RLI is unreachable from the notebook host, only from mlspace jobs.

import base64
import random
import shlex
import ssl
import tarfile
import tempfile
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

RLI_BASE_DEFAULT = "https://rli.sberdevices.ru"
DATASET_ROOT_DEFAULT = "/home/user/swe-bench-verified/datasets/swebench-verified"
_UPLOAD_CHUNK_CHARS = 48_000          # same as swe_bench_harbor_rli environment.py
_RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}

# _prepare_remote_layout + _install_uv from swe_bench_harbor_rli/src/environment.py
_BOOTSTRAP_LAYOUT_CMD = (
    "mkdir -p /logs /logs/agent /logs/verifier /logs/artifacts /tests"
    " && chmod -R 777 /logs /tests"
)
_UV_VERSION = "0.7.13"
_UV_ASSET_URL = (
    "https://github.com/astral-sh/uv/releases/download/"
    f"{_UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
)
_INSTALL_UV_CMD = f"""set -e
if ! command -v uv >/dev/null 2>&1; then
  tmpdir="$(mktemp -d)"
  cleanup() {{
    rm -rf "$tmpdir"
  }}
  trap cleanup EXIT
  mkdir -p /root/.local/bin /usr/local/bin
  curl --http1.1 --retry 5 --retry-delay 2 --retry-all-errors \
    --connect-timeout 20 --max-time 120 --retry-max-time 240 \
    -fL "{_UV_ASSET_URL}" -o "$tmpdir/uv.tar.gz"
  tar -xzf "$tmpdir/uv.tar.gz" -C "$tmpdir"
  uv_src="$(find "$tmpdir" -type f -name uv | head -n 1)"
  uvx_src="$(find "$tmpdir" -type f -name uvx | head -n 1)"
  test -n "$uv_src"
  test -n "$uvx_src"
  install -m 0755 "$uv_src" /root/.local/bin/uv
  install -m 0755 "$uvx_src" /root/.local/bin/uvx
  ln -sf /root/.local/bin/uv /usr/local/bin/uv
  ln -sf /root/.local/bin/uvx /usr/local/bin/uvx
fi
uv --version"""


class RliHttpError(Exception):
    def __init__(self, status, body=""):
        self.status = status
        self.body = (body or "")[:500]
        super().__init__(f"HTTP {status}: {self.body[:200]}")


class RliApiError(Exception):
    """RLI error envelope (status=error)."""

    def __init__(self, code, message, retryable):
        self.code = code
        self.retryable = bool(retryable)
        super().__init__(f"RLI error {code}: {message}")


class RliTransport:
    """Single seam for all RLI HTTP calls (create/get/step/close).

    Mirrors gigarli.RLIClient endpoints (gigarli/src/gigarli/client.py):
      POST /v1/sessions (Idempotency-Key), GET /v1/sessions/<id>,
      POST /v1/sessions/step, POST /v1/sessions/close.
    Envelope: {"status": "ok", "result": ...} | {"status": "error", "error": ...}.
    TLS verification off, like verify_ssl=False in the harbor RLI config.
    """

    def __init__(self, base_url, timeout=180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._ssl = ssl.create_default_context()
        self._ssl.check_hostname = False
        self._ssl.verify_mode = ssl.CERT_NONE

    def request(self, method, path, body=None, headers=None, timeout=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(
                req, timeout=timeout or self.timeout, context=self._ssl
            ) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                body_txt = e.read().decode("utf-8", errors="replace")
            except Exception:
                body_txt = ""
            raise RliHttpError(e.code, body_txt) from e
        if isinstance(payload, dict) and payload.get("status") == "error":
            err = payload.get("error") or {}
            raise RliApiError(err.get("code"), err.get("message"), err.get("retryable"))
        return payload.get("result", {}) if isinstance(payload, dict) else {}


class MockRliTransport(RliTransport):
    """Offline stand-in: canned session lifecycle + canned verifier outputs.

    Simulates create->RUNNING, exec rc=0, and a report.json in which every
    instance is resolved. Used by --grader rli-mock / ASYNC_VERIFY_RLI_MOCK=1
    to prove the pipeline logic without network access to RLI.
    """

    def __init__(self, base_url="mock://rli", timeout=5.0):
        super().__init__("https://mock.invalid", timeout)
        self.sessions = {}          # sid -> task_id
        self.calls = []             # (stage, detail) trace for debugging

    def _canned_report(self, task_id):
        return {
            task_id: {
                "patch_is_None": False,
                "patch_exists": True,
                "patch_successfully_applied": True,
                "resolved": True,
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["mock::f2p_test"], "failure": []},
                    "PASS_TO_PASS": {"success": ["mock::p2p_test"], "failure": []},
                },
            }
        }

    def request(self, method, path, body=None, headers=None, timeout=None):
        if method == "POST" and path == "/v1/sessions":
            sid = "mock-" + uuid.uuid4().hex[:12]
            self.sessions[sid] = (body or {}).get("env", {}).get("task_id", "")
            self.calls.append(("create_session", sid))
            return {"session_id": sid}
        if method == "GET" and path.startswith("/v1/sessions/"):
            sid = path.rsplit("/", 1)[-1]
            self.calls.append(("get_session", sid))
            return {"status": "RUNNING", "session_id": sid}
        if method == "POST" and path == "/v1/sessions/step":
            sid = (body or {}).get("session_id", "")
            cmd = ""
            try:
                cmd = body["actions"][0]["args"]["cmd"]
            except Exception:
                pass
            self.calls.append(("step", cmd[:120]))
            stdout = ""
            if "base64 -w0 /logs/verifier/report.json" in cmd:
                report = self._canned_report(self.sessions.get(sid, ""))
                stdout = base64.b64encode(
                    json.dumps(report).encode("utf-8")
                ).decode("ascii")
            elif "reward.txt" in cmd:
                stdout = "1"
            elif "/tests/test.sh" in cmd:
                stdout = "SWEBench results starts here\nPASSED\nSWEBench results ends here"
            return {
                "observation": {"output": stdout},
                "events": [
                    {
                        "action_index": 0,
                        "exit_code": 0,
                        "stdout_ref": stdout,
                        "stderr_ref": "",
                        "duration_ms": 1,
                    }
                ],
                "terminated": False,
            }
        if method == "POST" and path == "/v1/sessions/close":
            sid = (body or {}).get("session_id", "")
            self.sessions.pop(sid, None)
            self.calls.append(("close_session", sid))
            return {}
        raise RliHttpError(404, f"mock: unknown endpoint {method} {path}")


class RliBackend:
    """Grades one instance per RLI session; thread-pooled by the driver."""

    def __init__(self, args, transport=None, mock=False):
        self.dataset_root = Path(args.dataset_root)
        self.env_version = args.rli_env_version
        self.ttl_ms = args.rli_ttl_ms
        self.allow_internet_access = args.allow_internet_access
        self.verify_timeout_sec = args.rli_verify_timeout_sec
        self.ready_timeout_sec = args.rli_ready_timeout_sec
        self.ready_poll_sec = 2.0
        self.mock = mock
        self.transport = transport or (
            MockRliTransport() if mock
            else RliTransport(args.rli_base, timeout=args.rli_http_timeout)
        )

    # ---- retry wrapper: 408/429/5xx, transport errors, retryable API errors
    # (mirrors rli_policy classify_create_session_failure / retryable flags)
    def _call(self, what, fn, attempts=4):
        last = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except RliHttpError as e:
                last = e
                if e.status not in _RETRYABLE_HTTP:
                    raise
            except RliApiError as e:
                last = e
                if not e.retryable:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = e
            if attempt < attempts:
                backoff = min(2.0 * (2 ** (attempt - 1)), 30.0) + random.uniform(0, 1)
                log.info("rli %s failed (%s), retry %d/%d in %.1fs",
                         what, last, attempt, attempts - 1, backoff)
                time.sleep(backoff)
        raise last

    # ---- session API ------------------------------------------------------
    def create_session(self, iid):
        short_id = iid.split("__", 1)[1] if "__" in iid else iid
        body = {
            "env": {
                "name": f"swe-bench-{short_id}",
                "version": self.env_version,
                "task_id": iid,
            },
            "ttl_ms": int(self.ttl_ms),
            "allow_internet_access": self.allow_internet_access,
            "use_pip_proxy": False,
        }
        idem = str(uuid.uuid4())  # stable across retries, like _create_session_with_retries
        result = self._call(
            "create_session",
            lambda: self.transport.request(
                "POST", "/v1/sessions", body=body, headers={"Idempotency-Key": idem}
            ),
        )
        sid = result.get("session_id")
        if not sid:
            raise RuntimeError(f"create_session returned no session_id: {result}")
        return sid

    def wait_ready(self, sid):
        deadline = time.monotonic() + self.ready_timeout_sec
        while time.monotonic() < deadline:
            result = self._call(
                "get_session",
                lambda: self.transport.request("GET", f"/v1/sessions/{sid}"),
            )
            status = str(result.get("status", "")).strip().upper()
            if status == "RUNNING":
                return
            if status in {"CLOSED", "FAILED", "TERMINATED"}:
                raise RuntimeError(f"RLI session {sid} terminal status {status}")
            time.sleep(self.ready_poll_sec)
        raise RuntimeError(f"timed out waiting for RLI session {sid} to become ready")

    def close_session(self, sid):
        try:
            self._call(
                "close_session",
                lambda: self.transport.request(
                    "POST", "/v1/sessions/close", body={"session_id": sid}
                ),
                attempts=2,
            )
        except Exception:
            log.warning("failed to close RLI session %s (TTL will reap it):\n%s",
                        sid, traceback.format_exc())

    def exec(self, sid, cmd, deadline_sec=None, attempts=2):
        """One bash step. Returns (rc, stdout, stderr). attempts>1 only re-sends
        on transport-level failures (like max_verifier_step_retries=1)."""
        deadline_ms = int((deadline_sec or 600) * 1000)
        body = {
            "session_id": sid,
            "mode": "single",
            "actions": [
                {"type": "env_action", "name": "bash", "args": {"cmd": cmd}}
            ],
            "limits": {"deadline_ms": deadline_ms},
        }
        http_timeout = deadline_ms / 1000.0 + 120
        result = self._call(
            "step",
            lambda: self.transport.request(
                "POST", "/v1/sessions/step", body=body, timeout=http_timeout
            ),
            attempts=attempts,
        )
        # same mapping as environment._step_result_to_exec_result
        stdout = ((result.get("observation") or {}).get("output")) or ""
        stderr = ""
        rc = 0
        events = result.get("events") or []
        if events:
            ev = events[0]
            stdout = ev.get("stdout_ref") or stdout
            stderr = ev.get("stderr_ref") or ""
            rc = int(ev.get("exit_code", 0))
        return rc, stdout, stderr

    # ---- transfers (chunked base64 heredoc, as environment.upload_file) ----
    def upload_bytes(self, sid, data, target_path):
        payload = base64.b64encode(data).decode("ascii")
        chunks = [payload[i:i + _UPLOAD_CHUNK_CHARS]
                  for i in range(0, len(payload), _UPLOAD_CHUNK_CHARS)] or [""]
        target = shlex.quote(target_path)
        parent = shlex.quote(str(Path(target_path).parent))
        remote_b64_path = f"/tmp/rli-upload-{uuid.uuid4().hex}.b64"
        remote_b64 = shlex.quote(remote_b64_path)
        rc, out, err = self.exec(
            sid, f"mkdir -p {parent} && rm -f {remote_b64} {remote_b64}.*.part")
        if rc != 0:
            raise RuntimeError(f"upload init failed rc={rc}: {err or out}")
        for part_index, chunk in enumerate(chunks):
            remote_part = shlex.quote(f"{remote_b64_path}.{part_index}.part")
            cmd = (f"cat > {remote_part} <<'RLI_FILE_CHUNK_EOF'\n"
                   f"{chunk}\n"
                   "RLI_FILE_CHUNK_EOF\n")
            rc, out, err = self.exec(sid, cmd)
            if rc != 0:
                raise RuntimeError(f"upload chunk {part_index} failed rc={rc}: {err or out}")
        finalize = (
            f": > {remote_b64} && "
            f"for i in $(seq 0 {len(chunks) - 1}); do "
            f"cat {remote_b64}.${{i}}.part >> {remote_b64}; "
            "done && "
            f"base64 -d {remote_b64} > {target} && "
            f"rm -f {remote_b64} {remote_b64}.*.part"
        )
        rc, out, err = self.exec(sid, finalize)
        if rc != 0:
            raise RuntimeError(f"upload finalize failed rc={rc}: {err or out}")

    def upload_dir(self, sid, local_dir, target_dir):
        local_dir = Path(local_dir)
        remote_archive = f"/tmp/rli-upload-{uuid.uuid4().hex}.tar.gz"
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
            with tarfile.open(archive.name, "w:gz") as tar:
                for item in sorted(local_dir.iterdir()):
                    tar.add(item, arcname=item.name)
            archive.seek(0)
            self.upload_bytes(sid, Path(archive.name).read_bytes(), remote_archive)
        target = shlex.quote(target_dir)
        ra = shlex.quote(remote_archive)
        rc, out, err = self.exec(
            sid,
            f"mkdir -p {target} && "
            f"tar --no-same-owner --no-same-permissions -xzf {ra} -C {target} && "
            f"rm -f {ra}",
        )
        if rc != 0:
            raise RuntimeError(f"extract tests failed rc={rc}: {err or out}")

    # ---- the actual per-instance verification ------------------------------
    def grade_one(self, iid, patch_text):
        t0 = time.time()
        result = {
            "instance_id": iid,
            "grader": "rli",
            "resolved": False,
            "patch_applied": False,
        }
        if self.mock:
            result["rli_mock"] = True
        tests_dir = self.dataset_root / iid / "tests"
        if not (tests_dir / "test.sh").is_file():
            result["error"] = f"no tests dir in frozen dataset: {tests_dir}"
            return result
        sid = None
        try:
            sid = self.create_session(iid)
            result["session_id"] = sid
            self.wait_ready(sid)
            # bootstrap: layout + uv (uv best-effort, as _install_uv)
            rc, out, err = self.exec(sid, _BOOTSTRAP_LAYOUT_CMD)
            if rc != 0:
                raise RuntimeError(f"layout bootstrap failed rc={rc}: {err or out}")
            self.exec(sid, _INSTALL_UV_CMD, deadline_sec=300)
            # patch
            self.upload_bytes(sid, patch_text.encode("utf-8"), "/tmp/prediction.patch")
            rc, out, err = self.exec(
                sid, "cd /testbed && git apply -v /tmp/prediction.patch")
            if rc != 0:
                rc, out, err = self.exec(
                    sid, "cd /testbed && git apply --3way /tmp/prediction.patch")
            if rc != 0:
                rc, out, err = self.exec(
                    sid,
                    "cd /testbed && patch -p1 --fuzz=5 --batch -i /tmp/prediction.patch")
            result["patch_applied"] = (rc == 0)
            if rc != 0:
                result["error"] = f"patch apply failed: {(err or out)[:400]}"
                return result
            # tests + verifier (exactly what harbor's verifier runs: test.sh
            # applies the swebench test patch, runs pytest into $LOG_FILE, then
            # `uv run parser.py` -> /logs/verifier/report.json + reward.txt)
            self.upload_dir(sid, tests_dir, "/tests")
            rc, out, err = self.exec(
                sid,
                'export PATH="/root/.local/bin:$PATH" && cd /testbed && bash /tests/test.sh',
                deadline_sec=self.verify_timeout_sec,
            )
            result["test_sh_exit"] = rc
            # official verdict from report.json (harbor parser writes
            # report_map[iid]["resolved"])
            rc2, b64, _ = self.exec(sid, "base64 -w0 /logs/verifier/report.json")
            report_entry = None
            if rc2 == 0 and b64.strip():
                try:
                    report = json.loads(base64.b64decode(b64.strip()))
                    report_entry = report.get(iid) or next(iter(report.values()), None)
                except Exception:
                    log.warning("unparsable report.json for %s", iid)
            if isinstance(report_entry, dict):
                result["resolved"] = bool(report_entry.get("resolved", False))
                result["patch_applied"] = bool(
                    report_entry.get("patch_successfully_applied",
                                     result["patch_applied"]))
                ts = report_entry.get("tests_status") or {}
                f2p = ts.get("FAIL_TO_PASS") or {}
                p2p = ts.get("PASS_TO_PASS") or {}
                if ts:
                    result["f2p_ok"] = len(f2p.get("success") or [])
                    result["f2p_fail"] = len(f2p.get("failure") or [])
                    result["p2p_ok"] = len(p2p.get("success") or [])
                    result["p2p_fail"] = len(p2p.get("failure") or [])
            else:
                # fallback chain: reward.txt, then test.sh exit code
                rc3, reward_txt, _ = self.exec(
                    sid, "cat /logs/verifier/reward.txt 2>/dev/null")
                if rc3 == 0 and reward_txt.strip():
                    try:
                        result["resolved"] = float(reward_txt.strip()) >= 0.5
                        result["verdict_source"] = "reward.txt"
                    except ValueError:
                        result["resolved"] = (rc == 0)
                        result["verdict_source"] = "test_sh_exit"
                else:
                    result["resolved"] = (rc == 0)
                    result["verdict_source"] = "test_sh_exit"
            return result
        finally:
            if sid is not None:
                self.close_session(sid)
            result["duration_sec"] = round(time.time() - t0, 1)


# ---------------------------------------------------------------- scanning

def scan_trials(watch_roots):
    """Yield (trial_dir, patch_path) for every */agent/prediction.patch under the roots.
    Depth-limited, prunes heavy leaf dirs, tolerant to missing roots (appear later)."""
    seen = set()
    for root in watch_roots:
        root = Path(root)
        if not root.is_dir():
            log.debug("watch root does not exist (yet): %s", root)
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=True):
            d = Path(dirpath)
            depth = len(d.parts) - base_depth
            if depth >= WALK_MAX_DEPTH:
                dirnames[:] = []
                continue
            if d.name == "agent" and "prediction.patch" in filenames:
                trial = d.parent
                if trial not in seen:
                    seen.add(trial)
                    yield trial, d / "prediction.patch"
                dirnames[:] = []
                continue
            dirnames[:] = [n for n in dirnames
                           if n not in PRUNE_DIRS or (d / n / "prediction.patch").is_file()]


def read_harbor_reward(trial_dir: Path):
    """In-job verify verdict: verifier/reward.txt -> float, or None if absent/unparsable."""
    f = trial_dir / "verifier" / "reward.txt"
    try:
        txt = f.read_text().strip()
        return float(txt) if txt else None
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- driver

class Driver:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "runs").mkdir(exist_ok=True)
        self.state_path = self.out / "state.json"
        self.preds_path = self.out / "preds.json"
        self.results_path = self.out / "results_async.jsonl"
        self.parity_path = self.out / "parity_live.json"
        # state:
        #  trials: {trial_path: {iid, patch_mtime, patch_sha1, patch_size}}
        #  graded: {iid: {patch_sha1, trial, result, graded_at}}
        #  attempts: {iid: {patch_sha1, n}}
        self.state = {"trials": {}, "graded": {}, "attempts": {}}
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text())
                for key in ("trials", "graded", "attempts"):
                    self.state.setdefault(key, {})
                log.info("resumed state: %d trials known, %d iids graded",
                         len(self.state["trials"]), len(self.state["graded"]))
            except Exception:
                log.warning("state.json unreadable, starting fresh:\n%s", traceback.format_exc())
        self.batch_seq = 0
        # grading backend: --grader rli|rli-mock -> RLI sessions, else host script
        grader = str(args.grader)
        mock = grader == "rli-mock" or os.environ.get("ASYNC_VERIFY_RLI_MOCK") == "1"
        self.rli = RliBackend(args, mock=mock) if (grader in ("rli", "rli-mock") or mock) else None
        if self.rli:
            log.info("RLI backend active (mock=%s, base=%s, dataset=%s, ttl=%dms, verify_timeout=%ds)",
                     mock, args.rli_base, args.dataset_root, args.rli_ttl_ms,
                     args.rli_verify_timeout_sec)

    # -------------------------------------------------------- scan phase

    def scan(self):
        """Update state['trials'] from disk. Returns number of new/changed trials."""
        now = time.time()
        changed = 0
        for trial, patch in scan_trials(self.args.watch):
            try:
                st = patch.stat()
            except OSError:
                continue
            if st.st_size == 0:
                continue
            if now - st.st_mtime < self.args.min_age_sec:
                log.info("skip young patch (<%ds old): %s", self.args.min_age_sec, patch)
                continue
            key = str(trial)
            rec = self.state["trials"].get(key)
            if rec and rec.get("patch_mtime") == st.st_mtime and rec.get("patch_size") == st.st_size:
                continue
            try:
                data = patch.read_bytes()
            except OSError:
                continue
            if not data.strip():
                continue
            self.state["trials"][key] = {
                "iid": iid_from_trial(trial),
                "patch_mtime": st.st_mtime,
                "patch_sha1": sha1_bytes(data),
                "patch_size": st.st_size,
            }
            changed += 1
        if changed:
            log.info("scan: %d new/updated trials (total known: %d)",
                     changed, len(self.state["trials"]))
        return changed

    def best_candidates(self):
        """Per iid pick the trial with the LATEST patch mtime. -> {iid: (trial_path, rec)}"""
        best = {}
        for tpath, rec in self.state["trials"].items():
            iid = rec["iid"]
            cur = best.get(iid)
            if cur is None or rec["patch_mtime"] > cur[1]["patch_mtime"]:
                best[iid] = (tpath, rec)
        return best

    def pending(self, best):
        """iids whose best patch is not what we graded (new or changed), attempts allowing."""
        todo = []
        for iid, (tpath, rec) in sorted(best.items()):
            g = self.state["graded"].get(iid)
            if g and g.get("patch_sha1") == rec["patch_sha1"]:
                continue
            att = self.state["attempts"].get(iid)
            if att and att.get("patch_sha1") == rec["patch_sha1"] and att.get("n", 0) >= MAX_GRADE_ATTEMPTS:
                continue  # gave up on this exact patch; a NEW patch resets below
            todo.append(iid)
        return todo

    # -------------------------------------------------------- grade phase

    def write_preds(self, best):
        preds = {}
        for iid, (tpath, rec) in best.items():
            try:
                preds[iid] = {"model_patch": (Path(tpath) / "agent" / "prediction.patch").read_text(errors="replace")}
            except OSError:
                log.warning("patch vanished, skipping: %s", tpath)
        atomic_write_json(self.preds_path, preds)
        return preds

    def grade_batch(self, iids):
        """One grader invocation over a chunk of instance ids. Returns {iid: result_dict}."""
        self.batch_seq += 1
        tag = f"batch_{int(time.time())}_{self.batch_seq:03d}"
        list_file = self.out / f"{tag}.instances.txt"
        res_file = self.out / f"{tag}.results.jsonl"
        glog = self.out / f"{tag}.grader.log"
        atomic_write_text(list_file, "\n".join(iids) + "\n")
        cmd = ["bash", self.args.grader, str(self.preds_path), str(res_file),
               str(self.out / "runs"), str(self.args.par), str(list_file)]
        log.info("grading %d instances (par=%d): %s", len(iids), self.args.par, " ".join(iids[:6]) + ("..." if len(iids) > 6 else ""))
        t0 = time.time()
        try:
            with open(glog, "w") as lf:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=self.args.batch_timeout_sec, check=False)
        except subprocess.TimeoutExpired:
            log.error("grader batch timed out after %ds (%s)", self.args.batch_timeout_sec, tag)
        except Exception:
            log.error("grader invocation failed:\n%s", traceback.format_exc())
        results = {}
        try:
            if res_file.exists():
                for line in res_file.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if "instance_id" in r:
                            results[r["instance_id"]] = r
                    except json.JSONDecodeError:
                        log.warning("unparsable grader line: %.200s", line)
        except OSError:
            pass
        log.info("batch %s done in %.1f min: %d/%d results returned",
                 tag, (time.time() - t0) / 60, len(results), len(iids))
        return results

    def grade_batch_rli(self, iids, preds):
        """RLI backend: one session per instance, --par parallel sessions.
        Sessions are always closed (grade_one finally). Returns {iid: result}."""
        self.batch_seq += 1
        tag = f"rli_batch_{int(time.time())}_{self.batch_seq:03d}"
        log.info("RLI grading %d instances (par=%d): %s", len(iids), self.args.par,
                 " ".join(iids[:6]) + ("..." if len(iids) > 6 else ""))
        t0 = time.time()
        results = {}
        with ThreadPoolExecutor(max_workers=max(1, self.args.par)) as pool:
            futs = {
                pool.submit(self.rli.grade_one, iid, preds[iid]["model_patch"]): iid
                for iid in iids if iid in preds
            }
            for fut in as_completed(futs):
                iid = futs[fut]
                try:
                    r = fut.result()
                    results[iid] = r
                    log.info("rli verdict %s: resolved=%s (%.1fs)", iid,
                             r.get("resolved"), r.get("duration_sec", -1))
                except Exception:
                    log.error("rli grading failed for %s:\n%s", iid, traceback.format_exc())
        # raw batch dump for debugging, same layout as host batches
        try:
            atomic_write_text(
                self.out / f"{tag}.results.jsonl",
                "\n".join(json.dumps(r, ensure_ascii=False) for r in results.values())
                + ("\n" if results else ""))
        except OSError:
            pass
        log.info("batch %s done in %.1f min: %d/%d results returned",
                 tag, (time.time() - t0) / 60, len(results), len(iids))
        return results

    def record_results(self, iids, results, best):
        now = time.time()
        for iid in iids:
            rec = best[iid][1]
            att = self.state["attempts"].get(iid)
            if not att or att.get("patch_sha1") != rec["patch_sha1"]:
                att = {"patch_sha1": rec["patch_sha1"], "n": 0}
            att["n"] += 1
            self.state["attempts"][iid] = att
            r = results.get(iid)
            if r is None:
                log.warning("no grader verdict for %s (attempt %d/%d)", iid, att["n"], MAX_GRADE_ATTEMPTS)
                continue
            self.state["graded"][iid] = {
                "patch_sha1": rec["patch_sha1"],
                "trial": best[iid][0],
                "result": r,
                "graded_at": now,
            }

    # -------------------------------------------------------- outputs

    def write_outputs(self, best):
        # results_async.jsonl: one line per iid, latest verdict
        lines = []
        for iid in sorted(self.state["graded"]):
            g = self.state["graded"][iid]
            row = dict(g["result"])
            row["instance_id"] = iid
            row["_trial"] = g["trial"]
            row["_patch_sha1"] = g["patch_sha1"]
            lines.append(json.dumps(row, ensure_ascii=False))
        atomic_write_text(self.results_path, "\n".join(lines) + ("\n" if lines else ""))

        # parity_live.json: ours vs in-job harbor reward
        per_iid = {}
        n_both = n_agree = ours_only = harbor_only = 0
        for iid, g in self.state["graded"].items():
            ours = bool(g["result"].get("resolved", False))
            trial = Path(g["trial"])
            harbor = read_harbor_reward(trial)
            if harbor is None:
                # any other trial of the same iid with an in-job verdict (prefer latest)
                alts = [(r["patch_mtime"], Path(t)) for t, r in self.state["trials"].items()
                        if r["iid"] == iid and t != g["trial"]]
                for _, alt in sorted(alts, reverse=True):
                    harbor = read_harbor_reward(alt)
                    if harbor is not None:
                        break
            entry = {"ours_resolved": ours, "harbor_reward": harbor}
            if harbor is not None:
                hres = harbor >= 0.5
                entry["agree"] = (ours == hres)
                n_both += 1
                if ours == hres:
                    n_agree += 1
                elif ours:
                    ours_only += 1
                else:
                    harbor_only += 1
            else:
                entry["agree"] = None
            per_iid[iid] = entry
        summary = {
            "n_graded": len(self.state["graded"]),
            "n_both": n_both,
            "n_agree": n_agree,
            "agree_rate": round(n_agree / n_both, 4) if n_both else None,
            "ours_only_resolved": ours_only,
            "harbor_only_resolved": harbor_only,
            "n_trials_seen": len(self.state["trials"]),
            "n_iids_with_patch": len(best),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_write_json(self.parity_path, {"summary": summary, "per_instance": per_iid})
        return summary

    def save_state(self):
        atomic_write_json(self.state_path, self.state)

    # -------------------------------------------------------- main loop

    def pass_once(self):
        self.scan()
        self.save_state()
        best = self.best_candidates()
        todo = self.pending(best)
        if not todo:
            log.info("nothing pending (graded %d/%d iids)", len(self.state["graded"]), len(best))
            self.write_outputs(best)
            return
        preds = self.write_preds(best)
        todo = [i for i in todo if i in preds]
        for i in range(0, len(todo), self.args.batch_max):
            chunk = todo[i:i + self.args.batch_max]
            if self.rli is not None:
                results = self.grade_batch_rli(chunk, preds)
            else:
                results = self.grade_batch(chunk)
            self.record_results(chunk, results, best)
            self.save_state()
            summary = self.write_outputs(best)
            log.info("progress: %s", json.dumps(summary))

    def run(self):
        log.info("async_verify start: watch=%s out=%s par=%d poll=%ds once=%s",
                 self.args.watch, self.out, self.args.par, self.args.poll_sec, self.args.once)
        while True:
            try:
                self.pass_once()
            except KeyboardInterrupt:
                log.info("interrupted, saving state")
                self.save_state()
                raise
            except Exception:
                log.error("pass failed (will keep looping):\n%s", traceback.format_exc())
                try:
                    self.save_state()
                except Exception:
                    pass
            if self.args.once:
                break
            time.sleep(self.args.poll_sec)
        log.info("done (once=%s)", self.args.once)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--watch", nargs="+", required=True,
                   help="artifacts swe_bench dirs / harbor_jobs run dirs (may not exist yet)")
    p.add_argument("--out", required=True, help="output dir (state, results, logs, grader scratch)")
    p.add_argument("--par", type=int, default=6, help="grader parallelism (host conda evals), default 6")
    p.add_argument("--poll-sec", type=int, default=60, help="poll interval, default 60")
    p.add_argument("--once", action="store_true", help="single scan+grade pass, then exit")
    p.add_argument("--min-age-sec", type=int, default=30,
                   help="skip patches younger than this (partial writes), default 30")
    p.add_argument("--batch-max", type=int, default=25,
                   help="max instances per grader invocation, default 25")
    p.add_argument("--batch-timeout-sec", type=int, default=6 * 3600,
                   help="hard wall for one grader batch, default 6h")
    p.add_argument("--grader", default=GRADER_DEFAULT,
                   help=f"grader script path, or 'rli' / 'rli-mock' for the RLI "
                        f"session backend; default {GRADER_DEFAULT}")
    p.add_argument("--rli-base", default=RLI_BASE_DEFAULT,
                   help=f"RLI API base URL, default {RLI_BASE_DEFAULT}")
    p.add_argument("--allow-internet-access", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="allow internet and pip proxy in RLI sessions")
    p.add_argument("--rli-env-version", default="1.0.2",
                   help="RLI environment version, default 1.0.2")
    p.add_argument("--dataset-root", default=DATASET_ROOT_DEFAULT,
                   help=f"frozen swe-bench-verified dataset root (tests/ per instance), "
                        f"default {DATASET_ROOT_DEFAULT}")
    p.add_argument("--rli-ttl-ms", type=int, default=30 * 60 * 1000,
                   help="RLI session TTL, default 30 min")
    p.add_argument("--rli-verify-timeout-sec", type=int, default=15 * 60,
                   help="deadline for bash /tests/test.sh, default 15 min")
    p.add_argument("--rli-ready-timeout-sec", type=int, default=600,
                   help="max wait for session RUNNING, default 600")
    p.add_argument("--rli-http-timeout", type=float, default=180.0,
                   help="base HTTP timeout for RLI calls, default 180")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stderr),
                logging.FileHandler(out / "driver.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(message)s")
    Driver(args).run()


if __name__ == "__main__":
    main()
