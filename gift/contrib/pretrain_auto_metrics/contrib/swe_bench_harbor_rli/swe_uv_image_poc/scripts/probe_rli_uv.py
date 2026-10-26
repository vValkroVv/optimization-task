#!/usr/bin/env python3
from __future__ import annotations

import json
import ssl
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_BASE = "https://rli.sberdevices.ru"
ENV_NAME = "swe-bench-django-13343"
ENV_VERSION = "1.0.1"
TASK_ID = "django__django-13343"
TTL_MS = 1_800_000
USE_PIP_PROXY = True

READY_TIMEOUT_SEC = 600
READY_POLL_SEC = 5
HTTP_TIMEOUT_SEC = 180
STEP_HTTP_TIMEOUT_SEC = 1200
STEP_DEADLINE_MS = 1_080_000

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

REMOTE_COMMAND = """
set -eux
date -u
whoami
pwd
echo "PATH=$PATH"
command -v uv
uv --version
ls -l /opt/miniconda3/bin/uv /usr/local/bin/uv
test -f /root/.curlrc
cat /root/.curlrc
archive=/usr/local/share/.dev_logs/test_runtime.tar.zst
test -r "$archive"
tar --zstd -tf "$archive" >/dev/null
rm -rf /tmp/test_runtime
tar --zstd -xf "$archive" -C /tmp
/tmp/test_runtime/bin/python -c 'import swebench, datasets, fastcore; print("test_runtime deps ok")'
"""


def ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: int = HTTP_TIMEOUT_SEC,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_BASE.rstrip("/") + path,
        data=data,
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=timeout,
        context=ssl_context(),
    ) as response:
        body = response.read().decode("utf-8", errors="replace")
        return json.loads(body) if body.strip() else {}


def result(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("result")
    if not isinstance(value, dict):
        raise RuntimeError(f"RLI response has no result object: {payload!r}")
    return value


def wait_running(session_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    deadline = time.monotonic() + READY_TIMEOUT_SEC
    while time.monotonic() < deadline:
        response = request_json("GET", f"/v1/sessions/{session_id}")
        events.append(response)
        status = str(result(response).get("status", "")).upper()
        print(f"status={status}")
        if status == "RUNNING":
            return events
        if status in {"CLOSED", "FAILED", "TERMINATED"}:
            raise RuntimeError(f"RLI session reached terminal status: {status}")
        time.sleep(READY_POLL_SEC)
    raise TimeoutError(f"Timed out waiting for session {session_id}")


def close_session(session_id: str) -> dict[str, Any]:
    return request_json(
        "POST",
        "/v1/sessions/close",
        {"session_id": session_id},
        timeout=HTTP_TIMEOUT_SEC,
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = OUTPUT_DIR / f"probe_rli_uv_{TASK_ID}_{stamp}.json"

    artifact: dict[str, Any] = {
        "api_base": API_BASE,
        "env_name": ENV_NAME,
        "env_version": ENV_VERSION,
        "task_id": TASK_ID,
        "remote_command": REMOTE_COMMAND,
    }

    session_id: str | None = None
    try:
        create_payload = {
            "env": {
                "name": ENV_NAME,
                "version": ENV_VERSION,
                "task_id": TASK_ID,
            },
            "ttl_ms": TTL_MS,
            "use_pip_proxy": USE_PIP_PROXY,
        }
        print("creating session...")
        create_response = request_json("POST", "/v1/sessions", create_payload)
        artifact["create_response"] = create_response
        session_id = str(result(create_response)["session_id"])
        artifact["session_id"] = session_id
        print(f"session_id={session_id}")

        artifact["ready_events"] = wait_running(session_id)

        step_payload = {
            "session_id": session_id,
            "mode": "single",
            "actions": [
                {
                    "type": "env_action",
                    "name": "bash",
                    "args": {"cmd": REMOTE_COMMAND},
                }
            ],
            "limits": {"deadline_ms": STEP_DEADLINE_MS},
        }
        print("checking uv...")
        step_response = request_json(
            "POST",
            "/v1/sessions/step",
            step_payload,
            timeout=STEP_HTTP_TIMEOUT_SEC,
        )
        artifact["step_response"] = step_response
    finally:
        if session_id:
            print("closing session...")
            artifact["close_response"] = close_session(session_id)
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))

    print(f"wrote={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
