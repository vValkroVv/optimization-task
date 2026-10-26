#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import logging
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("LLMTUN_DIR", "/opt/llmtun"))
OUTBOX = ROOT / "outbox"
INBOX = ROOT / "inbox"
RUN_DIR = ROOT / "run"
PID_PATH = RUN_DIR / "bridge.pid"
LOG_PATH = Path(os.environ.get("LLMTUN_LOG_PATH", "/logs/agent/bridge.log"))
HOST = os.environ.get("LLMTUN_HOST", "127.0.0.1")
PORT = int(os.environ.get("LLMTUN_PORT", "8000"))
TIMEOUT_SEC = float(os.environ.get("LLMTUN_TIMEOUT", "1800"))
POLL_INTERVAL_SEC = float(os.environ.get("LLMTUN_POLL_INTERVAL", "0.1"))

LOGGER = logging.getLogger("llm_bridge")
STATS_LOCK = threading.Lock()
SERVED_REQUESTS = 0


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Tunnel envelope must be a JSON object")
    return payload


def _wait_for_response(request_id: str) -> dict[str, Any]:
    path = INBOX / f"{request_id}.resp.json.gz"
    deadline = time.monotonic() + TIMEOUT_SEC
    while time.monotonic() < deadline:
        try:
            payload = _read_gzip_json(path)
        except FileNotFoundError:
            time.sleep(POLL_INTERVAL_SEC)
            continue
        path.unlink(missing_ok=True)
        return payload
    raise TimeoutError(f"Timed out waiting for response {request_id}")


def _error_body(message: str, code: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "llm_tunnel_error",
            "code": code,
        }
    }


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = "RLI-LLM-Bridge/1.0"

    def log_message(self, message: str, *args: object) -> None:
        LOGGER.info("%s %s", self.client_address[0], message % args)

    def _send_json(self, status: int, body: Any) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_tunnel_error(self, status: int, message: str, code: str) -> None:
        self._send_json(status, _error_body(message, code))

    def _proxy(self, method: str, path: str, body: Any) -> None:
        global SERVED_REQUESTS

        request_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex}"
        request_path = OUTBOX / f"{request_id}.req.json.gz"
        _write_gzip_json(
            request_path,
            {
                "id": request_id,
                "method": method,
                "path": path,
                "body": body,
            },
        )

        try:
            response = _wait_for_response(request_id)
        except TimeoutError as exc:
            request_path.unlink(missing_ok=True)
            LOGGER.error("%s", exc)
            self._send_tunnel_error(504, str(exc), "tunnel_response_timeout")
            return
        except (OSError, ValueError) as exc:
            LOGGER.exception("Invalid tunnel response for %s", request_id)
            self._send_tunnel_error(502, str(exc), "invalid_tunnel_response")
            return

        if response.get("id") != request_id:
            self._send_tunnel_error(
                502,
                f"Unexpected response id {response.get('id')!r}",
                "response_id_mismatch",
            )
            return

        try:
            status = int(response["status"])
        except (KeyError, TypeError, ValueError):
            self._send_tunnel_error(
                502, "Response has no valid HTTP status", "invalid_status"
            )
            return

        if status < 100 or status > 599:
            self._send_tunnel_error(
                502, f"Invalid HTTP status {status}", "invalid_status"
            )
            return

        with STATS_LOCK:
            SERVED_REQUESTS += 1
        self._send_json(status, response.get("body", {}))

    def do_GET(self) -> None:
        path = self.path.partition("?")[0]
        if path == "/health":
            with STATS_LOCK:
                served = SERVED_REQUESTS
            self._send_json(
                200,
                {
                    "pending": len(list(OUTBOX.glob("*.req.json.gz"))),
                    "served": served,
                },
            )
            return
        if path == "/v1/models":
            self._proxy("GET", path, None)
            return
        self._send_tunnel_error(404, f"Unknown path {path}", "not_found")

    def do_POST(self) -> None:
        path = self.path.partition("?")[0]
        if path != "/v1/chat/completions":
            self._send_tunnel_error(404, f"Unknown path {path}", "not_found")
            return

        try:
            content_length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(content_length))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_tunnel_error(
                400, "Request body must be valid JSON", "invalid_json"
            )
            return

        if not isinstance(body, dict):
            self._send_tunnel_error(
                400, "Request body must be a JSON object", "invalid_json"
            )
            return
        if body.get("stream") is True:
            self._send_tunnel_error(
                400,
                "Streaming is not supported by the RLI LLM tunnel",
                "streaming_not_supported",
            )
            return

        self._proxy("POST", path, body)


def main() -> None:
    OUTBOX.mkdir(parents=True, exist_ok=True)
    INBOX.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    temporary_pid = PID_PATH.with_suffix(".tmp")
    temporary_pid.write_text(str(os.getpid()), encoding="utf-8")
    os.replace(temporary_pid, PID_PATH)

    server = ThreadingHTTPServer((HOST, PORT), BridgeHandler)
    server.daemon_threads = True
    LOGGER.info("Bridge listening on %s:%d", HOST, PORT)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        PID_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
