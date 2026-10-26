from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from swe_bench_harbor_rli.agents.llm_tunnel import bridge as llm_bridge


def _request_json(
    url: str,
    *,
    body: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        response = urlopen(request, timeout=2)
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())
    with response:
        return response.status, json.loads(response.read())


@contextmanager
def _running_bridge(
    monkeypatch,
    tmp_path: Path,
    *,
    timeout: float = 1,
) -> Iterator[str]:
    root = tmp_path / "llmtun"
    outbox = root / "outbox"
    inbox = root / "inbox"
    run_dir = root / "run"
    for directory in (outbox, inbox, run_dir):
        directory.mkdir(parents=True)

    monkeypatch.setattr(llm_bridge, "OUTBOX", outbox)
    monkeypatch.setattr(llm_bridge, "INBOX", inbox)
    monkeypatch.setattr(llm_bridge, "RUN_DIR", run_dir)
    monkeypatch.setattr(llm_bridge, "TIMEOUT_SEC", timeout)
    monkeypatch.setattr(llm_bridge, "POLL_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(llm_bridge, "SERVED_REQUESTS", 0)

    server = ThreadingHTTPServer(("127.0.0.1", 0), llm_bridge.BridgeHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _wait_for_request(outbox: Path) -> Path:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        requests = list(outbox.glob("*.req.json.gz"))
        if requests:
            return requests[0]
        time.sleep(0.01)
    raise AssertionError("Bridge did not publish a request")


def test_bridge_atomically_proxies_json_response(monkeypatch, tmp_path: Path) -> None:
    with _running_bridge(monkeypatch, tmp_path) as base_url:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _request_json,
                f"{base_url}/v1/chat/completions",
                body={"model": "agent-model", "messages": []},
            )
            request_path = _wait_for_request(llm_bridge.OUTBOX)
            envelope = llm_bridge._read_gzip_json(request_path)
            request_id = envelope["id"]

            assert envelope["path"] == "/v1/chat/completions"
            assert envelope["body"]["model"] == "agent-model"
            assert not list(llm_bridge.OUTBOX.glob("*.tmp"))

            llm_bridge._write_gzip_json(
                llm_bridge.INBOX / f"{request_id}.resp.json.gz",
                {
                    "id": request_id,
                    "status": 200,
                    "body": {"choices": [{"message": {"content": "done"}}]},
                },
            )
            status, body = future.result(timeout=2)

    assert status == 200
    assert body["choices"][0]["message"]["content"] == "done"


def test_bridge_rejects_streaming_without_publishing_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    with _running_bridge(monkeypatch, tmp_path) as base_url:
        status, body = _request_json(
            f"{base_url}/v1/chat/completions",
            body={"model": "agent-model", "messages": [], "stream": True},
        )

        assert status == 400
        assert body["error"]["code"] == "streaming_not_supported"
        assert not list(llm_bridge.OUTBOX.glob("*.req.json.gz"))


def test_bridge_returns_504_and_removes_timed_out_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    with _running_bridge(monkeypatch, tmp_path, timeout=0.05) as base_url:
        status, body = _request_json(
            f"{base_url}/v1/chat/completions",
            body={"model": "agent-model", "messages": []},
        )

        assert status == 504
        assert body["error"]["code"] == "tunnel_response_timeout"
        assert not list(llm_bridge.OUTBOX.glob("*.req.json.gz"))
