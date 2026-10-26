from __future__ import annotations

import asyncio
import gzip
import json
import logging
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable

import httpx
from harbor.environments.base import BaseEnvironment
from tenacity import retry, stop_after_attempt, wait_exponential

from ...src.errors import (
    EndpointUnavailableError,
    RLIExecProxyConnectionLostError,
)
from ...src.token_budget import account_output_tokens


_ROOT = "/opt/llmtun"
_OUTBOX = f"{_ROOT}/outbox"
_INBOX = f"{_ROOT}/inbox"
_RUN_DIR = f"{_ROOT}/run"
_BRIDGE_PATH = "/installed-agent/llm_bridge.py"
_BRIDGE_PYTHON = "/opt/openhands-sdk-venv/bin/python"
_BRIDGE_TIMEOUT_GRACE_SEC = 120
_IDEMPOTENT_EXEC_RETRIES = 2
_LOGGER = logging.getLogger(__name__)
_UNAVAILABLE_STATUS_CODES = {502, 503}

_POLL_COMMAND = f"""
if [ -s {_RUN_DIR}/agent.pid ]; then
  pid="$(cat {_RUN_DIR}/agent.pid)"
  if kill -0 "$pid" 2>/dev/null; then
    printf 'PID:alive\\n'
  else
    printf 'PID:dead\\n'
  fi
else
  printf 'PID:missing\\n'
fi
if [ -f {_RUN_DIR}/agent.exit ]; then
  printf 'EXIT:%s\\n' "$(cat {_RUN_DIR}/agent.exit)"
else
  printf 'EXIT:\\n'
fi
for request in {_OUTBOX}/*.req.json.gz; do
  [ -f "$request" ] || break
  printf 'FILE:%s\\n' "$request"
  break
done
""".strip()


@dataclass(frozen=True)
class _PollState:
    exit_code: int | None
    pid_state: str
    request_path: str | None


def _error_body(message: str, code: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "llm_tunnel_error",
            "code": code,
        }
    }


class RLILLMTunnel:
    def __init__(
        self,
        environment: BaseEnvironment,
        *,
        bridge_path: Path,
        llm_base_url: str,
        api_key: str | None,
        served_model: str | None = None,
        llm_timeout: int = 600,
        poll_interval: float = 0.7,
        unavailable_threshold: int = 5,
    ) -> None:
        if llm_timeout < 1:
            raise ValueError("llm_timeout must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if unavailable_threshold < 1:
            raise ValueError("unavailable_threshold must be positive")

        self._environment = environment
        self._bridge_path = bridge_path
        self._llm_base_url = llm_base_url.rstrip("/")
        self._api_key = api_key
        self._served_model = served_model
        self._llm_timeout = llm_timeout
        self._poll_interval = poll_interval
        self._unavailable_threshold = unavailable_threshold
        self._unavailable_failures = 0
        self._cache_dir = tempfile.TemporaryDirectory(prefix="rli-llm-tunnel-")
        self._responses: dict[str, Path] = {}

    @staticmethod
    def _raise_if_connection_lost(result: Any, operation: str) -> None:
        output = "\n".join(value for value in (result.stdout, result.stderr) if value)
        if result.return_code == -1 or "exec-proxy connection lost" in output.lower():
            raise RLIExecProxyConnectionLostError(
                f"RLI exec-proxy connection lost while {operation}: "
                f"{output or 'No output'}"
            )

    async def _exec(
        self,
        command: str,
        *,
        operation: str,
        user: str | int | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        result = await self._environment.exec(
            command,
            user=user,
            timeout_sec=timeout_sec,
        )
        self._raise_if_connection_lost(result, operation)
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed while {operation}: return_code={result.return_code}, "
                f"stdout={result.stdout!r}, stderr={result.stderr!r}"
            )
        return result

    async def _exec_idempotent(
        self,
        command: str,
        *,
        operation: str,
        user: str | int | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        for attempt in range(_IDEMPOTENT_EXEC_RETRIES + 1):
            try:
                return await self._exec(
                    command,
                    operation=operation,
                    user=user,
                    timeout_sec=timeout_sec,
                )
            except RLIExecProxyConnectionLostError:
                if attempt == _IDEMPOTENT_EXEC_RETRIES:
                    raise
                await asyncio.sleep(self._poll_interval)
        raise AssertionError("unreachable")

    async def install(self) -> None:
        agent_user = str(self._environment.default_user or "root")
        quoted_root = shlex.quote(_ROOT)
        await self._exec(
            (
                f"mkdir -p {_OUTBOX} {_INBOX} {_RUN_DIR} && "
                f"rm -f {_OUTBOX}/* {_INBOX}/* {_RUN_DIR}/* && "
                f"chown -R {shlex.quote(agent_user)} {quoted_root}"
            ),
            operation="preparing the LLM tunnel directories",
            user="root",
        )
        await self._environment.upload_file(
            self._bridge_path,
            _BRIDGE_PATH,
        )
        await self._exec(
            f"chmod 0755 {shlex.quote(_BRIDGE_PATH)}",
            operation="making the LLM bridge executable",
            user="root",
        )
        await self._exec(
            (
                "LLMTUN_TIMEOUT="
                f"{self._llm_timeout + _BRIDGE_TIMEOUT_GRACE_SEC} "
                f"setsid {_BRIDGE_PYTHON} {_BRIDGE_PATH} "
                "</dev/null >/dev/null 2>&1 &"
            ),
            operation="starting the LLM bridge",
        )
        await self._wait_until_healthy()

    async def _wait_until_healthy(self) -> None:
        health_command = f"{_BRIDGE_PYTHON} -c " + shlex.quote(
            "import urllib.request; "
            "response = urllib.request.urlopen("
            "'http://127.0.0.1:8000/health', timeout=2); "
            "raise SystemExit(0 if response.status == 200 else 1)"
        )
        last_result: Any = None
        for _ in range(20):
            last_result = await self._environment.exec(
                health_command,
                timeout_sec=5,
            )
            self._raise_if_connection_lost(
                last_result, "checking the LLM bridge health"
            )
            if last_result.return_code == 0:
                return
            await asyncio.sleep(0.25)
        raise RuntimeError(
            "LLM bridge did not become healthy: "
            f"stdout={getattr(last_result, 'stdout', None)!r}, "
            f"stderr={getattr(last_result, 'stderr', None)!r}"
        )

    async def shutdown(self) -> None:
        try:
            await self._exec(
                (
                    f"if [ -s {_RUN_DIR}/bridge.pid ]; then "
                    f"pid=$(cat {_RUN_DIR}/bridge.pid); "
                    'kill -TERM "$pid" 2>/dev/null || true; '
                    "for attempt in $(seq 1 20); do "
                    'kill -0 "$pid" 2>/dev/null || break; '
                    "sleep 0.1; "
                    "done; "
                    'kill -KILL "$pid" 2>/dev/null || true; '
                    "fi; "
                    f"rm -f {_RUN_DIR}/bridge.pid"
                ),
                operation="stopping the LLM bridge",
                user="root",
                timeout_sec=10,
            )
        finally:
            self._cache_dir.cleanup()

    async def run(self) -> int:
        timeout = httpx.Timeout(
            self._llm_timeout,
            connect=min(self._llm_timeout, 30),
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            while True:
                state = await self._poll()
                if state.request_path is not None:
                    await self._process_request(client, state.request_path)
                    continue
                if state.exit_code is not None:
                    return state.exit_code
                if state.pid_state != "alive":
                    raise RuntimeError(
                        "In-container agent stopped without publishing agent.exit"
                    )
                await asyncio.sleep(self._poll_interval)

    async def _poll(self) -> _PollState:
        result = await self._exec_idempotent(
            _POLL_COMMAND,
            operation="polling the in-container agent and LLM outbox",
        )
        exit_code: int | None = None
        pid_state: str | None = None
        request_path: str | None = None
        protocol_started = False

        for line in (result.stdout or "").splitlines():
            if line.startswith("PID:"):
                pid_state = line.removeprefix("PID:")
                protocol_started = True
            elif not protocol_started:
                continue
            elif line.startswith("EXIT:"):
                value = line.removeprefix("EXIT:")
                if value:
                    exit_code = int(value)
            elif line.startswith("FILE:"):
                request_path = line.removeprefix("FILE:")
            elif line:
                raise RuntimeError(f"Unexpected tunnel poll output: {line!r}")

        if pid_state not in {"alive", "dead", "missing"}:
            raise RuntimeError(f"Missing or invalid agent PID state: {pid_state!r}")
        return _PollState(exit_code, pid_state, request_path)

    async def _process_request(
        self,
        client: httpx.AsyncClient,
        request_path: str,
    ) -> None:
        remote_path = PurePosixPath(request_path)
        if str(remote_path.parent) != _OUTBOX:
            raise ValueError(f"Unexpected request path: {request_path!r}")
        if not remote_path.name.endswith(".req.json.gz"):
            raise ValueError(f"Unexpected request file: {request_path!r}")

        local_request = Path(self._cache_dir.name) / remote_path.name
        await self._environment.download_file(request_path, local_request)
        with gzip.open(local_request, "rt", encoding="utf-8") as stream:
            envelope = json.load(stream)
        if not isinstance(envelope, dict):
            raise ValueError("Tunnel request must be a JSON object")

        request_id = envelope.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Tunnel request has no valid id")
        expected_name = f"{request_id}.req.json.gz"
        if remote_path.name != expected_name:
            raise ValueError(f"Request id does not match filename: {request_id!r}")

        local_response = self._responses.get(request_id)
        if local_response is None:
            status, body = await self._call_upstream(client, envelope)
            local_response = Path(self._cache_dir.name) / f"{request_id}.resp.json.gz"
            with gzip.open(local_response, "wt", encoding="utf-8") as stream:
                json.dump(
                    {"id": request_id, "status": status, "body": body},
                    stream,
                    separators=(",", ":"),
                )
            self._responses[request_id] = local_response

        await self._deliver_response(
            request_id=request_id,
            request_path=request_path,
            local_response=local_response,
        )
        local_request.unlink(missing_ok=True)
        local_response.unlink(missing_ok=True)
        self._responses.pop(request_id, None)

    async def _call_upstream(
        self,
        client: httpx.AsyncClient,
        envelope: dict[str, Any],
    ) -> tuple[int, Any]:
        method = envelope.get("method")
        path = envelope.get("path")
        body = envelope.get("body")

        if (method, path) not in {
            ("POST", "/v1/chat/completions"),
            ("GET", "/v1/models"),
        }:
            raise ValueError(f"Unsupported upstream request: {method!r} {path!r}")
        if method == "POST" and not isinstance(body, dict):
            raise ValueError("Completion request body must be a JSON object")

        if isinstance(body, dict):
            body = dict(body)
            if self._served_model is not None:
                body["model"] = self._served_model

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = await self._request_upstream(
                client,
                method=method,
                path=path,
                headers=headers,
                body=body,
            )
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.RemoteProtocolError,
        ) as exc:
            status = 503
            response_body = _error_body(
                f"Model endpoint is unavailable: {exc}",
                "endpoint_unavailable",
            )
        except httpx.TimeoutException as exc:
            status = 504
            response_body = _error_body(
                f"Model request timed out: {exc}",
                "model_request_timeout",
            )
        else:
            status = response.status_code
            if response.content:
                try:
                    response_body = response.json()
                except ValueError:
                    response_body = _error_body(
                        "Model endpoint returned a non-JSON response",
                        "invalid_upstream_response",
                    )
            else:
                response_body = {}

        self._record_availability(status)
        if (
            method == "POST"
            and 200 <= status < 300
            and isinstance(response_body, dict)
        ):
            usage = response_body.get("usage")
            if isinstance(usage, dict):
                account_output_tokens(usage.get("completion_tokens"))
        return status, response_body

    async def _request_upstream(
        self,
        client: httpx.AsyncClient,
        *,
        method: str,
        path: str,
        headers: dict[str, str],
        body: Any,
        max_attempts: int = 20,
    ) -> httpx.Response:
        async def send_request() -> httpx.Response:
            response = await client.request(
                method,
                self._target_url(path),
                headers=headers,
                json=body if method == "POST" else None,
            )
            if response.status_code in _UNAVAILABLE_STATUS_CODES:
                response.raise_for_status()
            return response

        request_with_retry: Callable[[], Awaitable[httpx.Response]] = retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=0.5, min=1, max=10),
            reraise=True,
            before_sleep=lambda retry_state: _LOGGER.info(
                f"Retry attempt {retry_state.attempt_number}"
            ),
        )(send_request)
        try:
            return await request_with_retry()
        except (httpx.ReadError, httpx.HTTPStatusError) as exc:
            raise EndpointUnavailableError(
                f"Model endpoint request failed for {max_attempts} consecutive attempts"
            ) from exc

    def _record_availability(self, status: int) -> None:
        if status not in _UNAVAILABLE_STATUS_CODES:
            self._unavailable_failures = 0
            return
        self._unavailable_failures += 1
        if self._unavailable_failures >= self._unavailable_threshold:
            raise EndpointUnavailableError(
                "Model endpoint remained unavailable for "
                f"{self._unavailable_failures} consecutive requests"
            )

    def _target_url(self, path: str) -> str:
        if self._llm_base_url.endswith("/v1") and path.startswith("/v1/"):
            return f"{self._llm_base_url[:-3]}{path}"
        return f"{self._llm_base_url}{path}"

    async def _deliver_response(
        self,
        *,
        request_id: str,
        request_path: str,
        local_response: Path,
    ) -> None:
        temporary = f"{_INBOX}/{request_id}.resp.json.gz.tmp"
        final = f"{_INBOX}/{request_id}.resp.json.gz"
        await self._environment.upload_file(local_response, temporary)
        await self._exec_idempotent(
            (
                f"if [ -f {shlex.quote(temporary)} ]; then "
                f"mv -f {shlex.quote(temporary)} {shlex.quote(final)}; "
                "fi && "
                f"rm -f {shlex.quote(request_path)}"
            ),
            operation=f"delivering LLM response {request_id}",
        )
