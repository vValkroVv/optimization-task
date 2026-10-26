from __future__ import annotations

import asyncio
import base64
import json
import shlex
import tarfile
import tempfile
import time
import tomllib
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from random import uniform
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from httpx import HTTPStatusError, RequestError

from .errors import (
    RLIExecProxyConnectionLostError,
    RLILongRunningCommandLostError,
)
from .gigarli_bootstrap import ensure_gigarli_import_path
from .rli_policy import SWEBenchRLIPolicy, SWEBenchRLIPolicyConfig

ensure_gigarli_import_path()

from gigarli import (  # noqa: E402
    CloseSessionRequest,
    CreateSessionRequest,
    RequestLimits,
    RLIClient,
    RLIClientError,
    SessionEnv,
    StepAction,
    StepActionArgs,
    StepResult,
    StepRequest,
)


_UPLOAD_CHUNK_CHARS = 48_000
_DOWNLOAD_CHUNK_BYTES = 2_000_000
_DOWNLOAD_ROOT = "/opt/rli-downloads"
_RLI_MAX_STEP_DEADLINE_MS = 1_500_000
_LONG_COMMAND_STEP_DEADLINE_MS = 60_000
_LONG_COMMAND_POLL_INTERVAL_SEC = 30.0
_LONG_COMMAND_POLL_MAX_FAILURES = 3
_LONG_COMMAND_ROOT = "/logs/verifier/.rli-long-running"

_APPLY_PATCH_SHIM_PATH = Path(__file__).resolve().parent / "apply_patch_shim.py"
_REMOTE_APPLY_PATCH_PATH = "/usr/local/bin/apply_patch"
_REMOTE_WRITABLE_DIRS: tuple[str, ...] = (
    "/logs",
    "/logs/agent",
    "/logs/verifier",
    "/logs/artifacts",
    "/tests",
)
_REMOTE_CHMOD_ROOTS: tuple[str, ...] = ("/logs", "/tests")
_DEFAULT_WORKDIR = "/testbed"
_UV_VERSION = "0.7.13"
_UV_ASSET_URL = (
    "https://github.com/astral-sh/uv/releases/download/"
    f"{_UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
)
_INSTALL_UV_COMMAND = f"""set -e
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
_APT_FORCE_IPV4_COMMAND = """cat >/etc/apt/apt.conf.d/99force-ipv4 <<'EOF'
Acquire::ForceIPv4 "true";
EOF"""
_APT_CONFIGURE_SOURCES_COMMAND = """set -e
if [ -r /etc/os-release ]; then . /etc/os-release; fi
if [ "${ID:-}" = "ubuntu" ]; then
  codename="${VERSION_CODENAME:-noble}"
  rm -f /etc/apt/sources.list.d/ubuntu.sources
  printf '%s\n' \
    "deb https://archive.ubuntu.com/ubuntu/ ${codename} main restricted universe multiverse" \
    "deb https://archive.ubuntu.com/ubuntu/ ${codename}-updates main restricted universe multiverse" \
    "deb https://archive.ubuntu.com/ubuntu/ ${codename}-backports main restricted universe multiverse" \
    "deb https://security.ubuntu.com/ubuntu/ ${codename}-security main restricted universe multiverse" \
    >/etc/apt/sources.list
elif [ "${ID:-}" = "debian" ]; then
  codename="${VERSION_CODENAME:-bookworm}"
  rm -f /etc/apt/sources.list.d/debian.sources
  printf '%s\n' \
    "deb https://deb.debian.org/debian ${codename} main" \
    "deb https://deb.debian.org/debian ${codename}-updates main" \
    "deb https://security.debian.org/debian-security ${codename}-security main" \
    >/etc/apt/sources.list
else
  for path in \
    /etc/apt/sources.list \
    /etc/apt/sources.list.d/debian.sources \
    /etc/apt/sources.list.d/ubuntu.sources
  do
    [ -e "$path" ] || continue
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' "$path"
    sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' "$path"
    sed -i 's|http://security.ubuntu.com|https://security.ubuntu.com|g' "$path"
  done
fi
cat /etc/apt/sources.list 2>/dev/null || true"""


def _host_safe_mode(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:
    member.uid = member.gid = None
    member.uname = member.gname = None
    if member.isdir():
        member.mode = 0o2775
    else:
        member.mode = 0o755 if (member.mode or 0) & 0o100 else 0o644
    return member


def _normalize_rli_config(value: str | dict[str, Any] | None) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        loaded = json.loads(value)
    elif isinstance(value, dict):
        loaded = value
    else:
        raise TypeError(
            "rli_config_json must be a JSON string, dictionary, or None, "
            f"got {type(value).__name__}"
        )
    if not isinstance(loaded, dict):
        raise TypeError(
            "rli_config_json must decode to a JSON object, "
            f"got {type(loaded).__name__}"
        )
    return dict(loaded)


class RLIEnvironment(BaseEnvironment):
    @staticmethod
    def type() -> str:
        return "rli"

    def __init__(
        self,
        *args: Any,
        rli_config_json: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        config = _normalize_rli_config(rli_config_json)
        self._rli_config = config
        self._rli_session_id: str | None = None
        self._rli_task_id_override = config.get("task_id") or config.get("task_name")
        self._rli_env_prefix = config.get("env_prefix", "swe-bench")
        self._rli_env_version = config.get("env_version", "1.0.0")
        self._rli_ttl_ms = config.get("ttl_ms")
        self._rli_seed = config.get("seed")
        self._rli_use_pip_proxy = config.get(
            "use_pip_proxy",
            config.get("use-pip-proxy"),
        )
        self._rli_wait_for_ready = bool(config.get("wait_for_ready", True))
        self._rli_command_timeout_ms = config.get("command_timeout_ms")
        self._rli_verifier_command_timeout_ms = config.get(
            "verifier_command_timeout_ms"
        )
        self._rli_active_verifier_timeout_ms: int | None = None
        self._rli_transfer_retries = int(config.get("transfer_retries", 2))
        self._rli_transfer_retry_backoff = float(
            config.get("transfer_retry_backoff", 2.0)
        )
        self._rli_policy = SWEBenchRLIPolicy(
            SWEBenchRLIPolicyConfig(
                warmup_mode=str(config.get("warmup_mode", "poll_only")),
                enable_probe_warmup=bool(config.get("enable_probe_warmup", False)),
                max_bootstrap_step_retries=int(
                    config.get("bootstrap_step_retries", 2)
                ),
                max_transfer_retries=int(config.get("transfer_retries", 2)),
                max_verifier_step_retries=int(
                    config.get("verifier_step_retries", 1)
                ),
                max_agent_step_retries=int(config.get("agent_step_retries", 0)),
                probe_retry_backoff_sec=float(
                    config.get("probe_retry_backoff_sec", 2.0)
                ),
                max_create_session_retries=int(config.get("request_retries", 2)),
                create_session_retry_backoff_sec=float(
                    config.get("request_retry_backoff", 2.0)
                ),
                create_session_retry_backoff_max_sec=float(
                    config.get("create_session_retry_backoff_max_sec", 30.0)
                ),
                create_session_jitter_max_sec=float(
                    config.get("create_session_jitter_max_sec", 0.0)
                ),
            )
        )
        self._rli_bootstrap_apt = bool(config.get("bootstrap_apt", False))
        self._rli_bootstrap_uv = bool(config.get("bootstrap_uv", True))
        self._rli_trace_path = (
            Path(config["trace_path"]) if config.get("trace_path") else None
        )
        self._rli_client = RLIClient(
            base_url=config.get("api_base"),
            timeout=float(config.get("timeout", 60)),
            verify=bool(config.get("verify_ssl", False)),
            trace_path=self._rli_trace_path,
        )
        self._rli_create_session_jitter_max_sec = float(
            config.get(
                "create_session_jitter_max_sec",
                self._rli_policy.config.create_session_jitter_max_sec,
            )
        )
        self._rli_session_ready_timeout = float(
            config.get("session_ready_timeout", 600.0)
        )
        self._rli_session_ready_poll_interval = float(
            config.get("session_ready_poll_interval", 1.0)
        )
        super().__init__(*args, **kwargs)
        self._rli_default_workdir = (
            config.get("workdir")
            or config.get("default_workdir")
            or config.get("fallback_workdir")
            or self.task_env_config.workdir
            or _DEFAULT_WORKDIR
        )

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True, mounted=False)

    @contextmanager
    def use_verifier_command_timeout(self, timeout_sec: int) -> Iterator[None]:
        previous = self._rli_active_verifier_timeout_ms
        timeout_ms = int(timeout_sec * 1000)
        if self._rli_verifier_command_timeout_ms is not None:
            timeout_ms = min(timeout_ms, self._rli_verifier_command_timeout_ms)
        try:
            self._rli_active_verifier_timeout_ms = timeout_ms
            yield
        finally:
            self._rli_active_verifier_timeout_ms = previous

    def _validate_definition(self) -> None:
        return None

    def _infer_task_id(self) -> str:
        if self._rli_task_id_override:
            return str(self._rli_task_id_override)
        if self.environment_dir.name == "environment":
            return self.environment_dir.parent.name
        if self.environment_name.startswith(f"{self._rli_env_prefix}-"):
            return self.environment_name[len(self._rli_env_prefix) + 1 :]
        return self.environment_name

    def _build_env_name(self, task_id: str) -> str:
        if self._rli_config.get("env_name"):
            return str(self._rli_config["env_name"])
        short_id = task_id.split("__", 1)[1] if "__" in task_id else task_id
        return f"{self._rli_env_prefix}-{short_id}"

    async def start(self, force_build: bool) -> None:
        _ = force_build
        task_id = self._infer_task_id()
        agent_timeout = self._rli_config.get("agent_timeout_sec")
        if agent_timeout is None:
            task_config = tomllib.loads(
                (self.environment_dir.parent / "task.toml").read_text(encoding="utf-8")
            )
            agent_timeout = task_config["agent"]["timeout_sec"]
        ttl_ms = int((agent_timeout + 600) * 1000)
        if self._rli_ttl_ms is not None:
            ttl_ms = min(int(self._rli_ttl_ms), ttl_ms)
        allow_internet_access = self._rli_config.get("allow_internet_access", True)
        request = CreateSessionRequest(
            allow_internet_access=allow_internet_access,
            env=SessionEnv(
                name=self._build_env_name(task_id),
                version=str(self._rli_env_version),
                task_id=task_id,
            ),
            ttl_ms=ttl_ms,
            use_pip_proxy=False,
        )
        session = await self._create_session_with_retries(request)
        self._rli_session_id = session.session_id
        try:
            if self._rli_wait_for_ready:
                await self._wait_until_session_ready(session.session_id)
            if self._rli_policy.should_run_probe_warmup():
                await self._probe_warmup_session(
                    self._rli_policy.probe_warmup_command()
                )
            await self._install_apply_patch_shim()
            await self._prepare_remote_layout()
            await self._install_uv()
            await self._bootstrap_remote_apt()
        except Exception:
            await self._close_started_session_safely()
            raise

    async def _wait_until_session_ready(self, session_id: str) -> None:
        deadline = time.monotonic() + self._rli_session_ready_timeout
        while time.monotonic() < deadline:
            result = await self._rli_client.get_session(session_id)
            status = result.status
            if self._is_terminal_session_status(status):
                raise RuntimeError(
                    f"RLI session {session_id} reached terminal status {status}"
                )
            ready = self._extract_session_ready_state(status)
            if ready is True:
                return
            await asyncio.sleep(self._rli_session_ready_poll_interval)
        raise RuntimeError(f"Timed out waiting for session {session_id}")

    @staticmethod
    def _extract_session_ready_state(status_value: str | None) -> bool | None:
        if isinstance(status_value, str):
            status = status_value.strip().lower()
            if status == "running":
                return True
            if status in {
                "pending",
                "starting",
                "creating",
                "initializing",
                "provisioning",
            }:
                return False
        return None

    @staticmethod
    def _is_terminal_session_status(status_value: str | None) -> bool:
        if not isinstance(status_value, str):
            return False
        return status_value.strip().lower() in {"closed", "failed", "terminated"}

    async def _probe_warmup_session(self, command: str) -> None:
        attempts = max(1, self._rli_policy.config.max_bootstrap_step_retries + 1)
        last_result: ExecResult | None = None
        for attempt in range(1, attempts + 1):
            result = await self.exec(command, user="root")
            if result.return_code == 0:
                return
            last_result = result
            failure_kind = self._rli_policy.classify_step_failure(
                phase="probe",
                exit_code=result.return_code,
                error=result.stderr or result.stdout,
                incomplete=result.return_code in (-1, 124),
                step_index=0,
            )
            decision = self._rli_policy.should_retry_step(
                phase="probe",
                failure_kind=failure_kind,
                attempt=attempt,
            )
            if not decision.retry:
                break
            backoff = self._rli_policy.retry_backoff_seconds(
                phase="probe",
                attempt=attempt,
            )
            if backoff > 0:
                await asyncio.sleep(backoff)
        assert last_result is not None
        raise RuntimeError(
            f"Failed probe warmup command: return_code={last_result.return_code}, "
            f"stdout={last_result.stdout!r}, stderr={last_result.stderr!r}, "
            f"command={command!r}"
        )

    async def _create_session_with_retries(
        self,
        request: CreateSessionRequest,
    ) -> Any:
        idempotency_key = str(uuid.uuid4())
        jitter_max = max(
            0.0,
            float(getattr(self, "_rli_create_session_jitter_max_sec", 0.0)),
        )
        if jitter_max > 0:
            await asyncio.sleep(uniform(0.0, jitter_max))

        attempts = max(1, self._rli_policy.config.max_create_session_retries + 1)
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            retry_strategy: str | None = None
            try:
                return await self._rli_client.create_session(
                    request,
                    idempotency_key=idempotency_key,
                )
            except RLIClientError as exc:
                last_error = exc
                error = exc.error
                retry_strategy = error.retry_strategy
                failure_kind = self._rli_policy.classify_create_session_failure(
                    status_code=None,
                    transport_error=False,
                    retryable=error.retryable,
                    retry_strategy=error.retry_strategy,
                    error_code=error.code,
                    error_category=error.category,
                )
            except HTTPStatusError as exc:
                last_error = exc
                status_code = (
                    exc.response.status_code if exc.response is not None else None
                )
                error_payload: dict[str, Any] = {}
                if exc.response is not None:
                    try:
                        payload = exc.response.json()
                    except ValueError:
                        payload = None
                    if isinstance(payload, dict) and isinstance(
                        payload.get("error"), dict
                    ):
                        error_payload = payload["error"]
                retry_strategy_value = error_payload.get("retry_strategy")
                if isinstance(retry_strategy_value, str):
                    retry_strategy = retry_strategy_value
                failure_kind = self._rli_policy.classify_create_session_failure(
                    status_code=status_code,
                    transport_error=False,
                    retryable=error_payload.get("retryable"),
                    retry_strategy=retry_strategy,
                    error_code=error_payload.get("code"),
                    error_category=error_payload.get("category"),
                )
            except RequestError as exc:
                last_error = exc
                failure_kind = self._rli_policy.classify_create_session_failure(
                    status_code=None,
                    transport_error=True,
                )

            decision = self._rli_policy.should_retry_create_session(
                failure_kind=failure_kind,
                attempt=attempt,
            )
            if not decision.retry:
                break

            if retry_strategy in {
                "create_new_session",
                "wait_and_retry_step",
            }:
                idempotency_key = str(uuid.uuid4())

            backoff = self._rli_policy.create_session_retry_backoff_seconds(
                attempt=attempt,
            )
            if backoff > 0:
                await asyncio.sleep(backoff)

        assert last_error is not None
        raise last_error

    async def _install_apply_patch_shim(self) -> None:
        """Install an ``apply_patch`` shim into the RLI session.

        gpt-oss emits OpenAI-style ``apply_patch`` envelopes; the RLI base
        image does not ship that tool, so every such call would return
        rc=127. Best-effort: log and continue if the install fails — the
        agent will then fall back to its previous (worse) behaviour.
        """
        try:
            script_bytes = _APPLY_PATCH_SHIM_PATH.read_bytes()
        except OSError:
            return
        encoded = base64.b64encode(script_bytes).decode("ascii")
        target = shlex.quote(_REMOTE_APPLY_PATCH_PATH)
        try:
            await self._exec_bootstrap_command(
                f"mkdir -p $(dirname {target})",
                step_index=0,
                user="root",
            )
            await self._exec_bootstrap_command(
                f"printf %s {shlex.quote(encoded)} | base64 -d > {target}",
                step_index=1,
                user="root",
            )
            await self._exec_bootstrap_command(
                f"chmod 0755 {target}",
                step_index=2,
                user="root",
            )
        except Exception:
            return

    async def _prepare_remote_layout(self) -> None:
        """Create the /logs and /tests directories the Harbor verifier expects.

        SWE-bench RLI images ship neither directory (Harbor's own swebench
        Dockerfile would ``mkdir /logs``, but RLI uses the upstream images
        without that layer), so a root bootstrap step creates them.
        """
        dirs = " ".join(shlex.quote(path) for path in _REMOTE_WRITABLE_DIRS)
        roots = " ".join(shlex.quote(path) for path in _REMOTE_CHMOD_ROOTS)
        try:
            await self._exec_bootstrap_command(
                f"mkdir -p {dirs} && chmod -R 777 {roots}",
                step_index=0,
                user="root",
            )
        except Exception:
            return

    async def _install_uv(self) -> None:
        """Install ``uv`` into the RLI session for the Harbor verifier.

        ``test.sh`` runs ``uv run parser.py``; Harbor's swebench Dockerfile
        installs ``uv`` but RLI uses the upstream images without it. Best-effort:
        log nothing and continue if the install fails.
        """
        if not self._rli_bootstrap_uv:
            return
        try:
            await self._exec_bootstrap_command(
                _INSTALL_UV_COMMAND,
                step_index=0,
                user="root",
            )
        except Exception:
            return

    async def _bootstrap_remote_apt(self) -> None:
        """Best-effort fix for RLI images whose apt sources/timeouts are flaky.

        Some Debian/Ubuntu-backed RLI sessions can reach outbound HTTPS but fail
        on the default HTTP apt mirrors, especially when IPv6 is preferred. We
        observed that forcing IPv4 and rewriting common mirror URIs to HTTPS is
        enough to make later ``apt-get install ...`` calls succeed.

        Keep this non-fatal:
        - skip entirely when the image has no ``apt-get``
        - ignore missing source files
        - ignore bootstrap failures so tasks that do not need apt can still run
        """

        if not self._rli_bootstrap_apt:
            return

        rewrite_commands: list[str] = [
            _APT_FORCE_IPV4_COMMAND,
            _APT_CONFIGURE_SOURCES_COMMAND,
        ]

        # Keep only the cheap, global-safe part here. Running apt index warmup
        # for every task made ordinary runs much worse and could strand a
        # session inside one long bootstrap step.
        try:
            has_apt = await self._exec_bootstrap_command(
                "command -v apt-get >/dev/null 2>&1",
                step_index=0,
                user="root",
                raise_on_failure=False,
            )
            if has_apt.return_code != 0:
                return
            for step_index, command in enumerate(rewrite_commands, start=1):
                await self._exec_bootstrap_command(
                    command,
                    step_index=step_index,
                    user="root",
                )
        except Exception:
            return

    async def _close_started_session_safely(self) -> None:
        if self._rli_session_id is None:
            return
        try:
            await self._rli_client.close_session(
                CloseSessionRequest(session_id=self._rli_session_id)
            )
        except Exception:
            pass
        finally:
            self._rli_session_id = None

    async def stop(self, delete: bool) -> None:
        _ = delete
        if self._rli_session_id is None:
            return
        session_id = self._rli_session_id
        task = asyncio.current_task()
        self.logger.warning(
            "RLI stop BEGIN: sid=%s trial=%s cancelling=%s\n%s",
            session_id,
            self._harbor_session_id,
            task.cancelling() if task is not None else None,
            "".join(traceback.format_stack()),
        )
        try:
            await self._rli_client.close_session(
                CloseSessionRequest(session_id=session_id)
            )
        except asyncio.CancelledError:
            self.logger.warning(
                "RLI stop CANCELLED: sid=%s trial=%s cancelling=%s",
                session_id,
                self._harbor_session_id,
                task.cancelling() if task is not None else None,
            )
            raise
        except Exception:
            self.logger.exception(
                "RLI stop ERROR: sid=%s trial=%s",
                session_id,
                self._harbor_session_id,
            )
            raise
        else:
            self.logger.warning(
                "RLI stop END: sid=%s trial=%s",
                session_id,
                self._harbor_session_id,
            )
        finally:
            self._rli_session_id = None

    async def prepare_logs_for_host(self) -> None:
        return None

    @property
    def session_id(self) -> str:
        if self._rli_session_id is not None:
            return self._rli_session_id
        stored = getattr(self, "_harbor_session_id", None)
        if stored is not None:
            return stored
        raise RuntimeError("RLI session is not started")

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self._harbor_session_id = value

    def _require_session(self) -> str:
        if self._rli_session_id is None:
            raise RuntimeError("RLI session is not started")
        return self._rli_session_id

    def _wrap_command(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        user: str | int | None,
    ) -> str:
        merged_env = self._merge_env(env) or {}
        effective = command
        workdir = cwd or self.task_env_config.workdir or self._rli_default_workdir
        if workdir:
            effective = f"cd {shlex.quote(workdir)} && {effective}"
        if merged_env:
            prefix = "; ".join(
                f"export {key}={shlex.quote(str(value))}"
                for key, value in merged_env.items()
            )
            effective = f"{prefix}; {effective}"
        resolved_user = self._resolve_user(user)
        if resolved_user is not None and str(resolved_user) != "root":
            effective = f"su -s /bin/bash {shlex.quote(str(resolved_user))} -c {shlex.quote(effective)}"
        return effective

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        explicit_timeout_ms = (
            int(timeout_sec * 1000) if timeout_sec is not None else None
        )
        verifier_timeout_ms = self._rli_active_verifier_timeout_ms
        deadline_ms = explicit_timeout_ms or verifier_timeout_ms
        if deadline_ms is None:
            deadline_ms = self._rli_command_timeout_ms

        wrapped_command = self._wrap_command(
            command,
            cwd=cwd,
            env=env,
            user=user if user is not None else self.default_user,
        )
        if (
            verifier_timeout_ms is not None
            and "/tests/" in command
            and "/logs/verifier/" in command
        ):
            return await self._exec_long_running(
                wrapped_command,
                timeout_ms=deadline_ms or verifier_timeout_ms,
            )

        if verifier_timeout_ms is not None and deadline_ms is not None:
            deadline_ms = min(deadline_ms, _RLI_MAX_STEP_DEADLINE_MS)
        return await self._exec_step(wrapped_command, deadline_ms=deadline_ms)

    async def _exec_step(
        self,
        command: str,
        *,
        deadline_ms: int | None,
    ) -> ExecResult:
        try:
            result = await self._rli_client.step(
                StepRequest(
                    session_id=self._require_session(),
                    mode="single",
                    actions=[
                        StepAction(
                            type="env_action",
                            name="bash",
                            args=StepActionArgs(cmd=command),
                        )
                    ],
                    limits=(
                        RequestLimits(deadline_ms=deadline_ms)
                        if deadline_ms is not None
                        else None
                    ),
                )
            )
        except HTTPStatusError as exc:
            if exc.response.status_code == 410:
                request_id = None
                try:
                    payload = exc.response.json()
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    meta = payload.get("meta")
                    if isinstance(meta, dict):
                        request_id = meta.get("request_id")

                task = asyncio.current_task()
                self.logger.warning(
                    "RLI step returned 410: sid=%s trial=%s request_id=%s "
                    "verifier_active=%s cancelling=%s\n%s",
                    self._rli_session_id,
                    self._harbor_session_id,
                    request_id,
                    self._rli_active_verifier_timeout_ms is not None,
                    task.cancelling() if task is not None else None,
                    "".join(traceback.format_stack()),
                )
            raise
        mapped = self._step_result_to_exec_result(result)
        exec_result = ExecResult(
            stdout=mapped.stdout,
            stderr=mapped.stderr,
            return_code=mapped.return_code,
        )
        self._raise_if_verifier_connection_lost(exec_result)

        if (
            result.events
            and result.applied_limits is not None
            and result.applied_limits.deadline_ms is not None
        ):
            event = result.events[0]
            applied_deadline_ms = result.applied_limits.deadline_ms
            if (
                event.exit_code == -1
                and event.duration_ms >= applied_deadline_ms
            ):
                timeout_sec = applied_deadline_ms / 1000
                timeout_message = (
                    f"Command exceeded the {timeout_sec:g}-second execution "
                    "timeout and was terminated. Change the command to finish "
                    "faster "
                    "or split it into smaller commands."
                )
                stderr = exec_result.stderr or ""
                if stderr and not stderr.endswith("\n"):
                    stderr += "\n"
                exec_result = ExecResult(
                    stdout=exec_result.stdout,
                    stderr=f"{stderr}{timeout_message}",
                    return_code=124,
                )

        return exec_result

    def _raise_if_verifier_connection_lost(self, result: ExecResult) -> None:
        if self._rli_active_verifier_timeout_ms is None:
            return
        output = "\n".join(
            value for value in (result.stdout, result.stderr) if value
        )
        if (
            result.return_code == -1
            or "exec-proxy connection lost" in output.lower()
        ):
            raise RLIExecProxyConnectionLostError(
                "RLI exec-proxy connection lost while executing a verifier "
                f"command: {output or 'No output'}"
            )

    async def _exec_long_running(
        self,
        command: str,
        *,
        timeout_ms: int,
    ) -> ExecResult:
        job_dir = f"{_LONG_COMMAND_ROOT}/{uuid.uuid4().hex}"
        run_script = (
            "#!/bin/bash\n"
            "set +e\n"
            f"{command}\n"
            "status=$?\n"
            f"printf '%s\\n' \"$status\" > {shlex.quote(job_dir + '/exit-code')}\n"
            "exit \"$status\"\n"
        )
        encoded_script = base64.b64encode(run_script.encode()).decode("ascii")
        quoted_job_dir = shlex.quote(job_dir)
        start_command = (
            f"mkdir -p {quoted_job_dir} && "
            f"printf %s {shlex.quote(encoded_script)} | base64 -d "
            f"> {shlex.quote(job_dir + '/run.sh')} && "
            f"chmod 0700 {shlex.quote(job_dir + '/run.sh')} && "
            f"{{ nohup /bin/bash {shlex.quote(job_dir + '/run.sh')} "
            f"> {shlex.quote(job_dir + '/stdout')} "
            f"2> {shlex.quote(job_dir + '/stderr')} </dev/null & "
            f"printf '%s\\n' \"$!\" > {shlex.quote(job_dir + '/pid')}; }}"
        )
        start_result = await self._exec_step(
            start_command,
            deadline_ms=_LONG_COMMAND_STEP_DEADLINE_MS,
        )
        if start_result.return_code != 0:
            return start_result

        deadline = time.monotonic() + timeout_ms / 1000
        consecutive_poll_failures = 0
        try:
            while time.monotonic() < deadline:
                try:
                    poll_result = await self._exec_step(
                        (
                            f"if [ -f {shlex.quote(job_dir + '/exit-code')} ]; then "
                            f"printf 'DONE:'; cat {shlex.quote(job_dir + '/exit-code')}; "
                            f"elif [ -r {shlex.quote(job_dir + '/pid')} ] && "
                            f"kill -0 \"$(cat {shlex.quote(job_dir + '/pid')})\" "
                            "2>/dev/null; then printf 'RUNNING\\n'; "
                            "else printf 'LOST\\n'; fi"
                        ),
                        deadline_ms=_LONG_COMMAND_STEP_DEADLINE_MS,
                    )
                except RLIExecProxyConnectionLostError:
                    consecutive_poll_failures += 1
                    if (
                        consecutive_poll_failures
                        >= _LONG_COMMAND_POLL_MAX_FAILURES
                    ):
                        raise
                    await asyncio.sleep(_LONG_COMMAND_POLL_INTERVAL_SEC)
                    continue
                consecutive_poll_failures = 0
                state = poll_result.stdout.strip()
                if state.startswith("DONE:"):
                    return_code = int(state.removeprefix("DONE:").strip())
                    return ExecResult(
                        stdout="",
                        stderr="",
                        return_code=return_code,
                    )
                if state != "RUNNING":
                    raise RLILongRunningCommandLostError(
                        "RLI verifier process disappeared before writing its "
                        f"exit code: {state or 'No output'}"
                    )
                await asyncio.sleep(_LONG_COMMAND_POLL_INTERVAL_SEC)

            return ExecResult(
                stdout="",
                stderr=f"Verifier command timed out after {timeout_ms / 1000:g}s",
                return_code=124,
            )
        except asyncio.CancelledError:
            task = asyncio.current_task()
            self.logger.warning(
                "RLI long-running command CANCELLED before cleanup: "
                "sid=%s trial=%s cancelling=%s\n%s",
                self._rli_session_id,
                self._harbor_session_id,
                task.cancelling() if task is not None else None,
                "".join(traceback.format_stack()),
            )
            raise
        finally:
            try:
                await self._exec_step(
                    (
                        f"if [ -r {shlex.quote(job_dir + '/pid')} ]; then "
                        f"kill \"$(cat {shlex.quote(job_dir + '/pid')})\" "
                        f">/dev/null 2>&1 || true; fi; rm -rf {quoted_job_dir}"
                    ),
                    deadline_ms=_LONG_COMMAND_STEP_DEADLINE_MS,
                )
            except Exception:
                self.logger.warning(
                    "Failed to clean up RLI long-running command: %s",
                    job_dir,
                    exc_info=True,
                )

    async def is_dir(self, path: str) -> bool:
        result = await self.exec(
            f"test -d {shlex.quote(path)}",
            user=getattr(self, "default_user", "root"),
        )
        return result.return_code == 0

    @staticmethod
    def _step_result_to_exec_result(result: StepResult) -> ExecResult:
        output = result.observation.output if result.observation is not None else ""
        stdout = output or ""
        stderr = ""
        return_code = 0
        if result.events:
            event = result.events[0]
            stdout = event.stdout_ref or stdout
            stderr = event.stderr_ref or ""
            if event.error and event.error not in stderr:
                stderr = "\n".join(value for value in (stderr, event.error) if value)
            return_code = int(event.exit_code)
        return ExecResult(stdout=stdout, stderr=stderr, return_code=return_code)

    @staticmethod
    def _format_transfer_error(
        operation: str,
        source: Path | str,
        result: ExecResult,
    ) -> str:
        return (
            f"Failed to {operation} {source}: "
            f"return_code={result.return_code}, "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )

    async def _exec_transfer_command(
        self,
        command: str,
        *,
        operation: str,
        source: Path | str,
    ) -> None:
        last_error: BaseException | None = None
        attempts = max(1, self._rli_policy.config.max_transfer_retries + 1)
        retry_backoff = float(getattr(self, "_rli_transfer_retry_backoff", 0.0))
        for attempt in range(1, attempts + 1):
            try:
                result = await self.exec(command, user="root")
                if result.return_code == 0:
                    return
                last_error = RuntimeError(
                    self._format_transfer_error(operation, source, result)
                )
            except Exception as exc:
                last_error = exc
            if attempt < attempts and retry_backoff > 0:
                failure_kind = self._rli_policy.classify_upload_failure(
                    return_code=(
                        result.return_code
                        if "result" in locals() and result is not None
                        else -1
                    ),
                    stdout=(
                        result.stdout
                        if "result" in locals() and result is not None
                        else ""
                    ),
                    stderr=(
                        result.stderr
                        if "result" in locals() and result is not None
                        else ""
                    ),
                )
                decision = self._rli_policy.should_retry_transfer(
                    phase="upload",
                    failure_kind=failure_kind,
                    attempt=attempt,
                )
                if not decision.retry:
                    break
                await asyncio.sleep(retry_backoff * attempt)
        assert last_error is not None
        raise last_error

    async def _exec_bootstrap_command(
        self,
        command: str,
        *,
        step_index: int,
        user: str | int | None = None,
        raise_on_failure: bool = True,
    ) -> ExecResult:
        attempts = max(1, self._rli_policy.config.max_bootstrap_step_retries + 1)
        last_result: ExecResult | None = None
        for attempt in range(1, attempts + 1):
            result = await self.exec(
                command,
                user=user,
            )
            if result.return_code == 0:
                return result
            last_result = result
            failure_kind = self._rli_policy.classify_step_failure(
                phase="bootstrap",
                exit_code=result.return_code,
                error=result.stderr or result.stdout,
                incomplete=result.return_code in (-1, 124),
                step_index=step_index,
            )
            decision = self._rli_policy.should_retry_step(
                phase="bootstrap",
                failure_kind=failure_kind,
                attempt=attempt,
            )
            if not decision.retry:
                break
        assert last_result is not None
        if not raise_on_failure:
            return last_result
        raise RuntimeError(
            f"Failed bootstrap command: return_code={last_result.return_code}, "
            f"stdout={last_result.stdout!r}, stderr={last_result.stderr!r}, "
            f"command={command!r}"
        )

    async def _exec_download_command(
        self,
        command: str,
        *,
        source_path: str,
    ) -> ExecResult:
        attempts = max(1, self._rli_policy.config.max_transfer_retries + 1)
        retry_backoff = float(getattr(self, "_rli_transfer_retry_backoff", 0.0))
        last_result: ExecResult | None = None
        for attempt in range(1, attempts + 1):
            result = await self.exec(command, user="root")
            if result.return_code == 0:
                return result
            last_result = result
            failure_kind = self._rli_policy.classify_download_failure(
                source_path=source_path,
                return_code=result.return_code,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            decision = self._rli_policy.should_retry_transfer(
                phase="download",
                failure_kind=failure_kind,
                attempt=attempt,
            )
            if not decision.retry:
                break
            if retry_backoff > 0:
                await asyncio.sleep(retry_backoff * attempt)
        assert last_result is not None
        raise RuntimeError(
            f"Failed to download {source_path}: "
            f"return_code={last_result.return_code}, "
            f"stdout={last_result.stdout!r}, stderr={last_result.stderr!r}"
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        payload = base64.b64encode(source.read_bytes()).decode("ascii")
        chunks = [
            payload[index : index + _UPLOAD_CHUNK_CHARS]
            for index in range(0, len(payload), _UPLOAD_CHUNK_CHARS)
        ] or [""]
        target = shlex.quote(target_path)
        parent = shlex.quote(str(Path(target_path).parent))
        remote_b64_path = f"/tmp/rli-upload-{uuid.uuid4().hex}.b64"
        remote_b64 = shlex.quote(remote_b64_path)
        try:
            await self._exec_transfer_command(
                f"mkdir -p {parent} && rm -f {remote_b64} {remote_b64}.*.part",
                operation="initialize upload for",
                source=source,
            )
            for part_index, chunk in enumerate(chunks):
                remote_part = shlex.quote(f"{remote_b64_path}.{part_index}.part")
                command = (
                    f"cat > {remote_part} <<'RLI_FILE_CHUNK_EOF'\n"
                    f"{chunk}\n"
                    "RLI_FILE_CHUNK_EOF\n"
                )
                await self._exec_transfer_command(
                    command,
                    operation="upload chunk for",
                    source=source,
                )
            finalize = (
                f": > {remote_b64} && "
                f"for i in $(seq 0 {len(chunks) - 1}); do "
                f"cat {remote_b64}.${{i}}.part >> {remote_b64}; "
                "done && "
                f"base64 -d {remote_b64} > {target} && "
                f"rm -f {remote_b64} {remote_b64}.*.part"
            )
            await self._exec_transfer_command(
                finalize,
                operation="finalize upload for",
                source=source,
            )
        except Exception:
            try:
                await self.exec(f"rm -f {remote_b64} {remote_b64}.*.part", user="root")
            except Exception:
                pass
            raise

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        remote_archive_path = f"/tmp/rli-upload-{uuid.uuid4().hex}.tar.gz"
        remote_archive = shlex.quote(remote_archive_path)
        target = shlex.quote(target_dir)
        try:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                with tarfile.open(archive.name, "w:gz") as tar:
                    for item in source.iterdir():
                        tar.add(item, arcname=item.name)
                await self.upload_file(archive.name, remote_archive_path)
            await self._exec_transfer_command(
                # Use tar extraction flags that avoid ownership restoration,
                # which fails in some RLI images.
                " && ".join(
                    [
                        f"mkdir -p {target}",
                        (
                            "tar --no-same-owner --no-same-permissions "
                            f"-xzf {remote_archive} -C {target}"
                        ),
                        f"rm -f {remote_archive}",
                    ]
                ),
                operation="extract uploaded directory",
                source=source,
            )
        except Exception:
            try:
                await self.exec(f"rm -f {remote_archive}", user="root")
            except Exception:
                pass
            raise

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        result = await self._exec_download_command(
            f"base64 -w0 {shlex.quote(source_path)}",
            source_path=source_path,
        )
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_bytes(base64.b64decode((result.stdout or "").strip()))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        remote_archive_path = f"{_DOWNLOAD_ROOT}/{uuid.uuid4().hex}.tar.gz"
        remote_archive = shlex.quote(remote_archive_path)
        archive_path: Path | None = None
        try:
            size_result = await self._exec_download_command(
                (
                    f"mkdir -p {shlex.quote(_DOWNLOAD_ROOT)} && "
                    f"tar -C {shlex.quote(source_dir)} -czf {remote_archive} . && "
                    f"stat -c %s {remote_archive}"
                ),
                source_path=source_dir,
            )
            archive_size = int((size_result.stdout or "").strip())

            with tempfile.NamedTemporaryFile(
                suffix=".tar.gz",
                delete=False,
            ) as archive:
                archive_path = Path(archive.name)
                chunk_count = (
                    archive_size + _DOWNLOAD_CHUNK_BYTES - 1
                ) // _DOWNLOAD_CHUNK_BYTES
                for chunk_index in range(chunk_count):
                    result = await self._exec_download_command(
                        (
                            "set -o pipefail && "
                            f"dd if={remote_archive} "
                            f"bs={_DOWNLOAD_CHUNK_BYTES} "
                            f"skip={chunk_index} count=1 status=none | "
                            "base64 -w0"
                        ),
                        source_path=source_dir,
                    )
                    archive.write(
                        base64.b64decode(
                            (result.stdout or "").strip(),
                            validate=True,
                        )
                    )
                downloaded_size = archive.tell()

            if downloaded_size != archive_size:
                raise RuntimeError(
                    f"Downloaded archive size mismatch for {source_dir}: "
                    f"expected={archive_size}, actual={downloaded_size}"
                )

            target = Path(target_dir)
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(target, filter=_host_safe_mode)
        finally:
            if archive_path is not None:
                try:
                    archive_path.unlink()
                except FileNotFoundError:
                    pass
            try:
                await self.exec(f"rm -f {remote_archive}", user="root")
            except Exception:
                pass

    async def attach(self) -> None:
        raise NotImplementedError("RLI environment does not support attach")
