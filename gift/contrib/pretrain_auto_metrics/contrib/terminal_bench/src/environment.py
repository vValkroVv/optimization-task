from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shlex
import tarfile
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from random import uniform
from typing import Any, Iterator

try:
    import tomllib
except ImportError:  # pragma: no cover - Python <3.11 fallback, unused in prod
    tomllib = None  # type: ignore[assignment]

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from httpx import HTTPStatusError, RequestError

from .gigarli_bootstrap import ensure_gigarli_import_path
from .rli_policy import TerminalBenchRLIPolicy, TerminalBenchRLIPolicyConfig

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

# Child of Harbor's handler-bearing logger ON PURPOSE. Harbor attaches its DEBUG
# FileHandler (-> job.log) and INFO StreamHandler to the logger named
# "harbor.utils.logger" (harbor/job.py:_init_logger uses the module-level
# `logger = setup_logger(__name__)` from harbor/utils/logger.py, whose __name__ IS
# "harbor.utils.logger"), and every Harbor submodule reaches that handler via
# `logger.getChild(__name__)`. Log records propagate UP the dotted hierarchy, so ONLY
# descendants of "harbor.utils.logger" inherit those handlers -- a "harbor.*" or
# standalone "terminal_bench.*" name does NOT (the handler is not on the "harbor" root).
# Naming ours "harbor.utils.logger.terminal_bench.rli_environment" is what makes every
# diagnostic below (``[RLI SESSION TTL]``, ``[RLI EXEC DEADLINE]``,
# ``[RLI CONTAINER RESET]``, ``[RLI SESSION UNAVAILABLE]``) actually land in job.log so
# the infra-parity fixes can be verified in production; before this they were silently
# dropped (root logger default level WARNING, no handler on our namespace).
logger = logging.getLogger("harbor.utils.logger.terminal_bench.rli_environment")


class RLISessionUnavailableError(RuntimeError):
    """Raised when RLI's own session-creation endpoint is unavailable.

    ``POST /v1/sessions`` returning a persistent 503/410/transport error means
    the RLI control plane itself is degraded; this always happens before the
    agent is ever constructed (``_create_session_with_retries`` is the very
    first thing ``start()`` does), so retrying the whole trial can never mask
    a genuine agent failure -- no agent turn or verifier run has occurred yet.
    Harbor's queue retries the whole trial when this type is listed in
    ``--retry-include`` (see cli.py).
    """


class RLIContainerResetError(RuntimeError):
    """Raised when an RLI container is silently recycled mid-task.

    The RLI sandbox pod can be torn down and respawned while a task is running
    (k8s OOM-eviction / node reclamation). From our side every command after
    that point is non-deterministic — the exec-proxy loses its connection
    (``return_code == -1``) or returns 127 because the tools/bootstrap have
    vanished, and the verifier ends up with no reward file. Such a trial is
    inconsistent and must be re-run from scratch on a fresh container. We
    confirm the reset by probing our bootstrap artifacts before raising, so a
    genuine model failure (reward 0 with a healthy container) is never retried.
    Harbor's queue retries the whole trial when this exception type is listed in
    ``--retry-include`` (see cli.py).
    """


class RLIExecProxyBlipError(RuntimeError):
    """Raised when one exec() lost its exec-proxy connection (rc == -1) but the
    alive-marker probe confirms the container is HEALTHY.

    Retrying the failed command in-place is unsafe (a tmux ``send-keys`` may
    have been delivered before the proxy died — a retry would double-type),
    but retrying the WHOLE TRIAL is: the fresh session gets a fresh container
    and the verifier re-runs, so no model failure can hide behind it. Without
    this, Harbor surfaces the blip as a non-retryable RuntimeError and the
    trial becomes a permanent infra loss (run1–run3 forensics: mailman,
    −1.12 pt). Harbor retries the trial when this type is in
    ``--retry-include`` (see cli.py).
    """


_TMUX_GONE_PATTERNS = (
    "tmux: command not found",
    "error connecting to /tmp/tmux",
    "no server running on /tmp/tmux",
)


_DEFAULT_UPLOAD_CHUNK_KIB = 256
_RLI_MAX_STEP_DEADLINE_MS = 1_500_000
_LONG_COMMAND_STEP_DEADLINE_MS = 60_000
_LONG_COMMAND_POLL_INTERVAL_SEC = 30.0


class RLILongRunningCommandLostError(RuntimeError):
    pass


_APPLY_PATCH_SHIM_PATH = Path(__file__).resolve().parent / "apply_patch_shim.py"
_REMOTE_APPLY_PATCH_PATH = "/usr/local/bin/apply_patch"
_REMOTE_WRITABLE_ROOT = "/tmp/terminal-bench"
_REMOTE_PATH_REWRITES: tuple[tuple[str, str], ...] = (
    ("/logs/artifacts", f"{_REMOTE_WRITABLE_ROOT}/logs/artifacts"),
    ("/logs/agent", f"{_REMOTE_WRITABLE_ROOT}/logs/agent"),
    # Safe with the stricter regex in _rewrite_shell_paths: only rewrites
    # token-leading "/logs/..." paths, so already-rewritten targets do not
    # match a second time.
    ("/logs", f"{_REMOTE_WRITABLE_ROOT}/logs"),
    ("/tests", f"{_REMOTE_WRITABLE_ROOT}/tests"),
)
_REMOTE_PRECREATE_DIRS: tuple[str, ...] = (
    f"{_REMOTE_WRITABLE_ROOT}/logs",
    f"{_REMOTE_WRITABLE_ROOT}/logs/agent",
    f"{_REMOTE_WRITABLE_ROOT}/logs/verifier",
    # Rewrite target for "/logs/artifacts" in _REMOTE_PATH_REWRITES below --
    # must be pre-created too, or a first-touch "> /logs/artifacts/x" redirect
    # fails with "No such file or directory" (bash `>` never creates parents).
    f"{_REMOTE_WRITABLE_ROOT}/logs/artifacts",
    f"{_REMOTE_WRITABLE_ROOT}/tests",
)
_REMOTE_COMPAT_SYMLINKS: tuple[tuple[str, str], ...] = (
    ("/logs", f"{_REMOTE_WRITABLE_ROOT}/logs"),
    ("/tests", f"{_REMOTE_WRITABLE_ROOT}/tests"),
)
# Container-recycle canary. Written into the RLI container at the end of
# start() and probed later to detect that the container's filesystem was
# recycled mid-task. A healthy container never removes this marker, so its
# absence is an unambiguous pod OOM-eviction/recycle signal.
_ALIVE_MARKER_PATH = f"{_REMOTE_WRITABLE_ROOT}/.rli_session_alive"
_ALIVE_PROBE_COMMAND = f"test -f {_ALIVE_MARKER_PATH} -a -d {_REMOTE_WRITABLE_ROOT}"
_ALIVE_MARKER_COMMAND = (
    f"mkdir -p {_REMOTE_WRITABLE_ROOT} && "
    f"printf 'alive ts=%s\\n' \"$(date -u +%s)\" > {_ALIVE_MARKER_PATH}"
)
_APT_FORCE_IPV4_COMMAND = """cat >/etc/apt/apt.conf.d/99force-ipv4 <<'EOF'
Acquire::ForceIPv4 "true";
EOF"""
# Retry + stall-abort for curl/wget so the verifier's one-shot uv download
# survives transient RLI egress stalls (curl defaults have no retry/stall
# detection). Keeps the canonical verifier flow; only makes its curl resilient.
_CURL_ROBUSTNESS_COMMAND = """mkdir -p /etc/curl /tmp/terminal-bench/logs || true
cat > /root/.curlrc <<'EOF'
connect-timeout = 10
retry = 5
retry-all-errors
speed-time = 30
speed-limit = 1024
max-time = 120
EOF
cp /root/.curlrc /etc/curl/curlrc 2>/dev/null || true
{ [ -f /etc/wgetrc ] && { grep -q 'tries' /etc/wgetrc || printf 'tries = 5\\ntimeout = 30\\n' >> /etc/wgetrc; }; } 2>/dev/null || true
{ echo "=== net-diag $(date -u +%H:%M:%S) ==="; getent ahosts github.com | head -3; getent ahosts objects.githubusercontent.com | head -3; } > /tmp/terminal-bench/logs/net-diag.log 2>&1 || true"""
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
# Some RLI task images ship with dpkg half-configured (ca-certificates and
# perl packages stuck in the "installed but unconfigured" state because
# /etc/ssl/certs is read-only in the sandbox). The first `apt-get install`
# any agent or verifier runs then fails with "E: Sub-process /usr/bin/dpkg
# returned an error code (1)", and models burn many episodes wrestling
# `dpkg --configure -a` before doing real work (adaptive-rejection-sampler
# spent 11/18 episodes on this). Pre-settle dpkg once at session start:
# mark the wedged postinst scripts as succeeded (they only fail on the
# read-only symlink step; the certs themselves are already present).
_APT_SETTLE_DPKG_COMMAND = """set +e
command -v dpkg >/dev/null 2>&1 || exit 0
if ! dpkg --audit 2>/dev/null | grep -q .; then exit 0; fi
chmod -R u+w /etc/ssl/certs 2>/dev/null || true
dpkg --configure -a >/tmp/terminal-bench/logs/dpkg-settle.log 2>&1
if dpkg --audit 2>/dev/null | grep -q .; then
  for p in $(dpkg --audit 2>/dev/null | awk '/^ /{print $1}'); do
    printf '#!/bin/sh\\nexit 0\\n' > "/var/lib/dpkg/info/${p}.postinst" 2>/dev/null || true
  done
  dpkg --configure -a >>/tmp/terminal-bench/logs/dpkg-settle.log 2>&1 || true
fi
true"""


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
            f"rli_config_json must decode to a JSON object, got {type(loaded).__name__}"
        )
    return dict(loaded)


def _infer_dockerfile_workdir(environment_dir: Path) -> str | None:
    """Best-effort fallback for tasks whose Harbor env config has no workdir.

    RLI's ``POST /v1/sessions/step`` currently starts commands from ``/`` even
    when the task image declares ``WORKDIR /app``. Harbor does have a per-task
    workdir hook, but for our current terminal-bench runs it is often unset.
    As a harness-side workaround, read the task's Dockerfile and reuse its
    ``WORKDIR`` for every wrapped command.
    """

    dockerfile_path = environment_dir / "Dockerfile"
    try:
        dockerfile_text = dockerfile_path.read_text(encoding="utf-8")
    except OSError:
        return None

    workdir = None
    for raw_line in dockerfile_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        directive = line.split("#", 1)[0].strip()
        if re.match(r"(?i)^FROM\s+", directive):
            workdir = None
            continue
        match = re.match(r"(?i)^WORKDIR\s+(.+)$", directive)
        if match:
            value = match.group(1).strip()
            if value:
                workdir = value
    return workdir


def _read_task_toml_phase_timeouts_sec(environment_dir: Path) -> dict[str, float]:
    """Read ``{"agent": ..., "verifier": ...}`` ``timeout_sec`` straight from
    the task's own ``task.toml`` (sibling of ``environment_dir``, matching the
    existing ``_infer_dockerfile_workdir``/``_infer_task_id`` pattern of
    deriving per-task facts from the on-disk task directory).

    Returns ``{}`` (never ``None``) when ``tomllib`` is unavailable, the file
    is missing, or it fails to parse -- callers treat an empty dict the same
    as "nothing declared" and fall back to static config.
    """
    if tomllib is None:
        return {}
    task_toml_path = environment_dir.parent / "task.toml"
    try:
        text = task_toml_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = tomllib.loads(text)
    except Exception:
        return {}
    timeouts: dict[str, float] = {}
    for section in ("agent", "verifier"):
        section_data = data.get(section)
        if not isinstance(section_data, dict):
            continue
        value = section_data.get("timeout_sec")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            timeouts[section] = float(value)
    return timeouts


def _infer_task_declared_timeout_sec(environment_dir: Path) -> float | None:
    """Read the task's own ``[agent].timeout_sec`` / ``[verifier].timeout_sec``.

    Harbor's own ``Verifier.verify()`` calls
    ``environment.exec(command=test.sh, env=env)`` with NO ``timeout_sec``
    (harbor/verifier/verifier.py), so our RLI-side ``RequestLimits.deadline_ms``
    fallback (``exec()`` below) is the ONLY thing bounding how long the
    verifier's build/test process may run inside the RLI session. Harbor's own
    outer ``asyncio.wait_for(verifier.verify(), timeout=task.verifier.timeout_sec)``
    already allows up to the task's own declared budget (up to 12000s across
    the current terminal-bench-2 dataset: build-pov-ray=12000s,
    sam-cell-seg=7200s, fix-ocaml-gc=3600s, compile-compcert=2400s, ...), so a
    single static ~600s RLI-side deadline silently truncates the verifier
    process mid-build (observed as ``make: ... Hangup`` / a bare SIGHUP) long
    before Harbor's own timeout would ever fire -- a pure harness artifact,
    not a genuine timeout, and not fixable by retrying (the same static cap
    would truncate the retry identically).

    Return the larger of ``[agent].timeout_sec`` / ``[verifier].timeout_sec``
    (only one of the two phases is ever executing at a time within a single
    ``exec()`` call) so our RLI-side deadline can never be tighter than what
    Harbor itself would allow for whichever phase is currently running.
    """
    timeouts = _read_task_toml_phase_timeouts_sec(environment_dir)
    if not timeouts:
        return None
    return max(timeouts.values())


def _rewrite_remote_path(path: str) -> str:
    for source, target in _REMOTE_PATH_REWRITES:
        if path == source or path.startswith(f"{source}/"):
            return f"{target}{path[len(source) :]}"
    return path


def _rewrite_shell_paths(command: str) -> str:
    rewritten = command
    for source, target in _REMOTE_PATH_REWRITES:
        rewritten = re.sub(
            rf"(?:(?<=^)|(?<=[\s'\"=:(])){re.escape(source)}"
            rf"(?=($|/|\s|['\";|&()<>]))",
            target,
            rewritten,
        )
    return rewritten


class RLIEnvironment(BaseEnvironment):
    _rli_enable_container_recycle_detection = False
    _rli_session_confirmed_alive = False
    _rli_last_canary_ts = 0.0

    # Per-process, per-task attempt counter. Harbor's queue retries a failed
    # trial by constructing a brand-new RLIEnvironment and calling start() again
    # inside the same process, so this lets us report "attempt #N" across those
    # whole-trial retries (each start() bumps the count for its task_id).
    _CONTAINER_RESET_ATTEMPTS: dict[str, int] = {}

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
        self._rli_enable_container_recycle_detection = bool(
            config.get("enable_container_recycle_detection", False)
        )
        # Becomes True once start() finishes bootstrap — proof the container was
        # healthy and our alive-marker exists, so a later marker-disappearance is
        # a real mid-task recycle rather than "session never came up".
        self._rli_session_confirmed_alive = False
        self._rli_last_canary_ts = 0.0
        self._rli_task_id = ""
        self._rli_attempt = 0
        self._rli_task_id_override = config.get("task_id") or config.get("task_name")
        self._rli_env_prefix = config.get("env_prefix", "terminal-bench")
        self._rli_env_version = config.get("env_version", "1.0.0")
        self._rli_ttl_ms = config.get("ttl_ms")
        self._rli_seed = config.get("seed")
        self._rli_use_pip_proxy = config.get(
            "use_pip_proxy",
            config.get("use-pip-proxy"),
        )
        self._rli_wait_for_ready = bool(config.get("wait_for_ready", True))
        self._rli_command_timeout_ms = config.get("command_timeout_ms")
        self._rli_verifier_timeout_ms: int | None = None
        self._rli_transfer_retries = int(config.get("transfer_retries", 2))
        self._rli_transfer_retry_backoff = float(
            config.get("transfer_retry_backoff", 2.0)
        )
        upload_chunk_kib = config.get("upload_chunk_kib")
        if upload_chunk_kib is None:
            upload_chunk_kib = _DEFAULT_UPLOAD_CHUNK_KIB
        self._rli_upload_chunk_chars = int(upload_chunk_kib) * 1024
        if self._rli_upload_chunk_chars <= 0:
            raise ValueError("upload_chunk_kib must be greater than zero")
        self._rli_policy = TerminalBenchRLIPolicy(
            TerminalBenchRLIPolicyConfig(
                warmup_mode=str(config.get("warmup_mode", "poll_only")),
                enable_probe_warmup=bool(config.get("enable_probe_warmup", False)),
                max_bootstrap_step_retries=int(config.get("bootstrap_step_retries", 2)),
                max_transfer_retries=int(config.get("transfer_retries", 2)),
                max_verifier_step_retries=int(config.get("verifier_step_retries", 2)),
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
        self._rli_bootstrap_apt = bool(config.get("bootstrap_apt", True))
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
            or _infer_dockerfile_workdir(self.environment_dir)
        )
        # Fallback deadline_ms for exec() calls made without an explicit
        # timeout_sec (notably the verifier's own test.sh run -- see
        # _infer_task_declared_timeout_sec docstring). Prefer the task's own
        # declared agent/verifier timeout_sec (with a safety margin so our
        # RLI-side deadline is never tighter than Harbor's own outer
        # asyncio.wait_for), falling back to the static command_timeout_ms
        # config value when task.toml can't be read/parsed.
        declared_timeout_sec = _infer_task_declared_timeout_sec(self.environment_dir)
        self._rli_task_command_timeout_ms: int | None = None
        if declared_timeout_sec is not None:
            margin_sec = float(config.get("command_timeout_margin_sec", 120.0))
            self._rli_task_command_timeout_ms = int(
                (declared_timeout_sec + margin_sec) * 1000
            )
        logger.info(
            "[RLI EXEC DEADLINE] task=%s resolved fallback deadline_ms=%s "
            "(source=%s, task_toml_declared_timeout_sec=%s, "
            "static_command_timeout_ms=%s) -- applies to any exec() call "
            "issued without an explicit timeout_sec, in particular the "
            "verifier's test.sh run (harbor/verifier/verifier.py never "
            "passes timeout_sec).",
            self._infer_task_id(),
            self._rli_task_command_timeout_ms or self._rli_command_timeout_ms,
            "task.toml"
            if self._rli_task_command_timeout_ms is not None
            else "static_config",
            declared_timeout_sec,
            self._rli_command_timeout_ms,
        )

    def _resolve_session_ttl_ms(self) -> int:
        agent_timeout = self._rli_config.get("agent_timeout_sec")
        if agent_timeout is None:
            agent_timeout = _read_task_toml_phase_timeouts_sec(
                self.environment_dir
            )["agent"]
        ttl_ms = int((agent_timeout + 600) * 1000)
        if self._rli_ttl_ms is not None:
            ttl_ms = min(int(self._rli_ttl_ms), ttl_ms)
        return ttl_ms

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True, mounted=False)

    @contextmanager
    def use_verifier_command_timeout(self, timeout_sec: float) -> Iterator[None]:
        previous = self._rli_verifier_timeout_ms
        try:
            self._rli_verifier_timeout_ms = int(timeout_sec * 1000)
            yield
        finally:
            self._rli_verifier_timeout_ms = previous

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
        return f"{self._rli_env_prefix}-{task_id}"

    async def start(self, force_build: bool) -> None:
        _ = force_build
        task_id = self._infer_task_id()
        self._rli_task_id = task_id
        # Whole-trial retry attempt index for this task (1-based). Harbor retries
        # a failed trial by building a new RLIEnvironment + start() in-process,
        # so this counts across container resets, not just bootstrap retries.
        self._rli_attempt = RLIEnvironment._CONTAINER_RESET_ATTEMPTS.get(task_id, 0) + 1
        RLIEnvironment._CONTAINER_RESET_ATTEMPTS[task_id] = self._rli_attempt
        if self._rli_attempt > 1:
            logger.info(
                "[RLI] task=%s trial attempt #%d START (previous attempt(s) "
                "ended in RLIContainerResetError — container was recycled "
                "mid-task and the trial is being re-run on a fresh container)",
                task_id,
                self._rli_attempt,
            )
        allow_internet_access = self._rli_config.get("allow_internet_access", True)
        request = CreateSessionRequest(
            allow_internet_access=allow_internet_access,
            env=SessionEnv(
                name=self._build_env_name(task_id),
                version=str(self._rli_env_version),
                task_id=task_id,
            ),
            ttl_ms=self._resolve_session_ttl_ms(),
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
            if task_id == "mailman":
                await self._prepare_mailman_smtp_runtime()
            if task_id in {
                "git-multibranch",
                "configure-git-webserver",
            }:
                await self._prepare_local_ssh_runtime()
            if task_id == "git-multibranch":
                await self._prepare_git_multibranch_tls()
            await self._bootstrap_remote_apt()
            await self._setup_curl_robustness()
            await self._mark_session_alive()
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
            result = await self.exec(command, user="root", rewrite_paths=False)
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
        # This call happens before the agent is constructed (the very first
        # thing start() does) -- no agent turn or verifier run has occurred,
        # so converting a persistent RLI /v1/sessions outage into our own
        # narrowly-scoped, retryable exception type can never mask a genuine
        # agent/model failure. Only RLI-transport exception types raised from
        # THIS call are translated; unrelated RuntimeErrors elsewhere in the
        # codebase are untouched.
        if isinstance(last_error, (RLIClientError, HTTPStatusError, RequestError)):
            logger.info(
                "[RLI SESSION UNAVAILABLE] task=%s create_session exhausted "
                "%d attempt(s) against RLI's own control plane before the "
                "agent was constructed (error_type=%s: %s) -- raising "
                "RLISessionUnavailableError so Harbor retries the whole "
                "trial on a fresh attempt (retry-include=%s).",
                self._infer_task_id(),
                attempts,
                type(last_error).__name__,
                last_error,
                "RLISessionUnavailableError",
            )
            self._dump_rli_problem(
                "session_create_failed",
                f"create_session exhausted {attempts} attempt(s): "
                f"{type(last_error).__name__}: {last_error}; trial will be "
                f"retried (RLISessionUnavailableError is in the retry "
                f"allowlist)",
            )
            raise RLISessionUnavailableError(
                f"RLI session creation failed after {attempts} attempt(s): "
                f"{type(last_error).__name__}: {last_error}"
            ) from last_error
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
                rewrite_paths=False,
            )
            await self._exec_bootstrap_command(
                f"printf %s {shlex.quote(encoded)} | base64 -d > {target}",
                step_index=1,
                user="root",
                rewrite_paths=False,
            )
            await self._exec_bootstrap_command(
                f"chmod 0755 {target}",
                step_index=2,
                user="root",
                rewrite_paths=False,
            )
        except Exception:
            return

    async def _prepare_local_ssh_runtime(self) -> None:
        await self._exec_bootstrap_command(
            "mkdir -p /etc/ssh/ssh_config.d; "
            "printf 'Host localhost\\n    Port 10022\\n' "
            ">/etc/ssh/ssh_config.d/99-terminal-bench-rli.conf",
            step_index=0,
            user="root",
            rewrite_paths=False,
        )

    async def _prepare_mailman_smtp_runtime(self) -> None:
        await self._exec_bootstrap_command(
            "if [ -f /app/eval.py ]; then "
            "sed -i "
            '\'s/SMTP("localhost", 25)/SMTP("localhost", 2525)/g\' '
            "/app/eval.py; fi; "
            "if [ -f /etc/postfix/master.cf ]; then "
            "sed -i -E "
            "'s/^smtp([[:space:]]+inet[[:space:]])/2525\\1/' "
            "/etc/postfix/master.cf; "
            "sed -i -E "
            "'/^[[:space:]]*#/! "
            "s/^(([^[:space:]]+[[:space:]]+){4})"
            "y([[:space:]])/\\1n\\3/' "
            "/etc/postfix/master.cf; fi",
            step_index=0,
            user="root",
            rewrite_paths=False,
        )

    async def _prepare_git_multibranch_tls(self) -> None:
        await self._exec_bootstrap_command(
            "mkdir -p /tmp/terminal-bench; "
            "if [ ! -s /etc/ssl/certs/nginx-selfsigned.crt ]; then "
            "openssl req -x509 -new "
            "-key /etc/ssl/private/nginx-selfsigned.key "
            "-out /tmp/terminal-bench/nginx-selfsigned.crt "
            "-days 365 -subj /CN=localhost; "
            "sed -i "
            "'s|/etc/ssl/certs/nginx-selfsigned.crt|"
            "/tmp/terminal-bench/nginx-selfsigned.crt|g' "
            "/etc/nginx/sites-available/default; "
            "fi",
            step_index=0,
            user="root",
            rewrite_paths=False,
        )

    async def _prepare_remote_layout(self) -> None:
        """Create writable directories expected by Harbor/verifier code.

        Some RLI images run the task under ``/app`` with a read-only or absent
        ``/workspace``. We rewrite Harbor's hard-coded ``/tests`` and
        ``/logs/...`` paths into ``/tmp/terminal-bench/...``, but shell
        redirections such as ``> /logs/verifier/test-stdout.txt`` still fail
        unless the parent directory exists before the command starts.

        The local verifier rewrites paths inside the uploaded test script to
        the same writable layout.
        """
        try:
            for step_index, path in enumerate(_REMOTE_PRECREATE_DIRS):
                await self._exec_bootstrap_command(
                    f"mkdir -p {shlex.quote(path)}",
                    step_index=step_index,
                    user="root",
                    rewrite_paths=False,
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
            _APT_SETTLE_DPKG_COMMAND,
        ]

        # Keep only the cheap, global-safe part here. Running apt index warmup
        # for every task made ordinary runs much worse and could strand a
        # session inside one long bootstrap step.
        try:
            has_apt = await self._exec_bootstrap_command(
                "command -v apt-get >/dev/null 2>&1",
                step_index=0,
                user="root",
                rewrite_paths=False,
                raise_on_failure=False,
            )
            if has_apt.return_code != 0:
                return
            for step_index, command in enumerate(rewrite_commands, start=1):
                await self._exec_bootstrap_command(
                    command,
                    step_index=step_index,
                    user="root",
                    rewrite_paths=False,
                )
        except Exception:
            return

    async def _setup_curl_robustness(self) -> None:
        """Write curlrc/wgetrc retry+stall settings so the verifier's canonical
        ``uv`` download survives transient RLI egress stalls. Hard 60s deadline
        so a hung step can't eat the environment-start budget; best-effort.
        """
        try:
            await asyncio.wait_for(
                self._exec_bootstrap_command(
                    _CURL_ROBUSTNESS_COMMAND,
                    step_index=0,
                    user="root",
                    rewrite_paths=False,
                    raise_on_failure=False,
                ),
                timeout=60.0,
            )
        except Exception:
            return

    async def _mark_session_alive(self) -> None:
        """Write the alive-marker and flip ``_rli_session_confirmed_alive``.

        Called as the last step of start(), after all bootstrap artifacts are
        in place. Best-effort: if the marker write fails we still proceed —
        detection then simply never fires (safe under-fail, never a false
        retry).
        """
        if not self._rli_enable_container_recycle_detection:
            return
        try:
            result = await self._exec_bootstrap_command(
                _ALIVE_MARKER_COMMAND,
                step_index=0,
                user="root",
                rewrite_paths=False,
                raise_on_failure=False,
            )
        except Exception:
            return
        if result.return_code != 0:
            return
        self._rli_session_confirmed_alive = True
        self._rli_last_canary_ts = time.monotonic()

    async def _probe_container_alive(self) -> bool | None:
        """Probe the RLI container's bootstrap artifacts.

        Returns True if the alive-marker and writable root are present
        (container healthy, NOT recycled); False if either is gone (filesystem
        was recycled mid-task); None if the probe could not be executed at all
        (transport down after retries — treat as recycled).

        Transport failures (exception or exec-proxy return_code == -1) are
        RETRIED with a short backoff before concluding anything: a
        millisecond-long conn-lost blip on the probe itself must not be
        classified as "filesystem recycled" (observed 2026-07-22: an ~8-10s
        exec-proxy outage failed both a live trial's step AND its probe within
        1 ms, misclassifying a healthy pod). Only a CLEAN non-zero exit (the
        test command ran, marker genuinely absent) returns False immediately.
        """
        delays = (0.0, 5.0, 10.0)
        for attempt, delay in enumerate(delays, start=1):
            if delay:
                await asyncio.sleep(delay)
            try:
                result = await self.exec(
                    _ALIVE_PROBE_COMMAND,
                    user="root",
                    rewrite_paths=False,
                    is_probe=True,
                )
            except Exception:
                continue
            if result.return_code == -1:
                continue
            return result.return_code == 0
        return None

    _CANARY_PERIOD_SEC = 600.0

    _RLI_PROBLEM_LOG_DIR_ENV = "RLI_PROBLEM_LOG_DIR"

    def _rli_trace_events(self, session_id: str | None) -> list[str]:
        """Verbatim wire-trace lines for one session from rli_trace.jsonl.

        Falls back to the trailing create_session events when the problem
        happened before any session existed. Each line is truncated so bulk
        payloads (base64 file uploads) do not bloat the report.
        """
        path = self._rli_trace_path
        if path is None or not path.exists():
            return ["(no rli_trace.jsonl wire log configured for this run)"]
        needle = session_id if session_id else '"stage": "create_session"'
        matched: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if needle in line:
                    line = line.rstrip("\n")
                    if len(line) > 4000:
                        line = line[:4000] + f"...[truncated {len(line) - 4000} chars]"
                    matched.append(line)
        if session_id is None:
            matched = matched[-20:]
        if len(matched) > 800:
            omitted = len(matched) - 800
            matched = (
                matched[:200] + [f"...[{omitted} events omitted]..."] + matched[-600:]
            )
        return matched or ["(no wire-trace events matched this session)"]

    def _dump_rli_problem(self, kind: str, detail: str) -> None:
        """Write one self-contained, API-level problem report per occurrence.

        Active only when the RLI_PROBLEM_LOG_DIR env var names a directory.
        The report replays the exact RLI API calls of the affected session
        (timestamp, stage, request, response — verbatim from the wire trace),
        so the RLI team can understand and reproduce the issue WITHOUT
        reading this codebase. Best-effort by contract: never raises, never
        alters measurement behaviour.
        """
        try:
            out_dir_value = os.environ.get(self._RLI_PROBLEM_LOG_DIR_ENV, "").strip()
            if not out_dir_value:
                return
            out_dir = Path(out_dir_value)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            session_id = self._rli_session_id
            task_id = str(self._rli_task_id or "unknown-task")
            safe_task = (
                re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("_")[:80]
                or "unknown-task"
            )
            suffix = (session_id or "no-session")[:8]
            out_path = out_dir / f"{ts}__{safe_task}__{kind}__{suffix}.log"
            header = [
                "RLI PROBLEM REPORT (auto-generated, API-level, self-contained)",
                "generated_at_utc : "
                + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                f"problem_kind     : {kind}",
                f"problem_detail   : {detail}",
                "rli_api_base     : "
                + str(getattr(self._rli_client, "base_url", "unknown")),
                f"task_id          : {task_id}",
                f"session_id       : {session_id or '(never created)'}",
                f"trial_attempt    : {self._rli_attempt}",
                "",
                "Below: the exact RLI API calls of this session in",
                "chronological order, verbatim from the wire trace (one JSON",
                "per line: ts, stage, request, response). Bulk payloads are",
                "truncated.",
                "",
            ]
            events = self._rli_trace_events(session_id)
            out_path.write_text("\n".join(header + events) + "\n", encoding="utf-8")
            logger.info(
                "[RLI PROBLEM DUMP] task=%s kind=%s -> %s",
                task_id,
                kind,
                out_path,
            )
        except Exception:
            logger.warning("[RLI PROBLEM DUMP] failed to write report", exc_info=True)

    async def _raise_if_container_reset(
        self, *, trigger: str, healthy_blip: bool = True
    ) -> None:
        """Probe the alive-marker and raise RLIContainerResetError if the RLI
        container was recycled mid-task.

        Call this only after an operation failed in a way consistent with a
        mid-task recycle: a persistent exec-proxy break (``return_code == -1``),
        a tmux-gone signature, or a verifier download whose source dir is gone.
        Because such failures are non-deterministic (the command may or may not
        have actually run, and the agent's on-disk work may have vanished), the
        trial is unsafe to keep — but a real model failure on a HEALTHY
        container must NOT be retried. The marker resolves that: it is written
        during start() and a healthy container never removes it, so a missing
        marker is an unambiguous recycle signal. No-op when the session was
        never confirmed alive (e.g. failed during bootstrap).

        ``healthy_blip`` controls the HEALTHY-probe outcome. True (the rc=-1
        transport paths): the failure was the transport's, not the model's —
        raise RLIExecProxyBlipError so Harbor retries the whole trial (the
        command itself is never re-sent, so tmux keystrokes cannot be
        double-typed; the verifier re-runs on the retry, so no model failure
        is masked). False (tmux-gone / missing-reward / canary paths): a
        healthy container means the observed state is the task's own doing —
        return silently and let the caller's honest error stand.
        """
        if (
            not self._rli_enable_container_recycle_detection
            or not self._rli_session_confirmed_alive
        ):
            return
        alive = await self._probe_container_alive()
        if alive is True:
            if not healthy_blip:
                logger.info(
                    "[RLI HEALTHY-GENUINE] task=%s session=%s attempt=%d — "
                    "probe says HEALTHY; honest failure stands (trigger=%s)",
                    self._rli_task_id,
                    self._rli_session_id,
                    self._rli_attempt,
                    trigger,
                )
                return
            logger.info(
                "[RLI EXEC-PROXY BLIP] task=%s session=%s attempt=%d — an "
                "exec() call lost its exec-proxy connection (rc=-1) but the "
                "container probe reports HEALTHY; raising "
                "RLIExecProxyBlipError so Harbor retries the WHOLE trial on "
                "a fresh session (trigger=%s). The failed command itself is "
                "never re-sent — no double-typing risk.",
                self._rli_task_id,
                self._rli_session_id,
                self._rli_attempt,
                trigger,
            )
            self._dump_rli_problem("exec_proxy_blip_retried", trigger)
            raise RLIExecProxyBlipError(
                f"RLI exec-proxy blip on a healthy container "
                f"(task={self._rli_task_id}, session={self._rli_session_id}, "
                f"attempt={self._rli_attempt}, trigger={trigger}). The lost "
                f"step is non-deterministic; the trial is retried on a fresh "
                f"session."
            )
        self._rli_session_confirmed_alive = False
        probe_desc = (
            "bootstrap artifacts gone (marker/fs recycled)"
            if alive is False
            else "probe could not execute (exec-proxy connection lost)"
        )
        logger.info(
            "[RLI CONTAINER RESET] task=%s session=%s attempt=%d — container "
            "recycled mid-task (probe=%s). Harbor will close this session and "
            "re-run the whole trial on a fresh container "
            "(retry-include=RLIContainerResetError).",
            self._rli_task_id,
            self._rli_session_id,
            self._rli_attempt,
            probe_desc,
        )
        logger.info("[RLI CONTAINER RESET] trigger=%s", trigger)
        self._dump_rli_problem(
            "container_recycled_midtask",
            f"probe={probe_desc}; trigger={trigger}; trial will be retried "
            f"(RLIContainerResetError is in the retry allowlist)",
        )
        raise RLIContainerResetError(
            f"RLI container recycled mid-task (task={self._rli_task_id}, "
            f"session={self._rli_session_id}, attempt={self._rli_attempt}, "
            f"probe={probe_desc}, trigger={trigger}). Bootstrap artifacts are "
            f"gone and the last operation was non-deterministic, so we cannot "
            f"trust the trial state. Trial must be re-run from scratch."
        )

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
            await self._aclose_client_safely()
            return
        try:
            await self._rli_client.close_session(
                CloseSessionRequest(session_id=self._rli_session_id)
            )
        finally:
            self._rli_session_id = None
            await self._aclose_client_safely()

    async def _aclose_client_safely(self) -> None:
        aclose = getattr(self._rli_client, "aclose", None)
        if aclose is None:
            return
        try:
            await aclose()
        except Exception:
            pass

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
        rewrite_paths: bool,
    ) -> str:
        merged_env = self._merge_env(env) or {}
        # Keep Harbor's hard-coded /tests and /logs paths redirected into our
        # writable /tmp area by default. A few bootstrap commands, such as the
        # root-level compatibility symlinks created in _prepare_remote_layout,
        # must bypass this rewrite so they can reference literal "/tests" and
        # "/logs" on the remote host.
        effective = _rewrite_shell_paths(command) if rewrite_paths else command
        workdir = cwd or self.task_env_config.workdir or self._rli_default_workdir
        if workdir:
            effective = (
                f"cd {shlex.quote(_rewrite_remote_path(workdir))} && {effective}"
            )
        if merged_env:
            prefix = " ".join(
                f"{key}={shlex.quote(str(value))}" for key, value in merged_env.items()
            )
            effective = f"{prefix} {effective}"
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
        rewrite_paths: bool = True,
        is_probe: bool = False,
    ) -> ExecResult:
        if (
            self._rli_enable_container_recycle_detection
            and not is_probe
            and self._rli_session_confirmed_alive
            and time.monotonic() - self._rli_last_canary_ts > self._CANARY_PERIOD_SEC
        ):
            self._rli_last_canary_ts = time.monotonic()
            alive = await self._probe_container_alive()
            if alive is not True:
                await self._raise_if_container_reset(
                    trigger=f"periodic canary before exec: {command[:120]!r}",
                    healthy_blip=False,
                )
        is_verifier_command = (
            self._rli_verifier_timeout_ms is not None
            and "/tests/" in command
            and "/logs/verifier/" in command
        )
        is_oracle_command = "/solution/" in command and "/logs/agent/" in command
        if timeout_sec is not None:
            deadline_ms = int(timeout_sec * 1000)
        elif is_verifier_command:
            deadline_ms = self._rli_verifier_timeout_ms
        else:
            deadline_ms = self._rli_task_command_timeout_ms
            if deadline_ms is None:
                deadline_ms = self._rli_command_timeout_ms
        wrapped_command = self._wrap_command(
            command,
            cwd=cwd,
            env=env,
            user=user if user is not None else self.default_user,
            rewrite_paths=rewrite_paths,
        )
        if is_verifier_command:
            return await self._exec_long_running(wrapped_command, deadline_ms)

        if (
            is_oracle_command
            and deadline_ms is not None
            and deadline_ms > _RLI_MAX_STEP_DEADLINE_MS
        ):
            return await self._exec_long_running(wrapped_command, deadline_ms)

        if deadline_ms is not None:
            deadline_ms = min(deadline_ms, _RLI_MAX_STEP_DEADLINE_MS)
        result = await self._exec_step(wrapped_command, deadline_ms)
        if (
            self._rli_enable_container_recycle_detection
            and not is_probe
            and result.return_code == -1
        ):
            await self._raise_if_container_reset(
                trigger=f"exec return_code=-1, command={wrapped_command[:200]!r}"
            )
        if (
            self._rli_enable_container_recycle_detection
            and not is_probe
            and "tmux" in wrapped_command
            and result.return_code in (1, 127)
            and any(
                pattern in ((result.stdout or "") + (result.stderr or ""))
                for pattern in _TMUX_GONE_PATTERNS
            )
        ):
            await self._raise_if_container_reset(
                trigger=(
                    f"tmux-gone signature rc={result.return_code}, "
                    f"command={wrapped_command[:200]!r}"
                ),
                healthy_blip=False,
            )
        return result

    async def _exec_step(
        self,
        command: str,
        deadline_ms: int | None,
    ) -> ExecResult:
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
        return self._step_result_to_exec_result(result)

    async def _exec_long_running(
        self,
        command: str,
        deadline_ms: int,
    ) -> ExecResult:
        job_id = uuid.uuid4().hex
        job_dir = f"{_REMOTE_WRITABLE_ROOT}/long-running/{job_id}"
        stdout_path = f"{job_dir}/stdout"
        stderr_path = f"{job_dir}/stderr"
        exit_code_path = f"{job_dir}/exit-code"
        pid_path = f"{job_dir}/pid"
        script_path = f"{job_dir}/run.sh"
        run_script = (
            "#!/bin/bash\n"
            "set +e\n"
            f"{command}\n"
            "status=$?\n"
            f"printf '%s\\n' \"$status\" > {shlex.quote(exit_code_path)}\n"
            'exit "$status"\n'
        )
        encoded_script = base64.b64encode(run_script.encode()).decode("ascii")
        start_command = (
            f"mkdir -p {shlex.quote(job_dir)} && "
            f"printf %s {shlex.quote(encoded_script)} | base64 -d "
            f"> {shlex.quote(script_path)} && "
            f"chmod 0700 {shlex.quote(script_path)} && "
            f"{{ nohup /bin/bash {shlex.quote(script_path)} "
            f"> {shlex.quote(stdout_path)} "
            f"2> {shlex.quote(stderr_path)} </dev/null & "
            f"printf '%s\\n' \"$!\" > {shlex.quote(pid_path)}; }}"
        )
        cleanup_command = (
            f"if [ -r {shlex.quote(pid_path)} ]; then "
            f'kill "$(cat {shlex.quote(pid_path)})" '
            ">/dev/null 2>&1 || true; fi; "
            f"rm -rf {shlex.quote(job_dir)}"
        )

        started = await self._exec_step(
            start_command,
            _LONG_COMMAND_STEP_DEADLINE_MS,
        )
        if started.return_code != 0:
            return started

        deadline = time.monotonic() + deadline_ms / 1000
        poll_failure_attempt = 0
        try:
            while True:
                status = await self._exec_step(
                    (
                        f"if [ -f {shlex.quote(exit_code_path)} ]; then "
                        f"printf 'DONE:'; cat "
                        f"{shlex.quote(exit_code_path)}; "
                        f"elif [ -r {shlex.quote(pid_path)} ] && "
                        f'kill -0 "$(cat {shlex.quote(pid_path)})" '
                        "2>/dev/null; then printf 'RUNNING\\n'; "
                        "else printf 'LOST\\n'; fi"
                    ),
                    _LONG_COMMAND_STEP_DEADLINE_MS,
                )
                if status.return_code != 0:
                    poll_failure_attempt += 1
                    failure_kind = self._rli_policy.classify_step_failure(
                        phase="verifier",
                        exit_code=status.return_code,
                        error=status.stderr or status.stdout,
                        incomplete=status.return_code in (-1, 124),
                        step_index=0,
                    )
                    decision = self._rli_policy.should_retry_step(
                        phase="verifier",
                        failure_kind=failure_kind,
                        attempt=poll_failure_attempt,
                    )
                    remaining = deadline - time.monotonic()
                    if decision.retry and remaining > 0:
                        await asyncio.sleep(
                            min(_LONG_COMMAND_POLL_INTERVAL_SEC, remaining)
                        )
                        continue
                    raise RLILongRunningCommandLostError(
                        "RLI long-running command polling failed: "
                        f"return_code={status.return_code}"
                    )
                poll_failure_attempt = 0
                status_text = status.stdout.strip()
                if status_text == "LOST":
                    raise RLILongRunningCommandLostError(
                        "RLI long-running command disappeared before producing "
                        f"an exit code: {status_text}"
                    )
                if status_text.startswith("DONE:"):
                    return ExecResult(
                        stdout="",
                        stderr="",
                        return_code=int(status_text.removeprefix("DONE:").strip()),
                    )
                if status_text != "RUNNING":
                    raise RLILongRunningCommandLostError(
                        "Unexpected RLI long-running command state: "
                        f"{status_text or 'No output'}"
                    )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ExecResult(
                        stdout="",
                        stderr=(
                            f"Command timed out after {deadline_ms / 1000:g} seconds"
                        ),
                        return_code=124,
                    )
                await asyncio.sleep(min(_LONG_COMMAND_POLL_INTERVAL_SEC, remaining))
        finally:
            await asyncio.shield(
                self._exec_step(
                    cleanup_command,
                    _LONG_COMMAND_STEP_DEADLINE_MS,
                )
            )

    async def is_dir(self, path: str) -> bool:
        result = await self.exec(
            f"test -d {shlex.quote(path)}",
            user=getattr(self, "default_user", "root"),
            rewrite_paths=False,
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
        rewrite_paths: bool = True,
        raise_on_failure: bool = True,
    ) -> ExecResult:
        attempts = max(1, self._rli_policy.config.max_bootstrap_step_retries + 1)
        last_result: ExecResult | None = None
        for attempt in range(1, attempts + 1):
            result = await self.exec(
                command,
                user=user,
                rewrite_paths=rewrite_paths,
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
        # The verifier downloads /logs/verifier after running test.sh. When the
        # container was recycled mid-task that dir is gone, the exec here failed
        # with a normal non-zero rc (tar rc=2, not exec-proxy rc=-1), and Harbor
        # would otherwise surface this as RewardFileNotFoundError — which is in
        # the default retry-exclude set, i.e. a one-way reward=0. Confirm the
        # recycle via the marker before that happens so the trial is re-run.
        await self._raise_if_container_reset(
            trigger=(
                f"verifier download failed source={source_path} "
                f"return_code={last_result.return_code}"
            ),
            healthy_blip=(last_result.return_code == -1),
        )
        raise RuntimeError(
            f"Failed to download {source_path}: "
            f"return_code={last_result.return_code}, "
            f"stdout={last_result.stdout!r}, stderr={last_result.stderr!r}"
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        payload = base64.b64encode(source.read_bytes()).decode("ascii")
        chunks = [
            payload[index : index + self._rli_upload_chunk_chars]
            for index in range(0, len(payload), self._rli_upload_chunk_chars)
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
        result = await self._exec_download_command(
            # Keep pipefail so a failing tar does not get masked by base64.
            f"set -o pipefail && tar -C {shlex.quote(source_dir)} -czf - . | base64 -w0",
            source_path=source_dir,
        )
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as archive:
            archive.write(base64.b64decode((result.stdout or "").strip()))
            archive_path = Path(archive.name)
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(target, filter=_host_safe_mode)
        finally:
            try:
                archive_path.unlink()
            except FileNotFoundError:
                pass
        if source_dir.rstrip("/").endswith("/logs/verifier") and not any(
            target.glob("reward*")
        ):
            await self._raise_if_container_reset(
                trigger=(
                    f"verifier logs downloaded without reward file "
                    f"(source={source_dir})"
                ),
                healthy_blip=False,
            )
            self._dump_rli_problem(
                "reward_file_missing_container_alive",
                f"verifier logs downloaded from {source_dir} without a reward "
                f"file; alive-marker probe says the container is HEALTHY, so "
                f"Harbor will raise RewardFileNotFoundError (final, honest "
                f"fail, not retried)",
            )

    async def attach(self) -> None:
        raise NotImplementedError("RLI environment does not support attach")
