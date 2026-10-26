from __future__ import annotations

import asyncio
import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from httpx import HTTPStatusError, Request, Response

from terminal_bench.src.rli_policy import TerminalBenchRLIPolicy


def _load_environment_module():
    harbor_module = types.ModuleType("harbor")
    environments_module = types.ModuleType("harbor.environments")
    base_module = types.ModuleType("harbor.environments.base")
    capabilities_module = types.ModuleType("harbor.environments.capabilities")

    @dataclass
    class ExecResult:
        stdout: str = ""
        stderr: str = ""
        return_code: int = 0

    class BaseEnvironment:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs
            self.task_env_config = type("TaskEnvConfig", (), {"workdir": None})()
            self.environment_dir = Path.cwd()
            self.default_user = "root"
            # Real Harbor BaseEnvironment always sets this; RLIEnvironment.
            # _infer_task_id() reads it whenever environment_dir.name != "environment"
            # (e.g. Path.cwd() in this synthetic fixture).
            self.environment_name = "terminal-bench-test-task"

    @dataclass
    class EnvironmentCapabilities:
        disable_internet: bool = True
        mounted: bool = False

    base_module.BaseEnvironment = BaseEnvironment
    base_module.ExecResult = ExecResult
    capabilities_module.EnvironmentCapabilities = EnvironmentCapabilities

    sys.modules["harbor"] = harbor_module
    sys.modules["harbor.environments"] = environments_module
    sys.modules["harbor.environments.base"] = base_module
    sys.modules["harbor.environments.capabilities"] = capabilities_module

    sys.modules.pop("terminal_bench.src.environment", None)
    return importlib.import_module("terminal_bench.src.environment")


def _load_environment_module_with_session_id_assignment():
    harbor_module = types.ModuleType("harbor")
    environments_module = types.ModuleType("harbor.environments")
    base_module = types.ModuleType("harbor.environments.base")
    capabilities_module = types.ModuleType("harbor.environments.capabilities")

    @dataclass
    class ExecResult:
        stdout: str = ""
        stderr: str = ""
        return_code: int = 0

    class BaseEnvironment:
        def __init__(
            self,
            *,
            environment_dir: Path,
            environment_name: str,
            session_id: str,
            trial_paths=None,
            **kwargs,
        ):
            _ = kwargs
            self.environment_dir = environment_dir
            self.environment_name = environment_name
            self.session_id = session_id
            self.trial_paths = trial_paths
            self.default_user = None
            self.task_env_config = type("TaskEnvConfig", (), {"workdir": None})()

    @dataclass
    class EnvironmentCapabilities:
        disable_internet: bool = True
        mounted: bool = False

    base_module.BaseEnvironment = BaseEnvironment
    base_module.ExecResult = ExecResult
    capabilities_module.EnvironmentCapabilities = EnvironmentCapabilities

    sys.modules["harbor"] = harbor_module
    sys.modules["harbor.environments"] = environments_module
    sys.modules["harbor.environments.base"] = base_module
    sys.modules["harbor.environments.capabilities"] = capabilities_module

    sys.modules.pop("terminal_bench.src.environment", None)
    return importlib.import_module("terminal_bench.src.environment")


def _rli_client_error(
    *,
    code: str,
    category: str = "client_error",
    retryable: bool = True,
    retry_strategy: str | None = "wait_and_retry_step",
) -> Exception:
    from gigarli import ErrorDetails, ErrorResponse, ResponseMeta, RLIClientError

    return RLIClientError(
        ErrorResponse(
            error=ErrorDetails(
                id="err-1",
                code=code,
                category=category,
                message=code,
                retryable=retryable,
                retry_strategy=retry_strategy,
            ),
            meta=ResponseMeta(
                request_id="req-1",
                runtime_ms=1,
                version="rli-api-gateway@test",
                schema_version="1.0",
            ),
        )
    )


@pytest.mark.parametrize(
    ("configured_ttl", "agent_timeout", "expected_ttl"),
    [
        (64_800_000, None, 4_200_000),
        (2_000_000, None, 2_000_000),
        (64_800_000, 900, 1_500_000),
        (64_800_000, 7200, 7_800_000),
        (None, None, 4_200_000),
    ],
)
@pytest.mark.asyncio
async def test_start_creates_session_via_gigarli_request_model(
    tmp_path, configured_ttl, agent_timeout, expected_ttl
) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env.environment_dir = tmp_path / "environment"
    env.environment_dir.mkdir()
    (tmp_path / "task.toml").write_text(
        "[agent]\ntimeout_sec = 3600\n\n[verifier]\ntimeout_sec = 3600\n",
        encoding="utf-8",
    )
    env._rli_env_version = "1.0.0"
    env._rli_ttl_ms = configured_ttl
    env._rli_seed = None
    env._rli_use_pip_proxy = True
    env._rli_wait_for_ready = True
    env._rli_config = {"agent_timeout_sec": agent_timeout}
    env._rli_session_id = None
    env._rli_trace_path = None
    env._rli_policy = TerminalBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._build_env_name = lambda task_id: f"terminal-bench-{task_id}"
    env._infer_task_id = lambda: "mailman"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._prepare_mailman_smtp_runtime = AsyncMock()
    env._bootstrap_remote_apt = AsyncMock()

    create_session = AsyncMock(
        return_value=type("CreateSessionResult", (), {"session_id": "sess-123"})()
    )
    get_session = AsyncMock(
        return_value=type("GetSessionResult", (), {"status": "RUNNING"})()
    )
    env._rli_client = type(
        "FakeClient",
        (),
        {"create_session": create_session, "get_session": get_session},
    )()

    await env.start(force_build=False)

    request = create_session.await_args.args[0]
    assert request.env.name == "terminal-bench-mailman"
    assert request.env.version == "1.0.0"
    assert request.env.task_id == "mailman"
    assert request.ttl_ms == expected_ttl
    assert request.use_pip_proxy is True
    get_session.assert_awaited_once_with("sess-123")
    env._prepare_mailman_smtp_runtime.assert_awaited_once_with()
    assert env._rli_session_id == "sess-123"


@pytest.mark.asyncio
async def test_start_retries_create_session_on_retryable_http_status(
    monkeypatch,
) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_env_version = "1.0.0"
    env._rli_ttl_ms = 3_600_000
    env._rli_seed = None
    env._rli_use_pip_proxy = True
    env._rli_wait_for_ready = True
    env._rli_config = {"agent_timeout_sec": 3600}
    env._rli_session_id = None
    env._rli_trace_path = None
    env._rli_policy = TerminalBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._rli_request_retries = 2
    env._rli_request_retry_backoff = 2.0
    env._build_env_name = lambda task_id: f"terminal-bench-{task_id}"
    env._infer_task_id = lambda: "modernize-scientific-stack"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._bootstrap_remote_apt = AsyncMock()

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(environment_module.asyncio, "sleep", fake_sleep)

    retry_error = HTTPStatusError(
        "Server error '503 Service Unavailable' for url 'https://rli.sberdevices.ru/v1/sessions'",
        request=Request("POST", "https://rli.sberdevices.ru/v1/sessions"),
        response=Response(
            503, request=Request("POST", "https://rli.sberdevices.ru/v1/sessions")
        ),
    )

    create_session = AsyncMock(
        side_effect=[
            retry_error,
            type("CreateSessionResult", (), {"session_id": "sess-123"})(),
        ]
    )
    get_session = AsyncMock(
        return_value=type("GetSessionResult", (), {"status": "RUNNING"})()
    )
    env._rli_client = type(
        "FakeClient",
        (),
        {"create_session": create_session, "get_session": get_session},
    )()

    await env.start(force_build=False)

    assert create_session.await_count == 2
    assert (
        create_session.await_args_list[0].kwargs["idempotency_key"]
        == create_session.await_args_list[1].kwargs["idempotency_key"]
    )
    assert sleeps == [2.0]


def test_prepare_mailman_smtp_runtime_disables_postfix_chroot() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._exec_bootstrap_command = AsyncMock()

    asyncio.run(env._prepare_mailman_smtp_runtime())

    command = env._exec_bootstrap_command.await_args.args[0]
    assert 'SMTP("localhost", 2525)' in command
    assert "{4})y([[:space:]])/\\1n\\3/" in command


@pytest.mark.asyncio
async def test_start_retries_create_session_on_gateway_timeout(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_env_version = "1.0.0"
    env._rli_ttl_ms = 3_600_000
    env._rli_seed = None
    env._rli_use_pip_proxy = True
    env._rli_wait_for_ready = True
    env._rli_config = {"agent_timeout_sec": 3600}
    env._rli_session_id = None
    env._rli_trace_path = None
    env._rli_policy = TerminalBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._rli_request_retries = 2
    env._rli_request_retry_backoff = 2.0
    env._build_env_name = lambda task_id: f"terminal-bench-{task_id}"
    env._infer_task_id = lambda: "polyglot-c-py"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._bootstrap_remote_apt = AsyncMock()

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(environment_module.asyncio, "sleep", fake_sleep)

    retry_error = HTTPStatusError(
        "Server error '504 Gateway Timeout' for url 'https://rli.sberdevices.ru/v1/sessions'",
        request=Request("POST", "https://rli.sberdevices.ru/v1/sessions"),
        response=Response(
            504, request=Request("POST", "https://rli.sberdevices.ru/v1/sessions")
        ),
    )
    create_session = AsyncMock(
        side_effect=[
            retry_error,
            type("CreateSessionResult", (), {"session_id": "sess-123"})(),
        ]
    )
    get_session = AsyncMock(
        return_value=type("GetSessionResult", (), {"status": "RUNNING"})()
    )
    env._rli_client = type(
        "FakeClient",
        (),
        {"create_session": create_session, "get_session": get_session},
    )()

    await env.start(force_build=False)

    assert create_session.await_count == 2
    assert sleeps == [2.0]


@pytest.mark.asyncio
async def test_start_retries_create_session_on_retryable_rli_error_envelope(
    monkeypatch,
) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_env_version = "1.0.0"
    env._rli_ttl_ms = 3_600_000
    env._rli_seed = None
    env._rli_use_pip_proxy = True
    env._rli_wait_for_ready = True
    env._rli_config = {"agent_timeout_sec": 3600}
    env._rli_session_id = None
    env._rli_trace_path = None
    env._rli_policy = TerminalBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._rli_request_retries = 2
    env._rli_request_retry_backoff = 2.0
    env._build_env_name = lambda task_id: f"terminal-bench-{task_id}"
    env._infer_task_id = lambda: "build-pmars"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._bootstrap_remote_apt = AsyncMock()

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(environment_module.asyncio, "sleep", fake_sleep)

    create_session = AsyncMock(
        side_effect=[
            _rli_client_error(code="RESOURCE_EXHAUSTED"),
            type("CreateSessionResult", (), {"session_id": "sess-123"})(),
        ]
    )
    get_session = AsyncMock(
        return_value=type("GetSessionResult", (), {"status": "RUNNING"})()
    )
    env._rli_client = type(
        "FakeClient",
        (),
        {"create_session": create_session, "get_session": get_session},
    )()

    await env.start(force_build=False)

    assert create_session.await_count == 2
    assert sleeps == [2.0]


@pytest.mark.asyncio
async def test_start_closes_session_when_readiness_reaches_terminal_status() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_env_version = "1.0.0"
    env._rli_ttl_ms = 3_600_000
    env._rli_seed = None
    env._rli_use_pip_proxy = True
    env._rli_wait_for_ready = True
    env._rli_config = {"agent_timeout_sec": 3600}
    env._rli_session_id = None
    env._rli_trace_path = None
    env._rli_policy = TerminalBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._rli_request_retries = 0
    env._rli_request_retry_backoff = 0.0
    env._rli_create_session_jitter_max_sec = 0.0
    env._build_env_name = lambda task_id: f"terminal-bench-{task_id}"
    env._infer_task_id = lambda: "build-pmars"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._bootstrap_remote_apt = AsyncMock()

    create_session = AsyncMock(
        return_value=type("CreateSessionResult", (), {"session_id": "sess-123"})()
    )
    get_session = AsyncMock(
        return_value=type("GetSessionResult", (), {"status": "FAILED"})()
    )
    close_session = AsyncMock(return_value=None)
    env._rli_client = type(
        "FakeClient",
        (),
        {
            "create_session": create_session,
            "get_session": get_session,
            "close_session": close_session,
        },
    )()

    with pytest.raises(RuntimeError, match="FAILED"):
        await env.start(force_build=False)

    close_request = close_session.await_args.args[0]
    assert close_request.session_id == "sess-123"
    assert env._rli_session_id is None


@pytest.mark.asyncio
async def test_start_applies_create_session_startup_jitter(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_env_version = "1.0.0"
    env._rli_ttl_ms = 3_600_000
    env._rli_seed = None
    env._rli_use_pip_proxy = True
    env._rli_wait_for_ready = True
    env._rli_config = {"agent_timeout_sec": 3600}
    env._rli_session_id = None
    env._rli_trace_path = None
    env._rli_policy = TerminalBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._rli_request_retries = 0
    env._rli_request_retry_backoff = 2.0
    env._rli_create_session_jitter_max_sec = 3.0
    env._build_env_name = lambda task_id: f"terminal-bench-{task_id}"
    env._infer_task_id = lambda: "modernize-scientific-stack"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._bootstrap_remote_apt = AsyncMock()

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(environment_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(environment_module, "uniform", lambda a, b: 1.25)

    create_session = AsyncMock(
        return_value=type("CreateSessionResult", (), {"session_id": "sess-123"})()
    )
    get_session = AsyncMock(
        return_value=type("GetSessionResult", (), {"status": "RUNNING"})()
    )
    env._rli_client = type(
        "FakeClient",
        (),
        {"create_session": create_session, "get_session": get_session},
    )()

    await env.start(force_build=False)

    assert sleeps == [1.25]


@pytest.mark.asyncio
async def test_start_does_not_write_environment_level_rli_trace_events(
    monkeypatch,
) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_env_version = "1.0.0"
    env._rli_ttl_ms = 3_600_000
    env._rli_seed = None
    env._rli_use_pip_proxy = True
    env._rli_wait_for_ready = True
    env._rli_config = {"agent_timeout_sec": 3600}
    env._rli_session_id = None
    env._rli_trace_path = Path("/tmp/rli_trace.jsonl")
    env._rli_policy = TerminalBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._build_env_name = lambda task_id: f"terminal-bench-{task_id}"
    env._infer_task_id = lambda: "modernize-scientific-stack"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._bootstrap_remote_apt = AsyncMock()
    env._append_trace_event = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("environment-level RLI trace should not be written")
    )

    create_session = AsyncMock(
        return_value=type("CreateSessionResult", (), {"session_id": "sess-123"})()
    )
    get_session = AsyncMock(
        return_value=type("GetSessionResult", (), {"status": "RUNNING"})()
    )
    env._rli_client = type(
        "FakeClient",
        (),
        {"create_session": create_session, "get_session": get_session},
    )()

    await env.start(force_build=False)


@pytest.mark.asyncio
async def test_start_runs_probe_warmup_before_bootstrap_when_enabled() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_env_version = "1.0.0"
    env._rli_ttl_ms = 3_600_000
    env._rli_seed = None
    env._rli_use_pip_proxy = True
    env._rli_wait_for_ready = True
    env._rli_config = {"agent_timeout_sec": 3600}
    env._rli_session_id = None
    env._rli_trace_path = None
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._build_env_name = lambda task_id: f"terminal-bench-{task_id}"
    env._infer_task_id = lambda: "modernize-scientific-stack"
    env._rli_policy = TerminalBenchRLIPolicy(
        environment_module.TerminalBenchRLIPolicyConfig(
            enable_probe_warmup=True,
        )
    )
    env._rli_request_retries = 0
    env._rli_request_retry_backoff = 0.0
    env._rli_create_session_jitter_max_sec = 0.0

    calls: list[str] = []

    async def fake_probe(command: str, **kwargs) -> None:
        _ = kwargs
        calls.append(command)

    env._probe_warmup_session = fake_probe
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._bootstrap_remote_apt = AsyncMock()

    create_session = AsyncMock(
        return_value=type("CreateSessionResult", (), {"session_id": "sess-123"})()
    )
    get_session = AsyncMock(
        return_value=type("GetSessionResult", (), {"status": "RUNNING"})()
    )
    env._rli_client = type(
        "FakeClient",
        (),
        {"create_session": create_session, "get_session": get_session},
    )()

    await env.start(force_build=False)

    assert calls == ["pwd"]


@pytest.mark.asyncio
async def test_probe_warmup_retries_with_exponential_backoff(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_policy = TerminalBenchRLIPolicy(
        environment_module.TerminalBenchRLIPolicyConfig(
            enable_probe_warmup=True,
            max_bootstrap_step_retries=2,
        )
    )
    env.default_user = "root"

    results = [
        environment_module.ExecResult(return_code=-1, stdout="", stderr=""),
        environment_module.ExecResult(return_code=-1, stdout="", stderr=""),
        environment_module.ExecResult(return_code=0, stdout="", stderr=""),
    ]
    env.exec = AsyncMock(side_effect=results)

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(environment_module.asyncio, "sleep", fake_sleep)

    await env._probe_warmup_session("pwd")

    assert sleeps == [2.0, 4.0]


@pytest.mark.asyncio
async def test_bootstrap_remote_apt_configures_https_sources_and_force_ipv4() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_bootstrap_apt = True
    env._rli_policy = TerminalBenchRLIPolicy()

    calls: list[tuple[int, str, bool]] = []

    async def fake_exec_bootstrap_command(
        command: str,
        *,
        step_index: int,
        user: str | int | None = None,
        rewrite_paths: bool = True,
        raise_on_failure: bool = True,
    ):
        _ = user, raise_on_failure
        calls.append((step_index, command, rewrite_paths))
        return environment_module.ExecResult(return_code=0, stdout="", stderr="")

    env._exec_bootstrap_command = fake_exec_bootstrap_command

    await env._bootstrap_remote_apt()

    assert calls[0] == (0, "command -v apt-get >/dev/null 2>&1", False)
    assert "Acquire::ForceIPv4" in calls[1][1]
    assert calls[1][0] == 1
    assert calls[1][2] is False
    assert "rm -f /etc/apt/sources.list.d/ubuntu.sources" in calls[2][1]
    assert "https://archive.ubuntu.com/ubuntu/" in calls[2][1]
    assert "https://security.ubuntu.com/ubuntu/" in calls[2][1]
    assert calls[2][0] == 2
    assert calls[2][2] is False
    assert "dpkg --configure -a" in calls[3][1]
    assert calls[3][0] == 3
    assert calls[3][2] is False
    assert len(calls) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("return_code", "expected_confirmed", "expected_canary_ts"),
    [
        (0, True, 123.0),
        (1, False, 0.0),
    ],
)
async def test_mark_session_alive_requires_successful_marker_write(
    monkeypatch,
    return_code: int,
    expected_confirmed: bool,
    expected_canary_ts: float,
) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_enable_container_recycle_detection = True
    env._rli_session_confirmed_alive = False
    env._rli_last_canary_ts = 0.0
    env._exec_bootstrap_command = AsyncMock(
        return_value=environment_module.ExecResult(return_code=return_code)
    )
    monkeypatch.setattr(environment_module.time, "monotonic", lambda: 123.0)

    await env._mark_session_alive()

    assert env._rli_session_confirmed_alive is expected_confirmed
    assert env._rli_last_canary_ts == expected_canary_ts


@pytest.mark.asyncio
async def test_container_alive_probe_does_not_require_tests_symlink() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env.exec = AsyncMock(return_value=environment_module.ExecResult(return_code=0))

    assert await env._probe_container_alive() is True
    command = env.exec.await_args.args[0]
    assert environment_module._ALIVE_MARKER_PATH in command
    assert "/tests" not in command


@pytest.mark.asyncio
async def test_exec_calls_gigarli_step_and_maps_exec_result(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_command_timeout_ms = 60_000
    env._rli_task_command_timeout_ms = None
    env._rli_verifier_timeout_ms = None
    env._rli_enable_container_recycle_detection = False
    env.default_user = "root"
    env._rli_session_id = "sess-123"
    env._rli_trace_path = None
    env._wrap_command = lambda command, **_: f"wrapped::{command}"
    env._require_session = lambda: "sess-123"

    step = AsyncMock(
        return_value=type(
            "StepResult",
            (),
            {
                "observation": type(
                    "Observation", (), {"output": "stdout:\nhello\n"}
                )(),
                "events": [
                    type(
                        "StepEvent",
                        (),
                        {
                            "stdout_ref": "hello\n",
                            "stderr_ref": "root@rli:/app#",
                            "exit_code": 0,
                            "duration_ms": 512,
                        },
                    )()
                ],
                "terminated": False,
                "agent_id": "default",
                "applied_limits": None,
            },
        )()
    )
    env._rli_client = type("FakeClient", (), {"step": step})()

    result = await env.exec("pwd")

    request = step.await_args.args[0]
    assert request.session_id == "sess-123"
    assert request.mode == "single"
    assert request.actions[0].type == "env_action"
    assert request.actions[0].name == "bash"
    assert request.actions[0].args.cmd == "wrapped::pwd"
    assert request.limits.deadline_ms == 60_000
    assert result.stdout == "hello\n"
    assert result.stderr == "root@rli:/app#"
    assert result.return_code == 0


def test_use_verifier_command_timeout_restores_default() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_verifier_timeout_ms = None

    with env.use_verifier_command_timeout(3600):
        assert env._rli_verifier_timeout_ms == 3_600_000

    assert env._rli_verifier_timeout_ms is None


def test_exec_runs_long_oracle_solution_in_background() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env.default_user = "root"
    env._rli_command_timeout_ms = 600_000
    env._rli_verifier_timeout_ms = None
    env._wrap_command = lambda command, **kwargs: f"wrapped::{command}"
    env._exec_long_running = AsyncMock(
        return_value=environment_module.ExecResult(
            stdout="done\n",
            stderr="",
            return_code=0,
        )
    )
    env._exec_step = AsyncMock()

    result = asyncio.run(
        env.exec(
            "bash /solution/solve.sh > /logs/agent/oracle.txt",
            timeout_sec=3600,
        )
    )

    env._exec_long_running.assert_awaited_once_with(
        "wrapped::bash /solution/solve.sh > /logs/agent/oracle.txt",
        3_600_000,
    )
    env._exec_step.assert_not_awaited()
    assert result.return_code == 0


def test_exec_keeps_long_model_agent_command_on_direct_step() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env.default_user = "root"
    env._rli_command_timeout_ms = 600_000
    env._rli_verifier_timeout_ms = None
    env._wrap_command = lambda command, **kwargs: f"wrapped::{command}"
    env._exec_long_running = AsyncMock()
    env._exec_step = AsyncMock(
        return_value=environment_module.ExecResult(return_code=0)
    )

    asyncio.run(env.exec("python agent_command.py", timeout_sec=3600))

    env._exec_long_running.assert_not_awaited()
    env._exec_step.assert_awaited_once_with(
        "wrapped::python agent_command.py",
        1_500_000,
    )


def test_exec_long_running_retries_polling_and_cleans_up(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_policy = environment_module.TerminalBenchRLIPolicy()
    monkeypatch.setattr(environment_module.asyncio, "sleep", AsyncMock())
    env._exec_step = AsyncMock(
        side_effect=[
            environment_module.ExecResult(return_code=0),
            environment_module.ExecResult(return_code=-1),
            environment_module.ExecResult(return_code=-1),
            environment_module.ExecResult(stdout="DONE:0\n", return_code=0),
            environment_module.ExecResult(return_code=0),
        ]
    )

    result = asyncio.run(env._exec_long_running("sleep 1", 3_600_000))

    assert result.stdout == ""
    assert result.stderr == ""
    assert result.return_code == 0
    assert env._exec_step.await_count == 5
    start_command = env._exec_step.await_args_list[0].args[0]
    assert "nohup /bin/bash" in start_command
    assert "</dev/null &" in start_command
    assert "/pid" in start_command
    assert "tmux" not in start_command
    cleanup_command = env._exec_step.await_args_list[-1].args[0]
    assert 'kill "$(cat ' in cleanup_command
    assert "/pid" in cleanup_command
    assert "tmux" not in cleanup_command


def test_exec_long_running_fails_when_background_process_disappears() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._exec_step = AsyncMock(
        side_effect=[
            environment_module.ExecResult(return_code=0),
            environment_module.ExecResult(stdout="LOST", return_code=0),
            environment_module.ExecResult(return_code=0),
        ]
    )

    with pytest.raises(
        environment_module.RLILongRunningCommandLostError,
        match="disappeared before producing an exit code",
    ):
        asyncio.run(env._exec_long_running("sleep 1", 3_600_000))

    assert env._exec_step.await_count == 3
    poll_command = env._exec_step.await_args_list[1].args[0]
    assert "kill -0" in poll_command
    assert "/pid" in poll_command
    assert "tmux" not in poll_command


def test_exec_runs_verifier_command_in_background_below_rli_limit() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env.default_user = "root"
    env._rli_command_timeout_ms = 600_000
    env._rli_verifier_timeout_ms = 600_000
    env._wrap_command = lambda command, **kwargs: f"wrapped::{command}"
    env._exec_long_running = AsyncMock(
        return_value=environment_module.ExecResult(return_code=0)
    )
    env._exec_step = AsyncMock()

    asyncio.run(env.exec("bash /tests/test.sh > /logs/verifier/test-stdout.txt"))

    env._exec_long_running.assert_awaited_once_with(
        "wrapped::bash /tests/test.sh > /logs/verifier/test-stdout.txt",
        600_000,
    )
    env._exec_step.assert_not_awaited()


def test_session_id_property_exposes_active_rli_session() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_session_id = "sess-123"

    assert env.session_id == "sess-123"


def test_constructor_accepts_base_environment_session_id_assignment() -> None:
    environment_module = _load_environment_module_with_session_id_assignment()

    env = environment_module.RLIEnvironment(
        environment_dir=Path("/tmp/env"),
        environment_name="terminal-bench-modernize-scientific-stack",
        session_id="harbor-session",
        trial_paths=type("TrialPaths", (), {"agent_dir": Path("/tmp/agent")})(),
        rli_config_json={},
    )

    assert env.session_id == "harbor-session"


def test_session_id_property_raises_when_session_is_not_started() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_session_id = None

    with pytest.raises(RuntimeError, match="RLI session is not started"):
        _ = env.session_id


@pytest.mark.asyncio
async def test_is_dir_uses_exec_and_returns_true_for_existing_directory() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env.exec = AsyncMock(
        return_value=environment_module.ExecResult(stdout="", stderr="", return_code=0)
    )

    result = await env.is_dir("/tmp/terminal-bench/skills")

    env.exec.assert_awaited_once_with(
        "test -d /tmp/terminal-bench/skills",
        user="root",
        rewrite_paths=False,
    )
    assert result is True


@pytest.mark.asyncio
async def test_is_dir_returns_false_for_missing_directory() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env.exec = AsyncMock(
        return_value=environment_module.ExecResult(stdout="", stderr="", return_code=1)
    )

    result = await env.is_dir("/tmp/terminal-bench/missing")

    assert result is False


@pytest.mark.asyncio
async def test_stop_closes_session_via_gigarli_request_model(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_session_id = "sess-123"
    env._rli_trace_path = None

    close_session = AsyncMock(return_value=None)
    env._rli_client = type("FakeClient", (), {"close_session": close_session})()

    await env.stop(delete=False)

    request = close_session.await_args.args[0]
    assert request.session_id == "sess-123"
    assert env._rli_session_id is None


def test_init_passes_trace_path_to_gigarli_client(monkeypatch) -> None:
    environment_module = _load_environment_module()
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(environment_module, "RLIClient", FakeClient)

    environment_module.RLIEnvironment(
        rli_config_json={
            "trace_path": "/tmp/rli_trace.jsonl",
        }
    )

    assert captured["trace_path"] == Path("/tmp/rli_trace.jsonl")
