from __future__ import annotations

import base64
import importlib
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from httpx import HTTPStatusError, Request, Response

from swe_bench_harbor_rli.src.rli_policy import SWEBenchRLIPolicy


def test_wrap_command_exports_env_before_changing_workdir(tmp_path: Path) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env.task_env_config = type("TaskEnvConfig", (), {"workdir": None})()
    env._rli_default_workdir = str(tmp_path)
    env._merge_env = lambda value: value
    env._resolve_user = lambda value: value

    wrapped = env._wrap_command(
        'printf "%s|%s|%s" "$LLM_API_KEY" "$SPACE_VALUE" "$PWD"',
        cwd=None,
        env={"LLM_API_KEY": "dummy", "SPACE_VALUE": "two words"},
        user="root",
    )
    result = subprocess.run(
        ["bash", "-c", wrapped],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == f"dummy|two words|{tmp_path}"


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

    sys.modules.pop("swe_bench_harbor_rli.src.environment", None)
    return importlib.import_module("swe_bench_harbor_rli.src.environment")


def test_uv_bootstrap_uses_direct_release_asset_with_http11_retry() -> None:
    environment_module = _load_environment_module()
    command = environment_module._INSTALL_UV_COMMAND

    assert "install.sh" not in command
    assert "uv-x86_64-unknown-linux-gnu.tar.gz" in command
    assert "github.com/astral-sh/uv/releases/download" in command
    assert "--http1.1" in command
    assert "--retry 5" in command
    assert "--retry-all-errors" in command
    assert "/root/.local/bin/uv" in command
    assert "/usr/local/bin/uv" in command


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

    sys.modules.pop("swe_bench_harbor_rli.src.environment", None)
    return importlib.import_module("swe_bench_harbor_rli.src.environment")


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
    env._rli_policy = SWEBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._build_env_name = lambda task_id: f"swe-bench-{task_id}"
    env._infer_task_id = lambda: "modernize-scientific-stack"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._install_uv = AsyncMock()
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
    assert request.env.name == "swe-bench-modernize-scientific-stack"
    assert request.env.version == "1.0.0"
    assert request.env.task_id == "modernize-scientific-stack"
    assert request.ttl_ms == expected_ttl
    assert request.use_pip_proxy is True
    get_session.assert_awaited_once_with("sess-123")
    assert env._rli_session_id == "sess-123"


@pytest.mark.asyncio
async def test_start_retries_create_session_on_retryable_http_status(monkeypatch) -> None:
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
    env._rli_policy = SWEBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._rli_request_retries = 2
    env._rli_request_retry_backoff = 2.0
    env._build_env_name = lambda task_id: f"swe-bench-{task_id}"
    env._infer_task_id = lambda: "modernize-scientific-stack"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._install_uv = AsyncMock()
    env._bootstrap_remote_apt = AsyncMock()

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(environment_module.asyncio, "sleep", fake_sleep)

    retry_error = HTTPStatusError(
        "Server error '503 Service Unavailable' for url 'https://rli.sberdevices.ru/v1/sessions'",
        request=Request("POST", "https://rli.sberdevices.ru/v1/sessions"),
        response=Response(503, request=Request("POST", "https://rli.sberdevices.ru/v1/sessions")),
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
    assert create_session.await_args_list[0].kwargs["idempotency_key"] == create_session.await_args_list[1].kwargs["idempotency_key"]
    assert sleeps == [2.0]


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
    env._rli_policy = SWEBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._rli_request_retries = 2
    env._rli_request_retry_backoff = 2.0
    env._build_env_name = lambda task_id: f"swe-bench-{task_id}"
    env._infer_task_id = lambda: "polyglot-c-py"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._install_uv = AsyncMock()
    env._bootstrap_remote_apt = AsyncMock()

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(environment_module.asyncio, "sleep", fake_sleep)

    retry_error = HTTPStatusError(
        "Server error '504 Gateway Timeout' for url 'https://rli.sberdevices.ru/v1/sessions'",
        request=Request("POST", "https://rli.sberdevices.ru/v1/sessions"),
        response=Response(504, request=Request("POST", "https://rli.sberdevices.ru/v1/sessions")),
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
async def test_start_retries_create_session_with_new_key_on_wait_and_retry_step(
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
    env._rli_policy = SWEBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._rli_request_retries = 2
    env._rli_request_retry_backoff = 2.0
    env._build_env_name = lambda task_id: f"swe-bench-{task_id}"
    env._infer_task_id = lambda: "build-pmars"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._install_uv = AsyncMock()
    env._bootstrap_remote_apt = AsyncMock()

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(environment_module.asyncio, "sleep", fake_sleep)

    create_session = AsyncMock(
        side_effect=[
            _rli_client_error(
                code="RESOURCE_EXHAUSTED",
                retry_strategy="wait_and_retry_step",
            ),
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
        != create_session.await_args_list[1].kwargs["idempotency_key"]
    )
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
    env._rli_policy = SWEBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._rli_request_retries = 0
    env._rli_request_retry_backoff = 0.0
    env._rli_create_session_jitter_max_sec = 0.0
    env._build_env_name = lambda task_id: f"swe-bench-{task_id}"
    env._infer_task_id = lambda: "build-pmars"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._install_uv = AsyncMock()
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
async def test_upload_file_splits_base64_payload_and_finalizes(
    monkeypatch,
    tmp_path,
) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    commands: list[str] = []

    async def fake_transfer(command: str, *, operation: str, source) -> None:
        _ = operation, source
        commands.append(command)

    monkeypatch.setattr(env, "_exec_transfer_command", fake_transfer)
    monkeypatch.setattr(environment_module, "_UPLOAD_CHUNK_CHARS", 4)
    monkeypatch.setattr(
        environment_module.uuid,
        "uuid4",
        lambda: type("UUID", (), {"hex": "fixed"})(),
    )
    source = tmp_path / "payload.bin"
    source.write_bytes(b"abcdef")

    await env.upload_file(source, "/remote/data.bin")

    assert len(commands) == 4
    assert "YWJj" in commands[1]
    assert "ZGVm" in commands[2]
    assert "base64 -d /tmp/rli-upload-fixed.b64 > /remote/data.bin" in commands[3]
    assert "seq 0 1" in commands[3]


@pytest.mark.asyncio
async def test_download_file_decodes_payload_and_creates_parent(
    monkeypatch,
    tmp_path,
) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    download = AsyncMock(
        return_value=environment_module.ExecResult(
            stdout=base64.b64encode(b"payload").decode("ascii"),
            stderr="",
            return_code=0,
        )
    )
    monkeypatch.setattr(env, "_exec_download_command", download)
    target = tmp_path / "nested" / "payload.bin"

    await env.download_file("/remote/data.bin", target)

    assert target.read_bytes() == b"payload"
    download.assert_awaited_once_with(
        "base64 -w0 /remote/data.bin",
        source_path="/remote/data.bin",
    )


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
    env._rli_policy = SWEBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._rli_request_retries = 0
    env._rli_request_retry_backoff = 2.0
    env._rli_create_session_jitter_max_sec = 3.0
    env._build_env_name = lambda task_id: f"swe-bench-{task_id}"
    env._infer_task_id = lambda: "modernize-scientific-stack"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._install_uv = AsyncMock()
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
    env._rli_policy = SWEBenchRLIPolicy()
    env._rli_session_ready_timeout = 60.0
    env._rli_session_ready_poll_interval = 0.0
    env._build_env_name = lambda task_id: f"swe-bench-{task_id}"
    env._infer_task_id = lambda: "modernize-scientific-stack"
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._install_uv = AsyncMock()
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
    env._build_env_name = lambda task_id: f"swe-bench-{task_id}"
    env._infer_task_id = lambda: "modernize-scientific-stack"
    env._rli_policy = SWEBenchRLIPolicy(
        environment_module.SWEBenchRLIPolicyConfig(
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
    env._install_uv = AsyncMock()
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
    env._rli_policy = SWEBenchRLIPolicy(
        environment_module.SWEBenchRLIPolicyConfig(
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
    env._rli_policy = SWEBenchRLIPolicy()

    calls: list[tuple[int, str]] = []

    async def fake_exec_bootstrap_command(
        command: str,
        *,
        step_index: int,
        user: str | int | None = None,
        raise_on_failure: bool = True,
    ):
        _ = user, raise_on_failure
        calls.append((step_index, command))
        return environment_module.ExecResult(return_code=0, stdout="", stderr="")

    env._exec_bootstrap_command = fake_exec_bootstrap_command

    await env._bootstrap_remote_apt()

    assert calls[0] == (0, "command -v apt-get >/dev/null 2>&1")
    assert "Acquire::ForceIPv4" in calls[1][1]
    assert calls[1][0] == 1
    assert "rm -f /etc/apt/sources.list.d/ubuntu.sources" in calls[2][1]
    assert "https://archive.ubuntu.com/ubuntu/" in calls[2][1]
    assert "https://security.ubuntu.com/ubuntu/" in calls[2][1]
    assert calls[2][0] == 2
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_exec_calls_gigarli_step_and_maps_exec_result(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_command_timeout_ms = 60_000
    env._rli_active_verifier_timeout_ms = None
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
                "observation": type("Observation", (), {"output": "stdout:\nhello\n"})(),
                "events": [
                    type(
                        "StepEvent",
                        (),
                        {
                            "error": None,
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


def test_use_verifier_command_timeout_caps_and_restores_active_timeout() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_active_verifier_timeout_ms = None
    env._rli_verifier_command_timeout_ms = 1_500_000

    with env.use_verifier_command_timeout(2_100):
        assert env._rli_active_verifier_timeout_ms == 1_500_000

    assert env._rli_active_verifier_timeout_ms is None


@pytest.mark.asyncio
async def test_exec_runs_harbor_verifier_command_in_background() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_active_verifier_timeout_ms = 3_300_000
    env._rli_command_timeout_ms = 600_000
    env.default_user = "root"
    env._wrap_command = lambda command, **_: f"wrapped::{command}"
    env._exec_long_running = AsyncMock(
        return_value=environment_module.ExecResult(return_code=0)
    )

    result = await env.exec(
        "(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1"
    )

    assert result.return_code == 0
    env._exec_long_running.assert_awaited_once_with(
        "wrapped::(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1",
        timeout_ms=3_300_000,
    )


@pytest.mark.asyncio
async def test_exec_keeps_non_verifier_command_on_direct_step() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_active_verifier_timeout_ms = 3_300_000
    env._rli_command_timeout_ms = 600_000
    env.default_user = "root"
    env._wrap_command = lambda command, **_: f"wrapped::{command}"
    env._exec_step = AsyncMock(
        return_value=environment_module.ExecResult(return_code=0)
    )

    await env.exec("chmod +x /tests/test.sh")

    env._exec_step.assert_awaited_once_with(
        "wrapped::chmod +x /tests/test.sh",
        deadline_ms=1_500_000,
    )


@pytest.mark.asyncio
async def test_verifier_step_fails_fast_when_exec_proxy_is_lost() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_active_verifier_timeout_ms = 3_300_000

    with pytest.raises(
        environment_module.RLIExecProxyConnectionLostError,
        match="No output",
    ):
        env._raise_if_verifier_connection_lost(
            environment_module.ExecResult(return_code=-1)
        )


@pytest.mark.asyncio
async def test_long_running_command_polls_until_exit(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._exec_step = AsyncMock(
        side_effect=[
            environment_module.ExecResult(return_code=0),
            environment_module.ExecResult(stdout="RUNNING\n", return_code=0),
            environment_module.RLIExecProxyConnectionLostError("temporary loss"),
            environment_module.ExecResult(stdout="DONE:0\n", return_code=0),
            environment_module.ExecResult(return_code=0),
        ]
    )
    monkeypatch.setattr(environment_module.asyncio, "sleep", AsyncMock())

    result = await env._exec_long_running("run tests", timeout_ms=60_000)

    assert result == environment_module.ExecResult(
        stdout="",
        stderr="",
        return_code=0,
    )


@pytest.mark.asyncio
async def test_long_running_command_preserves_result_when_cleanup_fails() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env.logger = Mock()
    env._exec_step = AsyncMock(
        side_effect=[
            environment_module.ExecResult(return_code=0),
            environment_module.ExecResult(stdout="DONE:0\n", return_code=0),
            environment_module.RLIExecProxyConnectionLostError("cleanup failed"),
        ]
    )

    result = await env._exec_long_running("run tests", timeout_ms=60_000)

    assert result.return_code == 0
    env.logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_long_running_command_reports_disappeared_process() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._exec_step = AsyncMock(
        side_effect=[
            environment_module.ExecResult(return_code=0),
            environment_module.ExecResult(stdout="LOST\n", return_code=0),
            environment_module.ExecResult(return_code=0),
        ]
    )

    with pytest.raises(
        environment_module.RLILongRunningCommandLostError,
        match="disappeared",
    ):
        await env._exec_long_running("run tests", timeout_ms=60_000)


def test_build_env_name_maps_swe_bench_instance_id_to_registry_name() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_env_prefix = "swe-bench"
    env._rli_config = {}

    assert (
        env._build_env_name("astropy__astropy-12907")
        == "swe-bench-astropy-12907"
    )


def test_build_env_name_prefers_explicit_config_override() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_env_prefix = "swe-bench"
    env._rli_config = {"env_name": "swe-bench-custom"}

    assert env._build_env_name("astropy__astropy-12907") == "swe-bench-custom"


def test_session_id_property_exposes_active_rli_session() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_session_id = "sess-123"

    assert env.session_id == "sess-123"


def test_constructor_accepts_base_environment_session_id_assignment() -> None:
    environment_module = _load_environment_module_with_session_id_assignment()

    env = environment_module.RLIEnvironment(
        environment_dir=Path("/tmp/env"),
        environment_name="swe-bench-modernize-scientific-stack",
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

    result = await env.is_dir("/tmp/swe-bench/skills")

    env.exec.assert_awaited_once_with(
        "test -d /tmp/swe-bench/skills",
        user="root",
    )
    assert result is True


@pytest.mark.asyncio
async def test_is_dir_returns_false_for_missing_directory() -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env.exec = AsyncMock(
        return_value=environment_module.ExecResult(stdout="", stderr="", return_code=1)
    )

    result = await env.is_dir("/tmp/swe-bench/missing")

    assert result is False


@pytest.mark.asyncio
async def test_stop_closes_session_via_gigarli_request_model(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_session_id = "sess-123"
    env._rli_trace_path = None
    env._harbor_session_id = "harbor-session"
    env.logger = Mock()

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
