import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _install_harbor_stubs() -> None:
    harbor = types.ModuleType("harbor")
    environments = types.ModuleType("harbor.environments")
    base = types.ModuleType("harbor.environments.base")
    capabilities = types.ModuleType("harbor.environments.capabilities")

    class BaseEnvironment:
        pass

    class ExecResult:
        pass

    class EnvironmentCapabilities:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    base.BaseEnvironment = BaseEnvironment
    base.ExecResult = ExecResult
    capabilities.EnvironmentCapabilities = EnvironmentCapabilities

    sys.modules.setdefault("harbor", harbor)
    sys.modules.setdefault("harbor.environments", environments)
    sys.modules.setdefault("harbor.environments.base", base)
    sys.modules.setdefault("harbor.environments.capabilities", capabilities)


_install_harbor_stubs()

from contrib.terminal_bench.src import environment as environment_module  # noqa: E402
from contrib.terminal_bench.src.environment import (  # noqa: E402
    RLIEnvironment,
    _normalize_rli_config,
)

pytestmark = pytest.mark.unit


def _policy(max_transfer_retries: int = 0):
    return environment_module.TerminalBenchRLIPolicy(
        environment_module.TerminalBenchRLIPolicyConfig(
            max_transfer_retries=max_transfer_retries,
        )
    )


def test_normalize_rli_config_accepts_json_string():
    config = _normalize_rli_config(json.dumps({"env_prefix": "terminal-bench"}))

    assert config == {"env_prefix": "terminal-bench"}


def test_normalize_rli_config_accepts_dict():
    raw = {"env_prefix": "terminal-bench", "ttl_ms": 1000}

    config = _normalize_rli_config(raw)

    assert config == raw
    assert config is not raw


def test_normalize_rli_config_rejects_non_object_json():
    with pytest.raises(TypeError, match="JSON object"):
        _normalize_rli_config("[]")


@pytest.mark.parametrize("allow_internet_access", [False, True])
def test_start_passes_network_policy_and_disables_pip_proxy(allow_internet_access):
    env = object.__new__(RLIEnvironment)
    env._rli_task_id_override = "task-a"
    env._rli_env_prefix = "terminal-bench"
    env._rli_env_version = "1.0.0"
    env._rli_ttl_ms = None
    env._rli_seed = None
    env._rli_wait_for_ready = False
    env._rli_use_pip_proxy = True
    env._rli_config = {
        "agent_timeout_sec": 60,
        "allow_internet_access": allow_internet_access,
    }
    env._rli_policy = _policy()
    env._rli_request_retries = 0
    env._rli_request_retry_backoff = 0
    env._rli_create_session_jitter_max_sec = 0
    env._install_apply_patch_shim = AsyncMock()
    env._prepare_remote_layout = AsyncMock()
    env._bootstrap_remote_apt = AsyncMock()

    create_session = AsyncMock(return_value=SimpleNamespace(session_id="sid-1"))
    env._rli_client = SimpleNamespace(create_session=create_session)

    asyncio.run(env.start(force_build=False))

    request = create_session.await_args.args[0]
    assert env._rli_session_id == "sid-1"
    assert request.env.name == "terminal-bench-task-a"
    assert request.env.task_id == "task-a"
    assert request.use_pip_proxy is False
    assert request.allow_internet_access is allow_internet_access
    assert request.ttl_ms == (60 + 600) * 1000


def test_upload_file_sends_base64_in_chunks(tmp_path: Path):
    source = tmp_path / "payload.bin"
    source.write_bytes(b"0123456789abcdef")
    calls = []
    env = object.__new__(RLIEnvironment)
    env._rli_upload_chunk_chars = 8
    env._rli_policy = _policy()
    env._rli_transfer_retry_backoff = 0

    async def fake_exec(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    env.exec = fake_exec

    asyncio.run(env.upload_file(source, "/app/payload.bin"))

    commands = [command for command, _ in calls]
    assert commands[0].startswith("mkdir -p /app && rm -f /tmp/rli-upload-")
    assert sum("cat >" in command for command in commands) > 1
    assert "base64 -d /tmp/rli-upload-" in commands[-1]
    assert "for i in $(seq 0" in commands[-1]
    assert all(len(command) < 400 for command in commands if "cat >" in command)
    assert all(kwargs == {"user": "root"} for _, kwargs in calls)


def test_upload_dir_uses_chunked_upload_file_then_extract(tmp_path: Path):
    source = tmp_path / "tests"
    source.mkdir()
    (source / "test.txt").write_text("hello", encoding="utf-8")
    uploads = []
    calls = []
    env = object.__new__(RLIEnvironment)
    env._rli_policy = _policy()
    env._rli_transfer_retry_backoff = 0

    async def fake_upload_file(source_path, target_path):
        uploads.append((Path(source_path), target_path))

    async def fake_exec(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    env.upload_file = fake_upload_file
    env.exec = fake_exec

    asyncio.run(env.upload_dir(source, "/app/tests"))

    assert len(uploads) == 1
    assert uploads[0][0].name.endswith(".tar.gz")
    assert uploads[0][1].startswith("/tmp/rli-upload-")
    assert uploads[0][1].endswith(".tar.gz")
    assert len(calls) == 1
    assert "mkdir -p /app/tests" in calls[0][0]
    assert (
        "tar --no-same-owner --no-same-permissions -xzf /tmp/rli-upload-" in calls[0][0]
    )
    assert calls[0][1] == {"user": "root"}


def test_transfer_command_retries_idempotent_failures():
    env = object.__new__(RLIEnvironment)
    env._rli_policy = _policy(max_transfer_retries=2)
    env._rli_transfer_retry_backoff = 0
    calls = []

    async def fake_exec(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            stdout="temporary",
            stderr="",
            return_code=1 if len(calls) < 3 else 0,
        )

    env.exec = fake_exec

    asyncio.run(
        env._exec_transfer_command(
            "idempotent command",
            operation="upload chunk for",
            source="payload",
        )
    )

    assert len(calls) == 3
    assert all(call == ("idempotent command", {"user": "root"}) for call in calls)
