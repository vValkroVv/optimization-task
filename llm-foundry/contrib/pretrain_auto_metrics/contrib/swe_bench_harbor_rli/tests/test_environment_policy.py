from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass

import pytest

from swe_bench_harbor_rli.src.rli_policy import (
    SWEBenchRLIPolicy,
    SWEBenchRLIPolicyConfig,
)


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
        pass

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


@pytest.mark.asyncio
async def test_bootstrap_command_retries_early_session_instability(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_policy = SWEBenchRLIPolicy(
        SWEBenchRLIPolicyConfig(max_bootstrap_step_retries=2)
    )

    calls: list[tuple[str, str | None]] = []
    results = [
        environment_module.ExecResult(
            stdout="",
            stderr="exec-proxy connection lost; partial result preserved",
            return_code=124,
        ),
        environment_module.ExecResult(stdout="", stderr="", return_code=0),
    ]

    async def fake_exec(command: str, user: str | None = None):
        calls.append((command, user))
        return results.pop(0)

    monkeypatch.setattr(env, "exec", fake_exec)

    await env._exec_bootstrap_command(
        "mkdir -p /logs",
        step_index=0,
        user="root",
    )

    assert len(calls) == 2
    assert calls[0] == (
        "mkdir -p /logs",
        "root",
    )


@pytest.mark.asyncio
async def test_transfer_command_retries_opaque_upload_failure(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_policy = SWEBenchRLIPolicy(
        SWEBenchRLIPolicyConfig(max_transfer_retries=2)
    )

    calls: list[str] = []
    results = [
        environment_module.ExecResult(stdout="", stderr="", return_code=1),
        environment_module.ExecResult(stdout="", stderr="", return_code=0),
    ]

    async def fake_exec(command: str, user: str | None = None):
        _ = user
        calls.append(command)
        return results.pop(0)

    monkeypatch.setattr(env, "exec", fake_exec)

    await env._exec_transfer_command(
        "cat > /tmp/file.part <<'EOF'\nabc\nEOF\n",
        operation="upload chunk for",
        source="/tmp/source.txt",
    )

    assert len(calls) == 2


@pytest.mark.asyncio
async def test_download_command_does_not_retry_missing_remote_verifier_dir(
    monkeypatch,
) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_policy = SWEBenchRLIPolicy(
        SWEBenchRLIPolicyConfig(max_transfer_retries=2)
    )

    calls: list[str] = []

    async def fake_exec(command: str, user: str | None = None):
        _ = user
        calls.append(command)
        return environment_module.ExecResult(
            stdout=(
                "tar: /logs/verifier: Cannot open: "
                "No such file or directory"
            ),
            stderr="root@rli:/app#",
            return_code=2,
        )

    monkeypatch.setattr(env, "exec", fake_exec)

    with pytest.raises(RuntimeError, match="Failed to download /logs/verifier"):
        await env._exec_download_command(
            "set -o pipefail && tar -C /logs/verifier -czf - . | base64 -w0",
            source_path="/logs/verifier",
        )

    assert len(calls) == 1
