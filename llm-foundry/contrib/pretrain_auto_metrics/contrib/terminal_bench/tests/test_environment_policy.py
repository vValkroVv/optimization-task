from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass

import pytest

from terminal_bench.src.rli_policy import (
    TerminalBenchRLIPolicy,
    TerminalBenchRLIPolicyConfig,
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

    sys.modules.pop("terminal_bench.src.environment", None)
    return importlib.import_module("terminal_bench.src.environment")


@pytest.mark.asyncio
async def test_bootstrap_command_retries_early_session_instability(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_policy = TerminalBenchRLIPolicy(
        TerminalBenchRLIPolicyConfig(max_bootstrap_step_retries=2)
    )

    calls: list[tuple[str, str | None, bool]] = []
    results = [
        environment_module.ExecResult(
            stdout="",
            stderr="exec-proxy connection lost; partial result preserved",
            return_code=124,
        ),
        environment_module.ExecResult(stdout="", stderr="", return_code=0),
    ]

    async def fake_exec(
        command: str, user: str | None = None, rewrite_paths: bool = True
    ):
        calls.append((command, user, rewrite_paths))
        return results.pop(0)

    monkeypatch.setattr(env, "exec", fake_exec)

    await env._exec_bootstrap_command(
        "mkdir -p /tmp/terminal-bench/logs",
        step_index=0,
        user="root",
        rewrite_paths=False,
    )

    assert len(calls) == 2
    assert calls[0] == (
        "mkdir -p /tmp/terminal-bench/logs",
        "root",
        False,
    )


@pytest.mark.asyncio
async def test_transfer_command_retries_opaque_upload_failure(monkeypatch) -> None:
    environment_module = _load_environment_module()
    env = object.__new__(environment_module.RLIEnvironment)
    env._rli_policy = TerminalBenchRLIPolicy(
        TerminalBenchRLIPolicyConfig(max_transfer_retries=2)
    )

    calls: list[str] = []
    results = [
        environment_module.ExecResult(stdout="", stderr="", return_code=1),
        environment_module.ExecResult(stdout="", stderr="", return_code=0),
    ]

    async def fake_exec(
        command: str, user: str | None = None, rewrite_paths: bool = True
    ):
        _ = user, rewrite_paths
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
    env._rli_policy = TerminalBenchRLIPolicy(
        TerminalBenchRLIPolicyConfig(max_transfer_retries=2)
    )
    # Session was never confirmed alive (e.g. this synthetic env never ran
    # start()) -- _raise_if_container_reset() no-ops in that case and lets
    # this genuine failure stand as a plain RuntimeError, which is exactly
    # what this test is asserting.
    env._rli_enable_container_recycle_detection = False
    env._rli_session_confirmed_alive = False

    calls: list[str] = []

    async def fake_exec(
        command: str, user: str | None = None, rewrite_paths: bool = True
    ):
        _ = user, rewrite_paths
        calls.append(command)
        return environment_module.ExecResult(
            stdout=(
                "tar: /tmp/terminal-bench/logs/verifier: Cannot open: "
                "No such file or directory"
            ),
            stderr="root@rli:/app#",
            return_code=2,
        )

    monkeypatch.setattr(env, "exec", fake_exec)

    with pytest.raises(RuntimeError, match="Failed to download /logs/verifier"):
        await env._exec_download_command(
            "set -o pipefail && tar -C /tmp/terminal-bench/logs/verifier -czf - . | base64 -w0",
            source_path="/logs/verifier",
        )

    assert len(calls) == 1
