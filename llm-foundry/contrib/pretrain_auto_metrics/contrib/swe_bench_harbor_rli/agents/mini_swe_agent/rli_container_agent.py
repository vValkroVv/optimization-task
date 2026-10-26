from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import (
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from ...src.errors import RLIExecProxyConnectionLostError
from ..prediction_patch import export_prediction_patch
from ..llm_tunnel.tunnel import RLILLMTunnel


_DEFAULT_VERSION = "2.3.0"
_RUN_DIR = "/opt/llmtun/run"
_CONFIG_PATH = f"{_RUN_DIR}/mini-swe-agent.config.yaml"
_MODEL_TIMEOUT_GRACE_SEC = 180


def _pop_alias(kwargs: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in kwargs:
            return kwargs.pop(name)
    return default


def _present(value: Any) -> bool:
    return value is not None and value != ""


class RLIContainerMiniSweAgent(MiniSweAgent):
    def __init__(
        self,
        *args: Any,
        model_api_base: str | None = None,
        api_key: str | None = None,
        served_model: str | None = None,
        llm_timeout: int | str | None = None,
        max_tokens: int | str | None = None,
        temperature: float | str | None = None,
        top_p: float | str | None = None,
        command_timeout: int | str | None = None,
        reasoning_effort: str | None = None,
        config_file: str | None = None,
        version: str | None = None,
        **kwargs: Any,
    ) -> None:
        model_api_base = _pop_alias(
            kwargs,
            "model-api-base",
            "model_api_base",
            "api-base",
            "api_base",
            default=model_api_base,
        )
        api_key = _pop_alias(kwargs, "api-key", "api_key", default=api_key)
        served_model = _pop_alias(
            kwargs,
            "served-model",
            "served_model",
            default=served_model,
        )
        llm_timeout = _pop_alias(
            kwargs,
            "llm-timeout",
            "llm_timeout",
            default=llm_timeout,
        )
        max_tokens = _pop_alias(
            kwargs,
            "max-tokens",
            "max_tokens",
            default=max_tokens,
        )
        top_p = _pop_alias(kwargs, "top-p", "top_p", default=top_p)
        command_timeout = _pop_alias(
            kwargs,
            "command-timeout",
            "command_timeout",
            default=command_timeout,
        )
        reasoning_effort = _pop_alias(
            kwargs,
            "reasoning-effort",
            "reasoning_effort",
            default=reasoning_effort,
        )
        config_file = _pop_alias(
            kwargs,
            "config-file",
            "config_file",
            default=config_file,
        )

        self._model_api_base = str(model_api_base) if _present(model_api_base) else None
        self._host_api_key = str(api_key) if _present(api_key) else None
        self._served_model = str(served_model) if _present(served_model) else None
        self._llm_timeout = int(llm_timeout) if _present(llm_timeout) else 600
        self._max_tokens = int(max_tokens) if _present(max_tokens) else None
        self._temperature = float(temperature) if _present(temperature) else None
        self._top_p = float(top_p) if _present(top_p) else None
        self._command_timeout = (
            int(command_timeout) if _present(command_timeout) else None
        )
        if self._llm_timeout < 1:
            raise ValueError("llm_timeout must be positive")

        super().__init__(
            *args,
            reasoning_effort=reasoning_effort,
            config_file=config_file,
            version=version or _DEFAULT_VERSION,
            **kwargs,
        )

    @staticmethod
    def name() -> str:
        return "mini-swe-agent-rli"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                "test -x /root/.local/bin/mini-swe-agent && "
                "/root/.local/bin/mini-swe-agent --help >/dev/null"
            ),
        )

    def _augment_instruction(self, instruction: str) -> str:
        if not self.mcp_servers:
            return instruction

        mcp_info = (
            "\n\nMCP Servers:\nThe following MCP servers are available for this task.\n"
        )
        for server in self.mcp_servers:
            if server.transport == "stdio":
                args = " ".join(server.args)
                mcp_info += (
                    f"- {server.name}: stdio transport, command: "
                    f"{server.command} {args}\n"
                )
            else:
                mcp_info += (
                    f"- {server.name}: {server.transport} transport, "
                    f"url: {server.url}\n"
                )
        return instruction + mcp_info

    def _runner_env(self) -> dict[str, str]:
        env = dict(self._extra_env)
        env.update(
            {
                "MSWEA_CONFIGURED": "true",
                "MSWEA_COST_TRACKING": "ignore_errors",
                "MSWEA_SILENT_STARTUP": "1",
                "MSWEA_API_KEY": "dummy",
                "OPENAI_API_KEY": "dummy",
                "OPENAI_API_BASE": "http://127.0.0.1:8000/v1",
                "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            }
        )
        return env

    def _config_flags(self) -> str:
        flags = ["-c mini.yaml"]
        if self._config_yaml:
            flags.append(f"-c {_CONFIG_PATH}")

        assignments = [
            "model.model_kwargs.api_base=http://127.0.0.1:8000/v1",
            "model.model_kwargs.api_key=dummy",
            (
                "model.model_kwargs.timeout="
                f"{self._llm_timeout + _MODEL_TIMEOUT_GRACE_SEC}"
            ),
        ]
        if self._max_tokens is not None:
            assignments.append(f"model.model_kwargs.max_tokens={self._max_tokens}")
        if self._temperature is not None:
            assignments.append(f"model.model_kwargs.temperature={self._temperature}")
        if self._top_p is not None:
            assignments.append(f"model.model_kwargs.top_p={self._top_p}")
        if self._command_timeout is not None:
            assignments.append(f"environment.timeout={self._command_timeout}")
        if self._reasoning_effort:
            assignments.append(
                "model.model_kwargs.extra_body.reasoning_effort="
                f"{self._reasoning_effort}"
            )

        flags.extend(f"-c {shlex.quote(assignment)}" for assignment in assignments)
        return " ".join(flags)

    async def _upload_config(self, environment: BaseEnvironment) -> None:
        if not self._config_yaml:
            return

        local_config = self.logs_dir / "setup" / "mini-swe-agent.config.yaml"
        local_config.write_text(self._config_yaml, encoding="utf-8")
        await environment.upload_file(local_config, _CONFIG_PATH)

    async def _start_agent(
        self,
        instruction: str,
        environment: BaseEnvironment,
    ) -> None:
        if not self.model_name:
            raise ValueError("No LLM model specified")

        await self._upload_config(environment)
        cli_flags = self.build_cli_flags()
        mini_command = " ".join(
            part
            for part in (
                "/root/.local/bin/mini-swe-agent",
                "--yolo",
                f"--model={shlex.quote(self.model_name)}",
                f"--task={shlex.quote(instruction)}",
                f"--output={shlex.quote(str(self._mini_swe_agent_trajectory_path))}",
                cli_flags,
                self._config_flags(),
                "--exit-immediately",
            )
            if part
        )
        agent_script = f"""
printf '%s\\n' "$$" > {_RUN_DIR}/agent.pid.tmp
mv -f {_RUN_DIR}/agent.pid.tmp {_RUN_DIR}/agent.pid
set +e
set -o pipefail
if [ -f /etc/profile.d/testbed-conda.sh ]; then
    . /etc/profile.d/testbed-conda.sh
fi
cd /testbed
{mini_command} 2>&1 | stdbuf -oL tee /logs/agent/mini-swe-agent.txt
rc=${{PIPESTATUS[0]}}
printf '%s\\n' "$rc" > {_RUN_DIR}/agent.exit.tmp
mv -f {_RUN_DIR}/agent.exit.tmp {_RUN_DIR}/agent.exit
exit "$rc"
""".strip()
        command = (
            f"setsid bash -c {shlex.quote(agent_script)} "
            "</dev/null >/dev/null 2>&1 & "
            "for attempt in $(seq 1 50); do "
            f"[ -s {_RUN_DIR}/agent.pid ] && break; "
            "sleep 0.1; "
            "done; "
            f"test -s {_RUN_DIR}/agent.pid"
        )
        result = await environment.exec(
            command,
            env=self._runner_env(),
            user="root",
        )
        output = "\n".join(value for value in (result.stdout, result.stderr) if value)
        if result.return_code == -1 or "exec-proxy connection lost" in output.lower():
            raise RLIExecProxyConnectionLostError(
                "RLI exec-proxy connection lost while starting mini-swe-agent: "
                f"{output or 'No output'}"
            )
        if result.return_code != 0:
            raise RuntimeError(
                "Failed to start mini-swe-agent: "
                f"return_code={result.return_code}, output={output!r}"
            )

    async def _shutdown_agent(self, environment: BaseEnvironment) -> None:
        try:
            await environment.exec(
                (
                    f"if [ -s {_RUN_DIR}/agent.pid ]; then "
                    f"pid=$(cat {_RUN_DIR}/agent.pid); "
                    'if kill -0 -- "-$pid" 2>/dev/null; then '
                    'kill -TERM -- "-$pid" 2>/dev/null || true; '
                    "for attempt in $(seq 1 20); do "
                    'kill -0 -- "-$pid" 2>/dev/null || break; '
                    "sleep 0.1; "
                    "done; "
                    'kill -KILL -- "-$pid" 2>/dev/null || true; '
                    "fi; "
                    "fi; "
                    f"rm -f {_RUN_DIR}/agent.pid"
                ),
                user="root",
                timeout_sec=10,
            )
        except Exception as exc:
            self.logger.debug(
                "Failed to stop mini-swe-agent process group: %s",
                exc,
            )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        llm_base_url = (
            self._model_api_base
            or self._get_env("OPENAI_API_BASE")
            or self._get_env("OPENAI_BASE_URL")
        )
        if not llm_base_url:
            raise ValueError("No host-side LLM endpoint specified")
        api_key = self._host_api_key or self._get_env("OPENAI_API_KEY")

        tunnel = RLILLMTunnel(
            environment,
            bridge_path=(
                Path(__file__).parent.parent / "llm_tunnel" / "bridge.py"
            ),
            llm_base_url=llm_base_url,
            api_key=api_key,
            served_model=self._served_model,
            llm_timeout=self._llm_timeout,
        )
        tunnel_installed = False
        try:
            await tunnel.install()
            tunnel_installed = True
            await self._start_agent(
                self._augment_instruction(instruction),
                environment,
            )
            exit_code = await tunnel.run()
            if exit_code != 0:
                raise NonZeroAgentExitCodeError(
                    f"mini-swe-agent exited with code {exit_code}"
                )
        finally:
            try:
                if tunnel_installed:
                    await self._shutdown_agent(environment)
            finally:
                try:
                    if tunnel_installed:
                        await tunnel.shutdown()
                finally:
                    await export_prediction_patch(
                        environment,
                        self.logs_dir / "prediction.patch",
                        self.logger,
                    )
