from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import (
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
from harbor.agents.installed.openhands_sdk import OpenHandsSDK
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from ...src.errors import RLIExecProxyConnectionLostError
from ..llm_tunnel.tunnel import RLILLMTunnel
from ..prediction_patch import export_prediction_patch


_DEFAULT_VERSION = "1.36.0"
_RUN_DIR = "/opt/llmtun/run"
_RUNNER_PATH = "/installed-agent/run_agent.py"
_INSTALL_TIMEOUT_SEC = 900
_RUNNER_TIMEOUT_GRACE_SEC = 180


def _pop_alias(kwargs: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in kwargs:
            return kwargs.pop(name)
    return default


def _present(value: Any) -> bool:
    return value is not None and value != ""


class RLIOpenHandsSDKAgent(OpenHandsSDK):
    def __init__(
        self,
        *args: Any,
        model_api_base: str | None = None,
        api_key: str | None = None,
        served_model: str | None = None,
        llm_timeout: int | str | None = None,
        reasoning_effort: str | None = "high",
        max_iterations: int | str | None = None,
        temperature: float | str | None = None,
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
        reasoning_effort = _pop_alias(
            kwargs,
            "reasoning-effort",
            "reasoning_effort",
            default=reasoning_effort,
        )
        max_iterations = _pop_alias(
            kwargs,
            "max-iterations",
            "max_iterations",
            default=max_iterations,
        )

        self._model_api_base = str(model_api_base) if _present(model_api_base) else None
        self._host_api_key = str(api_key) if _present(api_key) else None
        self._served_model = str(served_model) if _present(served_model) else None
        self._llm_timeout = int(llm_timeout) if _present(llm_timeout) else 600
        if self._llm_timeout < 1:
            raise ValueError("llm_timeout must be positive")

        super().__init__(
            *args,
            reasoning_effort=reasoning_effort,
            max_iterations=(int(max_iterations) if _present(max_iterations) else None),
            temperature=float(temperature) if _present(temperature) else None,
            version=version or _DEFAULT_VERSION,
            **kwargs,
        )

    async def install(self, environment: BaseEnvironment) -> None:
        check_result = await environment.exec(
            command="[ -f /opt/openhands-sdk-venv/bin/python ] && "
            '/opt/openhands-sdk-venv/bin/python -c "import openhands.sdk" '
            "2>/dev/null",
        )
        if check_result.return_code != 0:
            version_spec = f"=={self._version}" if self._version else ""
            await self.exec_as_agent(
                environment,
                command=(
                    "set -euo pipefail; "
                    "uv venv --clear --python 3.12 "
                    "/opt/openhands-sdk-venv && "
                    "uv pip install "
                    "--python /opt/openhands-sdk-venv/bin/python "
                    f"openhands-sdk{version_spec} "
                    f"openhands-tools{version_spec} fastapi"
                ),
                timeout_sec=_INSTALL_TIMEOUT_SEC,
            )

        freeze_result = await self.exec_as_agent(
            environment,
            command=("uv pip freeze --python /opt/openhands-sdk-venv/bin/python"),
        )
        (self.logs_dir / "setup" / "pip-freeze.txt").write_text(
            freeze_result.stdout or "",
            encoding="utf-8",
        )
        runner_path = Path(__file__).with_name("openhands_sdk_runner.py")
        await environment.upload_file(runner_path, _RUNNER_PATH)
        await self.exec_as_root(
            environment,
            command=f"chmod 0755 {shlex.quote(_RUNNER_PATH)}",
        )

    def _runner_env(self) -> dict[str, str]:
        if not self.model_name:
            raise ValueError("No LLM model specified")

        env = dict(self._extra_env)
        env.update(
            {
                "DEBUG": "false",
                "LOG_LEVEL": "INFO",
                "LLM_API_KEY": "dummy",
                "LLM_BASE_URL": "http://127.0.0.1:8000/v1",
                "LLM_MODEL": self.model_name,
                "LLM_TIMEOUT": str(self._llm_timeout + _RUNNER_TIMEOUT_GRACE_SEC),
                "AGENT_LOGS_DIR": "/logs/agent",
                "TRAJECTORY_PATH": f"/logs/agent/{self._TRAJECTORY_FILENAME}",
                "LOAD_SKILLS": "1" if self._load_skills else "0",
                "SKILL_PATHS": ":".join(self._skill_paths),
            }
        )

        if self._reasoning_effort:
            env["LLM_REASONING_EFFORT"] = self._reasoning_effort
        if self._collect_token_ids:
            env["LITELLM_EXTRA_BODY"] = json.dumps({"return_token_ids": True})
        if self._max_iterations is not None:
            env["MAX_ITERATIONS"] = str(self._max_iterations)
        if self._temperature is not None:
            env["LLM_TEMPERATURE"] = str(self._temperature)

        if self.mcp_servers:
            mcp_list: list[dict[str, str | list[str]]] = []
            for server in self.mcp_servers:
                entry: dict[str, str | list[str]] = {
                    "name": server.name,
                    "transport": server.transport,
                }
                if server.transport == "stdio":
                    if server.command:
                        entry["command"] = server.command
                    if server.args:
                        entry["args"] = server.args
                elif server.url:
                    entry["url"] = server.url
                mcp_list.append(entry)
            env["MCP_SERVERS_JSON"] = json.dumps(mcp_list)

        return env

    async def _start_agent(
        self,
        instruction: str,
        environment: BaseEnvironment,
    ) -> None:
        agent_script = f"""
printf '%s\\n' "$$" > {_RUN_DIR}/agent.pid.tmp
mv -f {_RUN_DIR}/agent.pid.tmp {_RUN_DIR}/agent.pid
set +e
set -o pipefail
/opt/openhands-sdk-venv/bin/python {_RUNNER_PATH} \
    --instruction={shlex.quote(instruction)} \
    --logs-dir="$AGENT_LOGS_DIR" \
    --trajectory-path="$TRAJECTORY_PATH" \
    2>&1 | stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}
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
        result = await environment.exec(command, env=self._runner_env())
        output = "\n".join(value for value in (result.stdout, result.stderr) if value)
        if result.return_code == -1 or "exec-proxy connection lost" in output.lower():
            raise RLIExecProxyConnectionLostError(
                "RLI exec-proxy connection lost while starting OpenHands: "
                f"{output or 'No output'}"
            )
        if result.return_code != 0:
            raise RuntimeError(
                "Failed to start OpenHands: "
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
            self.logger.debug("Failed to stop OpenHands process group: %s", exc)

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
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
            bridge_path=Path(__file__).parent.parent / "llm_tunnel" / "bridge.py",
            llm_base_url=llm_base_url,
            api_key=api_key,
            served_model=self._served_model,
            llm_timeout=self._llm_timeout,
        )
        tunnel_installed = False
        try:
            await tunnel.install()
            tunnel_installed = True
            await self._start_agent(instruction, environment)
            exit_code = await tunnel.run()
            if exit_code != 0:
                raise NonZeroAgentExitCodeError(
                    f"OpenHands exited with code {exit_code}"
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
