from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import platform
import queue
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from litellm.exceptions import (
    APIConnectionError as LiteLLMAPIConnectionError,
)
from litellm.exceptions import BadGatewayError as LiteLLMBadGatewayError
from litellm.exceptions import (
    ServiceUnavailableError as LiteLLMServiceUnavailableError,
)
from minisweagent import __version__ as mini_swe_agent_version
from minisweagent.agents import get_agent
from minisweagent.config import get_config_from_spec
from minisweagent.exceptions import Submitted
from minisweagent.models import get_model
from minisweagent.utils.serialize import UNSET, recursive_merge
from pydantic import BaseModel, Field

from ...src.errors import EndpointUnavailableError, RLIExecProxyConnectionLostError
from ...src.token_budget import account_output_tokens
from ..prediction_patch import export_prediction_patch


_PROCESS_POLL_INTERVAL_SEC = 0.05
_PROCESS_SHUTDOWN_TIMEOUT_SEC = 5.0


class RLIMiniEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    timeout: int = 600


class RLIMiniEnvironment:
    def __init__(
        self,
        environment: BaseEnvironment,
        *,
        loop: asyncio.AbstractEventLoop,
        config: RLIMiniEnvironmentConfig | None = None,
    ) -> None:
        self._environment = environment
        self._loop = loop
        self.config = config or RLIMiniEnvironmentConfig()

    def execute(
        self,
        action: dict[str, Any],
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        command = action.get("command", "")
        effective_cwd = cwd or self.config.cwd or ""
        effective_timeout = timeout or self.config.timeout
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._environment.exec(
                    command,
                    cwd=effective_cwd or None,
                    env=self.config.env,
                    timeout_sec=effective_timeout,
                ),
                self._loop,
            )
            result = future.result()
            payload = _exec_result_to_payload(result)
        except Exception as exc:
            payload = _exception_to_payload(exc)
        _check_finished(payload)
        return payload

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return recursive_merge(
            self.config.model_dump(),
            platform.uname()._asdict(),
            dict(os.environ),
            kwargs,
        )

    def serialize(self) -> dict[str, Any]:
        return _serialize_environment(self.config, self.__class__)


class _RLIMiniProcessEnvironment:
    def __init__(
        self,
        *,
        request_queue: Any,
        response_queue: Any,
        config: RLIMiniEnvironmentConfig | None = None,
    ) -> None:
        self._request_queue = request_queue
        self._response_queue = response_queue
        self.config = config or RLIMiniEnvironmentConfig()

    def execute(
        self,
        action: dict[str, Any],
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        self._request_queue.put(
            {
                "type": "exec",
                "id": request_id,
                "action": action,
                "cwd": cwd,
                "timeout": timeout,
            }
        )
        while True:
            response = self._response_queue.get()
            if response.get("id") != request_id:
                continue
            if response.get("ok") is False:
                raise RuntimeError(response.get("error") or "agent IPC failed")
            payload = response["result"]
            _check_finished(payload)
            return payload

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return recursive_merge(
            self.config.model_dump(),
            platform.uname()._asdict(),
            dict(os.environ),
            kwargs,
        )

    def serialize(self) -> dict[str, Any]:
        return _serialize_environment(self.config, self.__class__)


def _exec_result_to_payload(result: Any) -> dict[str, Any]:
    output = result.stdout or ""
    if result.stderr:
        if output and not output.endswith("\n"):
            output = f"{output}\n"
        output = f"{output}{result.stderr}"
    return {
        "output": output,
        "returncode": result.return_code,
        "exception_info": "",
    }


def _exception_to_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "output": "",
        "returncode": -1,
        "exception_info": f"An error occurred while executing the command: {exc}",
        "extra": {
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        },
    }


def _check_finished(output: dict[str, Any]) -> None:
    lines = output.get("output", "").lstrip().splitlines(keepends=True)
    if (
        lines
        and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        and output["returncode"] == 0
    ):
        submission = "".join(lines[1:])
        raise Submitted(
            {
                "role": "exit",
                "content": submission,
                "extra": {
                    "exit_status": "Submitted",
                    "submission": submission,
                },
            }
        )


def _serialize_environment(
    config: RLIMiniEnvironmentConfig,
    environment_type: type,
) -> dict[str, Any]:
    return {
        "info": {
            "config": {
                "environment": config.model_dump(mode="json"),
                "environment_type": f"{environment_type.__module__}.{environment_type.__name__}",
            }
        }
    }


@contextmanager
def _patched_environ(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _pop_alias(kwargs: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in kwargs:
            return kwargs.pop(name)
    return default


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _as_string_dict(value: Any) -> dict[str, str]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("extra_env must be a dict or JSON object string")
    return {str(key): str(item) for key, item in value.items()}


def _run_mini_swe_agent_child(
    *,
    instruction: str,
    model_name: str,
    mini_config: dict[str, Any],
    env_config: dict[str, Any],
    env: dict[str, str],
    child_log_path: str,
    request_queue: Any,
    response_queue: Any,
    status_queue: Any,
) -> None:
    child_log = Path(child_log_path)
    child_log.parent.mkdir(parents=True, exist_ok=True)

    def log_child(message: str) -> None:
        with child_log.open("a", encoding="utf-8", errors="replace") as file:
            file.write(message.rstrip() + "\n")

    try:
        log_child("mini-swe-agent child starting")
        log_child(f"model_name={model_name}")
        log_child(
            "model_class="
            f"{((mini_config.get('model') or {}).get('model_class'))!r}"
        )
        log_child(
            "agent_output_path="
            f"{((mini_config.get('agent') or {}).get('output_path'))!r}"
        )
        config = dict(mini_config)
        config["model"] = dict(config.get("model") or {})
        config["model"]["model_name"] = model_name
        adapter = _RLIMiniProcessEnvironment(
            request_queue=request_queue,
            response_queue=response_queue,
            config=RLIMiniEnvironmentConfig.model_validate(env_config),
        )
        with _patched_environ(env):
            model = get_model(config=config.get("model", {}))
            agent = get_agent(
                model,
                adapter,
                config.get("agent", {}),
                default_type="default",
            )
            agent.run(instruction)
        status_queue.put({"status": "ok"})
        log_child("mini-swe-agent child finished ok")
    except BaseException as exc:
        tb = traceback.format_exc()
        log_child(
            "mini-swe-agent child failed: "
            f"{type(exc).__name__}: {exc}\n{tb}"
        )
        status_queue.put(
            {
                "status": "error",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": tb,
                "endpoint_unavailable": isinstance(
                    exc,
                    (
                        LiteLLMAPIConnectionError,
                        LiteLLMBadGatewayError,
                        LiteLLMServiceUnavailableError,
                    ),
                ),
            }
        )


class RLIMiniSweAgent(BaseAgent):
    SUPPORTS_ATIF = True

    def __init__(
        self,
        *args: Any,
        reasoning_effort: str | None = None,
        config_file: str | None = None,
        cost_limit: str | float | None = "0",
        model_class: str | None = None,
        step_limit: int | str | None = None,
        wall_time_limit_seconds: int | str | None = None,
        command_timeout: int | str | None = None,
        model_api_base: str | None = None,
        api_key: str | None = None,
        max_tokens: int | str | None = None,
        llm_timeout: int | str | None = None,
        temperature: float | str | None = None,
        top_p: float | str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
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
        cost_limit = _pop_alias(
            kwargs,
            "cost-limit",
            "cost_limit",
            default=cost_limit,
        )
        model_class = _pop_alias(
            kwargs,
            "model-class",
            "model_class",
            default=model_class,
        )
        step_limit = _pop_alias(
            kwargs,
            "step-limit",
            "step_limit",
            default=step_limit,
        )
        wall_time_limit_seconds = _pop_alias(
            kwargs,
            "wall-time-limit-seconds",
            "wall_time_limit_seconds",
            default=wall_time_limit_seconds,
        )
        command_timeout = _pop_alias(
            kwargs,
            "command-timeout",
            "command_timeout",
            default=command_timeout,
        )
        model_api_base = _pop_alias(
            kwargs,
            "model-api-base",
            "model_api_base",
            "api-base",
            "api_base",
            default=model_api_base,
        )
        api_key = _pop_alias(
            kwargs,
            "api-key",
            "api_key",
            default=api_key,
        )
        max_tokens = _pop_alias(
            kwargs,
            "max-tokens",
            "max_tokens",
            default=max_tokens,
        )
        llm_timeout = _pop_alias(
            kwargs,
            "llm-timeout",
            default=llm_timeout,
        )
        temperature = _pop_alias(
            kwargs,
            "temperature",
            default=temperature,
        )
        top_p = _pop_alias(
            kwargs,
            "top-p",
            "top_p",
            default=top_p,
        )
        extra_env = _pop_alias(
            kwargs,
            "extra-env",
            "extra_env",
            default=extra_env,
        )

        super().__init__(*args, **kwargs)
        self.reasoning_effort = reasoning_effort
        self.config_file = config_file
        self.cost_limit = _as_optional_float(cost_limit)
        self.model_class = model_class
        self.step_limit = _as_optional_int(step_limit)
        self.wall_time_limit_seconds = _as_optional_int(wall_time_limit_seconds)
        self.command_timeout = _as_optional_int(command_timeout)
        self.model_api_base = (
            str(model_api_base) if model_api_base not in (None, "") else None
        )
        self.api_key = str(api_key) if api_key not in (None, "") else None
        self.max_tokens = _as_optional_int(max_tokens)
        self.llm_timeout = _as_optional_int(llm_timeout)
        self.temperature = _as_optional_float(temperature)
        self.top_p = _as_optional_float(top_p)
        self.extra_env = _as_string_dict(extra_env)

    @staticmethod
    def name() -> str:
        return "mini-swe-agent"

    def version(self) -> str | None:
        return mini_swe_agent_version

    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _mini_swe_agent_trajectory_path(self) -> Path:
        return self.logs_dir / "mini-swe-agent.trajectory.json"

    @property
    def _atif_trajectory_path(self) -> Path:
        return self.logs_dir / "trajectory.json"

    @property
    def _child_log_path(self) -> Path:
        return self.logs_dir / "mini-swe-agent.child.log"

    def _build_mini_config(self) -> dict[str, Any]:
        configs = [get_config_from_spec("mini.yaml")]
        if self.config_file:
            configs.append(get_config_from_spec(self.config_file))

        model_kwargs: dict[str, Any] = {}
        if self.model_api_base is not None:
            model_kwargs["api_base"] = self.model_api_base
        if self.api_key is not None:
            model_kwargs["api_key"] = self.api_key
        if self.max_tokens is not None:
            model_kwargs["max_tokens"] = self.max_tokens
        if self.llm_timeout is not None:
            model_kwargs["timeout"] = self.llm_timeout
        if self.temperature is not None:
            model_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            model_kwargs["top_p"] = self.top_p
        if self.reasoning_effort:
            model_kwargs.setdefault("extra_body", {})[
                "reasoning_effort"
            ] = self.reasoning_effort

        model_config: dict[str, Any] = {
            "model_name": self.model_name or UNSET,
            "model_class": self.model_class or UNSET,
        }
        if model_kwargs:
            model_config["model_kwargs"] = model_kwargs

        configs.append(
            {
                "agent": {
                    "mode": "yolo",
                    "confirm_exit": False,
                    "cost_limit": self.cost_limit
                    if self.cost_limit is not None
                    else UNSET,
                    "output_path": str(self._mini_swe_agent_trajectory_path),
                    "step_limit": self.step_limit
                    if self.step_limit is not None
                    else UNSET,
                    "wall_time_limit_seconds": self.wall_time_limit_seconds
                    if self.wall_time_limit_seconds is not None
                    else UNSET,
                },
                "model": model_config,
                "environment": {
                    "timeout": self.command_timeout
                    if self.command_timeout is not None
                    else UNSET,
                },
            }
        )
        return recursive_merge(*configs)

    def _populate_context_from_trajectory(self, context: AgentContext) -> None:
        mini_path = self._mini_swe_agent_trajectory_path
        if not mini_path.exists():
            self.logger.debug(
                "Mini-swe-agent trajectory file %s does not exist",
                mini_path,
            )
            return

        try:
            mini_trajectory = json.loads(mini_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.debug("Failed to load mini-swe-agent trajectory: %s", exc)
            return

        n_input_tokens = 0
        n_output_tokens = 0
        n_cache_tokens = 0
        total_cost = (
            ((mini_trajectory.get("info") or {}).get("model_stats") or {}).get(
                "instance_cost"
            )
            or 0
        )
        for message in mini_trajectory.get("messages") or []:
            usage = (
                ((message.get("extra") or {}).get("response") or {}).get("usage")
                or {}
            )
            prompt_tokens_details = usage.get("prompt_tokens_details") or {}
            n_cache_tokens += prompt_tokens_details.get("cached_tokens") or 0
            n_input_tokens += usage.get("prompt_tokens") or 0
            n_output_tokens += usage.get("completion_tokens") or 0

        context.n_input_tokens = n_input_tokens
        context.n_output_tokens = n_output_tokens
        context.n_cache_tokens = n_cache_tokens
        context.cost_usd = total_cost

        try:
            from harbor.agents.installed.mini_swe_agent import (
                convert_and_save_trajectory,
            )

            convert_and_save_trajectory(
                mini_swe_agent_trajectory_path=mini_path,
                atif_trajectory_path=self._atif_trajectory_path,
                session_id=str(uuid.uuid4()),
            )
        except Exception as exc:
            self.logger.debug("Failed to convert trajectory to ATIF format: %s", exc)

    def populate_context_post_run(self, context: AgentContext) -> None:
        self._populate_context_from_trajectory(context)

    def _get_process_context(self) -> Any:
        return mp.get_context("spawn")

    async def _execute_process_request(
        self,
        *,
        environment: BaseEnvironment,
        env_config: RLIMiniEnvironmentConfig,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        action = request.get("action") or {}
        command = action.get("command", "")
        cwd = request.get("cwd") or env_config.cwd or ""
        timeout = request.get("timeout") or env_config.timeout
        result = await environment.exec(
            command,
            cwd=cwd or None,
            env=env_config.env,
            timeout_sec=timeout,
        )
        error = "\n".join(
            value for value in (result.stdout, result.stderr) if value
        )
        if (
            result.return_code == -1
            or "exec-proxy connection lost" in error.lower()
        ):
            raise RLIExecProxyConnectionLostError(
                "RLI exec-proxy connection lost while executing an agent "
                f"command: {error or 'No output'}"
            )
        return _exec_result_to_payload(result)

    async def _drain_process_requests(
        self,
        *,
        request_queue: Any,
        response_queue: Any,
        environment: BaseEnvironment,
        env_config: RLIMiniEnvironmentConfig,
    ) -> None:
        while True:
            try:
                request = request_queue.get_nowait()
            except queue.Empty:
                return
            if request.get("type") != "exec":
                response_queue.put(
                    {
                        "id": request.get("id"),
                        "ok": False,
                        "error": f"Unsupported request type: {request.get('type')}",
                    }
                )
                continue
            payload = await self._execute_process_request(
                environment=environment,
                env_config=env_config,
                request=request,
            )
            response_queue.put(
                {
                    "id": request.get("id"),
                    "ok": True,
                    "result": payload,
                }
            )

    @staticmethod
    def _terminate_process(process: Any) -> None:
        if not process.is_alive():
            process.join(timeout=0)
            return
        process.terminate()
        process.join(timeout=_PROCESS_SHUTDOWN_TIMEOUT_SEC)
        if process.is_alive():
            process.kill()
            process.join(timeout=_PROCESS_SHUTDOWN_TIMEOUT_SEC)

    @staticmethod
    def _collect_statuses(status_queue: Any, *, wait_for_first: bool = False) -> list[dict[str, Any]]:
        statuses: list[dict[str, Any]] = []
        if wait_for_first:
            try:
                statuses.append(status_queue.get(timeout=1.0))
            except queue.Empty:
                pass
        while True:
            try:
                statuses.append(status_queue.get_nowait())
            except queue.Empty:
                return statuses

    async def _wait_for_agent_process(
        self,
        *,
        process: Any,
        request_queue: Any,
        response_queue: Any,
        status_queue: Any,
        environment: BaseEnvironment,
        env_config: RLIMiniEnvironmentConfig,
    ) -> None:
        try:
            while process.is_alive():
                await self._drain_process_requests(
                    request_queue=request_queue,
                    response_queue=response_queue,
                    environment=environment,
                    env_config=env_config,
                )
                await asyncio.sleep(_PROCESS_POLL_INTERVAL_SEC)
            await self._drain_process_requests(
                request_queue=request_queue,
                response_queue=response_queue,
                environment=environment,
                env_config=env_config,
            )
        except asyncio.CancelledError:
            self._terminate_process(process)
            raise
        except BaseException:
            self._terminate_process(process)
            raise
        finally:
            if not process.is_alive():
                process.join(timeout=0)

        statuses = self._collect_statuses(
            status_queue,
            wait_for_first=process.exitcode == 0,
        )
        for status in statuses:
            if status.get("status") == "error":
                if status.get("endpoint_unavailable"):
                    raise EndpointUnavailableError(
                        "Model endpoint is unavailable: "
                        f"{status.get('exception_type')}: {status.get('message')}"
                    )
                raise RuntimeError(
                    "mini-swe-agent child process failed: "
                    f"{status.get('exception_type')}: {status.get('message')}\n"
                    f"{status.get('traceback') or ''}"
                )
        if process.exitcode not in (0, None):
            child_log_tail = ""
            child_log = self._child_log_path
            if child_log.is_file():
                child_log_tail = child_log.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[-4000:]
            raise RuntimeError(
                f"mini-swe-agent child process exited with code {process.exitcode}"
                + (
                    f"\nmini-swe-agent child log tail:\n{child_log_tail}"
                    if child_log_tail
                    else ""
                )
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        config = self._build_mini_config()
        env_config = RLIMiniEnvironmentConfig.model_validate(
            config.get("environment") or {}
        )
        env = {
            "MSWEA_CONFIGURED": "true",
            "MSWEA_COST_TRACKING": "ignore_errors",
            "MSWEA_SILENT_STARTUP": "1",
        }
        if self.model_api_base is not None:
            env["OPENAI_API_BASE"] = self.model_api_base
            env["OPENAI_BASE_URL"] = self.model_api_base
        if self.api_key is not None:
            env["OPENAI_API_KEY"] = self.api_key
            env["MSWEA_API_KEY"] = self.api_key
        env.update(self.extra_env)

        process_context = self._get_process_context()
        request_queue = process_context.Queue()
        response_queue = process_context.Queue()
        status_queue = process_context.Queue()
        process = process_context.Process(
            target=_run_mini_swe_agent_child,
            kwargs={
                "instruction": instruction,
                "model_name": self.model_name,
                "mini_config": config,
                "env_config": env_config.model_dump(mode="json"),
                "env": env,
                "child_log_path": str(self._child_log_path),
                "request_queue": request_queue,
                "response_queue": response_queue,
                "status_queue": status_queue,
            },
        )
        process.start()

        try:
            await self._wait_for_agent_process(
                process=process,
                request_queue=request_queue,
                response_queue=response_queue,
                status_queue=status_queue,
                environment=environment,
                env_config=env_config,
            )
        finally:
            await export_prediction_patch(
                environment,
                self.logs_dir / "prediction.patch",
                self.logger,
                env=env_config.env,
            )
            self._populate_context_from_trajectory(context)
            account_output_tokens(
                getattr(context, "n_output_tokens", None)
            )
