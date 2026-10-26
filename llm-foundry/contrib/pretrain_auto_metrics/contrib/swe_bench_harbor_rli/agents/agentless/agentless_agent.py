from __future__ import annotations

import base64
import json
import re
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from ...src.errors import EndpointUnavailableError
from ...src.token_budget import account_output_tokens

try:
    import openai as _openai
    _openai_version = getattr(_openai, "__version__", None)
except Exception:  # pragma: no cover
    _openai = None
    _openai_version = None


_DEFAULT_COMMAND_TIMEOUT_SEC = 600
_DEFAULT_REMOTE_PATCH_PATH = "/tmp/agentless_model.patch"


def _pop_alias(kwargs: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in kwargs:
            return kwargs.pop(name)
    return default


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _as_optional_int(value: Any) -> int | None:
    return int(value) if _present(value) else None


def _as_optional_float(value: Any) -> float | None:
    return float(value) if _present(value) else None


def _as_string_dict(value: Any) -> dict[str, str]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("extra_env must be a dict or JSON object string")
    return {str(key): str(item) for key, item in value.items()}


def _exec_result_to_payload(result: Any) -> dict[str, Any]:
    output = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    if stderr:
        if output and not output.endswith("\n"):
            output = f"{output}\n"
        output = f"{output}{stderr}"
    return {"output": output, "returncode": getattr(result, "return_code", None)}


def _extract_patch(text: str) -> str:
    """Pull a unified diff out of an LLM response (fenced or bare)."""
    if not text:
        return ""
    fenced = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", text, re.S)
    if fenced:
        candidate = fenced.group(1)
        if "diff --git" in candidate or "--- " in candidate or "+++ " in candidate:
            return candidate.strip("\n") + "\n"
    idx = text.find("diff --git")
    if idx == -1:
        idx = text.find("--- ")
    if idx != -1:
        return text[idx:].strip("\n") + "\n"
    return ""


class RLIAgentlessAgent(BaseAgent):
    """Procedural Agentless SWE harness: localize -> repair -> validate over RLI."""

    SUPPORTS_ATIF = False

    def __init__(
        self,
        *args: Any,
        model_api_base: str | None = None,
        api_key: str | None = None,
        max_tokens: int | str | None = None,
        llm_timeout: int | str | None = None,
        temperature: float | str | None = None,
        top_p: float | str | None = None,
        reasoning_effort: str | None = None,
        command_timeout: int | str | None = None,
        working_dir: str | None = None,
        remote_patch_path: str | None = None,
        context_command: str | None = None,
        apply_command: str | None = None,
        validate_command: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        model_api_base = _pop_alias(
            kwargs, "model-api-base", "model_api_base", "api-base", "api_base",
            default=model_api_base,
        )
        api_key = _pop_alias(kwargs, "api-key", "api_key", default=api_key)
        max_tokens = _pop_alias(kwargs, "max-tokens", "max_tokens", default=max_tokens)
        llm_timeout = _pop_alias(
            kwargs, "llm-timeout", default=llm_timeout
        )
        temperature = _pop_alias(kwargs, "temperature", default=temperature)
        top_p = _pop_alias(kwargs, "top-p", "top_p", default=top_p)
        reasoning_effort = _pop_alias(
            kwargs, "reasoning-effort", "reasoning_effort", default=reasoning_effort
        )
        command_timeout = _pop_alias(
            kwargs, "command-timeout", "command_timeout", default=command_timeout
        )
        working_dir = _pop_alias(
            kwargs, "working-dir", "working_dir", "cwd", default=working_dir
        )
        remote_patch_path = _pop_alias(
            kwargs, "remote-patch-path", "remote_patch_path", default=remote_patch_path
        )
        context_command = _pop_alias(
            kwargs, "context-command", "context_command", default=context_command
        )
        apply_command = _pop_alias(
            kwargs, "apply-command", "apply_command", default=apply_command
        )
        validate_command = _pop_alias(
            kwargs, "validate-command", "validate_command", default=validate_command
        )
        extra_env = _pop_alias(kwargs, "extra-env", "extra_env", default=extra_env)

        super().__init__(*args, **kwargs)

        self.model_api_base = str(model_api_base) if _present(model_api_base) else None
        self.api_key = str(api_key) if _present(api_key) else None
        self.max_tokens = _as_optional_int(max_tokens)
        self.llm_timeout = _as_optional_int(llm_timeout)
        self.temperature = _as_optional_float(temperature)
        self.top_p = _as_optional_float(top_p)
        self.reasoning_effort = str(reasoning_effort) if _present(reasoning_effort) else None
        self.command_timeout = (
            _as_optional_int(command_timeout) or _DEFAULT_COMMAND_TIMEOUT_SEC
        )
        self.working_dir = str(working_dir) if _present(working_dir) else None
        self.remote_patch_path = (
            str(remote_patch_path) if _present(remote_patch_path) else _DEFAULT_REMOTE_PATCH_PATH
        )
        self.context_command = (
            str(context_command) if _present(context_command) else "git ls-files | head -n 200"
        )
        self.apply_command = str(apply_command) if _present(apply_command) else None
        self.validate_command = str(validate_command) if _present(validate_command) else None
        self.extra_env = _as_string_dict(extra_env)
        self._usage = {"input": 0, "output": 0, "cache": 0}

    @staticmethod
    def name() -> str:
        return "agentless"

    def version(self) -> str | None:
        return _openai_version

    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _trajectory_path(self) -> Path:
        return self.logs_dir / "agentless.trajectory.json"

    def _model_id(self) -> str:
        name = self.model_name or ""
        return name[len("openai/"):] if name.startswith("openai/") else name

    def _env(self) -> dict[str, str]:
        env = dict(self.extra_env)
        if self.model_api_base is not None:
            env["OPENAI_API_BASE"] = self.model_api_base
            env["OPENAI_BASE_URL"] = self.model_api_base
        if self.api_key is not None:
            env["OPENAI_API_KEY"] = self.api_key
        return env

    def _client(self) -> Any:  # pragma: no cover
        if _openai is None:
            raise RuntimeError("openai package is not available in this environment")
        kwargs: dict[str, Any] = {
            "base_url": self.model_api_base,
            "api_key": self.api_key or "dummy",
        }
        if self.llm_timeout is not None:
            kwargs["timeout"] = self.llm_timeout
        return _openai.AsyncOpenAI(**kwargs)

    async def _chat_completion(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, int]]:
        """Single LLM seam (overridden in unit tests). Returns (content, usage)."""
        kwargs: dict[str, Any] = {"model": self._model_id(), "messages": messages}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        try:
            async with self._client() as client:  # pragma: no cover
                response = await client.chat.completions.create(**kwargs)
        except _openai.APITimeoutError:
            raise
        except (
            _openai.APIConnectionError,
            _openai.InternalServerError,
        ) as exc:
            if (
                isinstance(exc, _openai.InternalServerError)
                and getattr(exc, "status_code", None) not in (502, 503)
            ):
                raise
            raise EndpointUnavailableError(
                f"Model endpoint is unavailable: {exc}"
            ) from exc
        content = response.choices[0].message.content or ""
        usage_obj = getattr(response, "usage", None)
        usage = {
            "input": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "output": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            "cache": 0,
        }
        return content, usage

    def _accumulate(self, usage: dict[str, int]) -> None:
        self._usage["input"] += int(usage.get("input", 0) or 0)
        self._usage["output"] += int(usage.get("output", 0) or 0)
        self._usage["cache"] += int(usage.get("cache", 0) or 0)

    async def _collect_repo_context(self, environment: BaseEnvironment, env: dict[str, str]) -> str:
        result = await environment.exec(
            self.context_command,
            cwd=self.working_dir or None,
            env=env,
            timeout_sec=self.command_timeout,
        )
        return _exec_result_to_payload(result).get("output", "")

    def _write_patch_command(self, patch_path: str, patch_text: str) -> str:
        encoded = base64.b64encode(patch_text.encode("utf-8")).decode("ascii")
        parent = str(Path(patch_path).parent)
        return (
            f"mkdir -p {shlex.quote(parent)} && "
            f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(patch_path)}"
        )

    def _apply_patch_command(self, patch_path: str) -> str:
        if self.apply_command:
            return self.apply_command.replace("{patch}", shlex.quote(patch_path))
        quoted = shlex.quote(patch_path)
        return f"git apply --whitespace=nowarn {quoted} || patch -p1 < {quoted}"

    async def _apply_patch(
        self, environment: BaseEnvironment, env: dict[str, str], patch_text: str
    ) -> dict[str, Any]:
        await environment.exec(
            self._write_patch_command(self.remote_patch_path, patch_text),
            cwd=self.working_dir or None,
            env=env,
            timeout_sec=self.command_timeout,
        )
        result = await environment.exec(
            self._apply_patch_command(self.remote_patch_path),
            cwd=self.working_dir or None,
            env=env,
            timeout_sec=self.command_timeout,
        )
        return _exec_result_to_payload(result)

    def _localize_prompt(self, instruction: str, repo_context: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are Agentless, a precise software bug localizer.",
            },
            {
                "role": "user",
                "content": (
                    "Given the task and the repository file list, name the files "
                    "most likely to need changes.\n\n"
                    f"Task:\n{instruction}\n\nFiles:\n{repo_context}"
                ),
            },
        ]

    def _repair_prompt(
        self, instruction: str, repo_context: str, localization: str
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are Agentless, a software repair engine. Reply with a "
                    "single unified diff (git apply compatible) and nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{instruction}\n\n"
                    f"Likely files:\n{localization}\n\n"
                    f"Repository files:\n{repo_context}\n\n"
                    "Produce the fix as a unified diff."
                ),
            },
        ]

    def _write_trajectory(self, patch_text: str, apply_payload: dict[str, Any]) -> None:
        try:
            self._trajectory_path.write_text(
                json.dumps(
                    {"patch": patch_text, "apply": apply_payload, "usage": self._usage},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover
            self.logger.debug("Failed to write agentless trajectory: %s", exc)

    def _populate_context(self, context: AgentContext) -> None:
        context.n_input_tokens = self._usage["input"]
        context.n_output_tokens = self._usage["output"]
        context.n_cache_tokens = self._usage["cache"]
        context.cost_usd = 0.0

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        env = self._env()

        repo_context = await self._collect_repo_context(environment, env)
        localization, localize_usage = await self._chat_completion(
            self._localize_prompt(instruction, repo_context)
        )
        self._accumulate(localize_usage)

        repair_text, repair_usage = await self._chat_completion(
            self._repair_prompt(instruction, repo_context, localization)
        )
        self._accumulate(repair_usage)
        patch_text = _extract_patch(repair_text)

        apply_payload: dict[str, Any] = {}
        if patch_text:
            try:
                (self.logs_dir / "prediction.patch").write_text(
                    patch_text,
                    encoding="utf-8",
                )
            except OSError as exc:
                self.logger.debug(
                    "Failed to save agentless prediction patch: %s",
                    exc,
                )
            apply_payload = await self._apply_patch(environment, env, patch_text)
            if apply_payload.get("returncode") not in (0, None):
                self.logger.warning(
                    "agentless patch apply exited with code %s",
                    apply_payload.get("returncode"),
                )
            if self.validate_command:
                await environment.exec(
                    self.validate_command,
                    cwd=self.working_dir or None,
                    env=env,
                    timeout_sec=self.command_timeout,
                )
        else:
            self.logger.warning("agentless produced no applicable patch")

        self._write_trajectory(patch_text, apply_payload)
        self._populate_context(context)
        account_output_tokens(context.n_output_tokens)
