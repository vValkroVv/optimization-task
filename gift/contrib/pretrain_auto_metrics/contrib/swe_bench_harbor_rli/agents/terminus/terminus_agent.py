from __future__ import annotations

from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import LLMResponse
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from litellm.exceptions import (
    APIConnectionError as LiteLLMAPIConnectionError,
)
from litellm.exceptions import BadGatewayError as LiteLLMBadGatewayError
from litellm.exceptions import (
    ServiceUnavailableError as LiteLLMServiceUnavailableError,
)

from ...src.errors import EndpointUnavailableError
from ...src.token_budget import account_output_tokens
from ..prediction_patch import export_prediction_patch
from .tmux_session import TmuxSession

# Upper bound on how many times _query_llm may recurse on a max_tokens
# (OutputLengthExceededError) truncation before we stop and hand control back to
# the agent loop. The base Terminus2._query_llm recurses unconditionally on every
# truncation, so a model that persistently produces oversized responses keeps the
# agent wedged inside a single episode forever -- it never returns to the episode
# loop, never reaches a clean await/cancellation point, and the per-task agent
# timeout (harbor asyncio.wait_for) can never fire. Bounding the recursion makes
# the agent responsive to that timeout again.
DEFAULT_MAX_OUTPUT_LENGTH_RETRIES = 6


def _pop_alias(kwargs: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in kwargs:
            return kwargs.pop(name)
    return default


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _as_optional_int(value: Any) -> int | None:
    if not _present(value):
        return None
    return int(value)


def _as_optional_float(value: Any) -> float | None:
    if not _present(value):
        return None
    return float(value)


class RLITerminus2Agent(Terminus2):
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        try:
            await super().run(instruction, environment, context)
        finally:
            try:
                await export_prediction_patch(
                    environment,
                    self.logs_dir / "prediction.patch",
                    self.logger,
                )
            finally:
                account_output_tokens(
                    getattr(context, "n_output_tokens", None)
                )

    def __init__(
        self,
        *args,
        record_terminal_session: bool = False,
        model_api_base: str | None = None,
        api_key: str | None = None,
        max_tokens: int | str | None = None,
        llm_timeout: int | str | None = None,
        max_input_tokens: int | str | None = None,
        temperature: float | str | None = None,
        top_p: float | str | None = None,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> None:
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
        max_input_tokens = _pop_alias(
            kwargs,
            "max-input-tokens",
            "max_input_tokens",
            default=max_input_tokens,
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
        reasoning_effort = _pop_alias(
            kwargs,
            "reasoning-effort",
            "reasoning_effort",
            default=reasoning_effort,
        )

        if _present(model_api_base) and not _present(kwargs.get("api_base")):
            kwargs["api_base"] = str(model_api_base)
        parsed_temperature = _as_optional_float(temperature)
        if parsed_temperature is not None and kwargs.get("temperature") is None:
            kwargs["temperature"] = parsed_temperature
        if _present(reasoning_effort) and not _present(kwargs.get("reasoning_effort")):
            kwargs["reasoning_effort"] = str(reasoning_effort)

        llm_kwargs = dict(kwargs.pop("llm_kwargs", {}) or {})
        if _present(api_key):
            llm_kwargs.setdefault("api_key", str(api_key))
        if llm_kwargs:
            kwargs["llm_kwargs"] = llm_kwargs

        llm_call_kwargs = dict(kwargs.pop("llm_call_kwargs", {}) or {})
        parsed_max_tokens = _as_optional_int(max_tokens)
        if parsed_max_tokens is not None:
            llm_call_kwargs.setdefault("max_tokens", parsed_max_tokens)
        parsed_llm_timeout = _as_optional_int(llm_timeout)
        if parsed_llm_timeout is not None:
            llm_call_kwargs.setdefault("timeout", parsed_llm_timeout)
        parsed_top_p = _as_optional_float(top_p)
        if parsed_top_p is not None:
            llm_call_kwargs.setdefault("top_p", parsed_top_p)
        if llm_call_kwargs:
            kwargs["llm_call_kwargs"] = llm_call_kwargs

        # Register the served model's real input budget with LiteLLM. Without a
        # model_info, litellm.get_model_info("openai/pam-model") fails ("model
        # isn't mapped yet") and harbor falls back to a 1,000,000-token context
        # limit, so Terminus2 never summarizes and the chat grows until the
        # server truncates the response. max_input_tokens is the served context
        # length minus the proxy's reserved output budget; PAM supplies it.
        model_info = dict(kwargs.pop("model_info", {}) or {})
        parsed_max_input_tokens = _as_optional_int(max_input_tokens)
        if parsed_max_input_tokens is not None:
            model_info.setdefault("max_input_tokens", parsed_max_input_tokens)
        if model_info:
            kwargs["model_info"] = model_info

        super().__init__(
            *args,
            record_terminal_session=record_terminal_session,
            **kwargs,
        )

    async def setup(self, environment) -> None:
        if self._record_terminal_session:
            local_recording_path = environment.trial_paths.agent_dir / "recording.cast"
            remote_recording_path = EnvironmentPaths.agent_dir / "recording.cast"
        else:
            local_recording_path = None
            remote_recording_path = None

        self._session = TmuxSession(
            session_name=self.name(),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "terminus_2.pane",
            local_asciinema_recording_path=local_recording_path,
            remote_asciinema_recording_path=remote_recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
            extra_env=self._extra_env,
            user=environment.default_user,
        )
        await self._session.start()

    async def _query_llm(self, *args: Any, **kwargs: Any) -> LLMResponse:
        # Forward *args/**kwargs verbatim so this override stays compatible across
        # harbor versions whose base ``_query_llm`` signatures differ (e.g. 0.8.0
        # has an extra ``logging_paths`` positional that 0.9.0 dropped).
        depth = getattr(self, "_query_llm_recursion_depth", 0)
        if depth >= DEFAULT_MAX_OUTPUT_LENGTH_RETRIES:
            self.logger.warning(
                "Stopping _query_llm after %d consecutive max_tokens truncations; "
                "returning a placeholder response so the agent loop can continue.",
                depth,
            )
            return LLMResponse(
                content=(
                    "I kept exceeding the maximum output length. I will stop and "
                    "issue a single, much smaller command on the next turn."
                )
            )

        self._query_llm_recursion_depth = depth + 1
        try:
            return await super()._query_llm(*args, **kwargs)
        except (
            LiteLLMAPIConnectionError,
            LiteLLMBadGatewayError,
            LiteLLMServiceUnavailableError,
        ) as exc:
            raise EndpointUnavailableError(
                f"Model endpoint is unavailable: {exc}"
            ) from exc
        finally:
            self._query_llm_recursion_depth = depth
