from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _load_terminus_agent_module():
    harbor_module = types.ModuleType("harbor")
    agents_module = types.ModuleType("harbor.agents")
    base_agent_module = types.ModuleType("harbor.agents.base")
    environments_module = types.ModuleType("harbor.environments")
    environments_base_module = types.ModuleType("harbor.environments.base")
    names_module = types.ModuleType("harbor.models.agent.name")
    trial_paths_module = types.ModuleType("harbor.models.trial.paths")
    llms_module = types.ModuleType("harbor.llms")
    llms_base_module = types.ModuleType("harbor.llms.base")
    llms_chat_module = types.ModuleType("harbor.llms.chat")
    terminus_pkg = types.ModuleType("harbor.agents.terminus_2")
    terminus_module = types.ModuleType("harbor.agents.terminus_2.terminus_2")
    terminus_tmux_module = types.ModuleType("harbor.agents.terminus_2.tmux_session")
    litellm_module = types.ModuleType("litellm")
    litellm_exceptions_module = types.ModuleType("litellm.exceptions")

    class LLMResponse:
        def __init__(self, content: str = "", **kwargs):
            self.content = content
            self.kwargs = kwargs

    class Chat:
        pass

    class BaseAgent:
        def __init__(
            self, logs_dir: Path, model_name: str | None = None, *args, **kwargs
        ):
            _ = args, kwargs
            self.logs_dir = logs_dir
            self.model_name = model_name

    class BaseEnvironment:
        pass

    class AgentName:
        TERMINUS_2 = type("EnumValue", (), {"value": "terminus_2"})()

    class EnvironmentPaths:
        agent_dir = Path("/logs/agent")

    class HarborTmuxSession:
        pass

    class Terminus2(BaseAgent):
        def __init__(
            self,
            logs_dir: Path,
            model_name: str | None = None,
            tmux_pane_width: int = 160,
            tmux_pane_height: int = 40,
            record_terminal_session: bool = True,
            extra_env: dict[str, str] | None = None,
            *args,
            **kwargs,
        ) -> None:
            super().__init__(logs_dir, model_name, *args, **kwargs)
            self._tmux_pane_width = tmux_pane_width
            self._tmux_pane_height = tmux_pane_height
            self._record_terminal_session = record_terminal_session
            self._extra_env = extra_env
            self.constructor_kwargs = kwargs
            self._session = None

        @staticmethod
        def name() -> str:
            return AgentName.TERMINUS_2.value

    class APIConnectionError(Exception):
        pass

    class BadGatewayError(Exception):
        pass

    class ServiceUnavailableError(Exception):
        pass

    base_agent_module.BaseAgent = BaseAgent
    environments_base_module.BaseEnvironment = BaseEnvironment
    names_module.AgentName = AgentName
    trial_paths_module.EnvironmentPaths = EnvironmentPaths
    llms_base_module.LLMResponse = LLMResponse
    llms_chat_module.Chat = Chat
    terminus_module.Terminus2 = Terminus2
    terminus_tmux_module.TmuxSession = HarborTmuxSession
    litellm_exceptions_module.APIConnectionError = APIConnectionError
    litellm_exceptions_module.BadGatewayError = BadGatewayError
    litellm_exceptions_module.ServiceUnavailableError = ServiceUnavailableError

    sys.modules["harbor"] = harbor_module
    sys.modules["harbor.agents"] = agents_module
    sys.modules["harbor.agents.base"] = base_agent_module
    sys.modules["harbor.environments"] = environments_module
    sys.modules["harbor.environments.base"] = environments_base_module
    sys.modules["harbor.models.agent.name"] = names_module
    sys.modules["harbor.models.trial.paths"] = trial_paths_module
    sys.modules["harbor.llms"] = llms_module
    sys.modules["harbor.llms.base"] = llms_base_module
    sys.modules["harbor.llms.chat"] = llms_chat_module
    sys.modules["harbor.agents.terminus_2"] = terminus_pkg
    sys.modules["harbor.agents.terminus_2.terminus_2"] = terminus_module
    sys.modules["harbor.agents.terminus_2.tmux_session"] = terminus_tmux_module
    sys.modules["litellm"] = litellm_module
    sys.modules["litellm.exceptions"] = litellm_exceptions_module

    sys.modules.pop("terminal_bench.agents.terminus.tmux_session", None)
    sys.modules.pop("terminal_bench.agents.terminus.terminus_agent", None)
    return importlib.import_module("terminal_bench.agents.terminus.terminus_agent")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_name",
    ["APIConnectionError", "BadGatewayError", "ServiceUnavailableError"],
)
async def test_query_llm_stops_job_when_endpoint_is_unavailable(
    monkeypatch,
    exception_name,
) -> None:
    terminus_agent_module = _load_terminus_agent_module()
    exception_type = getattr(sys.modules["litellm.exceptions"], exception_name)

    async def fail_query(*args, **kwargs):
        _ = args, kwargs
        raise exception_type("endpoint unavailable")

    monkeypatch.setattr(
        terminus_agent_module.Terminus2,
        "_query_llm",
        fail_query,
        raising=False,
    )
    agent = terminus_agent_module.RLITerminus2Agent(
        logs_dir=Path("/tmp/logs"),
        model_name="model",
    )

    with pytest.raises(terminus_agent_module.EndpointUnavailableError):
        await agent._query_llm()


@pytest.mark.asyncio
async def test_setup_uses_local_tmux_session_without_recording_by_default(
    monkeypatch,
) -> None:
    terminus_agent_module = _load_terminus_agent_module()

    captured: dict[str, object] = {}

    class FakeTmuxSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr(terminus_agent_module, "TmuxSession", FakeTmuxSession)

    env = type(
        "FakeEnvironment",
        (),
        {
            "default_user": "root",
            "trial_paths": type("TrialPaths", (), {"agent_dir": Path("/tmp/agent")})(),
        },
    )()

    agent = terminus_agent_module.RLITerminus2Agent(
        logs_dir=Path("/tmp/logs"),
        model_name="model",
    )

    await agent.setup(env)

    assert agent._record_terminal_session is False
    assert captured["session_name"] == "terminus_2"
    assert captured["environment"] is env
    assert captured["logging_path"] == Path("/logs/agent/terminus_2.pane")
    assert captured["local_asciinema_recording_path"] is None
    assert captured["remote_asciinema_recording_path"] is None
    assert captured["user"] == "root"
    assert captured["started"] is True


@pytest.mark.asyncio
async def test_setup_enables_recording_when_requested(monkeypatch) -> None:
    terminus_agent_module = _load_terminus_agent_module()

    captured: dict[str, object] = {}

    class FakeTmuxSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr(terminus_agent_module, "TmuxSession", FakeTmuxSession)

    env = type(
        "FakeEnvironment",
        (),
        {
            "default_user": "sandbox",
            "trial_paths": type("TrialPaths", (), {"agent_dir": Path("/tmp/agent")})(),
        },
    )()

    agent = terminus_agent_module.RLITerminus2Agent(
        logs_dir=Path("/tmp/logs"),
        model_name="model",
        record_terminal_session=True,
    )

    await agent.setup(env)

    assert captured["local_asciinema_recording_path"] == Path(
        "/tmp/agent/recording.cast"
    )
    assert captured["remote_asciinema_recording_path"] == Path(
        "/logs/agent/recording.cast"
    )
    assert captured["user"] == "sandbox"


def test_common_model_kwargs_map_to_terminus_constructor() -> None:
    terminus_agent_module = _load_terminus_agent_module()

    agent = terminus_agent_module.RLITerminus2Agent(
        logs_dir=Path("/tmp/logs"),
        model_name="model",
        model_api_base="http://127.0.0.1:29999/v1",
        api_key="dummy",
        max_tokens="32768",
        max_input_tokens="98304",
        temperature="0.2",
        top_p="0.9",
        reasoning_effort="high",
        llm_call_kwargs={"presence_penalty": 0},
    )

    assert agent.constructor_kwargs["api_base"] == "http://127.0.0.1:29999/v1"
    assert agent.constructor_kwargs["temperature"] == 0.2
    assert agent.constructor_kwargs["reasoning_effort"] == "high"
    assert agent.constructor_kwargs["llm_kwargs"] == {"api_key": "dummy"}
    assert agent.constructor_kwargs["llm_call_kwargs"] == {
        "presence_penalty": 0,
        "max_tokens": 32768,
        "top_p": 0.9,
    }
    assert agent.constructor_kwargs["model_info"] == {"max_input_tokens": 98304}


def test_max_input_tokens_absent_omits_model_info() -> None:
    terminus_agent_module = _load_terminus_agent_module()

    agent = terminus_agent_module.RLITerminus2Agent(
        logs_dir=Path("/tmp/logs"),
        model_name="model",
        max_tokens="32768",
    )

    assert "model_info" not in agent.constructor_kwargs
