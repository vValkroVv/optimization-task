from __future__ import annotations

import importlib
import queue
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def mini_swe_agent_module(monkeypatch):
    harbor_base_module = types.ModuleType("harbor.agents.base")
    harbor_environment_module = types.ModuleType("harbor.environments.base")
    harbor_context_module = types.ModuleType("harbor.models.agent.context")
    litellm_exceptions_module = types.ModuleType("litellm.exceptions")
    minisweagent_module = types.ModuleType("minisweagent")
    minisweagent_agents_module = types.ModuleType("minisweagent.agents")
    minisweagent_config_module = types.ModuleType("minisweagent.config")
    minisweagent_exceptions_module = types.ModuleType("minisweagent.exceptions")
    minisweagent_models_module = types.ModuleType("minisweagent.models")
    minisweagent_serialize_module = types.ModuleType("minisweagent.utils.serialize")

    class BaseAgent:
        def __init__(self, *args, logs_dir, model_name=None, **kwargs):
            _ = args, kwargs
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name

    class BaseEnvironment:
        pass

    class AgentContext:
        pass

    class Submitted(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class BadGatewayError(Exception):
        pass

    class ServiceUnavailableError(Exception):
        pass

    harbor_base_module.BaseAgent = BaseAgent
    harbor_environment_module.BaseEnvironment = BaseEnvironment
    harbor_context_module.AgentContext = AgentContext
    litellm_exceptions_module.APIConnectionError = APIConnectionError
    litellm_exceptions_module.BadGatewayError = BadGatewayError
    litellm_exceptions_module.ServiceUnavailableError = ServiceUnavailableError
    minisweagent_module.__version__ = "test"
    minisweagent_agents_module.get_agent = lambda *args, **kwargs: None
    minisweagent_config_module.get_config_from_spec = lambda spec: {}
    minisweagent_exceptions_module.Submitted = Submitted
    minisweagent_models_module.get_model = lambda *args, **kwargs: None
    minisweagent_serialize_module.UNSET = object()
    minisweagent_serialize_module.recursive_merge = lambda *configs: configs[-1]

    modules = {
        "harbor.agents.base": harbor_base_module,
        "harbor.environments.base": harbor_environment_module,
        "harbor.models.agent.context": harbor_context_module,
        "litellm": types.ModuleType("litellm"),
        "litellm.exceptions": litellm_exceptions_module,
        "minisweagent": minisweagent_module,
        "minisweagent.agents": minisweagent_agents_module,
        "minisweagent.config": minisweagent_config_module,
        "minisweagent.exceptions": minisweagent_exceptions_module,
        "minisweagent.models": minisweagent_models_module,
        "minisweagent.utils.serialize": minisweagent_serialize_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "terminal_bench.agents.mini_swe_agent.mini_swe_agent"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_name",
    ["APIConnectionError", "BadGatewayError", "ServiceUnavailableError"],
)
async def test_child_endpoint_error_stops_job(
    mini_swe_agent_module,
    monkeypatch,
    tmp_path,
    exception_name,
) -> None:
    exception_type = getattr(sys.modules["litellm.exceptions"], exception_name)

    def fail_model(*args, **kwargs):
        _ = args, kwargs
        raise exception_type("endpoint unavailable")

    monkeypatch.setattr(mini_swe_agent_module, "get_model", fail_model)
    request_queue = queue.Queue()
    response_queue = queue.Queue()
    status_queue = queue.Queue()
    mini_swe_agent_module._run_mini_swe_agent_child(
        instruction="solve the task",
        model_name="openai/pam-model",
        mini_config={},
        env_config={},
        env={},
        request_queue=request_queue,
        response_queue=response_queue,
        status_queue=status_queue,
    )

    class Process:
        exitcode = 0

        @staticmethod
        def is_alive() -> bool:
            return False

        @staticmethod
        def join(timeout=None) -> None:
            _ = timeout

    agent = mini_swe_agent_module.RLIMiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/pam-model",
    )
    with pytest.raises(mini_swe_agent_module.EndpointUnavailableError):
        await agent._wait_for_agent_process(
            process=Process(),
            request_queue=request_queue,
            response_queue=response_queue,
            status_queue=status_queue,
            environment=object(),
            env_config=mini_swe_agent_module.RLIMiniEnvironmentConfig(),
        )
