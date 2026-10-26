from __future__ import annotations

import importlib
import queue
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def mini_swe_agent_module(monkeypatch):
    harbor_module = types.ModuleType("harbor")
    harbor_agents_module = types.ModuleType("harbor.agents")
    harbor_base_module = types.ModuleType("harbor.agents.base")
    harbor_environments_module = types.ModuleType("harbor.environments")
    harbor_environment_base_module = types.ModuleType("harbor.environments.base")
    harbor_models_module = types.ModuleType("harbor.models")
    harbor_agent_models_module = types.ModuleType("harbor.models.agent")
    harbor_context_module = types.ModuleType("harbor.models.agent.context")
    litellm_module = types.ModuleType("litellm")
    litellm_exceptions_module = types.ModuleType("litellm.exceptions")

    class Logger:
        def debug(self, *args, **kwargs) -> None:
            _ = args, kwargs

    class BaseAgent:
        def __init__(
            self,
            *args,
            logs_dir: Path,
            model_name: str | None = None,
            **kwargs,
        ) -> None:
            _ = args, kwargs
            self.logs_dir = logs_dir
            self.model_name = model_name
            self.logger = Logger()

    class BaseEnvironment:
        pass

    class AgentContext:
        pass

    class APIConnectionError(Exception):
        pass

    class BadGatewayError(Exception):
        pass

    class ServiceUnavailableError(Exception):
        pass

    harbor_base_module.BaseAgent = BaseAgent
    harbor_environment_base_module.BaseEnvironment = BaseEnvironment
    harbor_context_module.AgentContext = AgentContext
    litellm_exceptions_module.APIConnectionError = APIConnectionError
    litellm_exceptions_module.BadGatewayError = BadGatewayError
    litellm_exceptions_module.ServiceUnavailableError = ServiceUnavailableError

    minisweagent_module = types.ModuleType("minisweagent")
    minisweagent_agents_module = types.ModuleType("minisweagent.agents")
    minisweagent_config_module = types.ModuleType("minisweagent.config")
    minisweagent_exceptions_module = types.ModuleType("minisweagent.exceptions")
    minisweagent_models_module = types.ModuleType("minisweagent.models")
    minisweagent_utils_module = types.ModuleType("minisweagent.utils")
    minisweagent_serialize_module = types.ModuleType("minisweagent.utils.serialize")

    class Submitted(Exception):
        pass

    minisweagent_module.__version__ = "test"
    minisweagent_agents_module.get_agent = lambda *args, **kwargs: None
    minisweagent_config_module.get_config_from_spec = lambda spec: {}
    minisweagent_exceptions_module.Submitted = Submitted
    minisweagent_models_module.get_model = lambda *args, **kwargs: None
    minisweagent_serialize_module.UNSET = object()
    minisweagent_serialize_module.recursive_merge = lambda *configs: configs[-1]

    modules = {
        "harbor": harbor_module,
        "harbor.agents": harbor_agents_module,
        "harbor.agents.base": harbor_base_module,
        "harbor.environments": harbor_environments_module,
        "harbor.environments.base": harbor_environment_base_module,
        "harbor.models": harbor_models_module,
        "harbor.models.agent": harbor_agent_models_module,
        "harbor.models.agent.context": harbor_context_module,
        "litellm": litellm_module,
        "litellm.exceptions": litellm_exceptions_module,
        "minisweagent": minisweagent_module,
        "minisweagent.agents": minisweagent_agents_module,
        "minisweagent.config": minisweagent_config_module,
        "minisweagent.exceptions": minisweagent_exceptions_module,
        "minisweagent.models": minisweagent_models_module,
        "minisweagent.utils": minisweagent_utils_module,
        "minisweagent.utils.serialize": minisweagent_serialize_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "swe_bench_harbor_rli.agents.mini_swe_agent.mini_swe_agent"
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
        instruction="fix the bug",
        model_name="openai/pam-model",
        mini_config={},
        env_config={},
        env={},
        child_log_path=str(tmp_path / "child.log"),
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


def test_build_mini_config_maps_model_and_runtime_options(
    mini_swe_agent_module,
    monkeypatch,
    tmp_path,
) -> None:
    loaded_specs: list[str] = []
    monkeypatch.setattr(
        mini_swe_agent_module,
        "get_config_from_spec",
        lambda spec: loaded_specs.append(spec) or {},
    )
    monkeypatch.setattr(
        mini_swe_agent_module,
        "recursive_merge",
        lambda *configs: configs[-1],
    )
    agent = mini_swe_agent_module.RLIMiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/pam-model",
        config_file="swebench_backticks.yaml",
        model_class="litellm_textbased",
        step_limit="40",
        wall_time_limit_seconds="3600",
        command_timeout="120",
        model_api_base="http://model-proxy/v1",
        api_key="secret",
        max_tokens="2048",
        llm_timeout="6400",
        temperature="0.2",
        top_p="0.9",
        reasoning_effort="high",
        extra_env='{"CUSTOM": 1}',
    )

    config = agent._build_mini_config()

    assert loaded_specs == ["mini.yaml", "swebench_backticks.yaml"]
    assert config["agent"]["step_limit"] == 40
    assert config["agent"]["wall_time_limit_seconds"] == 3600
    assert config["environment"]["timeout"] == 120
    assert config["model"] == {
        "model_name": "openai/pam-model",
        "model_class": "litellm_textbased",
        "model_kwargs": {
            "api_base": "http://model-proxy/v1",
            "api_key": "secret",
            "max_tokens": 2048,
            "timeout": 6400,
            "temperature": 0.2,
            "top_p": 0.9,
            "extra_body": {"reasoning_effort": "high"},
        },
    }
    assert agent.extra_env == {"CUSTOM": "1"}


@pytest.mark.asyncio
async def test_execute_process_request_forwards_environment_options(
    mini_swe_agent_module,
) -> None:
    calls: list[dict] = []

    class Environment:
        async def exec(self, command: str, **kwargs):
            calls.append({"command": command, **kwargs})
            return SimpleNamespace(
                stdout="stdout",
                stderr="stderr",
                return_code=3,
            )

    agent = object.__new__(mini_swe_agent_module.RLIMiniSweAgent)
    config = mini_swe_agent_module.RLIMiniEnvironmentConfig(
        cwd="/testbed",
        env={"CUSTOM": "1"},
        timeout=42,
    )

    payload = await agent._execute_process_request(
        environment=Environment(),
        env_config=config,
        request={"action": {"command": "pytest"}, "cwd": "", "timeout": None},
    )

    assert calls == [
        {
            "command": "pytest",
            "cwd": "/testbed",
            "env": {"CUSTOM": "1"},
            "timeout_sec": 42,
        }
    ]
    assert payload == {
        "output": "stdout\nstderr",
        "returncode": 3,
        "exception_info": "",
    }


@pytest.mark.asyncio
async def test_execute_process_request_propagates_environment_error(
    mini_swe_agent_module,
) -> None:
    class Environment:
        async def exec(self, command: str, **kwargs):
            _ = command, kwargs
            raise TimeoutError("RLI step timed out")

    agent = object.__new__(mini_swe_agent_module.RLIMiniSweAgent)

    with pytest.raises(TimeoutError, match="RLI step timed out"):
        await agent._execute_process_request(
            environment=Environment(),
            env_config=mini_swe_agent_module.RLIMiniEnvironmentConfig(),
            request={"action": {"command": "pytest"}},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("return_code", "stdout", "stderr"),
    [
        (-1, "No output", ""),
        (124, "", "exec-proxy connection lost; partial result preserved"),
    ],
)
async def test_execute_process_request_fails_on_lost_exec_proxy(
    mini_swe_agent_module,
    return_code,
    stdout,
    stderr,
) -> None:
    class Environment:
        async def exec(self, command: str, **kwargs):
            _ = command, kwargs
            return SimpleNamespace(
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
            )

    agent = object.__new__(mini_swe_agent_module.RLIMiniSweAgent)

    with pytest.raises(
        mini_swe_agent_module.RLIExecProxyConnectionLostError,
        match="RLI exec-proxy connection lost",
    ):
        await agent._execute_process_request(
            environment=Environment(),
            env_config=mini_swe_agent_module.RLIMiniEnvironmentConfig(),
            request={"action": {"command": "pytest"}},
        )


@pytest.mark.asyncio
async def test_run_exports_patch_even_when_child_process_fails(
    mini_swe_agent_module,
    monkeypatch,
    tmp_path,
) -> None:
    agent = mini_swe_agent_module.RLIMiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/pam-model",
    )

    class Process:
        def __init__(self) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

    process = Process()

    class ProcessContext:
        @staticmethod
        def Queue():
            return object()

        @staticmethod
        def Process(**kwargs):
            _ = kwargs
            return process

    monkeypatch.setattr(agent, "_get_process_context", lambda: ProcessContext())
    monkeypatch.setattr(
        agent,
        "_build_mini_config",
        lambda: {"environment": {"timeout": 9}},
    )
    wait_for_process = AsyncMock(side_effect=RuntimeError("child failed"))
    export_patch = AsyncMock()
    populate_context = Mock()
    monkeypatch.setattr(agent, "_wait_for_agent_process", wait_for_process)
    monkeypatch.setattr(
        mini_swe_agent_module,
        "export_prediction_patch",
        export_patch,
    )
    monkeypatch.setattr(
        agent,
        "_populate_context_from_trajectory",
        populate_context,
    )
    environment = object()
    context = mini_swe_agent_module.AgentContext()

    with pytest.raises(RuntimeError, match="child failed"):
        await agent.run("fix the bug", environment, context)

    assert process.started is True
    export_args = export_patch.await_args.args
    assert export_args[0] is environment
    assert export_args[1] == tmp_path / "prediction.patch"
    assert export_args[2] is agent.logger
    assert export_patch.await_args.kwargs == {"env": {}}
    populate_context.assert_called_once_with(context)


@pytest.mark.asyncio
async def test_export_prediction_patch_writes_nonempty_git_diff(
    mini_swe_agent_module,
    tmp_path,
) -> None:
    calls: list[dict] = []

    class Environment:
        async def exec(self, command: str, **kwargs):
            calls.append({"command": command, **kwargs})
            return SimpleNamespace(
                stdout="diff --git a/a.py b/a.py\n+fixed\n",
                stderr="",
                return_code=0,
            )

    agent = mini_swe_agent_module.RLIMiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/pam-model",
    )
    await mini_swe_agent_module.export_prediction_patch(
        Environment(),
        tmp_path / "prediction.patch",
        agent.logger,
        env={"CUSTOM": "1"},
    )

    assert (tmp_path / "prediction.patch").read_text(encoding="utf-8") == (
        "diff --git a/a.py b/a.py\n+fixed\n"
    )
    assert len(calls) == 1
    assert "git diff --cached HEAD" in calls[0]["command"]
    assert calls[0]["env"] == {"CUSTOM": "1"}
    assert calls[0]["timeout_sec"] == 120
