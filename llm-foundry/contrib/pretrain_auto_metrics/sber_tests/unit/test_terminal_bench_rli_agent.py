import asyncio
import logging
import queue
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _recursive_merge(*dictionaries):
    result = {}
    for dictionary in dictionaries:
        if not dictionary:
            continue
        for key, value in dictionary.items():
            if value is UNSET:
                continue
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = _recursive_merge(result[key], value)
            elif isinstance(value, dict):
                result[key] = _recursive_merge(value)
            else:
                result[key] = value
    return result


UNSET = object()


def _install_harbor_stubs() -> None:
    harbor = types.ModuleType("harbor")
    agents = types.ModuleType("harbor.agents")
    agents_base = types.ModuleType("harbor.agents.base")
    installed = types.ModuleType("harbor.agents.installed")
    installed_mini = types.ModuleType("harbor.agents.installed.mini_swe_agent")
    environments = types.ModuleType("harbor.environments")
    environments_base = types.ModuleType("harbor.environments.base")
    capabilities = types.ModuleType("harbor.environments.capabilities")
    models = types.ModuleType("harbor.models")
    models_agent = types.ModuleType("harbor.models.agent")
    models_context = types.ModuleType("harbor.models.agent.context")

    class BaseAgent:
        def __init__(self, logs_dir, model_name=None, logger=None, *args, **kwargs):
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name
            self.logger = logger or logging.getLogger("test")

    class BaseEnvironment:
        pass

    class ExecResult:
        pass

    class EnvironmentCapabilities:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AgentContext:
        pass

    def convert_and_save_trajectory(
        mini_swe_agent_trajectory_path,
        atif_trajectory_path,
        session_id,
    ):
        atif_trajectory_path.write_text("{}", encoding="utf-8")

    agents_base.BaseAgent = BaseAgent
    environments_base.BaseEnvironment = BaseEnvironment
    environments_base.ExecResult = ExecResult
    capabilities.EnvironmentCapabilities = EnvironmentCapabilities
    models_context.AgentContext = AgentContext
    installed_mini.convert_and_save_trajectory = convert_and_save_trajectory

    sys.modules.setdefault("harbor", harbor)
    sys.modules.setdefault("harbor.agents", agents)
    sys.modules.setdefault("harbor.agents.base", agents_base)
    sys.modules.setdefault("harbor.agents.installed", installed)
    sys.modules.setdefault(
        "harbor.agents.installed.mini_swe_agent",
        installed_mini,
    )
    sys.modules.setdefault("harbor.environments", environments)
    sys.modules.setdefault("harbor.environments.base", environments_base)
    sys.modules.setdefault("harbor.environments.capabilities", capabilities)
    sys.modules.setdefault("harbor.models", models)
    sys.modules.setdefault("harbor.models.agent", models_agent)
    sys.modules.setdefault("harbor.models.agent.context", models_context)


def _install_mini_swe_agent_stubs() -> None:
    litellm = types.ModuleType("litellm")
    litellm_exceptions = types.ModuleType("litellm.exceptions")
    minisweagent = types.ModuleType("minisweagent")
    agents = types.ModuleType("minisweagent.agents")
    config = types.ModuleType("minisweagent.config")
    exceptions = types.ModuleType("minisweagent.exceptions")
    models = types.ModuleType("minisweagent.models")
    utils = types.ModuleType("minisweagent.utils")
    serialize = types.ModuleType("minisweagent.utils.serialize")

    class Submitted(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class BadGatewayError(Exception):
        pass

    class ServiceUnavailableError(Exception):
        pass

    litellm_exceptions.APIConnectionError = APIConnectionError
    litellm_exceptions.BadGatewayError = BadGatewayError
    litellm_exceptions.ServiceUnavailableError = ServiceUnavailableError
    minisweagent.__version__ = "2.3.0"
    agents.get_agent = lambda *args, **kwargs: None
    config.get_config_from_spec = lambda spec: {}
    exceptions.Submitted = Submitted
    models.get_model = lambda *args, **kwargs: None
    serialize.UNSET = UNSET
    serialize.recursive_merge = _recursive_merge

    sys.modules.setdefault("litellm", litellm)
    sys.modules.setdefault("litellm.exceptions", litellm_exceptions)
    sys.modules.setdefault("minisweagent", minisweagent)
    sys.modules.setdefault("minisweagent.agents", agents)
    sys.modules.setdefault("minisweagent.config", config)
    sys.modules.setdefault("minisweagent.exceptions", exceptions)
    sys.modules.setdefault("minisweagent.models", models)
    sys.modules.setdefault("minisweagent.utils", utils)
    sys.modules.setdefault("minisweagent.utils.serialize", serialize)


_install_harbor_stubs()
_install_mini_swe_agent_stubs()

from contrib.terminal_bench.agents.mini_swe_agent.mini_swe_agent import (  # noqa: E402
    RLIMiniEnvironment,
    RLIMiniEnvironmentConfig,
    RLIMiniSweAgent,
    _RLIMiniProcessEnvironment,
)
from minisweagent.exceptions import Submitted  # noqa: E402

pytestmark = pytest.mark.unit


class _FakeRLIEnvironment:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_sec": timeout_sec,
                "user": user,
            }
        )
        return self.result


def test_rli_mini_environment_executes_harbor_environment_in_event_loop():
    async def scenario():
        fake = _FakeRLIEnvironment(
            SimpleNamespace(stdout="stdout\n", stderr="stderr\n", return_code=7)
        )
        adapter = RLIMiniEnvironment(
            fake,
            loop=asyncio.get_running_loop(),
            config=RLIMiniEnvironmentConfig(
                env={"PAGER": "cat"},
                timeout=123,
            ),
        )

        result = await asyncio.to_thread(
            adapter.execute,
            {"command": "pwd"},
            cwd="/app",
        )

        assert result == {
            "output": "stdout\nstderr\n",
            "returncode": 7,
            "exception_info": "",
        }
        assert fake.calls == [
            {
                "command": "pwd",
                "cwd": "/app",
                "env": {"PAGER": "cat"},
                "timeout_sec": 123,
                "user": None,
            }
        ]

    asyncio.run(scenario())


def test_rli_mini_environment_raises_submitted_on_completion_marker():
    async def scenario():
        fake = _FakeRLIEnvironment(
            SimpleNamespace(
                stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nanswer",
                stderr="",
                return_code=0,
            )
        )
        adapter = RLIMiniEnvironment(
            fake,
            loop=asyncio.get_running_loop(),
            config=RLIMiniEnvironmentConfig(),
        )

        with pytest.raises(Submitted):
            await asyncio.to_thread(
                adapter.execute,
                {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
            )

    asyncio.run(scenario())


def test_rli_mini_swe_agent_builds_yolo_config(tmp_path, monkeypatch):
    import contrib.terminal_bench.agents.mini_swe_agent.mini_swe_agent as agent_module

    def fake_get_config_from_spec(spec):
        if spec == "mini.yaml":
            return {
                "agent": {"mode": "confirm", "cost_limit": 3.0},
                "environment": {"env": {"PAGER": "cat"}},
            }
        return {"model": {"model_kwargs": {"temperature": 0}}}

    monkeypatch.setattr(
        agent_module,
        "get_config_from_spec",
        fake_get_config_from_spec,
    )

    agent = RLIMiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/pam-model",
        config_file="custom.yaml",
        cost_limit="0",
        command_timeout="321",
        step_limit="9",
        model_api_base="http://127.0.0.1:29999/v1",
        api_key="dummy",
        max_tokens="32768",
        temperature="0.2",
        top_p="0.9",
        reasoning_effort="high",
    )

    config = agent._build_mini_config()

    assert config["agent"]["mode"] == "yolo"
    assert config["agent"]["confirm_exit"] is False
    assert config["agent"]["cost_limit"] == 0.0
    assert config["agent"]["step_limit"] == 9
    assert config["agent"]["output_path"] == str(
        tmp_path / "mini-swe-agent.trajectory.json"
    )
    assert config["environment"]["timeout"] == 321
    assert config["model"]["model_name"] == "openai/pam-model"
    assert config["model"]["model_kwargs"]["api_base"] == "http://127.0.0.1:29999/v1"
    assert config["model"]["model_kwargs"]["api_key"] == "dummy"
    assert config["model"]["model_kwargs"]["max_tokens"] == 32768
    assert config["model"]["model_kwargs"]["temperature"] == 0.2
    assert config["model"]["model_kwargs"]["top_p"] == 0.9
    assert config["model"]["model_kwargs"]["extra_body"] == {"reasoning_effort": "high"}


def test_process_environment_bridges_execute_requests():
    request_queue = queue.Queue()
    response_queue = queue.Queue()
    env = _RLIMiniProcessEnvironment(
        request_queue=request_queue,
        response_queue=response_queue,
        config=RLIMiniEnvironmentConfig(env={"PAGER": "cat"}, timeout=123),
    )
    captured = {}

    def respond():
        request = request_queue.get(timeout=1)
        captured.update(request)
        response_queue.put(
            {
                "id": request["id"],
                "ok": True,
                "result": {
                    "output": "done",
                    "returncode": 0,
                    "exception_info": "",
                },
            }
        )

    thread = threading.Thread(target=respond)
    thread.start()

    result = env.execute({"command": "pwd"}, cwd="/app", timeout=77)
    thread.join(timeout=1)

    assert result == {"output": "done", "returncode": 0, "exception_info": ""}
    assert captured["type"] == "exec"
    assert captured["action"] == {"command": "pwd"}
    assert captured["cwd"] == "/app"
    assert captured["timeout"] == 77


def test_agent_run_terminates_child_process_on_cancellation(tmp_path, monkeypatch):
    import contrib.terminal_bench.agents.mini_swe_agent.mini_swe_agent as agent_module

    class FakeProcess:
        def __init__(self):
            self.started = False
            self.terminated = False
            self.killed = False
            self._alive = False
            self.exitcode = None
            self.joins = []

        def start(self):
            self.started = True
            self._alive = True

        def is_alive(self):
            return self._alive

        def terminate(self):
            self.terminated = True
            self._alive = False
            self.exitcode = -15

        def kill(self):
            self.killed = True
            self._alive = False
            self.exitcode = -9

        def join(self, timeout=None):
            self.joins.append(timeout)

    class FakeContext:
        def __init__(self, process):
            self.process = process

        def Queue(self):
            return queue.Queue()

        def Process(self, target, kwargs):
            self.target = target
            self.kwargs = kwargs
            return self.process

    fake_process = FakeProcess()
    fake_context = FakeContext(fake_process)

    monkeypatch.setattr(agent_module, "_PROCESS_POLL_INTERVAL_SEC", 0.001)
    monkeypatch.setattr(
        agent_module,
        "get_config_from_spec",
        lambda spec: {"agent": {}, "environment": {}},
    )

    agent = RLIMiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/pam-model",
    )
    agent._get_process_context = lambda: fake_context
    agent._populate_context_from_trajectory = lambda context: None

    async def scenario():
        task = asyncio.create_task(agent.run("task", object(), SimpleNamespace()))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert fake_process.started is True
    assert fake_process.terminated is True
    assert fake_process.killed is False
