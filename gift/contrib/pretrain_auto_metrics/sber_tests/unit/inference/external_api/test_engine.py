import requests
import pytest
import sys

from node_runner.src.config_model import ExternalApiInferenceConfig
from node_runner.src.inference.external_api.engine import (
    ExternalApiEngine,
    ExternalApiProxyStartError,
)
from node_runner.src.inference.external_api.proxy.mappers import IftMapper, OpenAiMapper
from node_runner.src.logger import Logger
from node_runner.src.pam_proxies.models import SGLangChatCompletionRequest

pytestmark = pytest.mark.unit


def _config(formatting: str, base_url: str, **overrides) -> ExternalApiInferenceConfig:
    payload = {
        "formatting": formatting,
        "base_url": base_url,
        "model": "model",
    }
    if formatting == "ift":
        payload["user_pass"] = "user:pass"
    else:
        payload["api_key"] = "key"
    payload.update(overrides)
    return ExternalApiInferenceConfig.model_validate(payload)


def _cmd_value(cmd: list[str], option: str) -> str:
    return cmd[cmd.index(option) + 1]


def _engine(cfg) -> ExternalApiEngine:
    return ExternalApiEngine(
        inference_config=cfg,
    )


def test_proxy_env_uses_only_configured_secret(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXTERNAL_API_PROXY_USER_PASS", "stale:secret")
    monkeypatch.setenv("UNRELATED_VAR", "kept")

    cfg = _config("openai", "https://api.openai.com")

    env = _engine(cfg).proxy_env
    assert env["EXTERNAL_API_PROXY_API_KEY"] == "key"
    assert "EXTERNAL_API_PROXY_USER_PASS" not in env
    assert env["UNRELATED_VAR"] == "kept"


def test_proxy_env_supports_user_pass_without_api_key():
    cfg = _config("ift", "https://gigachat.ift.sberdevices.ru")

    env = _engine(cfg).proxy_env
    assert env["EXTERNAL_API_PROXY_USER_PASS"] == "user:pass"
    assert "EXTERNAL_API_PROXY_API_KEY" not in env


@pytest.mark.parametrize(
    ("formatting", "base_url", "expected"),
    [
        ("openai", "https://api.openai.com/v1", "https://api.openai.com"),
        (
            "openai",
            "https://api.openai.com/v1/chat/completions",
            "https://api.openai.com",
        ),
        (
            "openai",
            "https://openrouter.ai/api/v1",
            "https://openrouter.ai/api",
        ),
        (
            "openai",
            "https://openrouter.ai/api/v1/chat/completions",
            "https://openrouter.ai/api",
        ),
        (
            "ift",
            "https://gigachat.ift.sberdevices.ru/v1",
            "https://gigachat.ift.sberdevices.ru",
        ),
        (
            "ift",
            "https://gigachat.ift.sberdevices.ru/v2",
            "https://gigachat.ift.sberdevices.ru",
        ),
        (
            "ift",
            "https://gigachat.ift.sberdevices.ru/v2/chat/completions",
            "https://gigachat.ift.sberdevices.ru",
        ),
    ],
)
def test_proxy_cmd_uses_normalized_base_url(
    formatting: str, base_url: str, expected: str
):
    engine = _engine(_config(formatting, base_url))

    assert _cmd_value(engine.proxy_cmd, "--base-url") == expected


def test_proxy_cmd_passes_logging_level():
    engine = _engine(_config("openai", "https://api.openai.com", logging_level="debug"))

    assert _cmd_value(engine.proxy_cmd, "--logging-level") == "debug"


@pytest.mark.parametrize(
    ("formatting", "base_url", "expected"),
    [
        (
            "openai",
            "https://api.openai.com/v1",
            "https://api.openai.com/v1/chat/completions",
        ),
        (
            "openai",
            "https://api.openai.com/v1/chat/completions",
            "https://api.openai.com/v1/chat/completions",
        ),
        (
            "openai",
            "https://openrouter.ai/api/v1",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
        (
            "openai",
            "https://openrouter.ai/api/v1/chat/completions",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
        (
            "ift",
            "https://gigachat.ift.sberdevices.ru/v1",
            "https://gigachat.ift.sberdevices.ru/v2/chat/completions",
        ),
        (
            "ift",
            "https://gigachat.ift.sberdevices.ru/v2",
            "https://gigachat.ift.sberdevices.ru/v2/chat/completions",
        ),
        (
            "ift",
            "https://gigachat.ift.sberdevices.ru/v2/chat/completions",
            "https://gigachat.ift.sberdevices.ru/v2/chat/completions",
        ),
    ],
)
def test_normalized_base_url_combines_with_upstream_path(
    formatting: str, base_url: str, expected: str
):
    cfg = _config(formatting, base_url)
    mapper = IftMapper() if formatting == "ift" else OpenAiMapper()

    assert f"{cfg.base_url}/{mapper.upstream_path}" == expected


class _Response:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeSubprocessManager:
    def __init__(self, cmd, log_file, cwd, name, env=None):
        self.cmd = cmd
        self.log_file = log_file
        self.cwd = cwd
        self.name = name
        self.env = env
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def graceful_stop(self):
        self.stopped = True


class _FakeObmProxy:
    def __init__(self):
        self.api_port = 29999
        self.base_url = "http://127.0.0.1:29999"
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class TestExternalApiEngineLifecycle:
    def test_properties_are_derived_from_external_api_config(self):
        engine = _engine(_config("openai", "https://api.openai.com"))

        assert engine.router_port == 30000
        assert engine.inference_instances == 1
        assert engine.proxy_cwd.endswith("/node_runner")
        assert engine.proxy_cmd[:3] == [
            sys.executable,
            "-m",
            "src.inference.external_api.proxy.main",
        ]
        assert _cmd_value(engine.proxy_cmd, "--logging-level") == "info"

    def test_wait_for_proxy_to_be_ready_returns_true_for_200(self, monkeypatch):
        engine = _engine(_config("openai", "https://api.openai.com"))
        captured = {}

        def _get(url, timeout):
            captured["url"] = url
            captured["timeout"] = timeout
            return _Response(200)

        monkeypatch.setattr(
            "node_runner.src.inference.external_api.engine.requests.get",
            _get,
        )

        assert ExternalApiEngine.wait_for_proxy_to_be_ready.__wrapped__(engine) is True
        assert captured == {"url": "http://127.0.0.1:30000/ready", "timeout": 5}

    def test_wait_for_proxy_to_be_ready_raises_for_non_200(self, monkeypatch):
        engine = _engine(_config("openai", "https://api.openai.com"))
        monkeypatch.setattr(
            "node_runner.src.inference.external_api.engine.requests.get",
            lambda url, timeout: _Response(503),
        )

        with pytest.raises(ExternalApiProxyStartError):
            ExternalApiEngine.wait_for_proxy_to_be_ready.__wrapped__(engine)

    def test_wait_for_proxy_to_be_ready_propagates_request_error(self, monkeypatch):
        engine = _engine(_config("openai", "https://api.openai.com"))

        def _raise(*_args, **_kwargs):
            raise requests.exceptions.RequestException("connection failed")

        monkeypatch.setattr(
            "node_runner.src.inference.external_api.engine.requests.get",
            _raise,
        )

        with pytest.raises(requests.exceptions.RequestException):
            ExternalApiEngine.wait_for_proxy_to_be_ready.__wrapped__(engine)

    def test_start_launches_proxy_and_obm_proxy_on_master(self, monkeypatch, tmp_path):
        Logger.inference_log_file = tmp_path / "inference.log"
        infos = []
        monkeypatch.setattr(
            "node_runner.src.inference.external_api.engine.SubprocessManager",
            _FakeSubprocessManager,
        )
        monkeypatch.setattr(
            "node_runner.src.inference.external_api.engine.Logger.info",
            infos.append,
        )
        engine = _engine(_config("openai", "https://api.openai.com"))
        engine.obm_proxy_manager = _FakeObmProxy()
        monkeypatch.setattr(engine, "wait_for_proxy_to_be_ready", lambda: True)

        assert engine.start() is None

        assert engine.subprocess_manager is not None
        assert engine.subprocess_manager.started is True
        assert engine.subprocess_manager.name == "External API proxy"
        assert engine.obm_proxy_manager.started is True
        assert any("External API proxy started successfully" in msg for msg in infos)

    def test_start_launches_proxy_without_obm_proxy_on_master(
        self, monkeypatch, tmp_path
    ):
        Logger.inference_log_file = tmp_path / "inference.log"
        monkeypatch.setattr(
            "node_runner.src.inference.external_api.engine.SubprocessManager",
            _FakeSubprocessManager,
        )
        engine = _engine(_config("openai", "https://api.openai.com"))
        monkeypatch.setattr(engine, "wait_for_proxy_to_be_ready", lambda: True)

        engine.start()

        assert engine.subprocess_manager is not None
        assert engine.subprocess_manager.started is True
        assert engine.obm_proxy_manager is None

    def test_stop_stops_obm_proxy_and_subprocess(self):
        engine = _engine(_config("openai", "https://api.openai.com"))
        engine.obm_proxy_manager = _FakeObmProxy()
        subprocess_manager = _FakeSubprocessManager([], None, None, "proxy")
        engine.subprocess_manager = subprocess_manager

        assert engine.stop() is None

        assert engine.obm_proxy_manager.stopped is True
        assert subprocess_manager.stopped is True
        assert engine.subprocess_manager is None

    def test_stop_stops_subprocess_without_obm_proxy(self):
        engine = _engine(_config("openai", "https://api.openai.com"))
        subprocess_manager = _FakeSubprocessManager([], None, None, "proxy")
        engine.subprocess_manager = subprocess_manager

        engine.stop()

        assert subprocess_manager.stopped is True
        assert engine.subprocess_manager is None

    def test_stop_without_subprocess_still_stops_obm_proxy(self):
        engine = _engine(_config("openai", "https://api.openai.com"))
        engine.obm_proxy_manager = _FakeObmProxy()

        assert engine.stop() is None
        assert engine.obm_proxy_manager.stopped is True


class TestExternalApiProxyMappers:
    def test_openai_mapper_headers_and_request_mapping(self):
        mapper = OpenAiMapper()
        request = SGLangChatCompletionRequest.model_validate(
            {
                "model": "local",
                "messages": [{"role": "user", "content": "hi"}],
                "functions": [{"name": "lookup"}],
                "function_call": {"name": "lookup"},
                "top_k": 10,
            }
        )

        assert mapper.request_headers(None) == {"Content-Type": "application/json"}
        assert mapper.request_headers("key")["Authorization"] == "Bearer key"
        assert mapper.request_headers("Bearer key")["Authorization"] == "Bearer key"

        mapped = mapper.map_request(request, model="upstream-model")
        assert mapped["model"] == "upstream-model"
        assert mapped["tools"] == [{"type": "function", "function": {"name": "lookup"}}]
        assert mapped["tool_choice"] == {
            "type": "function",
            "function": {"name": "lookup"},
        }
        assert "top_k" not in mapped

    def test_openai_mapper_response_switches_to_tool_calls(self):
        mapper = OpenAiMapper()
        payload = {
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": "lookup",
                            "arguments": '{"id": 1}',
                        },
                    },
                    "finish_reason": "function_call",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }

        mapped = mapper.map_response(payload, headers={}, prefer_tool_calls=True)

        assert mapped["choices"][0]["finish_reason"] == "tool_calls"
        assert mapped["choices"][0]["message"]["tool_calls"][0]["function"] == {
            "name": "lookup",
            "arguments": '{"id": 1}',
        }

    def test_ift_mapper_maps_request_and_response_to_openai_shape(self):
        mapper = IftMapper()
        request = SGLangChatCompletionRequest.model_validate(
            {
                "model": "local",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"id": 1}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": '{"ok": true}',
                    },
                ],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "tool_choice": "required",
                "max_tokens": 10,
            }
        )

        assert mapper.request_headers(None) == {"Content-Type": "application/json"}
        assert mapper.request_headers("key")["Authorization"] == "Bearer key"
        mapped_request = mapper.map_request(request, model="ift-model")
        assert mapped_request["model"] == "ift-model"
        assert mapped_request["model_options"]["max_tokens"] == 10
        assert mapped_request["tool_config"] == {"mode": "forced"}
        assert mapped_request["messages"][2]["content"][0]["function_result"] == {
            "name": "lookup",
            "result": '{"ok": true}',
        }

        mapped_response = mapper.map_response(
            {
                "model": "ift-model",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"text": "answer"},
                            {
                                "function_call": {
                                    "name": "lookup",
                                    "arguments": {"id": 1},
                                }
                            },
                        ],
                    }
                ],
                "finish_reason": "tool_calls",
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
            headers={"X-Request-ID": "req-1"},
            prefer_tool_calls=True,
        )

        assert mapped_response["choices"][0]["finish_reason"] == "tool_calls"
        assert mapped_response["choices"][0]["message"]["tool_calls"][0][
            "function"
        ] == {
            "name": "lookup",
            "arguments": '{"id": 1}',
        }
        assert mapped_response["metadata"]["x_request_id"] == "req-1"


class TestExternalApiEngineAsSchedulerEngine:
    def test_the_api_holds_no_nodes_and_ignores_membership(self):
        engine = _engine(_config("openai", "https://api.openai.com"))
        units = engine.units()

        engine.attach(units)
        engine.detach(units)

        assert [unit.ranks for unit in units] == [()]
        assert engine.units(from_tail=True) == units
        assert engine.load(units) is None
        assert engine.worker_urls == []

    def test_the_proxy_can_be_released_and_brought_up_again(
        self, monkeypatch, tmp_path
    ):
        Logger.inference_log_file = tmp_path / "inference.log"
        events: list[str] = []

        class _Process(_FakeSubprocessManager):
            def start(self):
                events.append("api_proxy_start")

            def graceful_stop(self):
                events.append("api_proxy_stop")

        class _ObmProxy(_FakeObmProxy):
            def start(self):
                events.append("obm_start")

            def stop(self):
                events.append("obm_stop")

        monkeypatch.setattr(
            "node_runner.src.inference.external_api.engine.SubprocessManager", _Process
        )
        engine = _engine(_config("openai", "https://api.openai.com"))
        engine.obm_proxy_manager = _ObmProxy()
        monkeypatch.setattr(engine, "wait_for_proxy_to_be_ready", lambda: True)

        engine.start()
        engine.stop()
        engine.start()
        engine.close()
        engine.close()

        assert events == [
            "api_proxy_start",
            "obm_start",
            "obm_stop",
            "api_proxy_stop",
            "api_proxy_start",
            "obm_start",
            "obm_stop",
            "api_proxy_stop",
            "obm_stop",
        ]
