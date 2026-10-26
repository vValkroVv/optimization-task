import json
import os
from pathlib import Path

import pytest
from tenacity import stop_after_attempt, wait_fixed

from node_runner.src.config_model import ObmProxyConfig
from node_runner.src.obm_proxy.obm_proxy_manager import (
    ObmProxyManager,
    ObmProxyStartError,
)

pytestmark = pytest.mark.unit

# ============================================================================
# Constants for test configuration
# ============================================================================

# Port numbers
SGLANG_PORT = 30000
API_PORT = 29999
TEST_API_PORT = 31234

# Configuration values
REASONING_TOKEN_BUDGET = 128
DEFAULT_REASONING_TOKEN_BUDGET = 0
WORKERS_COUNT = 8

# Server and template configuration
CHAT_HANDLERS_MODE = "custom"
CHAT_TEMPLATE = "template.j2"
HF_CHECKPOINT_PATH = Path("/tmp/hf_checkpoint")
CONFIG_TOKENIZER_PATH = Path("/tmp/config_tokenizer")

# Environment variable names
ENV_TOKENIZER_PATH = "OBM_PROXY_TOKENIZER_PATH"
ENV_CHAT_HANDLERS_MODE = "OBM_PROXY_CHAT_HANDLERS_MODE"
ENV_CHAT_TEMPLATE = "OBM_PROXY_CHAT_TEMPLATE"
ENV_CHAT_TEMPLATE_KWARGS = "OBM_PROXY_CHAT_TEMPLATE_KWARGS"
ENV_SGLANG_PORT = "OBM_PROXY_SGLANG_PORT"
ENV_REASONING_TOKEN_BUDGET = "OBM_PROXY_REASONING_TOKEN_BUDGET"
ENV_LOGGING_LEVEL = "OBM_PROXY_LOGGING_LEVEL"

# Test values
USE_DEVSYSTEM = True

# URL and path constants
READY_URL_TEMPLATE = "http://localhost:{port}/ready"
PROXY_LOG_FILENAME = "proxy.log"
CHAT_COMPLETIONS_URL = "http://127.0.0.1:30000/v1/chat/completions"
COMPLETIONS_URL = "http://127.0.0.1:30000/v1/completions"
GENERATE_URL = "http://127.0.0.1:30000/generate"

# ============================================================================
# Helper functions
# ============================================================================


def create_test_config() -> ObmProxyConfig:
    """Create test OBM proxy configuration."""
    return ObmProxyConfig(
        sglang_port=SGLANG_PORT,
        api_port=API_PORT,
        chat_handlers_mode=CHAT_HANDLERS_MODE,
        chat_template=CHAT_TEMPLATE,
        chat_template_kwargs={"use_devsystem": USE_DEVSYSTEM},
        reasoning_token_budget=REASONING_TOKEN_BUDGET,
        workers=WORKERS_COUNT,
    )


def _setup_manager(
    config: ObmProxyConfig | None = None,
    hf_checkpoint_path: Path = HF_CHECKPOINT_PATH,
) -> ObmProxyManager:
    """Helper to setup OBM proxy manager with test configuration."""
    manager = ObmProxyManager()
    manager.setup(
        config or create_test_config(),
        hf_checkpoint_path,
        chat_completions_url=CHAT_COMPLETIONS_URL,
        completions_url=COMPLETIONS_URL,
        generate_url=GENERATE_URL,
    )
    return manager


class _Resp:
    def __init__(self, status_code: int):
        self.status_code = status_code


class TestObmProxyManager:
    def test_base_url_uses_localhost_and_api_port(self) -> None:
        manager = ObmProxyManager()
        manager.api_port = TEST_API_PORT
        assert manager.base_url == f"http://127.0.0.1:{TEST_API_PORT}"

    def test_setup_with_config_populates_fields(self) -> None:
        config = create_test_config()
        manager = _setup_manager(config)
        assert manager.api_port == API_PORT
        assert manager.chat_handlers_mode == CHAT_HANDLERS_MODE
        assert manager.tokenizer_path == HF_CHECKPOINT_PATH
        assert manager.bash_cmd[:4] == [
            "python3",
            "-m",
            "uvicorn",
            "obm_proxy.proxy.main:app",
        ]

    def test_setup_sets_default_reasoning_token_budget_when_missing(self) -> None:
        config = ObmProxyConfig(
            sglang_port=SGLANG_PORT,
            api_port=API_PORT,
            chat_handlers_mode=CHAT_HANDLERS_MODE,
            chat_template=CHAT_TEMPLATE,
            chat_template_kwargs={"use_devsystem": USE_DEVSYSTEM},
            workers=WORKERS_COUNT,
        )
        manager = _setup_manager(config)
        assert manager.reasoning_token_budget == DEFAULT_REASONING_TOKEN_BUDGET

    def test_setup_uses_config_tokenizer_path_when_provided(self) -> None:
        config = create_test_config().model_copy(
            update={"tokenizer_path": str(CONFIG_TOKENIZER_PATH)}
        )
        manager = _setup_manager(config)
        assert manager.tokenizer_path == CONFIG_TOKENIZER_PATH

    def test_setup_builds_expected_proxy_envs(self) -> None:
        manager = _setup_manager()
        envs = manager.proxy_envs
        assert envs[ENV_TOKENIZER_PATH] == HF_CHECKPOINT_PATH.absolute().as_posix()
        assert envs[ENV_CHAT_HANDLERS_MODE] == CHAT_HANDLERS_MODE
        assert envs[ENV_CHAT_TEMPLATE] == CHAT_TEMPLATE
        assert json.loads(envs[ENV_CHAT_TEMPLATE_KWARGS]) == {
            "use_devsystem": USE_DEVSYSTEM
        }
        assert envs[ENV_SGLANG_PORT] == str(SGLANG_PORT)
        assert envs[ENV_REASONING_TOKEN_BUDGET] == str(REASONING_TOKEN_BUDGET)
        assert envs[ENV_LOGGING_LEVEL] == "info"

    def test_setup_uses_config_logging_level(self) -> None:
        config = create_test_config().model_copy(update={"logging_level": "debug"})
        manager = _setup_manager(config)
        assert manager.proxy_envs[ENV_LOGGING_LEVEL] == "debug"
        assert manager.bash_cmd[manager.bash_cmd.index("--log-level") + 1] == "debug"

    def test_setup_does_not_mutate_os_environ(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV_TOKENIZER_PATH, raising=False)
        monkeypatch.delenv(ENV_SGLANG_PORT, raising=False)
        _setup_manager()
        assert os.getenv(ENV_TOKENIZER_PATH) is None
        assert os.getenv(ENV_SGLANG_PORT) is None

    def test_proxy_envs_use_config_tokenizer_path_when_provided(self) -> None:
        config = create_test_config().model_copy(
            update={"tokenizer_path": str(CONFIG_TOKENIZER_PATH)}
        )
        manager = _setup_manager(config)
        assert (
            manager.proxy_envs[ENV_TOKENIZER_PATH]
            == CONFIG_TOKENIZER_PATH.absolute().as_posix()
        )

    def test_proxy_envs_set_template_version_when_provided(self) -> None:
        config = create_test_config().model_copy(update={"template_version": "v4"})
        manager = _setup_manager(config)
        assert manager.proxy_envs["OBM_PROXY_TEMPLATE_VERSION"] == "v4"

    def test_subprocess_env_drops_stale_obm_proxy_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_CHAT_TEMPLATE, "stale-template")
        monkeypatch.setenv("OBM_PROXY_TEMPLATE_VERSION", "stale-version")
        monkeypatch.setenv("UNRELATED_VAR", "kept")
        config = ObmProxyConfig(
            sglang_port=SGLANG_PORT,
            api_port=API_PORT,
            chat_handlers_mode=CHAT_HANDLERS_MODE,
            chat_template=None,
            chat_template_kwargs={},
            workers=WORKERS_COUNT,
        )
        manager = _setup_manager(config)
        env = manager._subprocess_env()
        assert ENV_CHAT_TEMPLATE not in env
        assert "OBM_PROXY_TEMPLATE_VERSION" not in env
        assert env["UNRELATED_VAR"] == "kept"
        assert env[ENV_SGLANG_PORT] == str(SGLANG_PORT)

    @pytest.mark.parametrize(
        ("status_code", "should_raise"),
        [
            (200, False),
            (500, True),
        ],
    )
    def test_wait_for_obm_proxy_to_initialize(
        self, monkeypatch: pytest.MonkeyPatch, status_code: int, should_raise: bool
    ) -> None:
        manager = ObmProxyManager()
        manager.api_port = API_PORT
        monkeypatch.setattr(
            "node_runner.src.obm_proxy.obm_proxy_manager.requests.get",
            lambda url, timeout=None: _Resp(status_code),
        )
        if should_raise:
            monkeypatch.setattr(
                manager.wait_for_obm_proxy_to_initialize.retry,
                "stop",
                stop_after_attempt(1),
            )
            monkeypatch.setattr(
                manager.wait_for_obm_proxy_to_initialize.retry,
                "wait",
                wait_fixed(0),
            )
            with pytest.raises(ObmProxyStartError):
                manager.wait_for_obm_proxy_to_initialize()
            return
        assert manager.wait_for_obm_proxy_to_initialize() is True

    def test_wait_for_obm_proxy_to_initialize_uses_expected_ready_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = {"url": None}

        manager = ObmProxyManager()
        manager.api_port = TEST_API_PORT

        def _fake_get(url: str, timeout=None) -> _Resp:
            captured["url"] = url
            return _Resp(200)

        monkeypatch.setattr(
            "node_runner.src.obm_proxy.obm_proxy_manager.requests.get",
            _fake_get,
        )
        assert manager.wait_for_obm_proxy_to_initialize() is True
        assert captured["url"] == READY_URL_TEMPLATE.format(port=TEST_API_PORT)

    def test_start_runs_subprocess_and_waits(self, monkeypatch: pytest.MonkeyPatch):
        manager = _setup_manager()
        called = {"start": 0, "wait": 0}
        captured_env = {}

        class _SP:
            def __init__(self, cmd, log_file, cwd, run_name="", env=None):
                assert cmd == manager.bash_cmd
                # cwd is node_runner/src so that "obm_proxy.proxy.main:app" resolves
                assert cwd.endswith("src")
                captured_env.update(env or {})

            def start(self):
                called["start"] += 1

        monkeypatch.setattr(
            "node_runner.src.obm_proxy.obm_proxy_manager.SubprocessManager", _SP
        )
        monkeypatch.setattr(
            manager,
            "wait_for_obm_proxy_to_initialize",
            lambda: called.__setitem__("wait", 1),
        )
        manager.start()
        assert called == {"start": 1, "wait": 1}
        assert manager.subprocess_manager is not None
        assert captured_env[ENV_SGLANG_PORT] == str(SGLANG_PORT)
        assert captured_env[ENV_CHAT_TEMPLATE] == CHAT_TEMPLATE

    def test_stop_branches(self):
        manager = ObmProxyManager()
        manager.stop()
        assert manager.subprocess_manager is None

        manager = _setup_manager()
        manager.stop()
        assert manager.subprocess_manager is None

        class _SP:
            def __init__(self):
                self.stopped = 0

            def graceful_stop(self):
                self.stopped += 1

        sp = _SP()
        manager.subprocess_manager = sp
        manager.stop()
        assert sp.stopped == 1
