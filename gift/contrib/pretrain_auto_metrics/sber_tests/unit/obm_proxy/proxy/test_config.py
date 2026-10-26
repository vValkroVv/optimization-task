import os

import pytest

from node_runner.src.obm_proxy.proxy.config import (
    ProxySettings,
    env,
    env_bool,
    env_int,
    env_json_dict,
    env_required,
    load_settings,
    print_settings,
)

pytestmark = pytest.mark.unit


def _clear_obm_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("OBM_PROXY_"):
            monkeypatch.delenv(key, raising=False)


class TestProxyConfigEnvHelpers:
    def test_env_reads_prefixed_variable_or_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _clear_obm_proxy_env(monkeypatch)
        assert env("FOO", "fallback") == "fallback"
        monkeypatch.setenv("OBM_PROXY_FOO", "bar")
        assert env("FOO", "fallback") == "bar"

    def test_env_required_rejects_missing_or_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _clear_obm_proxy_env(monkeypatch)
        with pytest.raises(RuntimeError, match="Missing required env var"):
            env_required("TOKENIZER_PATH")
        monkeypatch.setenv("OBM_PROXY_TOKENIZER_PATH", "")
        with pytest.raises(RuntimeError, match="Missing required env var"):
            env_required("TOKENIZER_PATH")

    def test_env_int_parsing_and_validation(self, monkeypatch: pytest.MonkeyPatch):
        _clear_obm_proxy_env(monkeypatch)
        assert env_int("SGLANG_PORT", 30000) == 30000
        monkeypatch.setenv("OBM_PROXY_SGLANG_PORT", "32000")
        assert env_int("SGLANG_PORT", 30000) == 32000
        monkeypatch.setenv("OBM_PROXY_SGLANG_PORT", "bad")
        with pytest.raises(RuntimeError, match="Invalid int in env"):
            env_int("SGLANG_PORT", 30000)

    @pytest.mark.parametrize(
        ("raw_value", "expected"),
        [
            ("1", True),
            ("true", True),
            ("yes", True),
            ("on", True),
            ("0", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_env_bool_parsing(
        self, monkeypatch: pytest.MonkeyPatch, raw_value: str, expected: bool
    ):
        _clear_obm_proxy_env(monkeypatch)
        monkeypatch.setenv("OBM_PROXY_STATS_ENABLED", raw_value)
        assert env_bool("STATS_ENABLED", default=True) is expected

    def test_env_json_dict_parsing_and_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _clear_obm_proxy_env(monkeypatch)
        assert env_json_dict("CHAT_TEMPLATE_KWARGS", default={"x": 1}) == {"x": 1}
        monkeypatch.setenv("OBM_PROXY_CHAT_TEMPLATE_KWARGS", '{"reasoning": true}')
        assert env_json_dict("CHAT_TEMPLATE_KWARGS", default={}) == {"reasoning": True}

        monkeypatch.setenv("OBM_PROXY_CHAT_TEMPLATE_KWARGS", "[1,2,3]")
        with pytest.raises(RuntimeError, match="must be a JSON object"):
            env_json_dict("CHAT_TEMPLATE_KWARGS", default={})

        monkeypatch.setenv("OBM_PROXY_CHAT_TEMPLATE_KWARGS", "{broken")
        with pytest.raises(RuntimeError, match="Invalid JSON in env"):
            env_json_dict("CHAT_TEMPLATE_KWARGS", default={})


class TestProxyConfigLoading:
    def test_load_settings_uses_defaults_and_required_values(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _clear_obm_proxy_env(monkeypatch)
        monkeypatch.setenv("OBM_PROXY_TOKENIZER_PATH", "/tmp/tokenizer")

        settings = load_settings()

        assert settings == ProxySettings(
            sglang_port=30000,
            tokenizer_path="/tmp/tokenizer",
            reasoning_token_budget=0,
            template_version_override=None,
            chat_template_kwargs={},
            chat_template_path=None,
            stats_enabled=True,
            stats_interval_sec=60,
            chat_handlers_mode="pretrain",
            logging_level="info",
            chat_completions_url="http://127.0.0.1:30000/v1/chat/completions",
            completions_url="http://127.0.0.1:30000/v1/completions",
            generate_url="http://127.0.0.1:30000/generate",
        )

    def test_load_settings_applies_overrides_and_derived_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _clear_obm_proxy_env(monkeypatch)
        monkeypatch.setenv("OBM_PROXY_TOKENIZER_PATH", "/tmp/tokenizer")
        monkeypatch.setenv("OBM_PROXY_SGLANG_PORT", "30123")
        monkeypatch.setenv("OBM_PROXY_REASONING_TOKEN_BUDGET", "256")
        monkeypatch.setenv("OBM_PROXY_TEMPLATE_VERSION", "v4")
        monkeypatch.setenv(
            "OBM_PROXY_CHAT_TEMPLATE_KWARGS",
            '{"reasoning": true, "use_devsystem": false}',
        )
        monkeypatch.setenv("OBM_PROXY_CHAT_TEMPLATE", "/tmp/template.j2")
        monkeypatch.setenv("OBM_PROXY_STATS_ENABLED", "false")
        monkeypatch.setenv("OBM_PROXY_STATS_INTERVAL_SEC", "0")
        monkeypatch.setenv("OBM_PROXY_CHAT_HANDLERS_MODE", "opensource")
        monkeypatch.setenv("OBM_PROXY_LOGGING_LEVEL", "debug")

        settings = load_settings()

        assert settings.sglang_port == 30123
        assert settings.reasoning_token_budget == 256
        assert settings.template_version_override == "v4"
        assert settings.chat_template_kwargs == {
            "reasoning": True,
            "use_devsystem": False,
        }
        assert settings.chat_template_path == "/tmp/template.j2"
        assert settings.stats_enabled is False
        assert settings.stats_interval_sec == 1
        assert settings.chat_handlers_mode == "opensource"
        assert settings.logging_level == "debug"

    @pytest.mark.parametrize("bad_version", ["v3", "latest"])
    def test_load_settings_rejects_unsupported_template_version(
        self, monkeypatch: pytest.MonkeyPatch, bad_version: str
    ):
        _clear_obm_proxy_env(monkeypatch)
        monkeypatch.setenv("OBM_PROXY_TOKENIZER_PATH", "/tmp/tokenizer")
        monkeypatch.setenv("OBM_PROXY_TEMPLATE_VERSION", bad_version)
        with pytest.raises(
            RuntimeError, match="OBM_PROXY_TEMPLATE_VERSION must be one of"
        ):
            load_settings()

    def test_load_settings_rejects_unsupported_chat_handlers_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _clear_obm_proxy_env(monkeypatch)
        monkeypatch.setenv("OBM_PROXY_TOKENIZER_PATH", "/tmp/tokenizer")
        monkeypatch.setenv("OBM_PROXY_CHAT_HANDLERS_MODE", "invalid")
        with pytest.raises(
            RuntimeError, match="OBM_PROXY_CHAT_HANDLERS_MODE must be one of"
        ):
            load_settings()

    def test_print_settings_outputs_mode_and_conditional_sections(
        self, capsys: pytest.CaptureFixture[str]
    ):
        print_settings(
            ProxySettings(
                sglang_port=30000,
                tokenizer_path="/tmp/tokenizer",
                reasoning_token_budget=42,
                template_version_override="v4",
                chat_template_kwargs={"reasoning": True},
                chat_template_path="/tmp/template.j2",
                stats_enabled=True,
                stats_interval_sec=15,
                chat_handlers_mode="custom",
                logging_level="debug",
            )
        )
        output = capsys.readouterr().out
        assert "Template version: v4" in output
        assert "Reasoning token budget: 42" in output
        assert "Stats logging enabled, interval=15s" in output
        assert "Chat handlers mode: custom" in output
        assert "Logging level: debug" in output

    def test_print_settings_skips_optional_sections_when_disabled(
        self, capsys: pytest.CaptureFixture[str]
    ):
        print_settings(
            ProxySettings(
                sglang_port=30000,
                tokenizer_path="/tmp/tokenizer",
                reasoning_token_budget=42,
                template_version_override=None,
                chat_template_kwargs={},
                chat_template_path=None,
                stats_enabled=False,
                stats_interval_sec=15,
                chat_handlers_mode="pretrain",
                logging_level="info",
            )
        )
        output = capsys.readouterr().out
        assert "Reasoning token budget: 42" in output
        assert "Stats logging enabled" not in output
        assert "Chat handlers mode: pretrain" in output
