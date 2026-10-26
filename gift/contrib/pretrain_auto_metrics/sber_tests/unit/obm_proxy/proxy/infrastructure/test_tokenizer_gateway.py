from pathlib import Path

import pytest

from node_runner.src.obm_proxy.proxy.config import ProxySettings
from node_runner.src.obm_proxy.proxy.infrastructure.tokenizer_gateway import (
    TokenizerGateway,
)

pytestmark = pytest.mark.unit


class _DummyTokenizer:
    def __init__(self):
        self.chat_template = None
        self.calls = []

    def apply_chat_template(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [1, 2, 3]


class _FailFirstSetTokenizer(_DummyTokenizer):
    def __init__(self):
        object.__setattr__(self, "_fail_once", True)
        object.__setattr__(self, "chat_template", None)
        object.__setattr__(self, "calls", [])

    def __setattr__(self, name, value):
        if name == "chat_template" and getattr(self, "_fail_once", False):
            object.__setattr__(self, "_fail_once", False)
            raise RuntimeError("first set fails")
        object.__setattr__(self, name, value)


def _settings(tmp_path: Path, **overrides) -> ProxySettings:
    base = dict(
        sglang_port=30000,
        tokenizer_path=str(tmp_path / "tok"),
        reasoning_token_budget=0,
        template_version_override=None,
        chat_template_kwargs={},
        chat_template_path=None,
        stats_enabled=False,
        stats_interval_sec=60,
        chat_handlers_mode="pretrain",
        logging_level="info",
    )
    base.update(overrides)
    return ProxySettings(**base)


class TestTokenizerGateway:
    def test_get_loads_tokenizer_once_and_caches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        created = {"count": 0}

        def _fake_from_pretrained(path: str, **kwargs):
            created["count"] += 1
            assert path == str(tmp_path / "tok")
            assert kwargs == {"trust_remote_code": True}
            return _DummyTokenizer()

        monkeypatch.setattr(
            "node_runner.src.obm_proxy.proxy.infrastructure.tokenizer_gateway.AutoTokenizer.from_pretrained",
            _fake_from_pretrained,
        )

        gateway = TokenizerGateway(_settings(tmp_path))
        tok1 = gateway.get()
        tok2 = gateway.get()

        assert tok1 is tok2
        assert created["count"] == 1

    def test_load_applies_chat_template_from_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        template_path = tmp_path / "chat_template.jinja"
        template_path.write_text("{{ message }}", encoding="utf-8")
        tokenizer = _DummyTokenizer()
        monkeypatch.setattr(
            "node_runner.src.obm_proxy.proxy.infrastructure.tokenizer_gateway.AutoTokenizer.from_pretrained",
            lambda _path, **_kwargs: tokenizer,
        )

        gateway = TokenizerGateway(
            _settings(tmp_path, chat_template_path=str(template_path))
        )
        loaded = gateway.get()

        assert loaded is tokenizer
        assert loaded.chat_template == "{{ message }}"

    def test_load_warns_when_chat_template_file_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        tokenizer = _DummyTokenizer()
        missing_path = tmp_path / "missing_template.jinja"
        monkeypatch.setattr(
            "node_runner.src.obm_proxy.proxy.infrastructure.tokenizer_gateway.AutoTokenizer.from_pretrained",
            lambda _path, **_kwargs: tokenizer,
        )

        gateway = TokenizerGateway(
            _settings(tmp_path, chat_template_path=str(missing_path))
        )
        gateway.get()
        output = capsys.readouterr().out

        assert "Warning: chat-template file" in output
        assert tokenizer.chat_template is None

    def test_load_falls_back_to_setattr_when_first_assignment_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        template_path = tmp_path / "chat_template.jinja"
        template_path.write_text("{{ fallback }}", encoding="utf-8")
        tokenizer = _FailFirstSetTokenizer()
        monkeypatch.setattr(
            "node_runner.src.obm_proxy.proxy.infrastructure.tokenizer_gateway.AutoTokenizer.from_pretrained",
            lambda _path, **_kwargs: tokenizer,
        )

        gateway = TokenizerGateway(
            _settings(tmp_path, chat_template_path=str(template_path))
        )
        gateway.get()

        assert tokenizer.chat_template == "{{ fallback }}"

    def test_apply_chat_template_delegates_to_loaded_tokenizer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        tokenizer = _DummyTokenizer()
        monkeypatch.setattr(
            "node_runner.src.obm_proxy.proxy.infrastructure.tokenizer_gateway.AutoTokenizer.from_pretrained",
            lambda _path, **_kwargs: tokenizer,
        )
        gateway = TokenizerGateway(_settings(tmp_path))

        out = gateway.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tools=None,
            add_generation_prompt=True,
            tokenize=True,
        )

        assert out == [1, 2, 3]
        assert len(tokenizer.calls) == 1
