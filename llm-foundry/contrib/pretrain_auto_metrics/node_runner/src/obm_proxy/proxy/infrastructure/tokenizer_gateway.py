from __future__ import annotations

from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer

from ..config import ProxySettings


class TokenizerGateway:
    def __init__(self, settings: ProxySettings):
        self._settings = settings
        self._tokenizer: Optional[AutoTokenizer] = None

    def get(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    def _load(self) -> None:
        print(f"[obm_proxy] Loading tokenizer from {self._settings.tokenizer_path}")
        tok = AutoTokenizer.from_pretrained(
            self._settings.tokenizer_path, trust_remote_code=True
        )

        if self._settings.chat_template_path:
            template_path = Path(self._settings.chat_template_path)
            if not template_path.exists():
                print(
                    f"[obm_proxy] Warning: chat-template file {template_path} not found."
                )
            else:
                template_content = template_path.read_text(encoding="utf-8")
                try:
                    tok.chat_template = template_content
                except Exception:
                    setattr(tok, "chat_template", template_content)
                print(
                    "[obm_proxy] Loaded chat template into tokenizer.chat_template from file."
                )

        self._tokenizer = tok

    def apply_chat_template(self, *args, **kwargs) -> list[int]:
        return self.get().apply_chat_template(*args, **kwargs)
