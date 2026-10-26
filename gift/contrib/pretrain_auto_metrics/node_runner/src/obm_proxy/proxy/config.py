from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional


SUPPORTED_TEMPLATE_VERSIONS = ("v4", "v4.1", "v5")
SUPPORTED_CHAT_HANDLERS_MODES = ("custom", "opensource", "pretrain")


@dataclass(frozen=True, slots=True)
class ProxySettings:
    sglang_port: int
    tokenizer_path: str
    reasoning_token_budget: int
    template_version_override: Optional[str]
    chat_template_kwargs: dict[str, Any]
    chat_template_path: Optional[str]
    stats_enabled: bool
    stats_interval_sec: int
    chat_handlers_mode: str
    logging_level: str
    chat_completions_url: str = ""
    completions_url: str = ""
    generate_url: str = ""
    sampling_params: dict[str, Any] = field(default_factory=dict)
    priority: int | None = None


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(f"OBM_PROXY_{key}", default)


def env_required(key: str) -> str:
    val = env(key)
    if val is None or val == "":
        raise RuntimeError(f"Missing required env var: OBM_PROXY_{key}")
    return val


def env_int(key: str, default: int) -> int:
    val = env(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError as e:
        raise RuntimeError(f"Invalid int in env OBM_PROXY_{key}={val!r}") from e


def env_bool(key: str, default: bool) -> bool:
    val = env(key)
    if val is None or val == "":
        return default
    return str(val).lower() in {"1", "true", "yes", "on"}


def env_json_dict(key: str, default: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    raw = env(key)
    if raw is None or raw == "":
        return default or {}
    try:
        parsed = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON in env OBM_PROXY_{key}: {e}") from e
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Env OBM_PROXY_{key} must be a JSON object (dict), got {type(parsed).__name__}"
        )
    return parsed


def load_settings() -> ProxySettings:
    template_version_override = env("TEMPLATE_VERSION", None)
    if (
        template_version_override is not None
        and template_version_override not in SUPPORTED_TEMPLATE_VERSIONS
    ):
        raise RuntimeError(
            f"OBM_PROXY_TEMPLATE_VERSION must be one of {SUPPORTED_TEMPLATE_VERSIONS}, "
            f"got {template_version_override!r}"
        )

    chat_template_kwargs = env_json_dict("CHAT_TEMPLATE_KWARGS", default={})
    chat_handlers_mode = env("CHAT_HANDLERS_MODE", "pretrain")
    if chat_handlers_mode not in SUPPORTED_CHAT_HANDLERS_MODES:
        raise RuntimeError(
            f"OBM_PROXY_CHAT_HANDLERS_MODE must be one of {SUPPORTED_CHAT_HANDLERS_MODES}, "
            f"got {chat_handlers_mode!r}"
        )
    logging_level = env("LOGGING_LEVEL", "info")

    sglang_port = env_int("SGLANG_PORT", 30000)
    default_base_url = f"http://127.0.0.1:{sglang_port}"

    return ProxySettings(
        sglang_port=sglang_port,
        tokenizer_path=env_required("TOKENIZER_PATH"),
        reasoning_token_budget=env_int("REASONING_TOKEN_BUDGET", 0),
        template_version_override=template_version_override,
        chat_template_kwargs=chat_template_kwargs,
        chat_template_path=env("CHAT_TEMPLATE", None),
        stats_enabled=env_bool("STATS_ENABLED", True),
        stats_interval_sec=max(1, env_int("STATS_INTERVAL_SEC", 60)),
        chat_handlers_mode=chat_handlers_mode,
        logging_level=logging_level,
        sampling_params=env_json_dict("SAMPLING_PARAMS"),
        chat_completions_url=env(
            "CHAT_COMPLETIONS_URL",
            f"{default_base_url}/v1/chat/completions",
        ),
        completions_url=env("COMPLETIONS_URL", f"{default_base_url}/v1/completions"),
        generate_url=env("GENERATE_URL", f"{default_base_url}/generate"),
    )


def configure_logging(settings: ProxySettings) -> logging.Logger:
    logger = logging.getLogger("obm_proxy")
    logger.setLevel(settings.logging_level.upper())
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    return logger


def print_settings(settings: ProxySettings) -> None:
    print(
        f"[obm_proxy] Template version: {settings.template_version_override or 'auto-detect'}"
    )
    print(f"[obm_proxy] Chat template path: {settings.chat_template_path}")
    print(f"[obm_proxy] SGLang port: {settings.sglang_port}")
    print(f"[obm_proxy] Chat completions URL: {settings.chat_completions_url}")
    print(f"[obm_proxy] Completions URL: {settings.completions_url}")
    print(f"[obm_proxy] Generate URL: {settings.generate_url}")
    print(f"[obm_proxy] Tokenizer path: {settings.tokenizer_path}")
    print(f"[obm_proxy] Reasoning token budget: {settings.reasoning_token_budget}")
    print(f"[obm_proxy] Sampling overrides: {settings.sampling_params}")
    if settings.stats_enabled:
        print(
            f"[obm_proxy] Stats logging enabled, interval={settings.stats_interval_sec}s"
        )
    print(f"[obm_proxy] Chat handlers mode: {settings.chat_handlers_mode}")
    print(f"[obm_proxy] Logging level: {settings.logging_level}")
