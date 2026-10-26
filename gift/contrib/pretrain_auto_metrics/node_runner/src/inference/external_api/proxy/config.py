from __future__ import annotations

import json
import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExternalApiProxySettings:
    host: str
    port: int
    formatting: str
    base_url: str
    model: str
    api_key: str | None
    user_pass: str | None
    timeout: float
    logging_level: str
    max_retries: int
    retry_wait_sec: float
    max_concurrency: int
    request_customization: dict[str, Any]


def load_settings() -> ExternalApiProxySettings:
    parser = ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", "--proxy-port", dest="port", type=int, required=True)
    parser.add_argument("--formatting", choices=("openai", "ift"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument(
        "--logging-level",
        choices=("debug", "info", "warning", "error", "critical"),
        default="info",
    )
    parser.add_argument("--max-retries", type=int, required=True)
    parser.add_argument("--retry-wait-sec", type=float, required=True)
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument(
        "--request-customization",
        default="{}",
    )
    args, _ = parser.parse_known_args()

    request_customization = json.loads(args.request_customization)
    if not isinstance(request_customization, dict):
        raise RuntimeError("--request-customization must be a JSON object")

    return ExternalApiProxySettings(
        host=args.host,
        port=args.port,
        formatting=args.formatting,
        base_url=args.base_url,
        model=args.model,
        api_key=os.environ.get("EXTERNAL_API_PROXY_API_KEY"),
        user_pass=os.environ.get("EXTERNAL_API_PROXY_USER_PASS"),
        timeout=args.timeout,
        logging_level=args.logging_level,
        max_retries=args.max_retries,
        retry_wait_sec=args.retry_wait_sec,
        max_concurrency=args.max_concurrency,
        request_customization=request_customization,
    )


def log_settings(settings: ExternalApiProxySettings, logger: logging.Logger) -> None:
    customization_keys = ", ".join(sorted(settings.request_customization)) or "none"
    logger.info("Host: %s", settings.host)
    logger.info("Port: %s", settings.port)
    logger.info("Formatting: %s", settings.formatting)
    logger.info("Base URL: %s", settings.base_url)
    logger.info("Model: %s", settings.model)
    logger.info("Timeout: %s", settings.timeout)
    logger.info("Logging level: %s", settings.logging_level)
    logger.info("Max retries: %s", settings.max_retries)
    logger.info("Retry wait sec: %s", settings.retry_wait_sec)
    logger.info("Max concurrency: %s", settings.max_concurrency)
    logger.info("Request customization keys: %s", customization_keys)
