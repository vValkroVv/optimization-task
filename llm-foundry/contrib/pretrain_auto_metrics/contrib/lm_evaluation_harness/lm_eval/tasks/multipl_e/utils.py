"""
Utilities for MultiPL-E HumanEval lm_eval tasks.

Generation phase:
    - doc_to_text uses {{prompt}} directly (already formatted for the target language).
    - build_reference packs (language, tests, name) into a JSON string used as the
      reference passed to pass_at_1.
    - extract_code is a filter that prepends the prompt to each generation, producing
      a complete runnable program. Stop-token truncation is NOT needed: each per-language
      yaml passes the language-specific stop_tokens via generation_kwargs.until, which
      sglang honors as server-side stop sequences.

Validation phase:
    - pass_at_1 POSTs generated code + tests to the multipl_e_executor service
      via YT TaskProxy (gzip-compressed then base64, encoding="gzip+base64") and
      returns 1.0 on success
      (status OK, exit_code 0) or 0.0.
      Raises on any service/network error to stop the benchmark immediately.
      Requires env vars: YT_TOKEN, YT_TASKPROXY_BASE_URL.
"""

import base64
import gzip
import json
import logging
import os
import random
import time
from typing import Any, Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


eval_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy HTTP session singleton for the multipl_e_executor service
# ---------------------------------------------------------------------------

_session: requests.Session | None = None
_config: dict | None = None


def _get_session_and_config() -> tuple[requests.Session, dict]:
    """Lazily initialize a requests.Session with retry logic and load config from env vars."""
    global _session, _config
    if _session is not None:
        return _session, _config

    yt_token = os.environ.get("YT_TOKEN")
    base_url = os.environ.get("YT_TASKPROXY_BASE_URL")
    missing_variables = [
        name
        for name, value in (
            ("YT_TOKEN", yt_token),
            ("YT_TASKPROXY_BASE_URL", base_url),
        )
        if not value
    ]
    if missing_variables:
        eval_logger.warning(
            "Required environment variables are not set: %s. "
            "MultiPL-E code validation will return service errors.",
            ", ".join(missing_variables),
        )
        return None, None

    operation_alias = os.environ.get(
        "YT_TASKPROXY_OPERATION_ALIAS", "multipl_e_executor"
    )
    task_name = os.environ.get("YT_TASKPROXY_TASK_NAME", "run_multipl_e_executor")
    service = os.environ.get("YT_TASKPROXY_SERVICE", "port0")

    session = requests.Session()
    retries = Retry(
        total=20,
        connect=5,
        read=5,
        backoff_factor=5,
        backoff_jitter=10,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    session.headers.update(
        {
            "Authorization": f"Bearer {yt_token}",
            "x-yt-taskproxy-operation-alias": operation_alias,
            "x-yt-taskproxy-task-name": task_name,
            "x-yt-taskproxy-service": service,
            "Content-Type": "application/json",
        }
    )

    _config = {
        "url": f"{base_url.rstrip('/')}/execute",
    }

    eval_logger.info(
        "MultiPL-E executor session initialized: url=%s, alias=%s, task=%s",
        _config["url"],
        operation_alias,
        task_name,
    )

    _session = session
    return _session, _config


def build_reference(doc: Dict[str, Any]) -> str:
    """Pack (language, tests, name) into a JSON string used as the reference for pass_at_1."""
    return json.dumps(
        {
            "language": doc["language"],
            "tests": doc.get("tests", ""),
            "name": doc.get("name", ""),
        }
    )


def extract_code(resps: List[List[str]], docs: List[Dict[str, Any]]) -> List[List[str]]:
    """Prepend the prompt to each generated completion to form a runnable program."""
    return [
        [doc.get("prompt", "") + r for r in resp_list]
        for resp_list, doc in zip(resps, docs)
    ]


def pass_at_1(references: List[str], predictions: List[List[str]]) -> float:
    """
    POST generated code to the multipl_e_executor.

    Returns 1.0 if the program passed all tests, 0.0 otherwise.
    Raises on any service/network error to stop the benchmark immediately.
    """
    session, config = _get_session_and_config()
    if session is None:
        if os.environ.get("MULTIPL_E_CACHE_ONLY"):
            return 0.0
        raise RuntimeError(
            "MultiPL-E executor session not initialized "
            "(required environment variables are missing)"
        )

    ref = json.loads(references[0])
    language = ref["language"]
    tests = ref.get("tests", "")
    name = ref.get("name", "unknown")

    generated_code = predictions[0][0]  # single generation per doc (pass@1)
    full_program = generated_code + "\n" + tests

    program_gz64 = base64.b64encode(gzip.compress(full_program.encode("utf-8"))).decode(
        "ascii"
    )
    payload = {
        "input": {
            "language": language,
            "program": program_gz64,
            "encoding": "gzip+base64",
            "timeout": 13,
        }
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = session.post(config["url"], json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            delay = random.uniform(15, 30)
            eval_logger.warning(
                "Retry %d/%d for %s (%s): %s — sleeping %.1fs",
                attempt + 1,
                max_retries,
                name,
                language,
                e,
                delay,
            )
            time.sleep(delay)

    output = data.get("result", {}).get("output", {})
    passed = output.get("status") == "OK" and output.get("exit_code") == 0

    if not passed:
        eval_logger.debug(
            "FAIL %s: status=%s exit_code=%s stderr=%.200s",
            name,
            output.get("status"),
            output.get("exit_code"),
            output.get("stderr", ""),
        )

    return 1.0 if passed else 0.0
