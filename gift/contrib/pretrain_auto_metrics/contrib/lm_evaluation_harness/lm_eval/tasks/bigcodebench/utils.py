import base64
import gzip
import json
import logging
import os
import random
import time
from typing import Any, Dict, List

import requests


eval_logger = logging.getLogger(__name__)

PROGRAM_TIMEOUT_S = 120
WAIT_S = 10
POLL_DEADLINE_S = 900

_TESTS_PASSED_MARKER = "BCB_TESTS_PASSED"

_TEST_RUNNER = f"""

if __name__ == "__main__":
    import sys
    import unittest

    suite = unittest.TestLoader().loadTestsFromTestCase(TestCases)
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
    passed = result.wasSuccessful() and result.testsRun > 0
    if passed:
        print({_TESTS_PASSED_MARKER!r})
    sys.exit(0 if passed else 1)
"""


_session: requests.Session | None = None
_session_pid: int | None = None
_config: dict | None = None


def _get_session_and_config() -> tuple[requests.Session, dict]:
    global _session, _config, _session_pid
    if _session is not None and _session_pid == os.getpid():
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
            "BigCodeBench code validation will return service errors.",
            ", ".join(missing_variables),
        )
        return None, None

    operation_alias = os.environ.get("BCB_TASKPROXY_OPERATION_ALIAS", "bcb_executor")
    task_name = os.environ.get("BCB_TASKPROXY_TASK_NAME", "run_bcb_executor")
    service = os.environ.get("BCB_TASKPROXY_SERVICE", "port0")

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {yt_token}",
            "x-yt-taskproxy-operation-alias": operation_alias,
            "x-yt-taskproxy-task-name": task_name,
            "x-yt-taskproxy-service": service,
            "Content-Type": "application/json",
        }
    )

    _config = {"url": f"{base_url.rstrip('/')}/execute"}

    eval_logger.info(
        "BigCodeBench executor session initialized: url=%s, alias=%s, task=%s",
        _config["url"],
        operation_alias,
        task_name,
    )

    _session = session
    _session_pid = os.getpid()
    return _session, _config


def _call(session: requests.Session, url: str, payload: dict, name: str) -> dict:
    deadline = time.monotonic() + POLL_DEADLINE_S
    while True:
        try:
            response = session.post(url, json=payload, timeout=60)
        except requests.exceptions.RequestException as exc:
            response = None
            reason = str(exc)
        else:
            if response.status_code == 200:
                return response.json()["result"]["output"]
            if response.status_code not in (403, 404, 429, 500, 502, 503, 504):
                response.raise_for_status()
            reason = f"HTTP {response.status_code}: {response.text[:200]}"

        if time.monotonic() > deadline:
            raise RuntimeError(f"bcb_executor gave up on {name}: {reason}")
        retry_after = response.headers.get("Retry-After") if response is not None else None
        delay = float(retry_after) if retry_after else random.uniform(5, 15)
        eval_logger.warning("Retrying %s in %.0fs: %s", name, delay, reason)
        time.sleep(delay)


def build_reference(doc: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "task_id": doc["task_id"],
            "test": doc["test"],
            "entry_point": doc.get("entry_point", ""),
        }
    )


def extract_code(resps: List[List[str]], docs: List[Dict[str, Any]]) -> List[List[str]]:
    return [
        [doc.get("complete_prompt", "") + r for r in resp_list]
        for resp_list, doc in zip(resps, docs)
    ]


def pass_at_1(references: List[str], predictions: List[List[str]]) -> float:
    if os.environ.get("BCB_CACHE_ONLY"):
        return 0.0
    session, config = _get_session_and_config()
    if session is None:
        raise RuntimeError(
            "BigCodeBench executor session not initialized "
            "(required environment variables are missing)"
        )

    ref = json.loads(references[0])
    name = ref.get("task_id", "unknown")
    program = predictions[0][0] + "\n" + ref["test"] + _TEST_RUNNER

    program_gz64 = base64.b64encode(gzip.compress(program.encode("utf-8"))).decode("ascii")
    submit = {
        "input": {
            "program": program_gz64,
            "encoding": "gzip+base64",
            "timeout": PROGRAM_TIMEOUT_S,
            "wait": WAIT_S,
        }
    }
    output = _call(session, config["url"], submit, name)
    started = time.monotonic()
    while output.get("status") == "pending":
        if time.monotonic() - started > POLL_DEADLINE_S:
            raise RuntimeError(f"bcb_executor did not finish {name} within {POLL_DEADLINE_S}s")
        output = _call(session, config["url"], submit, name)

    passed = _TESTS_PASSED_MARKER in output.get("stdout", "")
    if not passed:
        eval_logger.debug(
            "FAIL %s: status=%s exit_code=%s stderr=%.200s",
            name,
            output.get("status"),
            output.get("exit_code"),
            output.get("stderr", ""),
        )
    return 1.0 if passed else 0.0
