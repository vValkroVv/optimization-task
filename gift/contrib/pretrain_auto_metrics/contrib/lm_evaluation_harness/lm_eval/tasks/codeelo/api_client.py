import base64
import gzip
import json
import os
import random
import re
import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


NAPI_SOURCE_TYPE_BY_LANGID = {
    91: "cpp.gcc14-64-msys2-g++23",
    70: "python.pypy3-64",
    87: "java21",
    75: "rust.2021",
    32: "go",
    88: "kotlin19",
    79: "csharp.dotnet-sdk-6",
    34: "v8.3",
    55: "v8.nodejs",
    67: "ruby.3",
    28: "d",
    51: "pas.pascalabc",
    12: "haskell.ghc",
    13: "perl.5",
    19: "ocaml",
    3: "pas.dpr",
    6: "php.5",
    20: "scala",
    43: "c.gcc11",
}

PROB_RE = re.compile(r"(\d+)([A-Z]\d?)")

_FINAL_VERDICTS_NOT = {"PENDING", "TESTING", "RUNNING"}


def _retrying_session(status_forcelist: list) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=20,
        connect=5,
        read=5,
        backoff_factor=5,
        backoff_jitter=10,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"],
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def _source_type(lang_id: int) -> str:
    source_type = NAPI_SOURCE_TYPE_BY_LANGID.get(lang_id)
    if source_type is None:
        raise ValueError(
            f"no NAPI sourceType mapped for lang_id={lang_id}; "
            f"add it to NAPI_SOURCE_TYPE_BY_LANGID"
        )
    return source_type


def _parse_prob(prob: str) -> tuple:
    m = PROB_RE.match(prob)
    if not m:
        raise ValueError(f"bad prob format: {prob!r}")
    return int(m.group(1)), m.group(2)


def _unwrap_problems(data: Any) -> list:
    if isinstance(data, dict):
        return data.get("napiProblemStatements") or []
    return data or []


class _BaseClient:
    def poll_until_verdict(
        self,
        submission_id: str,
        interval: float = 10.0,
        max_wait_s: float = 600.0,
    ) -> str:
        start = time.time()
        while time.time() - start < max_wait_s:
            v = self.get_verdict(submission_id)
            if v not in _FINAL_VERDICTS_NOT:
                return v
            time.sleep(interval)
        raise TimeoutError(
            f"submission {submission_id} not judged within {max_wait_s}s"
        )

    def submit_code(self, prob: str, lang_id: int, code: str) -> str:
        raise NotImplementedError

    def get_verdict(self, submission_id: str) -> str:
        raise NotImplementedError

    def judge(self, prob: str, lang_id: int, code: str) -> str:
        return self.poll_until_verdict(self.submit_code(prob, lang_id, code))

    def standings(self, contest_id: int) -> dict:
        raise RuntimeError("contest standings are served only by codeforces_proxy")


class CodeforcesNAPIClient(_BaseClient):
    def __init__(self) -> None:
        self.token = os.environ["CODEELO_TOKEN"]
        self.base_url = os.environ.get(
            "CODEELO_BASE_URL", "https://codeforces.com/api/v2"
        )
        self.session = _retrying_session([429, 500, 502, 503, 504])
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
        )

    def get_problems(self, contest_id: int) -> list:
        r = self.session.get(
            f"{self.base_url}/problemStatements",
            params={"contestId": contest_id},
            timeout=180,
        )
        r.raise_for_status()
        return _unwrap_problems(r.json())

    def submit_code(self, prob: str, lang_id: int, code: str) -> str:
        contest_id, problem_index = _parse_prob(prob)
        r = self.session.post(
            f"{self.base_url}/submissions",
            json={
                "contestId": contest_id,
                "problemIndex": problem_index,
                "sourceType": _source_type(lang_id),
                "sourceText": code,
            },
            timeout=60,
        )
        r.raise_for_status()
        return str(r.json()["napiSubmissionId"])

    def get_verdict(self, submission_id: str) -> str:
        r = self.session.get(
            f"{self.base_url}/submissionReports",
            params={"napiSubmissionIds": submission_id},
            timeout=30,
        )
        r.raise_for_status()
        reports = r.json().get("napiSubmissionReports") or []
        if not reports:
            return "PENDING"
        return reports[0].get("verdict") or "PENDING"


def _env(suffix: str, default: str) -> str:
    return (
        os.environ.get("CODEELO_TASKPROXY_" + suffix)
        or os.environ.get("YT_TASKPROXY_" + suffix)
        or default
    )


class CodeforcesProxyClient(_BaseClient):
    def __init__(self) -> None:
        base_url = (
            os.environ.get("CODEELO_TASKPROXY_BASE_URL")
            or os.environ.get("YT_TASKPROXY_BASE_URL")
            or ""
        )
        yt_token = os.environ.get("YT_TOKEN")
        missing = [
            name
            for name, value in (
                ("YT_TOKEN", yt_token),
                ("CODEELO_TASKPROXY_BASE_URL or YT_TASKPROXY_BASE_URL", base_url),
            )
            if not value
        ]
        if missing:
            raise RuntimeError("codeelo proxy transport needs " + ", ".join(missing))

        self.url = base_url.rstrip("/") + "/execute"
        self.session = _retrying_session([429, 500, 502, 503, 504])
        self.session.headers.update(
            {
                "Authorization": f"Bearer {yt_token}",
                "x-yt-taskproxy-operation-alias": _env(
                    "OPERATION_ALIAS", "codeforces_proxy"
                ),
                "x-yt-taskproxy-task-name": _env("TASK_NAME", "run_codeforces_proxy"),
                "x-yt-taskproxy-service": _env("SERVICE", "port0"),
                "Content-Type": "application/json",
                "Connection": "close",
            }
        )

    _RETRY_STATUSES = (403, 422)
    _MAX_ATTEMPTS = 5

    def _execute(self, payload: dict, timeout: float) -> dict:
        action = payload.get("action")
        last_error = None
        for attempt in range(self._MAX_ATTEMPTS):
            if attempt:
                time.sleep(0.5 * attempt)
            try:
                r = self.session.post(
                    self.url, json={"input": payload}, timeout=timeout
                )
            except requests.exceptions.RequestException as exc:
                last_error = f"codeforces_proxy action={action} failed: {exc}"
                time.sleep(random.uniform(15, 30))
                continue
            if r.status_code in self._RETRY_STATUSES:
                last_error = (
                    f"codeforces_proxy action={action} failed with "
                    f"{r.status_code}: {r.text[:500]}"
                )
                continue
            if r.status_code >= 400:
                raise RuntimeError(
                    f"codeforces_proxy action={action} "
                    f"failed with {r.status_code}: {r.text[:500]}"
                )
            try:
                return r.json().get("result", {}).get("output", {})
            except ValueError as exc:
                last_error = (
                    f"codeforces_proxy action={action} returned unparseable JSON: {exc}"
                )
                continue
        raise RuntimeError(f"{last_error} (after {self._MAX_ATTEMPTS} attempts)")

    @staticmethod
    def _gz64(text: str) -> str:
        return base64.b64encode(gzip.compress(text.encode("utf-8"))).decode("ascii")

    def get_problems(self, contest_id: int) -> list:
        output = self._execute(
            {
                "action": "get_problems",
                "contest_id": int(contest_id),
                "languages": ["en"],
                "resources": "text",
                "response_encoding": "gzip+base64",
            },
            timeout=180,
        )
        if output.get("encoding") != "gzip+base64":
            return _unwrap_problems(output.get("problems"))
        raw = gzip.decompress(
            base64.b64decode(output["problems_encoded"], validate=True)
        )
        problems = json.loads(raw)
        expected = output.get("problem_count")
        if expected is not None and len(problems) != expected:
            raise RuntimeError(
                f"codeforces_proxy returned {len(problems)} problems for contest "
                f"{contest_id}, expected {expected}"
            )
        return problems

    def judge(
        self, prob: str, lang_id: int, code: str, max_wait_s: float = 1800.0
    ) -> str:
        payload = {
            "action": "judge",
            "prob": prob,
            "source_type": _source_type(lang_id),
            "code": self._gz64(code),
            "encoding": "gzip+base64",
            "wait": 10,
        }
        deadline = time.time() + max_wait_s
        while True:
            output = self._execute(payload, timeout=60)
            if output.get("status") == "done":
                return output.get("verdict") or "PENDING"
            if time.time() > deadline:
                raise TimeoutError(
                    f"{prob} submission {output.get('submission_id')} "
                    f"not judged within {max_wait_s}s"
                )

    def standings(self, contest_id: int, max_wait_s: float = 900.0) -> dict:
        payload = {"action": "standings", "contest_id": int(contest_id), "wait": 10}
        deadline = time.time() + max_wait_s
        while True:
            output = self._execute(payload, timeout=60)
            if output.get("status") == "done":
                raw = gzip.decompress(
                    base64.b64decode(output["standings_encoded"], validate=True)
                )
                return json.loads(raw)
            if time.time() > deadline:
                raise TimeoutError(
                    f"standings for contest {contest_id} not ready within {max_wait_s}s"
                )


_CLIENT: Optional[_BaseClient] = None
_CLIENT_PID: Optional[int] = None


def _use_proxy() -> bool:
    transport = os.environ.get("CODEELO_TRANSPORT", "").strip().lower()
    if transport in ("proxy", "taskproxy"):
        return True
    if transport in ("direct", "napi"):
        return False
    if transport:
        raise ValueError(f"unknown CODEELO_TRANSPORT: {transport!r}")
    return bool(
        os.environ.get("CODEELO_TASKPROXY_BASE_URL")
        or os.environ.get("YT_TASKPROXY_BASE_URL")
    )


def get_client() -> _BaseClient:
    global _CLIENT, _CLIENT_PID
    pid = os.getpid()
    if _CLIENT is None or _CLIENT_PID != pid:
        _CLIENT = CodeforcesProxyClient() if _use_proxy() else CodeforcesNAPIClient()
        _CLIENT_PID = pid
    return _CLIENT
