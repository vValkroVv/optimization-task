import contextlib
import io
import json
import runpy
import shlex
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from job_submitter.infrastructure.job_dispatcher import request as jd_request
from job_submitter.infrastructure.job_dispatcher.request import build_send_command, send

pytestmark = pytest.mark.unit

MODULE_PATH = str(Path(jd_request.__file__).resolve())


@pytest.fixture
def urlopen(monkeypatch):
    urlopen = SimpleNamespace(calls=[], errors=[])

    def fake_urlopen(request, timeout, context):
        urlopen.calls.append(request)
        if urlopen.errors:
            raise urlopen.errors.pop(0)
        return contextlib.nullcontext()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    return urlopen


def test_main_sends_request_from_stdin(monkeypatch, urlopen) -> None:
    request = {"method": "POST", "url": "https://jd/api/add", "data": {"step": 1}, "headers": {"Authorization": "t"}}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request)))

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(MODULE_PATH, run_name="__main__")

    assert exc.value.code == 0
    assert json.loads(urlopen.calls[0].data) == {"step": 1}
    assert urlopen.calls[0].get_header("Authorization") == "t"


def test_send_fails_after_all_attempts(urlopen, capsys) -> None:
    urlopen.errors.extend(urllib.error.URLError("down") for _ in range(3))

    assert send("POST", "https://jd/api/add", {"step": 1}) is False
    assert len(urlopen.calls) == 3


def test_build_send_command_passes_request_to_module_stdin() -> None:
    command = build_send_command("POST", "https://jd/api/add", {"step": 1})

    python, module, here_string, payload = shlex.split(command)
    assert (python, module, here_string) == ("python3", MODULE_PATH, "<<<")
    assert json.loads(payload) == {"method": "POST", "url": "https://jd/api/add", "data": {"step": 1}, "headers": {}}
