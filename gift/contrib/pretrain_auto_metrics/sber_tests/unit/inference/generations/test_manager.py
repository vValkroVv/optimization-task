import sys
from pathlib import Path

import pytest
from tenacity import stop_after_attempt, wait_fixed

import requests

from node_runner.src.inference.generations.manager import GenerationsManager

pytestmark = pytest.mark.unit

PORT = 35000
DUMP_DIR = Path("/tmp/generations")
UPSTREAM = "http://127.0.0.1:30000"


class _Response:
    def __init__(self, status_code: int):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(str(self.status_code))


def _manager(upstream_url: str | None = None) -> GenerationsManager:
    return GenerationsManager(
        port=PORT, dump_dir=DUMP_DIR, log_profile="evaluated", upstream_url=upstream_url
    )


@pytest.mark.parametrize(
    ("upstream_url", "extra"),
    [(None, []), (UPSTREAM, ["--upstream-url", UPSTREAM])],
    ids=["replay", "record"],
)
def test_an_upstream_is_what_turns_the_server_into_a_recorder(upstream_url, extra):
    assert _manager(upstream_url).bash_cmd == [
        sys.executable,
        "-m",
        "src.inference.generations.server",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--dump-dir",
        DUMP_DIR.as_posix(),
        "--profile",
        "evaluated",
        *extra,
    ]


@pytest.mark.parametrize(("status", "ready"), [(200, True), (500, False)])
def test_the_wait_ends_only_on_a_ready_server(
    monkeypatch: pytest.MonkeyPatch, status, ready
):
    manager = _manager()
    monkeypatch.setattr(
        "node_runner.src.inference.generations.manager.requests.get",
        lambda url, timeout=None: _Response(status),
    )
    retry = manager.wait_for_server_to_initialize.retry
    monkeypatch.setattr(retry, "stop", stop_after_attempt(1))
    monkeypatch.setattr(retry, "wait", wait_fixed(0))

    if ready:
        manager.wait_for_server_to_initialize()
    else:
        with pytest.raises(requests.HTTPError):
            manager.wait_for_server_to_initialize()


def test_start_runs_the_server_from_node_runner_and_stop_ends_it(
    monkeypatch: pytest.MonkeyPatch,
):
    manager = _manager(UPSTREAM)
    events: list[str] = []

    class _SubprocessManager:
        def __init__(self, cmd, log_file, cwd, run_name=""):
            assert cwd.endswith("node_runner")

        def start(self):
            events.append("start")

        def graceful_stop(self):
            events.append("stop")

    monkeypatch.setattr(
        "node_runner.src.inference.generations.manager.SubprocessManager",
        _SubprocessManager,
    )
    monkeypatch.setattr(
        manager, "wait_for_server_to_initialize", lambda: events.append("wait")
    )

    manager.start()
    manager.stop()
    manager.stop()

    assert events == ["start", "wait", "stop"]
