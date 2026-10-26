import json
import runpy
import sys

import pytest
from fastapi.testclient import TestClient

from node_runner.src.inference.generations import server
from node_runner.src.inference.generations.store import ShardWriter

pytestmark = pytest.mark.unit

UPSTREAM = "http://127.0.0.1:30000"
BODY = {"input_ids": [1, 2]}
ANSWER = '{"text": "hi", "meta_info": {"id": "abc"}}'


class _UpstreamResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code
        self.text = ANSWER
        self.content = ANSWER.encode("utf-8")
        self.headers = {"content-type": "application/json"}


class _FakeUpstream:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url, json=None, headers=None):
        self.calls.append((url, json))
        self.headers = headers
        return _UpstreamResponse(self.status_code)

    async def aclose(self):
        pass


def _record(dump_dir, status=200):
    writer = ShardWriter(dump_dir, "evaluated")
    writer.write(path="/generate", body=BODY, status_code=status, response=ANSWER)
    writer.close()
    return dump_dir


class TestRecording:
    def test_routing_header_is_forwarded_but_not_part_of_replay_identity(
        self, tmp_path
    ):
        upstream = _FakeUpstream()
        with TestClient(
            server.recording_app(tmp_path, "evaluated", UPSTREAM)
        ) as client:
            client.app.state.http_client = upstream
            client.post(
                "/generate",
                json=BODY,
                headers={
                    "x-smg-routing-key": "dialog-a",
                    "authorization": "private-key",
                },
            )
        assert upstream.headers == {"x-smg-routing-key": "dialog-a"}
        replay = TestClient(server.replay_app(tmp_path, "evaluated"))
        response = replay.post(
            "/generate", json=BODY, headers={"x-smg-routing-key": "dialog-b"}
        )
        assert (response.status_code, response.text) == (200, ANSWER)

    @pytest.mark.parametrize("status", [200, 503])
    def test_an_answer_is_forwarded_untouched_and_replays_as_it_was(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, status
    ):
        upstream = _FakeUpstream(status)
        with TestClient(
            server.recording_app(tmp_path, "evaluated", UPSTREAM)
        ) as client:
            client.app.state.http_client = upstream
            recorded = client.post("/generate", json=BODY)

        replayed = TestClient(server.replay_app(tmp_path, "evaluated")).post(
            "/generate", json=BODY
        )

        assert upstream.calls == [(f"{UPSTREAM}/generate", BODY)]
        assert (recorded.status_code, recorded.text) == (status, ANSWER)
        assert (replayed.status_code, replayed.text) == (status, ANSWER)


class TestReplay:
    def test_a_miss_is_not_worth_retrying_and_says_why(self, tmp_path):
        app = server.replay_app(_record(tmp_path), "evaluated")

        response = TestClient(app).post("/generate", json={"input_ids": [9]})

        assert response.status_code == 404
        error = response.json()["error"]
        assert error.startswith(server.MISS_MARKER)
        assert "profile=evaluated path=/generate" in error
        assert json.dumps({"input_ids": [9]}) in error

    def test_a_long_request_is_cut_short_in_the_message(self, tmp_path):
        app = server.replay_app(_record(tmp_path), "evaluated")

        response = TestClient(app).post(
            "/generate", json={"input_ids": list(range(server.BODY_PREVIEW_CHARS))}
        )

        assert "... (truncated)" in response.json()["error"]


@pytest.mark.parametrize("upstream", [True, False], ids=["record", "replay"])
def test_readiness_is_answered_locally(tmp_path, upstream):
    dump_dir = _record(tmp_path)
    app = (
        server.recording_app(dump_dir, "evaluated", UPSTREAM)
        if upstream
        else server.replay_app(dump_dir, "evaluated")
    )

    with TestClient(app) as client:
        fake = _FakeUpstream()
        client.app.state.http_client = fake
        response = client.get("/ready")

    assert response.json() == {"ready": True}
    assert fake.calls == []


@pytest.mark.parametrize(
    ("upstream", "title"),
    [(True, "Generation recorder"), (False, "Generation replay")],
    ids=["record", "replay"],
)
def test_running_the_module_serves_the_app_its_arguments_ask_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path, upstream, title
):
    argv = ["server", "--host", "127.0.0.1", "--port", "30000"]
    argv += ["--dump-dir", str(_record(tmp_path)), "--profile", "evaluated"]
    if upstream:
        argv += ["--upstream-url", f"{UPSTREAM}/"]
    monkeypatch.setattr(sys, "argv", argv)
    served: list[tuple] = []
    monkeypatch.setattr(
        "uvicorn.run", lambda app, **kwargs: served.append((app.title, kwargs))
    )

    runpy.run_module(server.__name__, run_name="__main__")

    assert served == [
        (title, {"host": "127.0.0.1", "port": 30000, "log_level": "error"})
    ]
