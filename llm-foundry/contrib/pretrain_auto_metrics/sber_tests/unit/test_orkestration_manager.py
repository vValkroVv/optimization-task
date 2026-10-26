import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import requests
from fastapi.testclient import TestClient
from tenacity import wait_fixed

from node_runner.src import orkestration_manager as om_module
from node_runner.src.orkestration_manager import FastApiOrkestrationManager

pytestmark = pytest.mark.unit

PORT = 54321

servers: list["_Server"] = []


class _Server:

    def __init__(self, config):
        self.config = config
        self.should_exit = False
        self.serving = threading.Event()
        self.exited = threading.Event()
        servers.append(self)

    def run(self):
        self.serving.set()
        while not self.should_exit:
            time.sleep(0.01)
        self.exited.set()


@pytest.fixture
def uvicorn_spy(monkeypatch: pytest.MonkeyPatch) -> list[_Server]:
    servers.clear()
    monkeypatch.setattr(
        om_module.uvicorn, "Config", lambda app, **kw: SimpleNamespace(app=app, **kw)
    )
    monkeypatch.setattr(om_module.uvicorn, "Server", _Server)
    monkeypatch.setattr(
        FastApiOrkestrationManager.fetch_op.retry, "wait", wait_fixed(0.01)
    )
    monkeypatch.setattr(
        FastApiOrkestrationManager.wait_barrier.retry, "wait", wait_fixed(0.01)
    )
    return servers


def _setup(node_name: str, rank: int, nnodes: int = 2) -> FastApiOrkestrationManager:
    om = FastApiOrkestrationManager()
    om.setup(
        node_name=node_name,
        orkestration_port=PORT,
        global_nnodes=nnodes,
        global_node_rank=rank,
        global_dist_init_host="master",
    )
    return om


def _route_http_to(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    monkeypatch.setattr(
        om_module,
        "requests",
        SimpleNamespace(
            get=lambda url, timeout: client.get(urlsplit(url).path),
            post=lambda url, json, timeout: client.post(urlsplit(url).path, json=json),
        ),
    )


@pytest.fixture
def cluster(uvicorn_spy, monkeypatch: pytest.MonkeyPatch):
    master = _setup("mpimaster-0", rank=0)
    client = TestClient(uvicorn_spy[0].config.app)
    _route_http_to(monkeypatch, client)
    worker = _setup("mpiworker-0", rank=1)
    yield master, worker, client
    master.cleanup()


def _in_thread(fn, *args) -> tuple[threading.Thread, list]:
    result: list = []
    thread = threading.Thread(target=lambda: result.append(fn(*args)), daemon=True)
    thread.start()
    return thread, result


def _post_barrier(client: TestClient, node: str, stage: str, expected: int) -> dict:
    body = {"node_name": node, "stage": stage, "expected": expected, "node_rank": 0}
    return client.post("/barrier", json=body).json()


def test_master_serves_the_app_until_cleanup(uvicorn_spy):
    master = _setup("mpimaster-0", rank=0)

    assert (master.is_master, master.is_worker) == (True, False)
    assert [server.config.port for server in uvicorn_spy] == [PORT]
    assert uvicorn_spy[0].serving.wait(timeout=2)

    master.cleanup()

    assert uvicorn_spy[0].exited.is_set()


def test_worker_does_not_serve_and_reads_registered_nodes_from_master(cluster):
    master, worker, _ = cluster
    master.wait_barrier("REGISTRATION", expected=1)

    assert (worker.is_master, worker.is_worker) == (False, True)
    assert len(servers) == 1
    assert worker.get_registered_nodes() == {0: "testclient"}
    assert master.get_registered_nodes() == {0: "testclient"}


def test_cleanup_survives_a_failing_server_thread_join(
    uvicorn_spy, monkeypatch: pytest.MonkeyPatch
):
    class _Thread(threading.Thread):
        def join(self, timeout=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(om_module.threading, "Thread", _Thread)
    master = _setup("mpimaster-0", rank=0)

    master.cleanup()

    assert uvicorn_spy[0].exited.wait(timeout=2)


def test_worker_cleanup_is_a_noop(uvicorn_spy):
    _setup("mpiworker-0", rank=1).cleanup()


def test_barrier_releases_once_every_expected_node_arrived(cluster):
    master, _, client = cluster

    assert master.barrier_released("S") is False
    first = _post_barrier(client, "mpimaster-0", "S", expected=2)
    assert master.barrier_released("S") is False
    second = _post_barrier(client, "mpiworker-0", "S", expected=2)

    assert (first["released"], first["arrived"]) == (False, 1)
    assert (second["released"], second["arrived"]) == (True, 2)
    assert master.barrier_released("S") is True


def test_barrier_expecting_one_node_releases_on_first_arrival(cluster):
    master, _, client = cluster

    response = _post_barrier(client, "mpiworker-0", "S_FAIL", expected=1)

    assert response["released"] is True
    assert master.barrier_released("S_FAIL") is True


def test_repeated_arrival_of_one_node_does_not_release_the_barrier(cluster):
    master, _, client = cluster

    _post_barrier(client, "mpiworker-0", "S", expected=2)
    repeated = _post_barrier(client, "mpiworker-0", "S", expected=2)

    assert (repeated["released"], repeated["arrived"]) == (False, 1)
    assert master.barrier_released("S") is False


def test_barrier_waits_for_the_largest_expected_count_and_never_unreleases(cluster):
    _, _, client = cluster

    _post_barrier(client, "a", "S", expected=2)
    _post_barrier(client, "b", "S", expected=3)
    third = _post_barrier(client, "c", "S", expected=1)
    released_early = _post_barrier(client, "a", "T", expected=1)
    later = _post_barrier(client, "b", "T", expected=3)

    assert (third["released"], third["expected"]) == (True, 3)
    assert released_early["released"] is True
    assert (later["released"], later["arrived"]) == (True, 2)


def test_wait_barrier_expects_every_node_by_default_and_blocks_until_released(
    cluster,
):
    master, worker, _ = cluster

    thread, result = _in_thread(worker.wait_barrier, "S")
    time.sleep(0.1)
    assert thread.is_alive()
    assert master.barrier_released("S") is False

    master.wait_barrier("S")
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result == [True]


def test_ops_are_served_by_index_once_published(cluster):
    master, _, client = cluster

    before = client.get("/ops/0").status_code
    indices = [master.publish_op({"action": a}) for a in ("up", "down", "finish")]

    assert before == 404
    assert indices == [0, 1, 2]
    assert client.get("/ops/1").json() == {"action": "down"}
    assert client.get("/ops/3").status_code == 404


def test_negative_op_index_is_not_found(cluster):
    master, _, client = cluster
    master.publish_op({"action": "up"})
    master.publish_op({"action": "finish"})

    assert client.get("/ops/-1").status_code == 404


def test_concurrent_publishes_get_unique_contiguous_indices(cluster):
    master, _, client = cluster
    indices: dict[int, int] = {}
    start = threading.Barrier(16)

    def publish(n):
        start.wait()
        indices[n] = master.publish_op({"n": n})

    threads = [threading.Thread(target=publish, args=(n,)) for n in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert sorted(indices.values()) == list(range(16))
    assert all(client.get(f"/ops/{i}").json() == {"n": n} for n, i in indices.items())


def test_fetch_op_waits_until_the_op_is_published(cluster):
    master, worker, _ = cluster

    thread, result = _in_thread(worker.fetch_op, 0)
    time.sleep(0.1)
    assert thread.is_alive()
    master.publish_op({"action": "finish"})
    thread.join(timeout=2)

    assert result == [{"action": "finish"}]


def test_fetch_op_retries_connection_errors(cluster, monkeypatch):
    master, worker, _ = cluster
    master.publish_op({"action": "finish"})
    routed_get = om_module.requests.get
    calls = []

    def flaky_get(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise requests.ConnectionError("master is restarting")
        return routed_get(url, timeout=timeout)

    monkeypatch.setattr(om_module.requests, "get", flaky_get)

    assert worker.fetch_op(0) == {"action": "finish"}
    assert len(calls) == 3
