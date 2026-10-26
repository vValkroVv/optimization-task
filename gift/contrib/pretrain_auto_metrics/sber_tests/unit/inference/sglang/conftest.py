from types import SimpleNamespace

import pytest
import requests
from tenacity import stop_after_attempt, wait_none

from node_runner.src.inference.sglang import sglang_engine, sglang_router_manager
from node_runner.src.inference.sglang.sglang_inference import SglangInference
from node_runner.src.inference.sglang.sglang_router_manager import SglangRouter

READINESS_ATTEMPTS = 5
FOUR_NODES = {0: "10.0.0.0", 1: "10.0.0.1", 2: "10.0.0.2", 3: "10.0.0.3"}


def arg(engine, flag: str) -> str | None:
    bash = engine.inference.bash_cmd_list
    return bash[bash.index(flag) + 1] if flag in bash else None


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


class FakeRouter:
    def __init__(self, events: list[str]):
        self.events = events
        self.launches: list[list[str]] = []
        self.log_files: list = []
        self.calls: list[tuple[str, str, dict | None]] = []
        self.workers: list[dict] = []
        self.alive = False
        self.unhealthy_polls = 0
        self.unhealthy: set[str] = set()
        self.post_status = 200
        self.workers_status = 200
        self.workers_payload = None
        self._next_id = 0

    @property
    def served(self) -> list[str]:
        return sorted(worker["url"] for worker in self.workers)

    def mutations(self) -> list[tuple[str, str, dict | None]]:
        return [call for call in self.calls if call[0] != "GET"]

    def die(self) -> None:
        self.alive = False

    def _add(self, url: str) -> None:
        self._next_id += 1
        self.workers.append({"id": f"w{self._next_id}", "url": url})

    def process(self, cmd, log_file, cwd):
        router = self
        self.log_files.append(log_file)

        class _Process:
            def start(self):
                router.events.append("router_start")
                router.launches.append(cmd)
                router.alive = True
                router.workers = []
                for token in cmd:
                    if token.startswith("http://"):
                        router._add(token)

            def is_alive(self):
                return router.alive

            def raise_for_exit_code(self):
                raise RuntimeError("router process failed")

            def graceful_stop(self):
                if router.alive:
                    router.events.append("router_stop")
                router.alive = False

        return _Process()

    def _path(self, url: str) -> str:
        if not self.alive:
            raise requests.ConnectionError(f"router is down: {url}")
        return "/" + url.split("/", 3)[3]

    def get(self, url, timeout):
        path = self._path(url)
        self.calls.append(("GET", path, None))
        if self.workers_payload is not None:
            return FakeResponse(self.workers_status, self.workers_payload)
        healthy = self.unhealthy_polls == 0
        self.unhealthy_polls = max(0, self.unhealthy_polls - 1)
        workers = [
            {**worker, "is_healthy": healthy and worker["url"] not in self.unhealthy}
            for worker in self.workers
        ]
        return FakeResponse(self.workers_status, {"workers": workers})

    def post(self, url, json, timeout):
        path = self._path(url)
        self.calls.append(("POST", path, json))
        if self.post_status < 400:
            self._add(json["url"])
        return FakeResponse(self.post_status, {})

    def delete(self, url, timeout):
        path = self._path(url)
        self.calls.append(("DELETE", path, None))
        worker_id = path.rsplit("/", 1)[-1]
        self.workers = [w for w in self.workers if w["id"] != worker_id]
        return FakeResponse(200, {})


@pytest.fixture
def events() -> list[str]:
    return []


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> FakeRouter:
    fake = FakeRouter(events)
    monkeypatch.setattr(sglang_router_manager, "SubprocessManager", fake.process)
    monkeypatch.setattr(
        sglang_router_manager,
        "requests",
        SimpleNamespace(
            get=fake.get,
            post=fake.post,
            delete=fake.delete,
            exceptions=requests.exceptions,
        ),
    )
    readiness = SglangRouter.wait_for_router_to_initialize.retry
    monkeypatch.setattr(readiness, "wait", wait_none())
    monkeypatch.setattr(readiness, "stop", stop_after_attempt(READINESS_ATTEMPTS))
    return fake


def dp_load(running, waiting, capacity=192, usage=0.1, **disaggregation):
    load = {
        "num_running_reqs": running,
        "num_waiting_reqs": waiting,
        "max_running_requests": capacity,
        "token_usage": usage,
    }
    if disaggregation:
        load["disaggregation"] = disaggregation
    return load


class FakeWorkers:
    def __init__(self):
        self.replies: dict[str, object] = {}
        self.calls: list[tuple[str, float | None]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, timeout))
        reply = self.replies.get(url.removesuffix("/v1/loads"), [])
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, FakeResponse):
            return reply
        return FakeResponse(200, {"loads": reply})


@pytest.fixture
def workers(monkeypatch: pytest.MonkeyPatch) -> FakeWorkers:
    fake = FakeWorkers()
    monkeypatch.setattr(
        sglang_engine,
        "requests",
        SimpleNamespace(get=fake.get, RequestException=requests.RequestException),
    )
    return fake


@pytest.fixture
def scrapers(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> list[tuple]:
    built: list[tuple] = []

    class _Scraper:
        def __init__(self, log_profile, model_name, targets):
            self.sources = [target.source for target in targets]
            built.extend((target.source, target.url) for target in targets)

        def start(self):
            events.extend(f"scraper_start:{source}" for source in self.sources)

        def stop(self):
            events.extend(f"scraper_stop:{source}" for source in self.sources)

    monkeypatch.setattr(sglang_engine, "MetricsScraper", _Scraper)
    return built


@pytest.fixture
def skip_dg_cache_setup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        SglangInference, "_setup_deepgemm_cache_dir", lambda self, nfs, local: None
    )


class FakeProxy:
    def __init__(self, name: str, events: list[str], port: int = 0):
        self.name = name
        self.events = events
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"

    def start(self):
        self.events.append(f"{self.name}_start")

    def stop(self):
        self.events.append(f"{self.name}_stop")


@pytest.fixture
def record_worker(monkeypatch: pytest.MonkeyPatch, events: list[str]):

    def record(engine) -> None:
        for method, event in (
            ("start_inference", "inference_start"),
            ("wait_for_inference_to_initialize", "inference_ready"),
            ("copy_dg_cache_to_nfs", "dg_copy"),
            ("stop", "inference_stop"),
        ):
            monkeypatch.setattr(
                engine.inference, method, lambda event=event: events.append(event)
            )

    return record


@pytest.fixture
def proxies(events: list[str]):

    def install(engine) -> None:
        engine.obm_proxy_manager = FakeProxy("obm", events)
        engine.generations_manager = FakeProxy("recorder", events)

    return install
