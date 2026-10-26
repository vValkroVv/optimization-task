from pathlib import Path

import pytest
import requests

from node_runner.src.config_model import RouterConfig
from node_runner.src.inference.sglang.sglang_router_manager import (
    RouterStartError,
    SglangRouter,
)
from node_runner.src.logger import Logger

from .conftest import READINESS_ATTEMPTS

pytestmark = pytest.mark.unit

U0 = "http://h0:30001"
U1 = "http://h1:30001"
U2 = "http://h2:30001"
U3 = "http://h3:30001"


def _router(**kwargs) -> SglangRouter:
    return SglangRouter(
        RouterConfig(host="127.0.0.1", port=30000, max_concurrent_requests=128),
        **kwargs,
    )


def _polls(router) -> int:
    return sum(1 for call in router.calls if call[:2] == ("GET", "/workers"))


class TestSglangRouterLaunch:
    def test_the_router_launches_with_its_config_and_workers(self, router):
        _router().start_router({U1, U0})

        cmd = router.launches[0]
        assert cmd[:3] == [
            "/home/user/sglang/bin/python",
            "-m",
            "sglang_router.launch_router",
        ]
        assert "--max-concurrent-requests" in cmd
        assert "--queue-size" in cmd
        assert cmd[-3:] == ["--worker-urls", U0, U1]
        assert router.served == [U0, U1]

    def test_pd_router_splits_workers_into_prefill_and_decode(self, router):
        pd_router = _router(prefill_urls=frozenset({U0}), bootstrap_port=8998)

        pd_router.start_router({U0, U1})

        assert router.launches[0][-6:] == [
            "--pd-disaggregation",
            "--prefill",
            U0,
            "8998",
            "--decode",
            U1,
        ]

    def test_router_log_file_follows_profile(self, router, tmp_path: Path):
        Logger.node_log_dir = tmp_path

        _router(log_profile="judge").start_router(set())

        assert router.log_files == [tmp_path / "judge_router.log"]

    def test_start_returns_only_once_the_router_serves_every_worker(self, router):
        router.unhealthy_polls = 2

        _router().start_router({U0})

        assert _polls(router) == 3

    def test_a_dead_router_process_fails_without_waiting(self, router):
        sglang_router = _router()
        sglang_router.start_router({U0})
        polls_before = _polls(router)
        router.die()

        with pytest.raises(RuntimeError, match="router process failed"):
            sglang_router.wait_for_router_to_initialize(set())
        assert _polls(router) == polls_before

    @pytest.mark.parametrize(
        ("status", "payload"),
        [
            pytest.param(500, {"workers": []}, id="http_error"),
            pytest.param(200, {"workers": []}, id="missing_worker"),
            pytest.param(
                200,
                {"workers": [{"url": U0, "is_healthy": False}]},
                id="unhealthy_worker",
            ),
            pytest.param(200, ValueError("bad json"), id="invalid_json"),
        ],
    )
    def test_a_router_that_never_gets_ready_gives_up_after_its_budget(
        self, router, status, payload
    ):
        router.workers_status = status
        router.workers_payload = payload

        with pytest.raises(RouterStartError):
            _router().start_router({U0})
        assert _polls(router) == READINESS_ATTEMPTS

    def test_an_unreachable_router_is_retried_not_fatal(self, router, monkeypatch):
        sglang_router = _router()
        sglang_router.start_router({U0})
        answers = iter([requests.ConnectionError("refused")])
        get = router.get

        def _flaky_get(url, timeout):
            for error in answers:
                raise error
            return get(url, timeout)

        monkeypatch.setattr(
            "node_runner.src.inference.sglang.sglang_router_manager.requests.get",
            _flaky_get,
        )

        sglang_router.wait_for_router_to_initialize(set())

    def test_stop_stops_the_process_and_is_a_noop_without_one(self, router, events):
        _router().stop_router()
        sglang_router = _router()
        sglang_router.start_router({U0})

        sglang_router.stop_router()

        assert events == ["router_start", "router_stop"]
        assert not sglang_router.is_running


class TestSglangRouterMembership:
    def test_add_workers_registers_each_and_waits_until_all_are_healthy(self, router):
        sglang_router = _router()
        sglang_router.start_router({U0})
        router.unhealthy_polls = 2
        polls_before = _polls(router)

        sglang_router.add_workers({U2, U1})

        assert router.mutations() == [
            ("POST", "/workers", {"url": U1}),
            ("POST", "/workers", {"url": U2}),
        ]
        assert router.served == [U0, U1, U2]
        assert _polls(router) - polls_before == 3

    @pytest.mark.parametrize("status", [409, 500])
    def test_a_rejected_worker_raises(self, router, status):
        sglang_router = _router()
        sglang_router.start_router({U0})
        router.post_status = status

        with pytest.raises(requests.HTTPError):
            sglang_router.add_workers({U1})
        assert sglang_router.worker_urls == {U0}

    @pytest.mark.parametrize(
        ("kwargs", "worker_args"),
        [
            pytest.param({}, ["--worker-urls", U0, U1], id="plain"),
            pytest.param(
                {"prefill_urls": frozenset({U0}), "bootstrap_port": 8998},
                ["--pd-disaggregation", "--prefill", U0, "8998", "--decode", U1],
                id="pd",
            ),
        ],
    )
    def test_add_workers_on_a_dead_router_relaunches_with_old_and_new_workers(
        self, router, kwargs, worker_args
    ):
        sglang_router = _router(**kwargs)
        sglang_router.start_router({U0})
        router.die()

        sglang_router.add_workers({U1})

        assert len(router.launches) == 2
        assert router.launches[-1][-len(worker_args) :] == worker_args
        assert router.mutations() == []
        assert router.served == [U0, U1]

    def test_remove_workers_deletes_only_the_listed_workers_by_router_id(self, router):
        sglang_router = _router()
        sglang_router.start_router({U0, U1, U2})

        sglang_router.remove_workers({U1})

        assert router.mutations() == [("DELETE", "/workers/w2", None)]
        assert router.served == [U0, U2]

    def test_a_worker_listed_with_a_trailing_slash_is_still_removed(self, router):
        sglang_router = _router()
        sglang_router.start_router({U0, U1})
        router.workers[1]["url"] = f"{U1}/"

        sglang_router.remove_workers({U1})

        assert router.mutations() == [("DELETE", "/workers/w2", None)]
        assert router.served == [U0]

    @pytest.mark.parametrize(
        ("change", "served"),
        [
            pytest.param(lambda r: r.remove_workers({U2}), [U0, U1], id="remove"),
            pytest.param(lambda r: r.add_workers({U3}), [U0, U1, U2, U3], id="add"),
        ],
    )
    def test_a_dead_worker_does_not_block_changes_of_the_others(
        self, router, change, served
    ):
        sglang_router = _router()
        sglang_router.start_router({U0, U1, U2})
        router.unhealthy = {U0}
        polls_before = _polls(router)

        change(sglang_router)

        assert router.served == served
        assert _polls(router) - polls_before < READINESS_ATTEMPTS
