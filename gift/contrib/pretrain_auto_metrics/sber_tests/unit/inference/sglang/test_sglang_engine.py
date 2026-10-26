from pathlib import Path

import pytest
import requests

from node_runner.src.config_model import RouterConfig, SglangInferenceConfig
from node_runner.src.inference.engine import EngineLoad, EngineUnit
from node_runner.src.inference.sglang.sglang_engine import SglangEngine

from .conftest import FOUR_NODES, FakeResponse, arg, dp_load

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("skip_dg_cache_setup")]

THREE_NODES = {0: "h0", 1: "h1", 2: "h2"}


def _engine(
    *,
    rank: int = 0,
    nodes: dict[int, str] | None = None,
    tp: int = 1,
    pp: int = 1,
    gpus_per_node: int = 8,
    enable_metrics: bool = True,
    router_config: RouterConfig | None = None,
    record_generations_dir: str | None = None,
) -> SglangEngine:
    inference_config = SglangInferenceConfig(
        tp=tp, pp=pp, sglang_communication_port=61000, enable_metrics=enable_metrics
    )
    return SglangEngine(
        hf_checkpoint_path=Path("/tmp/model"),
        nodes=nodes or {0: "127.0.0.1"},
        inference_config=inference_config,
        global_node_rank=rank,
        gpus_per_node=gpus_per_node,
        router_config=router_config or RouterConfig(port=30000, prometheus_port=29000),
        record_generations_dir=record_generations_dir,
    )


def _ranks(units: list[EngineUnit]) -> list[tuple[int, ...]]:
    return [unit.ranks for unit in units]


class TestSglangEngineUnits:
    def test_multinode_units_serve_from_their_head_host_behind_the_router(self):
        engine = _engine(nodes=FOUR_NODES, tp=16)

        assert engine.units() == [
            EngineUnit(ranks=(0, 1), worker_urls=("http://10.0.0.0:30001",)),
            EngineUnit(ranks=(2, 3), worker_urls=("http://10.0.0.2:30001",)),
        ]

    def test_units_follow_rank_order_not_registration_order(self):
        engine = _engine(nodes={3: "d", 1: "b", 2: "c", 0: "a"}, tp=16)

        assert engine.units() == [
            EngineUnit(ranks=(0, 1), worker_urls=("http://a:30001",)),
            EngineUnit(ranks=(2, 3), worker_urls=("http://c:30001",)),
        ]

    @pytest.mark.parametrize(
        ("tp", "pp", "gpus_per_node", "nnodes", "expected_ranks"),
        [
            pytest.param(1, 1, 8, 2, [(0,), (1,)], id="tp_below_node"),
            pytest.param(8, 1, 8, 2, [(0,), (1,)], id="tp_fills_node"),
            pytest.param(16, 1, 8, 4, [(0, 1), (2, 3)], id="tp_spans_two_nodes"),
            pytest.param(16, 3, 8, 6, [(0, 1, 2, 3, 4, 5)], id="pp_counts_to_group"),
            pytest.param(16, 1, 4, 8, [(0, 1, 2, 3), (4, 5, 6, 7)], id="gpn4"),
        ],
    )
    def test_units_group_nodes_by_gpus_of_one_instance(
        self, tp, pp, gpus_per_node, nnodes, expected_ranks
    ):
        nodes = {rank: f"10.0.0.{rank}" for rank in range(nnodes)}

        units = _engine(nodes=nodes, tp=tp, pp=pp, gpus_per_node=gpus_per_node).units()

        assert _ranks(units) == expected_ranks
        assert all(unit.standalone for unit in units)

    def test_every_node_of_the_pool_sees_the_same_units(self):
        engines = [_engine(rank=rank, nodes=FOUR_NODES, tp=16) for rank in FOUR_NODES]

        assert all(e.units() == engines[0].units() for e in engines)
        assert all(
            e.units(from_tail=True) == engines[0].units(from_tail=True) for e in engines
        )

    def test_tail_order_is_the_reverse_and_leaves_head_order_alone(self):
        engine = _engine(nodes=FOUR_NODES)
        head_first = engine.units()

        assert engine.units(from_tail=True) == list(reversed(head_first))
        assert engine.units() == head_first

    def test_worker_urls_are_sorted_so_metrics_read_them_in_a_stable_order(self):
        engine = _engine(nodes={0: "10.0.0.2", 1: "10.0.0.1"})

        assert engine.worker_urls == ["http://10.0.0.1:30001", "http://10.0.0.2:30001"]

    def test_nodes_not_divisible_by_group_size_are_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            _engine(nodes=THREE_NODES, tp=16)

    @pytest.mark.parametrize(
        ("nnodes", "tp", "pp", "gpus_per_node", "expected"),
        [
            pytest.param(2, 4, 1, 8, 4, id="two_instances_per_node"),
            pytest.param(4, 16, 1, 8, 2, id="two_node_instances"),
            pytest.param(6, 16, 3, 8, 1, id="pp_takes_gpus_too"),
        ],
    )
    def test_inference_instances_fill_every_gpu_of_every_node(
        self, nnodes, tp, pp, gpus_per_node, expected
    ):
        nodes = {rank: f"10.0.0.{rank}" for rank in range(nnodes)}

        engine = _engine(nodes=nodes, tp=tp, pp=pp, gpus_per_node=gpus_per_node)

        assert engine.inference_instances == expected


class TestSglangEngineWorkerLaunch:
    @pytest.mark.parametrize(
        ("rank", "nnodes", "tp", "pp", "gpus_per_node", "expected"),
        [
            pytest.param(0, 1, 8, 1, 8, (None, None, None), id="single_node"),
            pytest.param(1, 2, 4, 1, 8, (None, None, None), id="node_per_unit"),
            pytest.param(3, 4, 16, 1, 8, ("2", "1", "10.0.0.2:61000"), id="tp16_r3"),
            pytest.param(5, 6, 16, 1, 8, ("2", "1", "10.0.0.4:61000"), id="tp16_r5"),
            pytest.param(5, 6, 16, 3, 8, ("6", "5", "10.0.0.0:61000"), id="pp3_r5"),
            pytest.param(3, 8, 16, 1, 4, ("4", "3", "10.0.0.0:61000"), id="gpn4_r3"),
            pytest.param(4, 8, 16, 1, 4, ("4", "0", "10.0.0.4:61000"), id="gpn4_r4"),
        ],
    )
    def test_each_node_joins_its_unit_on_the_worker_port(
        self, rank, nnodes, tp, pp, gpus_per_node, expected
    ):
        nodes = {r: f"10.0.0.{r}" for r in range(nnodes)}

        engine = _engine(
            rank=rank, nodes=nodes, tp=tp, pp=pp, gpus_per_node=gpus_per_node
        )

        assert arg(engine, "--port") == "30001"
        assert (
            arg(engine, "--nnodes"),
            arg(engine, "--node-rank"),
            arg(engine, "--dist-init-addr"),
        ) == expected


class TestSglangEngineWorkerLifecycle:
    @pytest.mark.parametrize(
        ("rank", "expected"),
        [
            pytest.param(
                0,
                [
                    "inference_start",
                    "inference_ready",
                    "dg_copy",
                    "scraper_start:engine",
                ],
                id="unit_head_shares_the_deepgemm_cache",
            ),
            pytest.param(
                1,
                ["inference_start", "inference_ready", "scraper_start:engine"],
                id="other_nodes_do_not",
            ),
        ],
    )
    def test_start_launches_only_this_nodes_worker(
        self, router, scrapers, record_worker, events, rank, expected
    ):
        engine = _engine(rank=rank, nodes={0: "h0", 1: "h1"}, tp=16)
        record_worker(engine)

        engine.start()

        assert events == expected
        assert router.launches == []

    def test_restart_after_release_relaunches_the_worker_and_its_scraper(
        self, scrapers, record_worker, events
    ):
        engine = _engine()
        record_worker(engine)

        engine.start()
        engine.stop()
        engine.start()

        assert events == [
            "inference_start",
            "inference_ready",
            "dg_copy",
            "scraper_start:engine",
            "scraper_stop:engine",
            "inference_stop",
            "inference_start",
            "inference_ready",
            "dg_copy",
            "scraper_start:engine",
        ]

    def test_a_failed_start_propagates_and_scrapes_nothing(
        self, monkeypatch, scrapers, record_worker, events
    ):
        engine = _engine()
        record_worker(engine)

        def _not_ready():
            raise RuntimeError("worker died")

        monkeypatch.setattr(
            engine.inference, "wait_for_inference_to_initialize", _not_ready
        )

        with pytest.raises(RuntimeError, match="worker died"):
            engine.start()
        assert scrapers == []

    def test_stop_stops_only_this_nodes_worker(
        self, router, scrapers, record_worker, proxies, events
    ):
        engine = _engine(nodes={0: "h0", 1: "h1"})
        record_worker(engine)
        proxies(engine)
        engine.start()
        engine.attach(engine.units())
        events.clear()

        engine.stop()

        assert events == ["scraper_stop:engine", "inference_stop"]
        assert router.alive
        assert router.served == ["http://h0:30001", "http://h1:30001"]


class TestSglangEngineRouterMembership:
    def test_first_attach_starts_the_router_with_exactly_those_workers(
        self, router, scrapers, proxies, events
    ):
        engine = _engine(nodes=THREE_NODES)
        proxies(engine)

        engine.attach(engine.units()[:1])

        assert router.launches[0][-2:] == ["--worker-urls", "http://h0:30001"]
        assert router.served == ["http://h0:30001"]
        assert events == [
            "router_start",
            "recorder_start",
            "obm_start",
            "scraper_start:router",
        ]
        assert scrapers == [("router", "http://127.0.0.1:29000/metrics")]

    def test_attaching_every_unit_at_once_launches_the_router_once(
        self, router, scrapers
    ):
        engine = _engine(nodes=THREE_NODES)

        engine.attach(engine.units())

        assert len(router.launches) == 1
        assert router.mutations() == []
        assert router.served == [
            "http://h0:30001",
            "http://h1:30001",
            "http://h2:30001",
        ]

    def test_later_attach_adds_workers_to_the_running_router(
        self, router, scrapers, proxies, events
    ):
        engine = _engine(nodes=THREE_NODES)
        proxies(engine)
        first, second, third = engine.units()
        engine.attach([first])
        events.clear()

        engine.attach([third, second])

        assert router.mutations() == [
            ("POST", "/workers", {"url": "http://h1:30001"}),
            ("POST", "/workers", {"url": "http://h2:30001"}),
        ]
        assert len(router.launches) == 1
        assert events == []
        assert router.served == [
            "http://h0:30001",
            "http://h1:30001",
            "http://h2:30001",
        ]

    def test_detaching_some_units_removes_only_their_workers(
        self, router, scrapers, proxies, events
    ):
        engine = _engine(nodes=THREE_NODES)
        proxies(engine)
        first, second, third = engine.units()
        engine.attach([first, second, third])
        events.clear()

        engine.detach([second])

        assert router.mutations() == [("DELETE", "/workers/w2", None)]
        assert events == []
        assert router.served == ["http://h0:30001", "http://h2:30001"]

    @pytest.mark.parametrize(
        "detached",
        [pytest.param(2, id="exactly_attached"), pytest.param(3, id="superset")],
    )
    def test_detaching_every_attached_unit_stops_the_router_and_proxies(
        self, router, scrapers, proxies, events, detached
    ):
        engine = _engine(nodes=THREE_NODES)
        proxies(engine)
        units = engine.units()
        engine.attach(units[:2])
        events.clear()

        engine.detach(units[:detached])

        assert router.mutations() == []
        assert events == [
            "scraper_stop:router",
            "obm_stop",
            "recorder_stop",
            "router_stop",
        ]
        assert not router.alive

    def test_reattach_after_full_detach_restarts_router_with_only_new_units(
        self, router, scrapers, proxies, events
    ):
        engine = _engine(nodes=THREE_NODES)
        proxies(engine)
        first, second, _ = engine.units()
        engine.attach([first, second])
        engine.detach([first, second])
        events.clear()

        engine.attach([second])

        assert router.launches[-1][-2:] == ["--worker-urls", "http://h1:30001"]
        assert router.served == ["http://h1:30001"]
        assert events == [
            "router_start",
            "recorder_start",
            "obm_start",
            "scraper_start:router",
        ]

    def test_drain_timeout_puts_the_unit_back_without_restarting_the_router(
        self, router, scrapers
    ):
        engine = _engine(nodes={0: "h0", 1: "h1"})
        first, second = engine.units()
        engine.attach([first, second])
        engine.detach([second])

        engine.attach([second])

        assert len(router.launches) == 1
        assert router.served == ["http://h0:30001", "http://h1:30001"]

    @pytest.mark.parametrize(
        "detached", [pytest.param(1, id="unit"), pytest.param(0, id="none")]
    )
    def test_detach_before_any_attach_is_harmless(self, router, scrapers, detached):
        engine = _engine(nodes={0: "h0", 1: "h1"})

        engine.detach(engine.units()[:detached])

        assert router.calls == []
        assert router.launches == []

    def test_attach_after_router_crash_relaunches_with_old_and_new_workers(
        self, router, scrapers, proxies, events
    ):
        engine = _engine(nodes={0: "h0", 1: "h1"})
        proxies(engine)
        first, second = engine.units()
        engine.attach([first])
        router.die()

        engine.attach([second])

        assert len(router.launches) == 2
        assert router.served == ["http://h0:30001", "http://h1:30001"]
        assert events.count("obm_start") == 1

    def test_detach_after_router_crash_drops_the_worker_from_the_next_launch(
        self, router, scrapers
    ):
        engine = _engine(nodes={0: "h0", 1: "h1", 2: "h2"})
        first, second, third = engine.units()
        engine.attach([first, second])
        router.die()

        engine.detach([second])
        engine.attach([third])

        assert router.served == ["http://h0:30001", "http://h2:30001"]


class TestSglangEngineLoad:
    def test_load_sums_every_dp_rank_of_every_worker(self, workers):
        engine = _engine(nodes={0: "h0", 1: "h1"})
        workers.replies = {
            "http://h0:30001": [dp_load(3, 2, usage=0.4), dp_load(1, 0, usage=0.2)],
            "http://h1:30001": [dp_load(0, 3, capacity=64, usage=0.9)],
        }

        load = engine.load(engine.units())

        assert load == EngineLoad(running=4, waiting=5, capacity=448, token_usage=0.9)
        assert [url for url, _ in workers.calls] == [
            "http://h0:30001/v1/loads",
            "http://h1:30001/v1/loads",
        ]

    @pytest.mark.parametrize(
        "replies",
        [
            pytest.param({}, id="no_dp_ranks"),
            pytest.param(
                {"http://h0:30001": [dp_load(0, 0, capacity=0, usage=0.0)]},
                id="all_idle",
            ),
        ],
    )
    def test_idle_workers_report_zero_load(self, workers, replies):
        engine = _engine()
        workers.replies = replies

        assert engine.load(engine.units()) == EngineLoad(
            running=0, waiting=0, capacity=0, token_usage=0.0
        )

    def test_no_units_report_zero_load_without_requests(self, workers):
        assert _engine().load([]) == EngineLoad(
            running=0, waiting=0, capacity=0, token_usage=0.0
        )
        assert workers.calls == []

    @pytest.mark.parametrize(
        "failure",
        [
            pytest.param(requests.ConnectionError("refused"), id="unreachable"),
            pytest.param(requests.Timeout("slow"), id="timeout"),
            pytest.param(FakeResponse(503, []), id="http_error"),
            pytest.param(FakeResponse(200, ValueError("bad json")), id="invalid_json"),
        ],
    )
    def test_any_unanswered_worker_makes_load_unknown(self, workers, failure):
        engine = _engine(nodes={0: "h0", 1: "h1"})
        workers.replies = {"http://h1:30001": failure}

        assert engine.load(engine.units()) is None

    def test_a_hung_worker_ends_the_probe_at_once(self, workers):
        engine = _engine(nodes=THREE_NODES)
        workers.replies = {"http://h0:30001": requests.Timeout("slow")}

        assert engine.load(engine.units()) is None
        assert len(workers.calls) == 1

    def test_every_probe_is_time_bounded(self, workers):
        engine = _engine(nodes=THREE_NODES)

        engine.load(engine.units())

        assert all(
            timeout is not None and 0 < timeout <= 15 for _, timeout in workers.calls
        )

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"num_reqs": 1}, id="dict"),
            pytest.param(None, id="null"),
            pytest.param([{"num_reqs": 1}], id="missing_field"),
            pytest.param(["x"], id="not_an_object"),
        ],
    )
    def test_a_malformed_load_body_makes_load_unknown(self, workers, body):
        engine = _engine()
        workers.replies = {"http://127.0.0.1:30001": FakeResponse(200, body)}

        assert engine.load(engine.units()) is None


class TestSglangEngineScrapers:
    @pytest.mark.parametrize(
        ("enable_metrics", "expected"),
        [
            pytest.param(
                True,
                [
                    ("engine", "http://127.0.0.1:30001/metrics"),
                    ("router", "http://127.0.0.1:29100/metrics"),
                ],
                id="engine_and_router",
            ),
            pytest.param(
                False,
                [("router", "http://127.0.0.1:29100/metrics")],
                id="router_only",
            ),
        ],
    )
    def test_engine_and_router_are_scraped_on_their_ports(
        self, router, scrapers, record_worker, enable_metrics, expected
    ):
        engine = _engine(
            enable_metrics=enable_metrics,
            router_config=RouterConfig(port=30000, prometheus_port=29100),
        )
        record_worker(engine)

        engine.start()
        engine.attach(engine.units())

        assert scrapers == expected


class TestSglangEngineClose:
    def test_close_stops_what_reads_a_process_before_the_process(
        self, router, scrapers, record_worker, proxies, events
    ):
        engine = _engine()
        record_worker(engine)
        proxies(engine)
        engine.start()
        engine.attach(engine.units())
        events.clear()

        engine.close()

        assert events == [
            "scraper_stop:router",
            "obm_stop",
            "recorder_stop",
            "router_stop",
            "scraper_stop:engine",
            "inference_stop",
        ]

    @pytest.mark.parametrize(
        "lifecycle",
        [
            pytest.param([], id="never_started"),
            pytest.param(["start", "stop"], id="after_release"),
            pytest.param(["start", "attach", "close"], id="twice"),
        ],
    )
    def test_close_is_safe_in_any_state(self, router, scrapers, lifecycle):
        engine = _engine(nodes={0: "h0", 1: "h1"})
        actions = {
            "start": engine.start,
            "stop": engine.stop,
            "attach": lambda: engine.attach(engine.units()),
            "close": engine.close,
        }
        engine.inference.start_inference = lambda: None
        engine.inference.wait_for_inference_to_initialize = lambda: None
        engine.inference.copy_dg_cache_to_nfs = lambda: None
        for action in lifecycle:
            actions[action]()

        engine.close()

        assert not router.alive


class TestSglangEngineRecorder:
    def test_the_recorder_stands_between_callers_and_the_router(self):
        engine = _engine(record_generations_dir="/tmp/generations")

        assert engine.direct_base_url == "http://127.0.0.1:35000"
        assert engine.generations_manager.bash_cmd[-2:] == [
            "--upstream-url",
            "http://127.0.0.1:30000",
        ]

    def test_without_a_dump_dir_callers_reach_the_router_directly(self):
        assert _engine().direct_base_url == "http://127.0.0.1:30000"
