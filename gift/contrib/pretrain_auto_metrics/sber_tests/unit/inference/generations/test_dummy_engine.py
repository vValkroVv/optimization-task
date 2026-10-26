import pytest

from node_runner.src.inference.generations.dummy_engine import DummyEngine

pytestmark = pytest.mark.unit

ROUTER_PORT = 30000
DUMP_DIR = "/tmp/generations"


def _engine() -> DummyEngine:
    return DummyEngine(
        replay_generations_dir=DUMP_DIR,
        port=ROUTER_PORT,
    )


def _record_lifecycle(engine: DummyEngine, events: list[str]) -> None:
    engine.generations_manager.start = lambda: events.append("replay_start")
    engine.generations_manager.stop = lambda: events.append("replay_stop")


class TestDummyEngineAddressing:
    def test_callers_address_the_replay_exactly_as_they_addressed_the_router(self):
        engine = _engine()

        assert engine.router_port == ROUTER_PORT
        assert engine.base_url == f"http://127.0.0.1:{ROUTER_PORT}"

    def test_one_replay_process_serves_the_whole_run(self):
        assert _engine().inference_instances == 8


class TestDummyEngineLifecycle:
    def test_the_master_starts_and_stops_the_replay(self):
        engine = _engine()
        events: list[str] = []
        _record_lifecycle(engine, events)

        engine.start()
        engine.stop()

        assert events == ["replay_start", "replay_stop"]

    def test_the_obm_proxy_starts_after_the_replay_and_stops_before_it(self):
        engine = _engine()
        events: list[str] = []
        _record_lifecycle(engine, events)
        engine.obm_proxy_manager = type(
            "_Proxy",
            (),
            {
                "start": lambda self: events.append("proxy_start"),
                "stop": lambda self: events.append("proxy_stop"),
            },
        )()

        engine.start()
        engine.stop()

        assert events == ["replay_start", "proxy_start", "proxy_stop", "replay_stop"]

    def test_the_replay_holds_no_nodes_so_it_runs_on_the_master_only(self):
        engine = _engine()
        units = engine.units()

        engine.attach(units)
        engine.detach(units)

        assert [unit.ranks for unit in units] == [()]
        assert engine.units(from_tail=True) == units
        assert engine.load(units) is None
        assert engine.worker_urls == []

    def test_the_replay_can_be_released_and_brought_up_again(self):
        engine = _engine()
        events: list[str] = []
        _record_lifecycle(engine, events)

        engine.start()
        engine.stop()
        engine.start()
        engine.close()

        assert events == ["replay_start", "replay_stop", "replay_start", "replay_stop"]
