import pytest

import node_runner.src.inference.engine as engine_module
from node_runner.src.config_model import ObmProxyConfig
from node_runner.src.inference.engine import NODELESS_UNIT, EngineUnit, InferenceEngine
from node_runner.src.inference.generations.dummy_engine import DummyEngine
from node_runner.src.logger import Logger

pytestmark = pytest.mark.unit


class _DummyEngine(InferenceEngine):
    def __init__(self):
        super().__init__(router_port=30000)
        self.started = False
        self.stopped = False

    @property
    def inference_instances(self) -> int:
        return 1

    def start(self) -> None:
        self.started = True

    def stop(self):
        self.stopped = True


class TestInferenceEngineAbstract:
    def test_an_engine_without_workers_reports_none(self):
        assert _DummyEngine().worker_urls == []

    def test_inference_engine_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            InferenceEngine()

    def test_start_and_stop_contract(self):
        engine = _DummyEngine()
        assert engine.base_url == "http://127.0.0.1:30000"
        assert engine.start() is None
        assert engine.started is True
        assert engine.stop() is None
        assert engine.stopped is True

    def test_close_stops_an_engine_without_a_router(self):
        engine = _DummyEngine()
        engine.close()
        assert engine.stopped is True

    def test_log_file_follows_log_profile(self, tmp_path):
        Logger.node_log_dir = tmp_path
        engine = _DummyEngine()
        engine.log_profile = "judge"

        assert engine.inference_log_file == tmp_path / "judge_inference.log"

    def test_base_urls_use_direct_or_obm_proxy(self):
        engine = _DummyEngine()

        assert engine.direct_base_url == "http://127.0.0.1:30000"
        assert (
            engine.chat_completions_url == "http://127.0.0.1:30000/v1/chat/completions"
        )
        assert engine.completions_url == "http://127.0.0.1:30000/v1/completions"
        assert engine.generate_url == "http://127.0.0.1:30000/generate"

        engine.obm_proxy_manager = type(
            "_Proxy",
            (),
            {"base_url": "http://127.0.0.1:29999", "api_port": 29999},
        )()
        assert engine.base_url == "http://127.0.0.1:29999"

    def test_a_nodeless_engine_has_one_unit_and_ignores_membership(self):
        engine = _DummyEngine()
        units = engine.units()

        engine.attach(units)
        engine.detach(units)

        assert units == [NODELESS_UNIT] == engine.units(from_tail=True)
        assert NODELESS_UNIT == EngineUnit(ranks=(), worker_urls=(), standalone=True)
        assert engine.load(units) is None
        assert engine.worker_urls == []


class TestEngineUnitAllocationCost:
    @pytest.mark.parametrize(
        "prefill,decode,candidate_ranks,expected",
        [
            pytest.param(3, 3, [(1,), (1, 4)], (1, 4), id="balance-equal-roles"),
            pytest.param(
                5, 3, [(1, 2, 3), (1,)], (1,), id="lose-fewer-at-equal-balance"
            ),
            pytest.param(
                5, 4, [(6,), (1, 2, 3, 4)], (1, 2, 3, 4), id="majority-before-balance"
            ),
            pytest.param(
                4,
                5,
                [(1,), (5, 6, 7, 8)],
                (5, 6, 7, 8),
                id="majority-before-balance-mirrored",
            ),
        ],
    )
    def test_pd_candidates_follow_role_and_balance_priorities(
        self, prefill, decode, candidate_ranks, expected
    ):
        available = [
            EngineUnit((r,), (), prefill_instances=1) for r in range(prefill)
        ] + [
            EngineUnit((r,), (), decode_instances=1)
            for r in range(prefill, prefill + decode)
        ]
        candidates = [EngineUnit(ranks, ()) for ranks in candidate_ranks]

        chosen = min(candidates, key=lambda unit: unit.allocation_cost(available))

        assert chosen.ranks == expected

    def test_free_ranks_precede_donating_another_instance_to_improve_balance(self):
        available = [
            EngineUnit((0, 3), (), prefill_instances=1, decode_instances=1),
            EngineUnit((1,), (), prefill_instances=1),
            EngineUnit((2,), (), prefill_instances=1),
        ]
        candidates = [EngineUnit((2,), ()), EngineUnit((4,), ())]

        chosen = min(candidates, key=lambda unit: unit.allocation_cost(available))

        assert chosen.ranks == (4,)

    @pytest.mark.parametrize(
        "available", [[], [NODELESS_UNIT], [EngineUnit((0, 1), ())]]
    )
    def test_non_pd_and_nodeless_engines_preserve_candidate_order(self, available):
        candidates = [EngineUnit((3,), ()), EngineUnit((2,), ())]

        assert (
            sorted(candidates, key=lambda unit: unit.allocation_cost(available))
            == candidates
        )


class TestInferenceEngineObmProxy:
    def test_an_obm_proxy_needs_the_checkpoint(self):
        with pytest.raises(ValueError, match="hf_checkpoint_path is required"):
            DummyEngine(
                replay_generations_dir="/tmp/generations",
                port=30000,
                obm_proxy_config=ObmProxyConfig(),
                hf_checkpoint_path=None,
            )

    def test_the_obm_proxy_forwards_to_the_engine_and_becomes_its_base_url(
        self, monkeypatch, tmp_path
    ):
        calls = {}

        class _ProxyManager:
            base_url = "http://127.0.0.1:29999"

            def setup(self, *args, **kwargs):
                calls["args"] = args
                calls["kwargs"] = kwargs

        monkeypatch.setattr(engine_module, "ObmProxyManager", _ProxyManager)
        config = ObmProxyConfig()
        checkpoint_path = tmp_path / "hf"

        engine = DummyEngine(
            replay_generations_dir="/tmp/generations",
            port=30000,
            obm_proxy_config=config,
            hf_checkpoint_path=checkpoint_path,
        )

        assert engine.base_url == "http://127.0.0.1:29999"
        assert calls["args"] == (config, checkpoint_path)
        assert calls["kwargs"] == {
            "chat_completions_url": "http://127.0.0.1:30000/v1/chat/completions",
            "completions_url": "http://127.0.0.1:30000/v1/completions",
            "generate_url": "http://127.0.0.1:30000/generate",
        }
