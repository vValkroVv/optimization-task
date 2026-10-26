from pathlib import Path

import pytest

from node_runner.src.config_model import (
    AutometricsConfig,
    GlobalsConfig,
    RouterConfig,
    SglangPdInferenceConfig,
)
from node_runner.src.inference.engine import EngineLoad, EngineUnit
from node_runner.src.inference.inference_manager import InferenceManager
from node_runner.src.inference.sglang.sglang_pd_engine import (
    SglangPdEngine,
    split_nodes_for_pd,
)

from .conftest import FOUR_NODES, arg, dp_load

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("skip_dg_cache_setup")]


def _pd_config(inference: dict) -> SglangPdInferenceConfig:
    return AutometricsConfig.model_validate(
        {
            "globals": {},
            "router": {},
            "evaluators": {},
            "preparation": {},
            "convertation": {},
            "model_info": {"model_class": "giga", "size": "10b", "username": "u"},
            "inference": inference,
        }
    ).inference


def _make_engine(
    *,
    global_node_rank: int = 0,
    nodes_for_inference: dict[int, str],
    inference: dict | None = None,
) -> SglangPdEngine:
    InferenceManager.setup(
        inference_config=_pd_config(
            inference or {"prefill": {"tp": 16}, "decode": {"tp": 16}}
        ),
        global_node_rank=global_node_rank,
        gpus_per_node=8,
        globals_config=GlobalsConfig(),
        hf_checkpoint_path=Path("/tmp/model"),
        router_config=RouterConfig(port=30000),
        obm_proxy_config=None,
        registered_nodes=nodes_for_inference,
    )
    return InferenceManager.setup_inference_engine()


def _single_node_groups(fraction: float) -> dict:
    return {
        "shared_params": {"tp": 8},
        "prefill_fraction": fraction,
        "prefill": {},
        "decode": {},
    }


class TestSglangPdEngineUnits:
    def test_the_first_unit_is_the_smallest_working_prefill_decode_pair(self):
        engine = _make_engine(nodes_for_inference=FOUR_NODES)

        assert engine.units() == [
            EngineUnit(
                ranks=(0, 1, 2, 3),
                worker_urls=("http://10.0.0.0:30001", "http://10.0.0.2:30001"),
                standalone=True,
                prefill_instances=1,
                decode_instances=1,
            )
        ]

    @pytest.mark.parametrize(
        ("nnodes", "fraction", "from_tail", "expected_ranks"),
        [
            pytest.param(
                6, 0.5, False, [(0, 3), (4,), (5,), (1,), (2,)], id="half_from_head"
            ),
            pytest.param(
                6, 0.5, True, [(2, 5), (4,), (3,), (1,), (0,)], id="half_from_tail"
            ),
            pytest.param(
                8,
                0.25,
                False,
                [(0, 2), (3,), (4,), (5,), (6,), (7,), (1,)],
                id="quarter_from_head",
            ),
            pytest.param(4, 0.25, False, [(0, 1), (2,), (3,)], id="one_prefill_group"),
        ],
    )
    def test_extra_decode_units_come_before_extra_prefill_units(
        self, nnodes, fraction, from_tail, expected_ranks
    ):
        engine = _make_engine(
            nodes_for_inference=_generate_ips(nnodes),
            inference=_single_node_groups(fraction),
        )

        units = engine.units(from_tail=from_tail)

        assert [unit.ranks for unit in units] == expected_ranks
        assert [unit.standalone for unit in units] == [True] + [False] * (
            len(units) - 1
        )

    def test_every_node_of_the_pool_sees_the_same_units(self):
        nodes = _generate_ips(6)
        engines = [
            _make_engine(
                global_node_rank=rank,
                nodes_for_inference=nodes,
                inference=_single_node_groups(0.5),
            )
            for rank in nodes
        ]

        for from_tail in (False, True):
            expected = engines[0].units(from_tail=from_tail)
            assert all(e.units(from_tail=from_tail) == expected for e in engines)

    def test_units_cover_every_node_once_with_one_leading_standalone_pair(self):
        for nnodes in range(2, 11):
            nodes = _generate_ips(nnodes)
            for prefill_tp, decode_tp in ((8, 8), (8, 16), (16, 8), (16, 16)):
                for fraction in (0.001, 0.3, 0.5, 0.7, 0.999):
                    inference = {
                        "prefill_fraction": fraction,
                        "prefill": {"tp": prefill_tp},
                        "decode": {"tp": decode_tp},
                    }
                    try:
                        engine = _make_engine(
                            nodes_for_inference=nodes, inference=inference
                        )
                    except ValueError:
                        continue
                    for from_tail in (False, True):
                        units = engine.units(from_tail=from_tail)
                        case = f"{nnodes=} {inference=} {from_tail=}"
                        ranks = [r for unit in units for r in unit.ranks]
                        assert sorted(ranks) == list(nodes), case
                        assert [u.standalone for u in units] == [True] + [False] * (
                            len(units) - 1
                        ), case
                        assert len(units[0].worker_urls) == 2, case
                        assert (
                            units[0].prefill_instances,
                            units[0].decode_instances,
                        ) == (1, 1), case
                        prefill_nodes, _ = engine.sides[0]
                        decode_nodes, _ = engine.sides[1]
                        assert (
                            sum(u.prefill_instances for u in units)
                            == len(prefill_nodes) * 8 // prefill_tp
                        ), case
                        assert (
                            sum(u.decode_instances for u in units)
                            == len(decode_nodes) * 8 // decode_tp
                        ), case

    @pytest.mark.parametrize(
        ("nnodes", "inference", "expected"),
        [
            pytest.param(4, None, 2, id="tp16_both_sides"),
            pytest.param(
                8,
                {
                    "shared_params": {"tp": 16},
                    "prefill_fraction": 0.3,
                    "prefill": {},
                    "decode": {},
                },
                4,
                id="matches_flat_engine",
            ),
        ],
    )
    def test_inference_instances_count_the_whole_pool(
        self, nnodes, inference, expected
    ):
        nodes = _generate_ips(nnodes)
        for rank in nodes:
            engine = _make_engine(
                global_node_rank=rank, nodes_for_inference=nodes, inference=inference
            )
            assert engine.inference_instances == expected

    def test_worker_urls_cover_both_sides(self):
        for rank in FOUR_NODES:
            engine = _make_engine(global_node_rank=rank, nodes_for_inference=FOUR_NODES)
            assert engine.worker_urls == [
                "http://10.0.0.0:30001",
                "http://10.0.0.2:30001",
            ]

    def test_an_impossible_split_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Cannot split"):
            _make_engine(
                nodes_for_inference={0: "h0"}, inference=_single_node_groups(0.5)
            )


class TestSglangPdEngineWorkerLaunch:
    @pytest.mark.parametrize(
        ("rank", "expected"),
        [
            pytest.param(0, ("prefill", "0", "10.0.0.0:61000"), id="prefill_head"),
            pytest.param(1, ("prefill", "1", "10.0.0.0:61000"), id="prefill_tail"),
            pytest.param(2, ("decode", "0", "10.0.0.2:61000"), id="decode_head"),
            pytest.param(3, ("decode", "1", "10.0.0.2:61000"), id="decode_tail"),
        ],
    )
    def test_each_node_runs_its_side_relative_to_that_side(self, rank, expected):
        engine = _make_engine(
            global_node_rank=rank,
            nodes_for_inference=FOUR_NODES,
            inference={
                "shared_params": {"tp": 16, "sglang_communication_port": 61000},
                "prefill": {},
                "decode": {},
            },
        )

        assert (
            arg(engine, "--disaggregation-mode"),
            arg(engine, "--node-rank"),
            arg(engine, "--dist-init-addr"),
        ) == expected


class TestSglangPdEngineRouter:
    def test_the_router_starts_on_the_anchor_pair(self, router, scrapers):
        engine = _make_engine(nodes_for_inference=FOUR_NODES)

        engine.attach(engine.units())

        assert router.launches[0][-6:] == [
            "--pd-disaggregation",
            "--prefill",
            "http://10.0.0.0:30001",
            "8998",
            "--decode",
            "http://10.0.0.2:30001",
        ]
        assert "--worker-urls" not in router.launches[0]

    def test_a_configured_bootstrap_port_reaches_the_router(self, router, scrapers):
        engine = _make_engine(
            nodes_for_inference=FOUR_NODES,
            inference={
                "prefill": {"tp": 16, "disaggregation_bootstrap_port": 9100},
                "decode": {"tp": 16},
            },
        )

        engine.attach(engine.units())

        assert router.launches[0][-4:-2] == ["http://10.0.0.0:30001", "9100"]

    def test_attaching_every_unit_registers_both_sides(self, router, scrapers):
        engine = _make_engine(nodes_for_inference=_generate_ips(8))

        engine.attach(engine.units())

        cmd = router.launches[0]
        assert cmd.count("--prefill") == 2
        assert cmd.count("--decode") == 2
        assert router.mutations() == []

    def test_later_units_join_the_running_router_as_their_side(self, router, scrapers):
        engine = _make_engine(
            nodes_for_inference=_generate_ips(8), inference=_single_node_groups(0.5)
        )
        anchor, extra_decode, *_, extra_prefill = engine.units()
        engine.attach([anchor])

        engine.attach([extra_decode])
        engine.attach([extra_prefill])

        assert router.mutations() == [
            (
                "POST",
                "/workers",
                {"url": "http://10.0.0.5:30001", "worker_type": "decode"},
            ),
            (
                "POST",
                "/workers",
                {
                    "url": "http://10.0.0.3:30001",
                    "worker_type": "prefill",
                    "bootstrap_port": 8998,
                },
            ),
        ]
        assert len(router.launches) == 1

    def test_base_url_stays_on_router_port(self):
        engine = _make_engine(nodes_for_inference=FOUR_NODES)
        assert engine.direct_base_url == "http://127.0.0.1:30000"


class TestSglangPdEngineLoad:
    def test_anchor_load_counts_prefill_work_once_and_kv_in_transfer_apart(
        self, workers
    ):
        engine = _make_engine(nodes_for_inference=FOUR_NODES)
        workers.replies = {
            "http://10.0.0.0:30001": [
                dp_load(2, 7, capacity=128, usage=0.99, mode="prefill")
            ],
            "http://10.0.0.2:30001": [
                dp_load(
                    4,
                    5,
                    capacity=192,
                    usage=0.3,
                    mode="decode",
                    decode_transfer_queue_reqs=3,
                )
            ],
        }

        assert engine.load(engine.units()) == EngineLoad(
            running=6, waiting=9, capacity=192, token_usage=0.3, kv_in_transfer=3
        )
        assert [url for url, _ in workers.calls] == [
            "http://10.0.0.0:30001/v1/loads",
            "http://10.0.0.2:30001/v1/loads",
        ]

    def test_prefill_extra_with_kv_still_leaving_is_not_empty(self, workers):
        engine = _make_engine(
            nodes_for_inference=_generate_ips(8), inference=_single_node_groups(0.5)
        )
        *_, prefill_extra = engine.units()
        (url,) = prefill_extra.worker_urls
        workers.replies = {
            url.removesuffix("/v1/loads"): [
                dp_load(0, 0, mode="prefill", prefill_inflight_queue_reqs=2)
            ]
        }

        load = engine.load([prefill_extra])

        assert (load.running, load.waiting, load.kv_in_transfer) == (0, 0, 2)


class TestSglangPdEngineLifecycle:
    @pytest.mark.parametrize(
        ("rank", "expected"),
        [
            pytest.param(
                0, ["inference_start", "inference_ready", "dg_copy"], id="prefill_head"
            ),
            pytest.param(
                1, ["inference_start", "inference_ready"], id="prefill_worker"
            ),
            pytest.param(
                2, ["inference_start", "inference_ready", "dg_copy"], id="decode_head"
            ),
        ],
    )
    def test_start_launches_only_the_side_worker(
        self, router, record_worker, events, rank, expected
    ):
        engine = _make_engine(
            global_node_rank=rank,
            nodes_for_inference=FOUR_NODES,
            inference={
                "shared_params": {"tp": 16, "enable_metrics": False},
                "prefill": {},
                "decode": {},
            },
        )
        record_worker(engine)

        engine.start()

        assert events == expected
        assert router.launches == []


def _generate_ips(n_nodes: int) -> dict[int, str]:
    return {i: f"10.0.0.{i}" for i in range(n_nodes)}


class TestSplitNodesForPd:
    @pytest.mark.parametrize(
        (
            "nnodes",
            "prefill_fraction",
            "prefill_group_nnodes",
            "decode_group_nnodes",
            "expected_prefill_nnodes",
        ),
        [
            pytest.param(8, 0.5, 1, 1, 4, id="half_split_single_node_groups"),
            pytest.param(10, 0.5, 2, 2, 4, id="half_split_two_node_groups_rounds_down"),
            pytest.param(10, 0.25, 2, 2, 2, id="fraction_rounds_to_group_multiple"),
            pytest.param(
                10, 0.001, 1, 1, 1, id="tiny_fraction_keeps_one_prefill_instance"
            ),
            pytest.param(
                10, 0.999, 1, 1, 9, id="huge_fraction_keeps_one_decode_instance"
            ),
            pytest.param(
                4, 0.01, 2, 2, 2, id="tiny_fraction_keeps_one_two_node_prefill_instance"
            ),
            pytest.param(
                8, 0.99, 2, 2, 6, id="huge_fraction_keeps_one_two_node_decode_instance"
            ),
            pytest.param(2, 0.5, 1, 1, 1, id="two_nodes_split_into_minimal_pd"),
            pytest.param(
                10, 0.5, 1, 4, 6, id="indivisible_decode_remainder_snaps_to_valid_split"
            ),
            pytest.param(
                8, 0.5, 2, 3, 2, id="asymmetric_groups_use_the_only_valid_split"
            ),
            pytest.param(
                6, 0.5, 1, 2, 2, id="tie_between_valid_splits_prefers_more_decode"
            ),
            pytest.param(
                12,
                0.5,
                4,
                2,
                4,
                id="tie_with_multinode_prefill_groups_prefers_more_decode",
            ),
            pytest.param(9, 0.5, 3, 3, 3, id="odd_nnodes_with_three_node_groups"),
        ],
    )
    def test_split_counts(
        self,
        nnodes: int,
        prefill_fraction: float,
        prefill_group_nnodes: int,
        decode_group_nnodes: int,
        expected_prefill_nnodes: int,
    ):
        nodes = _generate_ips(nnodes)

        prefill_nodes, decode_nodes = split_nodes_for_pd(
            nodes, prefill_fraction, prefill_group_nnodes, decode_group_nnodes
        )

        assert len(prefill_nodes) == expected_prefill_nnodes
        assert len(decode_nodes) == nnodes - expected_prefill_nnodes

    def test_partition_is_exact_and_prefill_takes_lowest_ranks(self):
        nodes = {rank: f"10.0.0.{rank}" for rank in (4, 5, 6, 7, 8, 9)}

        prefill_nodes, decode_nodes = split_nodes_for_pd(nodes, 0.5, 1, 1)

        assert prefill_nodes == {4: "10.0.0.4", 5: "10.0.0.5", 6: "10.0.0.6"}
        assert decode_nodes == {7: "10.0.0.7", 8: "10.0.0.8", 9: "10.0.0.9"}
        assert min(nodes) in prefill_nodes

    @pytest.mark.parametrize(
        ("nnodes", "prefill_group_nnodes", "decode_group_nnodes"),
        [
            pytest.param(1, 1, 1, id="single_node_cannot_host_both_sides"),
            pytest.param(2, 2, 2, id="both_sides_need_all_nodes"),
            pytest.param(5, 2, 2, id="odd_nnodes_with_even_groups"),
            pytest.param(4, 3, 3, id="remainder_smaller_than_decode_group"),
        ],
    )
    def test_impossible_splits_raise(
        self,
        nnodes: int,
        prefill_group_nnodes: int,
        decode_group_nnodes: int,
    ):
        with pytest.raises(ValueError, match="Cannot split"):
            split_nodes_for_pd(
                _generate_ips(nnodes), 0.5, prefill_group_nnodes, decode_group_nnodes
            )

    def test_split_correction_is_logged(self, monkeypatch: pytest.MonkeyPatch):
        from node_runner.src.logger import Logger

        messages: list[str] = []
        monkeypatch.setattr(
            Logger, "info", classmethod(lambda cls, msg: messages.append(msg))
        )

        split_nodes_for_pd(_generate_ips(10), 0.5, 1, 1)
        assert messages == []

        split_nodes_for_pd(_generate_ips(10), 0.001, 1, 1)
        assert len(messages) == 1
        assert "closest valid split: 1 prefill / 9 decode" in messages[0]

    def test_every_group_and_fraction_combination_splits_or_raises_correctly(self):
        fractions = (0.001, 0.3, 0.5, 0.7, 0.999)
        for nnodes in range(1, 17):
            nodes = _generate_ips(nnodes)
            for prefill_group in range(1, 5):
                for decode_group in range(1, 5):
                    valid_splits_exist = any(
                        (nnodes - candidate) % decode_group == 0
                        for candidate in range(prefill_group, nnodes, prefill_group)
                    )
                    for fraction in fractions:
                        case = (
                            f"nnodes={nnodes} prefill_group={prefill_group} "
                            f"decode_group={decode_group} fraction={fraction}"
                        )
                        if not valid_splits_exist:
                            with pytest.raises(ValueError, match="Cannot split"):
                                split_nodes_for_pd(
                                    nodes, fraction, prefill_group, decode_group
                                )
                            continue

                        prefill_nodes, decode_nodes = split_nodes_for_pd(
                            nodes, fraction, prefill_group, decode_group
                        )
                        assert len(prefill_nodes) % prefill_group == 0, case
                        assert len(prefill_nodes) >= prefill_group, case
                        assert len(decode_nodes) % decode_group == 0, case
                        assert len(decode_nodes) >= decode_group, case
                        assert (
                            prefill_nodes.keys() | decode_nodes.keys() == nodes.keys()
                        ), case
                        assert not prefill_nodes.keys() & decode_nodes.keys(), case
                        assert max(prefill_nodes) < min(decode_nodes), case
                        assert min(nodes) in prefill_nodes, case
