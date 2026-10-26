from dataclasses import replace

from ...config_model import SglangInferenceConfig, SglangPdInferenceConfig
from ...logger import Logger
from .sglang_engine import EngineUnit, SglangEngine, _group_nnodes
from .sglang_router_manager import SglangRouter


class SglangPdEngine(SglangEngine):
    def _sides(
        self, nodes: dict[int, str], inference_config: SglangPdInferenceConfig
    ) -> list[tuple[dict[int, str], SglangInferenceConfig]]:
        prefill_nodes, decode_nodes = split_nodes_for_pd(
            nodes,
            inference_config.prefill_fraction,
            _group_nnodes(inference_config.prefill, self.gpus_per_node),
            _group_nnodes(inference_config.decode, self.gpus_per_node),
        )
        return [
            (prefill_nodes, inference_config.prefill),
            (decode_nodes, inference_config.decode),
        ]

    def _make_router(self) -> SglangRouter:
        prefill_nodes, prefill_config = self.sides[0]
        prefill_units = self._side_units(prefill_nodes, prefill_config)
        return SglangRouter(
            self.router_config,
            log_profile=self.log_profile,
            prefill_urls=frozenset(unit.worker_urls[0] for unit in prefill_units),
            bootstrap_port=prefill_config.disaggregation_bootstrap_port,
        )

    def units(self, from_tail: bool = False) -> list[EngineUnit]:
        (prefill_nodes, prefill_config), (decode_nodes, decode_config) = self.sides
        prefill_units = self._side_units(prefill_nodes, prefill_config)
        decode_units = self._side_units(decode_nodes, decode_config)
        if from_tail:
            prefill_units.reverse()
            decode_units.reverse()
        first_prefill = prefill_units.pop(0)
        first_decode = decode_units.pop(0)
        minimal_pd_pair = EngineUnit(
            ranks=first_prefill.ranks + first_decode.ranks,
            worker_urls=first_prefill.worker_urls + first_decode.worker_urls,
            prefill_instances=1,
            decode_instances=1,
        )
        units = [minimal_pd_pair]
        for extra_unit in decode_units + prefill_units:
            units.append(replace(extra_unit, standalone=False))
        return units


def split_nodes_for_pd(
    nodes_for_inference: dict[int, str],
    prefill_fraction: float,
    prefill_group_nnodes: int,
    decode_group_nnodes: int,
) -> tuple[dict[int, str], dict[int, str]]:
    nnodes = len(nodes_for_inference)
    prefill_nnodes_options = []
    for candidate in range(prefill_group_nnodes, nnodes, prefill_group_nnodes):
        decode_nnodes = nnodes - candidate
        if decode_nnodes % decode_group_nnodes == 0:
            prefill_nnodes_options.append(candidate)
    if not prefill_nnodes_options:
        raise ValueError(
            f"Cannot split {nnodes} nodes for PD disaggregation: "
            f"prefill needs a non-zero multiple of {prefill_group_nnodes} "
            f"nodes and decode a non-zero multiple of "
            f"{decode_group_nnodes} nodes"
        )
    requested_prefill_nnodes = nnodes * prefill_fraction
    prefill_nnodes = min(
        prefill_nnodes_options,
        key=lambda candidate: abs(candidate - requested_prefill_nnodes),
    )
    requested_whole_prefill_nnodes = int(requested_prefill_nnodes)
    if prefill_nnodes != requested_whole_prefill_nnodes:
        Logger.info(
            f"prefill_fraction={prefill_fraction} requests "
            f"{requested_prefill_nnodes:g} of {nnodes} nodes for prefill; "
            f"using the closest valid split: {prefill_nnodes} prefill / "
            f"{nnodes - prefill_nnodes} decode"
        )
    ranks = sorted(nodes_for_inference)
    prefill_ranks = ranks[:prefill_nnodes]
    decode_ranks = ranks[prefill_nnodes:]
    return (
        {rank: nodes_for_inference[rank] for rank in prefill_ranks},
        {rank: nodes_for_inference[rank] for rank in decode_ranks},
    )
