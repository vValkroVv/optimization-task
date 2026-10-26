from math import ceil
from pathlib import Path
from typing import Optional

import requests

from ...config_model import (
    InferenceConfig,
    ObmProxyConfig,
    RouterConfig,
    SglangInferenceConfig,
)
from ...logger import Logger
from ...trace import MetricsScraper, ScrapeTarget
from ..engine import EngineLoad, EngineUnit, InferenceEngine, _worker_urls
from .sglang_inference import SglangInference
from .sglang_router_manager import SglangRouter

WORKER_PORT_OFFSET = 1


class SglangEngine(InferenceEngine):
    def __init__(
        self,
        hf_checkpoint_path: Path,
        nodes: dict[int, str],
        inference_config: InferenceConfig,
        global_node_rank: int,
        gpus_per_node: int,
        router_config: RouterConfig,
        obm_proxy_config: ObmProxyConfig | None = None,
        log_profile: str = "evaluated",
        model_name: Optional[str] = None,
        record_generations_dir: str | None = None,
    ):
        super().__init__(
            router_port=router_config.port,
            log_profile=log_profile,
            model_name=model_name,
        )
        self.gpus_per_node = gpus_per_node
        self.sides = self._sides(nodes, inference_config)
        side_nodes, side_config = next(
            (nodes, config) for nodes, config in self.sides if global_node_rank in nodes
        )
        self.inference_config = side_config
        self.global_node_rank = global_node_rank
        self.router_config = router_config
        self.router = self._make_router()
        self.scraper: Optional[MetricsScraper] = None
        self.router_scraper: Optional[MetricsScraper] = None
        self.unit = next(
            unit
            for unit in self._side_units(side_nodes, side_config)
            if global_node_rank in unit.ranks
        )
        self.group_node_rank = self.unit.ranks.index(global_node_rank)
        group_head_host = side_nodes[self.unit.ranks[0]]
        (group_worker_url,) = self.unit.worker_urls
        self.inference = SglangInference(
            hf_checkpoint_path=hf_checkpoint_path,
            inference_config=self.inference_config,
            log_profile=log_profile,
            port=self._worker_port,
            url=group_worker_url,
            nnodes=len(self.unit.ranks),
            node_rank=self.group_node_rank,
            dist_init_address=(
                f"{group_head_host}:{self.inference_config.sglang_communication_port}"
            ),
        )
        self._setup_recorder(record_generations_dir)
        self._setup_obm_proxy(obm_proxy_config, hf_checkpoint_path)

    @property
    def _worker_port(self) -> int:
        return self.router_config.port + WORKER_PORT_OFFSET

    @property
    def inference_instances(self) -> int:
        instances = 0
        for side_nodes, side_config in self.sides:
            side_gpus = len(side_nodes) * self.gpus_per_node
            gpus_per_instance = side_config.tp * side_config.pp
            instances += side_gpus // gpus_per_instance
        return instances

    def _sides(
        self, nodes: dict[int, str], inference_config: SglangInferenceConfig
    ) -> list[tuple[dict[int, str], SglangInferenceConfig]]:
        return [(nodes, inference_config)]

    def _make_router(self) -> SglangRouter:
        return SglangRouter(self.router_config, log_profile=self.log_profile)

    def units(self, from_tail: bool = False) -> list[EngineUnit]:
        ((side_nodes, side_config),) = self.sides
        units = self._side_units(side_nodes, side_config)
        if from_tail:
            units.reverse()
        return units

    def _side_units(
        self, side_nodes: dict[int, str], side_config: SglangInferenceConfig
    ) -> list[EngineUnit]:
        group_nnodes = _group_nnodes(side_config, self.gpus_per_node)
        if len(side_nodes) % group_nnodes != 0:
            raise ValueError("Nnodes must be divisible by nodes per group")
        ranks = sorted(side_nodes)
        units = []
        for start in range(0, len(ranks), group_nnodes):
            group_ranks = tuple(ranks[start : start + group_nnodes])
            head_host = side_nodes[group_ranks[0]]
            units.append(
                EngineUnit(
                    ranks=group_ranks,
                    worker_urls=(f"http://{head_host}:{self._worker_port}",),
                    prefill_instances=int(side_config.disaggregation_mode == "prefill"),
                    decode_instances=int(side_config.disaggregation_mode == "decode"),
                )
            )
        return units

    def load(self, units: list[EngineUnit]) -> EngineLoad | None:
        running = waiting = capacity = kv_in_transfer = 0
        token_usage = 0.0
        for worker_url in _worker_urls(units):
            try:
                response = requests.get(
                    f"{worker_url}/v1/loads",
                    params={"include": "core,disagg"},
                    timeout=15,
                )
                response.raise_for_status()
                for dp_load in response.json()["loads"]:
                    disaggregation = dp_load.get("disaggregation") or {}
                    running += dp_load["num_running_reqs"]
                    if disaggregation.get("mode") == "prefill":
                        waiting += dp_load["num_waiting_reqs"]
                        kv_in_transfer += disaggregation.get(
                            "prefill_inflight_queue_reqs", 0
                        )
                        continue
                    in_transfer = disaggregation.get("decode_transfer_queue_reqs", 0)
                    waiting += dp_load["num_waiting_reqs"] - in_transfer
                    kv_in_transfer += in_transfer
                    capacity += dp_load["max_running_requests"]
                    token_usage = max(token_usage, dp_load["token_usage"])
            except (requests.RequestException, ValueError, KeyError, TypeError):
                return None
        return EngineLoad(
            running=running,
            waiting=waiting,
            capacity=capacity,
            token_usage=token_usage,
            kv_in_transfer=kv_in_transfer,
        )

    def attach(self, units: list[EngineUnit]) -> None:
        worker_urls = set(_worker_urls(units))
        router_serves_workers = bool(self.router.worker_urls)
        if router_serves_workers:
            self.router.add_workers(worker_urls)
            return
        Logger.info(
            f"Starting {self.log_profile} SGLang router on "
            f"global_node_rank={self.global_node_rank}, "
            f"port={self.router_config.port}, "
            f"worker_urls={sorted(worker_urls)}"
        )
        self.router.start_router(worker_urls)
        self._start_proxies()
        prometheus_port = self.router_config.prometheus_port
        self.router_scraper = self._start_scraper(
            ScrapeTarget(
                source="router",
                endpoint=f":{prometheus_port}",
                url=f"http://127.0.0.1:{prometheus_port}/metrics",
            )
        )

    def detach(self, units: list[EngineUnit]) -> None:
        worker_urls = set(_worker_urls(units))
        detaches_every_router_worker = worker_urls.issuperset(self.router.worker_urls)
        if detaches_every_router_worker:
            self._stop_router()
            return
        self.router.remove_workers(worker_urls)

    def start(self) -> None:
        Logger.info(
            f"Starting {self.log_profile} SGLang inference on "
            f"global_node_rank={self.global_node_rank}, "
            f"disaggregation_mode={self.inference_config.disaggregation_mode}, "
            f"group_node_rank={self.group_node_rank}, "
            f"group_nnodes={len(self.unit.ranks)}, "
            f"sglang_url={self.inference.url}"
        )
        self.inference.start_inference()
        self.inference.wait_for_inference_to_initialize()
        if self.group_node_rank == 0:
            self.inference.copy_dg_cache_to_nfs()
        if self.inference_config.enable_metrics:
            self.scraper = self._start_scraper(
                ScrapeTarget(
                    source="engine",
                    endpoint=f":{self._worker_port}",
                    url=f"http://127.0.0.1:{self._worker_port}/metrics",
                )
            )

    def stop(self) -> None:
        if self.scraper is not None:
            self.scraper.stop()
            self.scraper = None
        self.inference.stop()

    def close(self) -> None:
        self._stop_router()
        self.stop()

    def _stop_router(self) -> None:
        if self.router_scraper is not None:
            self.router_scraper.stop()
            self.router_scraper = None
        self._stop_proxies()
        self.router.stop_router()

    def _start_scraper(self, target: ScrapeTarget) -> MetricsScraper:
        scraper = MetricsScraper(
            log_profile=self.log_profile,
            model_name=self.model_name,
            targets=[target],
        )
        scraper.start()
        return scraper


def _group_nnodes(config: SglangInferenceConfig, gpus_per_node: int) -> int:
    return ceil(config.tp * config.pp / gpus_per_node)
