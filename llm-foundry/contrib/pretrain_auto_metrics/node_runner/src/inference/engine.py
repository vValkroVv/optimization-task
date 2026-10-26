from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config_model import ObmProxyConfig
from ..logger import Logger
from ..obm_proxy.obm_proxy_manager import ObmProxyManager
from .generations.manager import RECORDER_PORT_OFFSET, GenerationsManager


@dataclass(frozen=True)
class EngineUnit:
    ranks: tuple[int, ...]
    worker_urls: tuple[str, ...]
    standalone: bool = True
    prefill_instances: int = 0
    decode_instances: int = 0

    def allocation_cost(
        self, available: list["EngineUnit"]
    ) -> tuple[bool, int, int, int]:
        remaining_prefill = remaining_decode = 0
        lost_prefill = lost_decode = 0
        ranks = set(self.ranks)
        for unit in available:
            if ranks.isdisjoint(unit.ranks):
                remaining_prefill += unit.prefill_instances
                remaining_decode += unit.decode_instances
            else:
                lost_prefill += unit.prefill_instances
                lost_decode += unit.decode_instances
        prefill_instances = remaining_prefill + lost_prefill
        decode_instances = remaining_decode + lost_decode
        minority_loss = 0
        if prefill_instances > decode_instances:
            minority_loss = lost_decode
        elif decode_instances > prefill_instances:
            minority_loss = lost_prefill
        lost_instances = lost_prefill + lost_decode
        return (
            lost_instances > 0,
            minority_loss,
            abs(remaining_prefill - remaining_decode),
            lost_instances,
        )


@dataclass(frozen=True)
class EngineLoad:
    running: int
    waiting: int
    capacity: int
    token_usage: float
    kv_in_transfer: int = 0


NODELESS_UNIT = EngineUnit(ranks=(), worker_urls=())


def _worker_urls(units: list[EngineUnit]) -> list[str]:
    urls = []
    for unit in units:
        urls.extend(unit.worker_urls)
    return urls


class InferenceEngine(ABC):
    def __init__(
        self,
        router_port: int,
        log_profile: str = "evaluated",
        model_name: Optional[str] = None,
    ) -> None:
        self.router_port = router_port
        self.log_profile = log_profile
        self.obm_proxy_manager: Optional[ObmProxyManager] = None
        self.generations_manager: Optional[GenerationsManager] = None
        self.model_name = model_name

    @property
    def inference_log_file(self) -> Path:
        return Logger.inference_log_file_for(self.log_profile)

    @property
    def direct_base_url(self) -> str:
        if self.generations_manager is not None:
            return f"http://127.0.0.1:{self.generations_manager.port}"
        return f"http://127.0.0.1:{self.router_port}"

    @property
    def base_url(self) -> str:
        if self.obm_proxy_manager is not None:
            return self.obm_proxy_manager.base_url
        return self.direct_base_url

    @property
    def worker_urls(self) -> list[str]:
        return sorted(_worker_urls(self.units()))

    def units(self, from_tail: bool = False) -> list[EngineUnit]:
        return [NODELESS_UNIT]

    def load(self, units: list[EngineUnit]) -> EngineLoad | None:
        return None

    def attach(self, units: list[EngineUnit]) -> None:
        pass

    def detach(self, units: list[EngineUnit]) -> None:
        pass

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/completions"

    @property
    def generate_url(self) -> str:
        return f"{self.base_url}/generate"

    def _setup_recorder(self, record_generations_dir: str | None) -> None:
        if record_generations_dir is None:
            return
        self._setup_generations(
            record_generations_dir,
            port=self.router_port + RECORDER_PORT_OFFSET,
            upstream_url=f"http://127.0.0.1:{self.router_port}",
        )

    def _setup_generations(
        self, dump_dir: str, *, port: int, upstream_url: str | None = None
    ) -> None:
        self.generations_manager = GenerationsManager(
            port=port,
            dump_dir=Path(dump_dir),
            log_profile=self.log_profile,
            upstream_url=upstream_url,
        )

    def _start_proxies(self) -> None:
        if self.generations_manager is not None:
            self.generations_manager.start()
        if self.obm_proxy_manager is not None:
            self.obm_proxy_manager.start()

    def _stop_proxies(self) -> None:
        if self.obm_proxy_manager is not None:
            self.obm_proxy_manager.stop()
        if self.generations_manager is not None:
            self.generations_manager.stop()

    def _setup_obm_proxy(
        self,
        config: ObmProxyConfig | None,
        hf_checkpoint_path: Path | None,
    ) -> None:
        if config is None:
            return
        if hf_checkpoint_path is None:
            raise ValueError("hf_checkpoint_path is required to setup OBM proxy")

        obm_proxy_manager = ObmProxyManager()
        obm_proxy_manager.setup(
            config,
            hf_checkpoint_path,
            chat_completions_url=self.chat_completions_url,
            completions_url=self.completions_url,
            generate_url=self.generate_url,
        )
        self.obm_proxy_manager = obm_proxy_manager

    @property
    @abstractmethod
    def inference_instances(self) -> int:
        pass  # pragma: no cover

    @abstractmethod
    def start(
        self,
    ) -> None:
        """
        Start the inference engine on this node.
        """
        pass  # pragma: no cover

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the inference engine.

        Args:
            None

        Returns:
            None
        """
        pass  # pragma: no cover

    def close(self) -> None:
        self.stop()
