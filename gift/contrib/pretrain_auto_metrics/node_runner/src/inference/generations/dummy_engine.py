from pathlib import Path
from typing import Optional

from ...config_model import ObmProxyConfig
from ...logger import Logger
from ..engine import InferenceEngine


class DummyEngine(InferenceEngine):
    def __init__(
        self,
        replay_generations_dir: str,
        port: int,
        obm_proxy_config: ObmProxyConfig | None = None,
        hf_checkpoint_path: Path | None = None,
        log_profile: str = "evaluated",
        model_name: Optional[str] = None,
    ):
        super().__init__(
            router_port=port,
            log_profile=log_profile,
            model_name=model_name,
        )
        self._setup_generations(replay_generations_dir, port=port)
        self._setup_obm_proxy(obm_proxy_config, hf_checkpoint_path)

    @property
    def inference_instances(self) -> int:
        return 8

    def start(self) -> None:
        Logger.info(
            f"Starting {self.log_profile} generation replay on port={self.router_port}"
        )
        self._start_proxies()

    def stop(self) -> None:
        self._stop_proxies()
