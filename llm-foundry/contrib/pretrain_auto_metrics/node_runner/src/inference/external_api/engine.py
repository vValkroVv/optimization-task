import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ...config_model import ExternalApiInferenceConfig, ObmProxyConfig
from ...logger import Logger
from ...subprocess_manager import SubprocessManager
from ..engine import InferenceEngine


class ExternalApiProxyStartError(Exception):
    pass


class ExternalApiEngine(InferenceEngine):
    def __init__(
        self,
        inference_config: ExternalApiInferenceConfig,
        obm_proxy_config: ObmProxyConfig | None = None,
        hf_checkpoint_path: Path | None = None,
        log_profile: str = "evaluated",
        model_name: Optional[str] = None,
    ):
        super().__init__(
            router_port=inference_config.port,
            log_profile=log_profile,
            model_name=model_name,
        )
        self.inference_config = inference_config
        self.subprocess_manager: Optional[SubprocessManager] = None
        self._setup_obm_proxy(obm_proxy_config, hf_checkpoint_path)

    @property
    def inference_instances(self) -> int:
        return 1

    @cached_property
    def proxy_env(self) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("EXTERNAL_API_PROXY_")
        }
        if self.inference_config.api_key is not None:
            env["EXTERNAL_API_PROXY_API_KEY"] = self.inference_config.api_key
        if self.inference_config.user_pass is not None:
            env["EXTERNAL_API_PROXY_USER_PASS"] = self.inference_config.user_pass
        return env

    @cached_property
    def proxy_cmd(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "src.inference.external_api.proxy.main",
            "--host",
            "0.0.0.0",
            *self.inference_config.to_bash_list(),
        ]

    @cached_property
    def proxy_cwd(self) -> str:
        return Path(__file__).parents[3].as_posix()

    @retry(
        stop=stop_after_attempt(60),
        wait=wait_fixed(5),
        reraise=True,
        retry=retry_if_exception_type(
            (ExternalApiProxyStartError, requests.exceptions.RequestException)
        ),
    )
    def wait_for_proxy_to_be_ready(self) -> bool:
        response = requests.get(f"{self.direct_base_url}/ready", timeout=5)
        if response.status_code == 200:
            return True
        raise ExternalApiProxyStartError("External API proxy did not start")

    def start(
        self,
    ) -> None:
        Logger.info(
            f"Starting {self.log_profile} External API proxy on "
            f"port={self.router_port}, upstream_base_url={self.inference_config.base_url}"
        )
        self.subprocess_manager = SubprocessManager(
            self.proxy_cmd,
            self.inference_log_file,
            self.proxy_cwd,
            "External API proxy",
            env=self.proxy_env,
        )
        Logger.info(f"Starting External API proxy with command: {self.proxy_cmd}")
        self.subprocess_manager.start()
        self.wait_for_proxy_to_be_ready()
        self._start_proxies()
        Logger.info("External API proxy started successfully")
        return

    def stop(self) -> None:
        self._stop_proxies()
        if self.subprocess_manager is None:
            return
        self.subprocess_manager.graceful_stop()
        self.subprocess_manager = None
