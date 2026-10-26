import json
import os
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ..config_model import ObmProxyConfig
from ..logger import Logger
from ..subprocess_manager import SubprocessManager


class ObmProxyStartError(Exception):
    pass


class ObmProxyManager:
    def __init__(self):
        self.sglang_port = None
        self.api_port = None
        self.chat_template = None
        self.chat_template_kwargs = None
        self.reasoning_token_budget = 0
        self.sampling_params = {}
        self.template_version = None
        self.bash_cmd = None
        self.subprocess_manager: Optional[SubprocessManager] = None
        self.chat_handlers_mode = None
        self.tokenizer_path = None
        self.chat_completions_url = None
        self.completions_url = None
        self.generate_url = None
        self.logging_level = "info"
        self.proxy_envs: dict[str, str] = {}

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    def _build_envs(self) -> dict[str, str]:
        envs = {
            "OBM_PROXY_TOKENIZER_PATH": self.tokenizer_path.absolute().as_posix(),
            "OBM_PROXY_CHAT_HANDLERS_MODE": self.chat_handlers_mode or "pretrain",
            "OBM_PROXY_CHAT_TEMPLATE_KWARGS": json.dumps(
                self.chat_template_kwargs or {}
            ),
            "OBM_PROXY_SGLANG_PORT": str(self.sglang_port),
            "OBM_PROXY_CHAT_COMPLETIONS_URL": self.chat_completions_url,
            "OBM_PROXY_COMPLETIONS_URL": self.completions_url,
            "OBM_PROXY_GENERATE_URL": self.generate_url,
            "OBM_PROXY_REASONING_TOKEN_BUDGET": str(self.reasoning_token_budget),
            "OBM_PROXY_LOGGING_LEVEL": self.logging_level,
            "OBM_PROXY_SAMPLING_PARAMS": json.dumps(self.sampling_params),
        }
        if self.chat_template:
            envs["OBM_PROXY_CHAT_TEMPLATE"] = self.chat_template
        if self.template_version:
            envs["OBM_PROXY_TEMPLATE_VERSION"] = self.template_version
        return envs

    def _subprocess_env(self) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("OBM_PROXY_")
        }
        env.update(self.proxy_envs)
        return env

    def setup(
        self,
        config: ObmProxyConfig,
        hf_checkpoint_path: Path,
        *,
        chat_completions_url: str,
        completions_url: str,
        generate_url: str,
    ):
        self.sglang_port = config.sglang_port
        self.api_port = config.api_port
        self.chat_template = config.chat_template
        self.chat_template_kwargs = config.chat_template_kwargs
        self.reasoning_token_budget = config.reasoning_token_budget
        self.sampling_params = config.sampling_params.model_dump(exclude_unset=True)
        self.template_version = config.template_version
        self.workers = config.workers
        self.logging_level = config.logging_level
        self.chat_handlers_mode = config.chat_handlers_mode
        self.tokenizer_path = (
            Path(config.tokenizer_path)
            if config.tokenizer_path is not None
            else hf_checkpoint_path
        )
        self.chat_completions_url = chat_completions_url
        self.completions_url = completions_url
        self.generate_url = generate_url
        self.proxy_envs = self._build_envs()
        self.bash_cmd = [
            "python3",
            "-m",
            "uvicorn",
            "obm_proxy.proxy.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.api_port),
            "--log-level",
            self.logging_level,
            "--workers",
            str(self.workers),
        ]
        Logger.info(
            f"OBM proxy initialized successfully with config: {config} and command: {self.bash_cmd}"
            f"with chat_handlers_mode: {config.chat_handlers_mode} and chat_template: {config.chat_template}"
        )

    @retry(
        stop=stop_after_attempt(60),
        wait=wait_fixed(5),
        reraise=True,
        retry=retry_if_exception_type(
            (ObmProxyStartError, requests.exceptions.RequestException)
        ),
    )
    def wait_for_obm_proxy_to_initialize(self):
        response = requests.get(f"http://localhost:{self.api_port}/ready", timeout=10)
        if response.status_code == 200:
            return True
        raise ObmProxyStartError("OBM proxy did not start in 5 minutes")

    def start(self):
        Logger.info(f"Starting OBM proxy with command: {self.bash_cmd}")
        obm_proxy_path = Path(__file__).resolve().parent.parent.as_posix()
        self.subprocess_manager = SubprocessManager(
            self.bash_cmd,
            Logger.obm_proxy_log_file,
            obm_proxy_path,
            env=self._subprocess_env(),
        )
        self.subprocess_manager.start()
        self.wait_for_obm_proxy_to_initialize()
        Logger.info("OBM proxy started successfully")

    def stop(self):
        if self.subprocess_manager is None:
            return
        self.subprocess_manager.graceful_stop()
        self.subprocess_manager = None
