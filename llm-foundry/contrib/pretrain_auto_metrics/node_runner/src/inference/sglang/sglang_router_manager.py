from typing import Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ...config_model import RouterConfig
from ...logger import Logger
from ...subprocess_manager import SubprocessManager


class RouterStartError(Exception):
    pass


class SglangRouter:
    def __init__(
        self,
        router_config: RouterConfig,
        log_profile: str = "evaluated",
        prefill_urls: frozenset[str] = frozenset(),
        bootstrap_port: int | None = None,
    ):
        self.log_profile = log_profile
        self.subprocess_manager: Optional[SubprocessManager] = None
        self._cmd_to_run = [
            "/home/user/sglang/bin/python",
            "-m",
            "sglang_router.launch_router",
            *router_config.to_bash_list(),
        ]
        self._base_url = f"http://localhost:{router_config.port}"
        self.worker_urls: set[str] = set()
        self._prefill_urls = prefill_urls
        self._prefill_bootstrap_port = bootstrap_port
        self._pd_disaggregation = bootstrap_port is not None
        Logger.info(f"Router ready to start with command: {self._cmd_to_run}")

    def _check_router_is_ready(self, must_be_healthy: set[str]) -> bool:
        url = f"{self._base_url}/workers"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return False

            workers = response.json().get("workers", [])
            if len(workers) != len(self.worker_urls):
                return False

            return all(
                worker.get("is_healthy") is True
                for worker in workers
                if worker["url"].rstrip("/") in must_be_healthy
            )
        except (requests.exceptions.RequestException, ValueError, KeyError):
            return False

    @retry(
        stop=stop_after_attempt(480),
        wait=wait_fixed(5),
        reraise=True,
        retry=retry_if_exception_type(RouterStartError),
    )
    def wait_for_router_to_initialize(self, must_be_healthy: set[str]):
        if not self.subprocess_manager.is_alive():
            self.subprocess_manager.raise_for_exit_code()
        if self._check_router_is_ready(must_be_healthy):
            Logger.info("Router started successfully")
            return

        raise RouterStartError("Router did not start in 40 minutes")

    @property
    def is_running(self) -> bool:
        return (
            self.subprocess_manager is not None and self.subprocess_manager.is_alive()
        )

    def _worker_args(self) -> list[str]:
        if not self._pd_disaggregation:
            return ["--worker-urls", *sorted(self.worker_urls)]
        prefill_worker_urls = self.worker_urls & self._prefill_urls
        decode_worker_urls = self.worker_urls - self._prefill_urls
        args = ["--pd-disaggregation"]
        for url in sorted(prefill_worker_urls):
            args.extend(["--prefill", url, str(self._prefill_bootstrap_port)])
        for url in sorted(decode_worker_urls):
            args.extend(["--decode", url])
        return args

    def _worker_spec(self, url: str) -> dict:
        if not self._pd_disaggregation:
            return {"url": url}
        if url in self._prefill_urls:
            return {
                "url": url,
                "worker_type": "prefill",
                "bootstrap_port": self._prefill_bootstrap_port,
            }
        return {"url": url, "worker_type": "decode"}

    def add_workers(self, worker_urls: set[str]) -> None:
        if not self.is_running:
            Logger.warning(f"Restarting crashed {self.log_profile} SGLang router")
            self.start_router(self.worker_urls | worker_urls)
            return
        for url in sorted(worker_urls):
            response = requests.post(
                f"{self._base_url}/workers", json=self._worker_spec(url), timeout=15
            )
            response.raise_for_status()
        self.worker_urls.update(worker_urls)
        self.wait_for_router_to_initialize(worker_urls)

    def remove_workers(self, worker_urls: set[str]) -> None:
        if not self.is_running:
            self.worker_urls -= worker_urls
            return
        workers_response = requests.get(f"{self._base_url}/workers", timeout=15)
        workers_response.raise_for_status()
        for worker in workers_response.json().get("workers", []):
            if worker["url"].rstrip("/") in worker_urls:
                delete_response = requests.delete(
                    f"{self._base_url}/workers/{worker['id']}", timeout=15
                )
                delete_response.raise_for_status()
        self.worker_urls.difference_update(worker_urls)
        self.wait_for_router_to_initialize(set())

    def start_router(self, worker_urls: set[str]) -> None:
        self.worker_urls = set(worker_urls)
        cmd = [*self._cmd_to_run, *self._worker_args()]
        Logger.info(f"Starting router with command: {cmd}")
        self.subprocess_manager = SubprocessManager(
            cmd, Logger.router_log_file_for(self.log_profile), None
        )
        self.subprocess_manager.start()
        self.wait_for_router_to_initialize(self.worker_urls)

    def stop_router(self) -> None:
        self.worker_urls = set()
        if self.subprocess_manager:
            self.subprocess_manager.graceful_stop()
