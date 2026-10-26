import sys
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ...logger import Logger
from ...subprocess_manager import SubprocessManager

RECORDER_PORT_OFFSET = 5000


class GenerationsManager:
    def __init__(
        self,
        *,
        port: int,
        dump_dir: Path,
        log_profile: str,
        upstream_url: Optional[str] = None,
    ) -> None:
        self.port = port
        self.bash_cmd = [
            sys.executable,
            "-m",
            "src.inference.generations.server",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--dump-dir",
            dump_dir.as_posix(),
            "--profile",
            log_profile,
        ]
        if upstream_url is not None:
            self.bash_cmd += ["--upstream-url", upstream_url]
        self.subprocess_manager: Optional[SubprocessManager] = None

    @retry(
        stop=stop_after_attempt(120),
        wait=wait_fixed(5),
        reraise=True,
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def wait_for_server_to_initialize(self) -> None:
        requests.get(
            f"http://127.0.0.1:{self.port}/ready", timeout=10
        ).raise_for_status()

    def start(self) -> None:
        Logger.info(f"Starting generations server with command: {self.bash_cmd}")
        self.subprocess_manager = SubprocessManager(
            self.bash_cmd,
            Logger.node_log_file,
            Path(__file__).parents[3].as_posix(),
            "Generations server",
        )
        self.subprocess_manager.start()
        self.wait_for_server_to_initialize()
        Logger.info("Generations server started successfully")

    def stop(self) -> None:
        if self.subprocess_manager is None:
            return
        self.subprocess_manager.graceful_stop()
        self.subprocess_manager = None
