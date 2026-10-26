import shutil
import subprocess
import threading
import time
from typing import Optional

from ...logger import Logger
from .model import NVIDIA_SMI_METRICS, MetricSample
from .scraper import SCRAPE_INTERVAL_SECONDS
from .stream import MetricsStream

NVIDIA_SMI_QUERY = ("index", "uuid", *NVIDIA_SMI_METRICS)
NVIDIA_SMI_TIMEOUT_SECONDS = 10.0


class NvidiaSmiSampler:
    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._unavailable = False

    def start(self) -> None:
        if self._thread is not None:
            return
        if shutil.which("nvidia-smi") is None:
            Logger.warning("gpu metrics disabled: nvidia-smi not found")
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=NVIDIA_SMI_TIMEOUT_SECONDS + SCRAPE_INTERVAL_SECONDS)
        if self._thread.is_alive():
            Logger.error(
                "gpu metrics sampler did not stop in time, "
                "its last samples may be missing from the jsonl"
            )

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                Logger.warning(f"gpu metrics sample tick failed: {e}")
            self._stop_event.wait(SCRAPE_INTERVAL_SECONDS)

    def _tick(self) -> None:
        rows = self._query()
        if rows is not None:
            MetricsStream.write(self._samples(rows))

    def _query(self) -> Optional[list[str]]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={','.join(NVIDIA_SMI_QUERY)}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            self._report_unavailable(str(e))
            return None
        if result.returncode != 0:
            self._report_unavailable(result.stderr.strip())
            return None
        if self._unavailable:
            self._unavailable = False
            Logger.info("gpu metrics available again")
        return result.stdout.splitlines()

    def _report_unavailable(self, reason: str) -> None:
        if self._unavailable:
            return
        self._unavailable = True
        Logger.warning(f"gpu metrics unavailable, nvidia-smi failed: {reason}")

    def _samples(self, rows: list[str]) -> list[MetricSample]:
        ts = time.time()
        samples = []
        for row in rows:
            index, uuid, *values = (cell.strip() for cell in row.split(","))
            for metric, raw in zip(NVIDIA_SMI_METRICS.values(), values):
                value = self._parse_value(raw)
                if value is None:
                    continue
                samples.append(
                    MetricSample(
                        log_profile="node",
                        model_name=None,
                        source="nvidia_smi",
                        endpoint=f"gpu:{index}",
                        metric=metric,
                        kind="gauge",
                        labels={"uuid": uuid},
                        ts=ts,
                        value=value,
                    )
                )
        return samples

    @staticmethod
    def _parse_value(raw: str) -> Optional[float]:
        if not raw or raw.startswith("["):
            return None
        try:
            return float(int(raw, 16)) if raw.startswith("0x") else float(raw)
        except ValueError:
            return None
