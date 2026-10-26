import threading
import time
from typing import Optional

import requests

from ...logger import Logger
from .model import MetricSample, ScrapeTarget
from .parser import MetricsParser
from .stream import MetricsStream

SCRAPE_INTERVAL_SECONDS = 10.0
REQUEST_TIMEOUT_SECONDS = 5.0


class MetricsScraper:
    def __init__(
        self,
        log_profile: str,
        model_name: Optional[str],
        targets: list[ScrapeTarget],
    ):
        self._log_profile = log_profile
        self._model_name = model_name
        self._targets = targets
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._unreachable: set[str] = set()

    def start(self) -> None:
        if not self._targets or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(
            timeout=len(self._targets) * REQUEST_TIMEOUT_SECONDS
            + SCRAPE_INTERVAL_SECONDS
        )
        if self._thread.is_alive():
            Logger.error(
                f"metrics scraper for {self._log_profile} did not stop in time, "
                "its last samples may be missing from the jsonl"
            )

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                Logger.warning(f"metrics scrape tick failed: {e}")
            self._stop_event.wait(SCRAPE_INTERVAL_SECONDS)

    def _tick(self) -> None:
        samples: list[MetricSample] = []
        for target in self._targets:
            parser = self._fetch(target)
            if parser is not None:
                samples.extend(self._samples(target, parser))
        MetricsStream.write(samples)

    def _fetch(self, target: ScrapeTarget) -> Optional[MetricsParser]:
        try:
            response = requests.get(target.url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if target.key not in self._unreachable:
                self._unreachable.add(target.key)
                Logger.warning(
                    f"metrics scrape target unreachable, "
                    f"{target.key} ({target.url}): {e}"
                )
            return None
        if target.key in self._unreachable:
            self._unreachable.discard(target.key)
            Logger.info(
                f"metrics scrape target reachable again: {target.key} ({target.url})"
            )
        return MetricsParser(response.text)

    def _samples(
        self, target: ScrapeTarget, parser: MetricsParser
    ) -> list[MetricSample]:
        ts = time.time()
        samples = []
        for metric, kind in target.metrics.items():
            for name, labels, value in parser.values_of(metric, kind):
                samples.append(
                    MetricSample(
                        log_profile=self._log_profile,
                        model_name=self._model_name,
                        source=target.source,
                        endpoint=target.endpoint,
                        metric=name,
                        kind=kind,
                        labels=labels,
                        ts=ts,
                        value=value,
                    )
                )
        return samples
