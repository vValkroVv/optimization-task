import re
from dataclasses import dataclass
from typing import Iterator

_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(\{(?P<labels>[^}]*)\})?\s+(?P<value>[^\s]+)\s*$"
)
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    name: str
    labels: dict[str, str]
    value: float


class MetricsParser:
    """One ``/metrics`` page in exposition format, parsed into samples. Every
    rank of the page is kept with its labels: the trace collects all nodes and
    ranks as they are, aggregation is left to the analysis."""

    def __init__(self, text: str) -> None:
        self._samples = list(self._parse(text))

    def values_of(
        self, metric: str, kind: str
    ) -> list[tuple[str, dict[str, str], float]]:
        if kind == "histogram":
            names = {f"{metric}_bucket", f"{metric}_sum", f"{metric}_count"}
        else:
            names = {metric}
        return [
            (sample.name, dict(sample.labels), sample.value)
            for sample in self._samples
            if sample.name in names
        ]

    @staticmethod
    def _parse(text: str) -> Iterator[Sample]:
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            match = _LINE_RE.match(line)
            if match is None:
                continue
            try:
                value = float(match.group("value"))
            except ValueError:
                continue
            raw_labels = match.group("labels")
            labels = dict(_LABEL_RE.findall(raw_labels)) if raw_labels else {}
            yield Sample(match.group("name"), labels, value)
