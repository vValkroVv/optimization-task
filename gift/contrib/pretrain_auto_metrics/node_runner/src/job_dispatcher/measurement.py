import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MeasurementUpload:
    key_prefix: str
    rel_paths: list[Path]


class MeasurementFiles:
    def __init__(self, metrics_dir: Path, recorded_mtime_ns: int = 0) -> None:
        self.metrics_dir = Path(metrics_dir)
        self.recorded_mtime_ns = recorded_mtime_ns
        self._baseline = self._state()
        self._current: Optional[dict[Path, tuple[int, int]]] = None

    def _state(self) -> dict[Path, tuple[int, int]]:
        state: dict[Path, tuple[int, int]] = {}
        for dir_path, _, file_names in os.walk(self.metrics_dir):
            rel_dir = Path(dir_path).relative_to(self.metrics_dir)
            for file_name in file_names:
                try:
                    stat = os.stat(Path(dir_path, file_name))
                except OSError:
                    continue
                state[rel_dir / file_name] = (stat.st_mtime_ns, stat.st_size)
        return state

    def _scan(self) -> dict[Path, tuple[int, int]]:
        if self._current is None:
            self._current = self._state()
        return self._current

    def changed(self) -> list[Path]:
        return sorted(
            rel_path
            for rel_path, entry in self._scan().items()
            if self._baseline.get(rel_path) != entry
            or entry[0] > self.recorded_mtime_ns
        )

    def paths(self) -> set[Path]:
        return {self.metrics_dir / rel_path for rel_path in self.changed()}

    def mtime_ns(self) -> int:
        return max((mtime for mtime, _ in self._scan().values()), default=0)
