from __future__ import annotations

from pathlib import Path

PRESSURE_PATH = Path("/proc/pressure/io")
FULL_AVG10_FIELD = 1
FULL_AVG60_FIELD = 2


class IoPressure:
    """The share, in %, of the last seconds in which every task waited for IO (the "full" line of the pressure file)."""

    def __init__(self, path: Path = PRESSURE_PATH) -> None:
        self._path = path

    def full_avg10(self) -> int:
        return self._full(FULL_AVG10_FIELD)

    def full_avg60(self) -> int:
        return self._full(FULL_AVG60_FIELD)

    def _full(self, field: int) -> int:
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.startswith("full"):
                return int(float(line.split()[field].split("=")[1]))
        raise ValueError(f"{self._path} has no full line")
