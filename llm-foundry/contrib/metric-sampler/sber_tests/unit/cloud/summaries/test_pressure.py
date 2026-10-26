from __future__ import annotations

import pytest

from metric_sampler.cloud.summaries.pressure import IoPressure

pytestmark = pytest.mark.unit

_PRESSURE = (
    "some avg10=1.00 avg60=2.00 avg300=3.00 total=4\n"
    "full avg10={avg10} avg60={avg60} avg300=7.00 total=8\n"
)


@pytest.mark.parametrize(
    ("avg10", "avg60", "expected"),
    [("0.00", "0.00", (0, 0)), ("99.99", "50.50", (99, 50)), ("12", "7.1", (12, 7)), ("100.00", "3.00", (100, 3))],
)
def test_io_pressure_reads_the_full_line_and_drops_the_fraction_of_each_average(
    tmp_path, avg10, avg60, expected,
) -> None:
    path = tmp_path / "io"
    path.write_text(_PRESSURE.format(avg10=avg10, avg60=avg60), encoding="utf-8")
    pressure = IoPressure(path)

    assert (pressure.full_avg10(), pressure.full_avg60()) == expected


def test_io_pressure_without_a_full_line_is_an_error(tmp_path) -> None:
    path = tmp_path / "io"
    path.write_text("some avg10=1.00 avg60=2.00 avg300=3.00 total=4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has no full line"):
        IoPressure(path).full_avg10()
