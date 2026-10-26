import pytest

from stand.activation_kernel_checks.kernel_table import kernel_totals

pytestmark = pytest.mark.unit


def test_kernel_totals_sum_the_launches_and_microseconds_of_each_kernel_of_a_trace():
    events = [
        {"cat": "kernel", "name": "stats", "dur": 30.5},
        {"cat": "cpu_op", "name": "aten::zeros", "dur": 2.0},
        {"cat": "kernel", "name": "finalize", "dur": 6.0},
        {"cat": "kernel", "name": "stats", "dur": 31.5},
        {"ph": "M", "name": "process_name"},
    ]

    assert kernel_totals(events) == {"stats": (2, 62.0), "finalize": (1, 6.0)}
