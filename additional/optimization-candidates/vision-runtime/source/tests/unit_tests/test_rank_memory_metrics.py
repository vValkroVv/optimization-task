"""Per-rank allocator evidence independent of rank-zero metric backends."""

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.tools import utils


def test_capacity_report_retains_byte_counts():
    stats = {
        "allocated_bytes.all.current": 101,
        "allocated_bytes.all.peak": 202,
        "active_bytes.all.peak": 303,
        "reserved_bytes.all.current": 404,
        "reserved_bytes.all.peak": 505,
        "num_alloc_retries": 0,
        "num_ooms": 0,
    }
    with (
        mock.patch.object(utils, "_capacity_bytes", 1024),
        mock.patch.object(utils.device_module, "memory_stats", return_value=stats),
    ):
        capacity, _ = utils.device_capacity_report()
    assert capacity["memory/allocated_bytes"] == 101
    assert capacity["memory/max_allocated_bytes"] == 202
    assert capacity["memory/max_active_bytes"] == 303
    assert capacity["memory/reserved_bytes"] == 404
    assert capacity["memory/max_reserved_bytes"] == 505


@pytest.mark.parametrize("rank", [0, 7, 127])
def test_memory_rows_saved_for_every_rank_before_peak_reset(tmp_path, rank):
    capacity = {
        "memory/max_reserved_pct": 50.0,
        "memory/max_allocated_bytes": 123456,
        "memory/max_reserved_bytes": 234567,
    }
    with (
        mock.patch.object(MetricsProcessor, "_build_metric_logger") as backend,
        mock.patch("torch.distributed.get_rank", return_value=rank),
        mock.patch("torchtitan.components.metrics.reset_peak_stats") as reset,
        mock.patch(
            "torchtitan.components.metrics.device_capacity_report",
            return_value=(capacity, 1.0),
        ),
    ):
        memory_path = tmp_path / "memory" / f"rank{rank}.jsonl"
        rows_at_reset = []
        reset.side_effect = lambda: rows_at_reset.append(
            len(memory_path.read_text().splitlines()) if memory_path.exists() else 0
        )
        processor = MetricsProcessor(
            MetricsProcessor.Config(log_freq=1, save_rank_memory_stats=True),
            parallel_dims=SimpleNamespace(non_data_parallel_size=1),
            dump_folder=str(tmp_path),
        )
        for step in [1, 2]:
            processor.log(step, 1.0, 1.1, 0.5)
            rows = [
                json.loads(line)
                for line in (tmp_path / "memory" / f"rank{rank}.jsonl").read_text().splitlines()
            ]
            assert [row["step"] for row in rows] == list(range(1, step + 1))
            assert rows[-1]["rank"] == rank
            assert rows[-1]["memory/max_allocated_bytes"] == 123456
            assert rows[-1]["memory/max_reserved_bytes"] == 234567
        processor.close()
        assert rows_at_reset == [0, 1, 2]
        assert backend.return_value.log.call_count == 2


def test_rank_memory_disabled_does_not_create_files(tmp_path):
    with (
        mock.patch.object(MetricsProcessor, "_build_metric_logger"),
        mock.patch("torchtitan.components.metrics.reset_peak_stats"),
    ):
        processor = MetricsProcessor(
            MetricsProcessor.Config(),
            parallel_dims=SimpleNamespace(non_data_parallel_size=1),
            dump_folder=str(tmp_path),
        )
        processor.close()
    assert not list(tmp_path.iterdir())
