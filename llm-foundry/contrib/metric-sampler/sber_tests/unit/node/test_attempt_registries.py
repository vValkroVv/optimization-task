from __future__ import annotations

import json
from pathlib import Path

import pytest

from metric_sampler.contract.catalog import METRICS
from metric_sampler.contract.raw_layout import AttemptDir
from metric_sampler.node.attempt_registries import AttemptRegistries

pytestmark = pytest.mark.unit


@pytest.fixture
def attempt_dir(tmp_path: Path) -> AttemptDir:
    attempt_dir = AttemptDir(path=tmp_path / "exp" / "run" / "attempt=a1", exp="exp", run="run", attempt="a1")
    attempt_dir.path.mkdir(parents=True)
    return attempt_dir


def test_the_metrics_registry_lists_every_catalog_metric_with_its_merge_rules(attempt_dir: AttemptDir) -> None:
    AttemptRegistries(attempt_dir).write_metrics()

    registry = json.loads(attempt_dir.metrics_path.read_text())

    assert [entry["metric_id"] for entry in registry] == [metric.metric_id for metric in METRICS]
    assert [entry["name"] for entry in registry] == [metric.name for metric in METRICS]
    assert list(registry[0]) == [
        "metric_id", "name", "family", "value_type", "node_merge", "unit", "description", "bins",
        "summarize_experts", "detailed", "job_identical", "required",
    ]
    by_name = {entry["name"]: entry for entry in registry}
    assert by_name["router/tokens_per_expert"]["summarize_experts"] is True
    assert by_name["loss/total"]["summarize_experts"] is False
    assert by_name["loss/total"]["bins"] is None
    assert by_name["loss/total"]["detailed"] is False
    assert by_name["loss/total"]["job_identical"] is False
    assert by_name["router/expert_bias_before_update"]["job_identical"] is True
    assert by_name["grad/clip_total_norm"]["job_identical"] is True
    assert by_name["expert_loss/hist"]["detailed"] is True
    assert by_name["loss/token_hist_by_stream"]["bins"] == {"start": 0.0, "width": 0.125, "count": 256}
    assert by_name["loss/total"]["family"] == "scalars"


@pytest.mark.parametrize("stream_names", [("all",), ("all", "code", "web")])
def test_the_streams_registry_numbers_the_stream_names_in_order(attempt_dir: AttemptDir, stream_names) -> None:
    AttemptRegistries(attempt_dir).write_streams(stream_names)

    assert json.loads(attempt_dir.streams_path.read_text()) == [
        {"stream_id": index, "name": name} for index, name in enumerate(stream_names)
    ]
