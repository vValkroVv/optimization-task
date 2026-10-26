"""The small callback preset supports common edits without copying diagnostics."""

from pathlib import Path

import yaml

from torchtitan.config import load_document
from torchtitan.config.build import build_trainer_config
from torchtitan.config.schema.callbacks.builtin.monitoring import (
    MonitoringCallbackConfig,
)
from torchtitan.config.schema.callbacks.sources.modelling_source import (
    ModellingSourceConfig,
)
from torchtitan.config.schema.callbacks.sources.optimizer_source import (
    OptimizerSourceConfig,
)


def entry_path(tmp_path):
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "config_version": 1,
                "imports": [
                    str(root / "configs/experiments/llama3_debug.yaml"),
                    str(root / "configs/presets/callbacks/basic.yaml"),
                ],
            }
        )
    )
    return path


def test_optimizer_default_metric_edits_do_not_change_another_config():
    first, second = OptimizerSourceConfig(), OptimizerSourceConfig()
    first.metrics[2].pipeline.clear()
    assert second.metrics[2].pipeline[0].op == "abs_max"


def test_basic_preset_decodes_with_its_full_source_set(tmp_path):
    document, _ = load_document(str(entry_path(tmp_path)))
    config = build_trainer_config(document, with_model_spec=False)
    monitoring = config.callbacks["monitoring"]
    assert isinstance(monitoring, MonitoringCallbackConfig)
    assert list(monitoring.sources) == ["trainer", "data", "modelling", "optimizer"]


def test_basic_preset_allows_interval_source_and_metric_edits(tmp_path):
    path = entry_path(tmp_path)
    document, _ = load_document(
        str(path),
        [
            ("callbacks.monitoring.log_interval", "20"),
            ("callbacks.monitoring.sources.optimizer", "null"),
            (
                "callbacks.monitoring.sources.modelling.metrics",
                "[{name: norm, pipeline: [{op: norm}], reduce: {group: dp, op: mean}}]",
            ),
        ],
    )
    config = build_trainer_config(document, with_model_spec=False)
    assert list(config.callbacks) == ["system", "monitoring", "timers"]
    monitoring = config.callbacks["monitoring"]
    assert isinstance(monitoring, MonitoringCallbackConfig)
    assert monitoring.log_interval == 20
    assert list(monitoring.sources) == ["trainer", "data", "modelling"]
    modelling = monitoring.sources["modelling"]
    assert isinstance(modelling, ModellingSourceConfig)
    assert [metric.name for metric in modelling.metrics] == ["norm"]
