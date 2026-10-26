"""FT startup logs internal and converted architectures before model allocation."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.unit_tests.config_helpers import load_experiment_config
from torchtitan.config.schema.experiments.ft.trainer import FaultTolerantTrainerConfig


@pytest.mark.parametrize(
    "converters",
    [
        "[]",
        "[{type: lora, target_modules: [wo]}, {type: freeze, selectors: [layers.0]}]",
    ],
)
def test_ft_constructor_logs_runtime_architecture_before_build(monkeypatch, converters):
    import torchtitan.experiments.ft.trainer as ft

    original = load_experiment_config(
        "llama3_debug.yaml", "--model.config.converters", converters
    )
    config = FaultTolerantTrainerConfig(
        dataloader=original.dataloader, loss=original.loss, tokenizer=None
    )
    config.model_spec = original.model_spec
    assert config.model_spec is not None
    tree = config.model_spec.model
    # Runtime inspection must not accidentally make internal schemas public YAML.
    with pytest.raises(ValueError, match="runtime model trees are internal"):
        tree.to_dict()
    expected = config.model_spec.to_dict()["model"]
    logger = Mock()
    monkeypatch.setattr(ft, "logger", logger)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(ft.utils, "device_type", "cpu")
    monkeypatch.setattr(ft.utils, "device_module", SimpleNamespace(set_device=Mock()))
    monkeypatch.setattr(ft.utils, "GarbageCollection", lambda **kwargs: object())
    monkeypatch.setattr(ft.utils, "set_default_dtype", lambda dtype: nullcontext())
    monkeypatch.setattr(ft.dist_utils, "set_determinism", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        type(config.dataloader), "build", lambda self, **kwargs: object()
    )
    monkeypatch.setattr(
        tree.owner(), "prepare_config", classmethod(lambda cls, *args, **kwargs: None)
    )

    def init_distributed(self):
        self.ft_manager = SimpleNamespace(
            get_dp_info=lambda degree, rank: (degree, rank)
        )
        return SimpleNamespace(dp_enabled=False)

    monkeypatch.setattr(ft.FaultTolerantTrainer, "init_distributed", init_distributed)

    class ReachedModelBuildError(Exception):
        pass

    def stop_at_build(self, **kwargs):
        assert self is tree
        raise ReachedModelBuildError

    monkeypatch.setattr(type(tree), "build", stop_at_build)
    with pytest.raises(ReachedModelBuildError):
        ft.FaultTolerantTrainer(config)
    messages = [call.args[0] for call in logger.info.call_args_list]
    architecture = next(
        message for message in messages if message.startswith("Building ")
    )
    assert json.loads(architecture.split(" with ", 1)[1]) == expected
