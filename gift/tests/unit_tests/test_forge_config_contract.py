"""Forge schema construction and serialization stay independent of runtime."""

from typing import Any, cast

from torchtitan.config.configurable import Configurable
from torchtitan.config.schema.experiments.forge.engine import ForgeEngineConfig


def test_default_forge_config_has_no_unbuilt_model_spec():
    config = ForgeEngineConfig()
    assert config.model_spec is None


def test_forge_uses_common_serialization_and_omits_runtime_objects():
    config = ForgeEngineConfig()
    config.model_spec = cast(Any, object())
    assert ForgeEngineConfig.to_dict is Configurable.Config.to_dict
    data = config.to_dict()
    assert "model_spec" not in data
    assert data["optimizer"]["lr"] == config.optimizer.lr


def test_forge_default_construction_and_dump_are_torch_free():
    from tests.unit_tests.torch_free import run_blocked

    result = run_blocked(
        "from torchtitan.config.schema.experiments.forge.engine import ForgeEngineConfig\n"
        "config = ForgeEngineConfig()\n"
        "assert config.model_spec is None\n"
        "assert 'model_spec' not in config.to_dict()\n"
    )
    assert result.returncode == 0
