"""Backend factories validate typed model capabilities before training starts."""

import sys
from types import ModuleType
from typing import ClassVar
from unittest.mock import Mock

import pytest

from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.config.configurable import config
from torchtitan.distributed.backends import build_sharding_backend


@config
class GFSDPOnlyModelConfig(BaseModelConfig):
    supported_sharding_backends: ClassVar[tuple[str, ...]] = ("gfsdp2",)


def test_fsdp_runtime_accepts_the_default_model_capability():
    assert (
        build_sharding_backend(ParallelismConfig(), BaseModelConfig()).name == "fsdp2"
    )


def test_runtime_refuses_a_backend_absent_from_the_model_contract():
    with pytest.raises(ValueError, match="not supported by model type"):
        build_sharding_backend(ParallelismConfig(), GFSDPOnlyModelConfig())


def test_gfsdp_factory_forwards_init_option(monkeypatch):
    backend = Mock()
    constructor = Mock(return_value=backend)
    module = ModuleType("torchtitan.distributed.backends.gfsdp2")
    vars(module)["GFSDP2Backend"] = constructor
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = ParallelismConfig(fsdp_backend="gfsdp2", gfsdp_match_fsdp2_init=True)
    model_config = GFSDPOnlyModelConfig()
    assert build_sharding_backend(config, model_config) is backend
    constructor.assert_called_once_with(match_fsdp2_init=True)


def test_gfsdp_factory_refuses_cpu_offload_before_constructing_backend(monkeypatch):
    constructor = Mock()
    module = ModuleType("torchtitan.distributed.backends.gfsdp2")
    vars(module)["GFSDP2Backend"] = constructor
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="training.enable_cpu_offload"):
        build_sharding_backend(
            ParallelismConfig(fsdp_backend="gfsdp2"),
            GFSDPOnlyModelConfig(),
            cpu_offload=True,
        )
    constructor.assert_not_called()


def test_gfsdp_factory_refuses_unsupported_model_before_constructing_backend(
    monkeypatch,
):
    constructor = Mock()
    module = ModuleType("torchtitan.distributed.backends.gfsdp2")
    vars(module)["GFSDP2Backend"] = constructor
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="not supported by model type"):
        build_sharding_backend(
            ParallelismConfig(fsdp_backend="gfsdp2"), BaseModelConfig()
        )
    constructor.assert_not_called()
