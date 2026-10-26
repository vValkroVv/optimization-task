"""Pure configuration capability checks and CPU CP wrapper boundaries."""

from unittest.mock import Mock

import pytest
import torch.nn as nn

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.softmax.kernels.base import (
    InnerAttentionConfig,
)
from torchtitan.distributed.context_parallel import (
    _inner_attention_classes,
    apply_cp_to_forward,
    CP_SUPPORTED_ATTENTION,
)
from torchtitan.protocols.module import Module


def test_discovery_skips_unbound_bases_and_finds_concrete_descendants():
    @config
    class AbstractKernelConfig(InnerAttentionConfig):
        pass

    class NewKernel(Module):
        @config
        class Config(AbstractKernelConfig):
            pass

    derived = _inner_attention_classes()
    assert NewKernel in derived
    assert Module not in derived
    assert set(CP_SUPPORTED_ATTENTION) <= set(derived)

    # An explicitly bound schema variant is valid even when the owner's Config
    # names the original schema. Discovery must not silently discard it.
    @config
    class VariantConfig(AbstractKernelConfig):
        pass

    VariantConfig._owner = NewKernel
    assert NewKernel in _inner_attention_classes()


def test_discovery_does_not_hide_broken_concrete_bindings(monkeypatch):
    @config
    class BrokenKernelConfig(InnerAttentionConfig):
        pass

    monkeypatch.setattr(
        BrokenKernelConfig, "_owner_spec", "missing_cp_kernel_for_test:Kernel"
    )
    with pytest.raises(ModuleNotFoundError, match="missing_cp_kernel_for_test"):
        _inner_attention_classes()


def test_sdpa_wraps_nested_kernels_without_distributed_execution(monkeypatch):
    import torchtitan.distributed.context_parallel as cp

    kernel = CP_SUPPORTED_ATTENTION[1].Config().build()
    original = Mock(return_value="output")
    kernel.forward = original
    model = nn.Sequential(nn.Sequential(kernel))
    dispatcher = Mock()
    monkeypatch.setattr(cp, "_enable_context_parallel_dispatcher", dispatcher)

    class LocalDTensor:
        def __init__(self, value):
            self.value = value

        @classmethod
        def from_local(cls, value, mesh, placement, run_check):
            assert mesh == "mesh" and not run_check
            assert len(placement) == 1 and placement[0].dim == 1
            return cls(value)

    monkeypatch.setattr(cp, "DTensor", LocalDTensor)
    apply_cp_to_forward(model, "mesh")
    assert kernel.forward("q", "k", "v") == "output"
    dispatcher.assert_called_once_with()
    assert [arg.value for arg in original.call_args.args] == ["q", "k", "v"]


def test_all_kernel_capabilities_match_runtime_support():
    # Import the package so discovery includes every built-in registered kernel.
    from torchtitan.models.common.attention.softmax import kernels  # noqa: F401

    for owner in _inner_attention_classes():
        schema = owner.Config
        assert schema.supports_context_parallel == issubclass(
            owner, CP_SUPPORTED_ATTENTION
        ), schema.__name__


def test_new_kernel_cannot_be_silently_skipped_in_nested_stack():
    class UnsupportedKernel(Module):
        @config
        class Config(InnerAttentionConfig):
            pass

    supported = CP_SUPPORTED_ATTENTION[1].Config().build()
    model = nn.Sequential(supported, nn.Sequential(UnsupportedKernel()))
    with pytest.raises(NotImplementedError, match="UnsupportedKernel"):
        apply_cp_to_forward(model, None)


def _model_document(experiment):
    from torchtitan.config.cli import load_document

    document, _ = load_document(f"configs/experiments/{experiment}.yaml", [])
    return document.data["model"]


def _materialize(document):
    from torchtitan.config.build import materialize_model_config
    from torchtitan.config.decode import decode
    from torchtitan.config.sections import ModelSection

    return materialize_model_config(decode(ModelSection, document).config)


def test_each_builtin_kernel_cp_validation_matches_runtime_whitelist():
    from torchtitan.config.configs import ParallelismConfig
    from torchtitan.config.configurable import family_of, type_name_of

    checked = set()
    for owner in _inner_attention_classes():
        name = type_name_of(owner.Config)
        if name is None:
            continue  # Unregistered test-only kernels are checked separately.
        document = _model_document("llama3_debug")
        attention = document["config"]["blocks"]["dense"]["attention"]
        attention["inner_attention"] = {"type": name}
        tree = _materialize(document)
        for degree in (1, 2):
            issues = tree.validate_parallelism(
                ParallelismConfig(context_parallel_degree=degree)
            )
            rejected = any(
                i.path == "parallelism.context_parallel_degree" for i in issues
            )
            assert rejected == (
                degree > 1 and not issubclass(owner, CP_SUPPORTED_ATTENTION)
            ), (name, degree, issues)
        checked.add(name)
    assert checked == set(family_of(InnerAttentionConfig).types)


@pytest.mark.parametrize("attention_type", ["gdn", "kda"])
@pytest.mark.parametrize("position", ["late_main", "late_mtp"])
def test_recurrent_attention_is_rejected_in_every_placed_position(
    attention_type, position
):
    from torchtitan.config.configs import ParallelismConfig

    document = _model_document("gift_3_5_debug")
    model = document["config"]
    model["blocks"]["gdn_dense"]["attention"]["type"] = attention_type
    model["layers"] = [{"block": "mtp"}]
    model.pop("mtp", None)
    if position == "late_main":
        model["layers"].append({"block": "gdn_dense"})
    else:
        model["mtp"] = {"layers": [{"block": "mtp"}, {"block": "gdn_dense"}]}
    tree = _materialize(document)
    assert not tree.validate_parallelism(ParallelismConfig(context_parallel_degree=1))
    issues = tree.validate_parallelism(ParallelismConfig(context_parallel_degree=2))
    assert any(
        i.path == "parallelism.context_parallel_degree"
        and f"'{attention_type}'" in i.message
        for i in issues
    ), issues
