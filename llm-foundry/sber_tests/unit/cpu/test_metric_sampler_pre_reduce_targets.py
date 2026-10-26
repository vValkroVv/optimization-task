import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("composer", reason="Requires the CPU training runtime")

from torch import nn  # noqa: E402

from llmfoundry.callbacks.metric_sampler import MetricSampler  # noqa: E402
from llmfoundry.models.layers.moe import AbstractGMMMoeBlock  # noqa: E402


class _ExpertBlock(AbstractGMMMoeBlock):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _SuperTensorModule(nn.Module):
    def __init__(self, root_module, super_tensor):
        super().__init__()
        self.root_module = root_module
        self.super_tensor = super_tensor


class _SuperTensor:
    def __init__(self, name, process_group):
        self.name = name
        self._fsdp_process_group = process_group
        self.is_trainable = True


class _GigaFSDP:
    def __init__(self, super_tensors, layernorm_super_tensor):
        self._zero_stage = 1
        self._comm_stream = object()
        self._super_tensors = super_tensors
        self._layernorm_supertensor = layernorm_super_tensor
        self._module = nn.Module()
        self._module.add_module(
            "moe",
            _SuperTensorModule(_ExpertBlock(), super_tensors[0]),
        )
        self._module.add_module(
            "dense",
            _SuperTensorModule(nn.Linear(2, 2), super_tensors[1]),
        )


def _targets(monkeypatch, expert_group_is_default):
    default_group = object()
    expert_group = default_group if expert_group_is_default else object()
    expert = _SuperTensor("model.layers.1.block_sparse_moe.flatten", expert_group)
    dense = _SuperTensor("model.layers.1.self_attn.flatten", default_group)
    layernorm = _SuperTensor("model.norm", default_group)
    wrapper = _GigaFSDP([expert, dense, layernorm], layernorm)
    monkeypatch.setitem(
        sys.modules,
        "giga_fsdp",
        SimpleNamespace(GigaFSDP=_GigaFSDP, SuperTensorModule=_SuperTensorModule),
    )
    sampler = MetricSampler.__new__(MetricSampler)
    return sampler._gigafsdp_pre_reduce_targets(
        SimpleNamespace(
            gigafsdp_enabled=True, model=SimpleNamespace(modules=lambda: [wrapper])
        ),
        num_layers=4,
    )


@pytest.mark.parametrize(
    "expert_group_is_default", [True, False], ids=["ep-1-aliased", "ep-many"]
)
def test_moe_super_tensor_is_expert_independent_of_process_group(
    monkeypatch, expert_group_is_default
):
    targets = _targets(monkeypatch, expert_group_is_default)

    assert [(target[1], target[2]) for target in targets] == [
        ("grad/pre_reduce_sq_expert", 1),
        ("grad/pre_reduce_sq_dense", 1),
        ("grad/pre_reduce_sq_dense", -1),
    ]


def test_unowned_trainable_super_tensor_is_not_silently_classified_as_dense(
    monkeypatch,
):
    default_group = object()
    expert = _SuperTensor("model.layers.1.block_sparse_moe.flatten", default_group)
    dense = _SuperTensor("model.layers.1.self_attn.flatten", default_group)
    unknown = _SuperTensor("unknown.trainable.flatten", default_group)
    wrapper = _GigaFSDP([expert, dense, unknown], layernorm_super_tensor=None)
    monkeypatch.setitem(
        sys.modules,
        "giga_fsdp",
        SimpleNamespace(GigaFSDP=_GigaFSDP, SuperTensorModule=_SuperTensorModule),
    )
    sampler = MetricSampler.__new__(MetricSampler)

    with pytest.raises(ValueError, match="could not determine the owning module"):
        sampler._gigafsdp_pre_reduce_targets(
            SimpleNamespace(
                gigafsdp_enabled=True, model=SimpleNamespace(modules=lambda: [wrapper])
            ),
            num_layers=4,
        )
