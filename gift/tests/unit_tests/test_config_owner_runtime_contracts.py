"""CPU evidence for initialization, conversion and checkpointed MoE balancing."""

from dataclasses import field

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from torchtitan.components.lora import LoRAConverter
from torchtitan.config import load_document
from torchtitan.config.build import prepare_document
from torchtitan.config.configurable import config
from torchtitan.config.schema.components.lora import LoRAConverterConfig
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.models.common.moe.load_balance import register_moe_load_balancing_hook


def _prepared(recipe, overrides=()):
    document, _ = load_document(f"configs/experiments/{recipe}.yaml", list(overrides))
    return prepare_document(document).model_config


class CPUExperts(nn.Module):
    """Replace the GPU grouped-GEMM boundary; keep routing and counters real."""

    def forward(self, x, *args):
        return x.tanh()

    def sync_combine(self):
        pass


class Hooks:
    def register_step_pre_hook(self, function):
        self.function = function


class LocalDims:
    full_dtensor = False

    def get_optional_mesh(self, name):
        return None


@pytest.mark.parametrize("requires_grad", [False, True])
def test_moe_balancing_does_not_assume_checkpoint_recomputation(requires_grad):
    cfg = _prepared("qwen3_moe_debug").layers[0].moe
    moe = cfg.build()
    moe.init_states(buffer_device=torch.device("cpu"))
    moe.experts = CPUExperts()
    moe.requires_grad_(False)
    holder = nn.Module()
    holder.block = checkpoint_wrapper(moe)
    hooks = Hooks()
    register_moe_load_balancing_hook(hooks, [holder], LocalDims())
    x = torch.zeros(1, 1, cfg.experts.dim, requires_grad=requires_grad)
    output = holder.block(x)
    nn.Linear(cfg.experts.dim, 1)(output).sum().backward()
    counts = moe.tokens_per_expert_E.clone()
    assert counts.max().item() == (2 if requires_grad else 1)
    expected = cfg.load_balance_coeff * torch.sign(counts.mean() - counts)
    expected -= expected.mean()
    assert expected.abs().sum() > 0
    hooks.function()
    assert torch.equal(expected, moe.expert_bias_E)
    assert torch.count_nonzero(moe.tokens_per_expert_E) == 0


@config
class DictionaryModelConfig(BaseModelConfig):
    projections: dict[str, list[LinearConfig]] = field(default_factory=dict)


def test_lora_replaces_nested_dictionary_projections():
    model = DictionaryModelConfig(
        projections={"group": [LinearConfig(in_features=4, out_features=4)]}
    ).materialize()
    LoRAConverter(LoRAConverterConfig(rank=2)).convert(model)
    cfg = model.projections["group"][0]
    assert type(cfg)._lora_applied
    model.validate_materialized()
    module = cfg.build()
    assert module(torch.zeros(1, 4)).shape == (1, 4)


@config
class NamedModelConfig(BaseModelConfig):
    projections: dict[str, LinearConfig] = field(default_factory=dict)


def test_lora_replaces_dictionary_projections():
    model = NamedModelConfig(
        projections={"input": LinearConfig(in_features=4, out_features=4)}
    ).materialize()
    LoRAConverter(LoRAConverterConfig(rank=2)).convert(model)
    cfg = model.projections["input"]
    assert type(cfg)._lora_applied
    assert cfg.build()(torch.zeros(1, 4)).shape == (1, 4)


def test_restated_flux_embedder_initializes_identically():
    original = _prepared("flux_debug").time_in_config
    restated = _prepared(
        "flux_debug",
        [
            (
                "model.config.time_in_config",
                "{in_dim: 256, hidden_dim: 1536, "
                "in_layer: {in_features: 256, out_features: 1536, bias: true}, "
                "out_layer: {in_features: 1536, out_features: 1536, bias: true}}",
            )
        ],
    ).time_in_config

    def weights(cfg):
        torch.manual_seed(42)
        module = cfg.build()
        module.init_states()
        return module.state_dict()

    expected, actual = weights(original), weights(restated)
    assert expected.keys() == actual.keys()
    assert all(torch.equal(value, actual[key]) for key, value in expected.items())


def test_flux_partial_initializer_is_complete_at_runtime():
    cfg = _prepared(
        "flux_debug",
        [("model.config.img_in", "{bias: true, param_init: {weight: {type: zeros}}}")],
    ).img_in
    module = cfg.build()
    module.init_states()
    assert torch.count_nonzero(module.weight) == 0
    assert torch.count_nonzero(module.bias) == 0
