import pytest
import torch


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_import(unified_module):

    module = unified_module("flash_attn.losses.cross_entropy", "CrossEntropyLoss")
    assert module is not None

    module = unified_module("llmfoundry.models.layers.blocks", "LlamaDecoderLayer")
    assert module is not None

    module = unified_module(
        "composer.optim.decoupled_weight_decay", "DecoupledFusedAdamW"
    )
    assert module is not None

    module = unified_module("torch", "sqrt")
    assert module(torch.tensor(4.0)) == 2.0
