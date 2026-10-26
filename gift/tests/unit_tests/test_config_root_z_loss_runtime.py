"""Root regularization reaches the built module's hook installation."""

import torch

from torchtitan.config.cli import load_document
from torchtitan.config.decode import decode
from torchtitan.config.schema.models.common.regularization import HiddenZLoss
from torchtitan.config.sections import ModelSection


def test_compact_root_z_loss_attaches_input_hook_to_runtime_module():
    document, _ = load_document("configs/experiments/llama3_debug.yaml")
    compact = decode(ModelSection, document.data["model"], path="model").config
    compact.hidden_z_loss = HiddenZLoss(coef=0.01, on="input")
    runtime = compact.materialize()
    # The attachment path is shared by runtime owner constructors; no model
    # forward or external attention kernels are needed to inspect its effect.
    module = runtime._attach_config(torch.nn.Identity())
    assert module._aux_loss_kinds == ("hidden_z_loss",)
    assert len(module._forward_pre_hooks) == 1
    assert not module._forward_hooks
