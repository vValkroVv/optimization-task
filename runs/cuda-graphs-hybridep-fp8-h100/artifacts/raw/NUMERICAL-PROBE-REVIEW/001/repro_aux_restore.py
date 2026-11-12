"""CPU regression for native aux scale versus checkpoint inventory equality."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[6]
STAGED = ROOT / "runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-PROBE-IMPLEMENTATION/001/staged/gift"
subject = runpy.run_path(str(STAGED / "tests/unit_tests/test_campaign_numerical_probe.py"))["probe_module"]
from torchtitan.components.aux_loss import AuxLossTracker
from torchtitan.models.common.regularization import declare_aux_loss, aux_loss_args


def make_model_and_tracker():
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(1))
    declare_aux_loss(model, "router_aux_loss", "router_z_loss")
    tracker = AuxLossTracker(
        [model],
        parallel_dims=SimpleNamespace(pp_enabled=False),
        device=torch.device("cpu"),
    )
    return model, tracker


saved, old_tracker = make_model_and_tracker()
old_tracker.weigh(valid_tokens=torch.tensor(8 * 4096), global_valid_tokens=torch.tensor(256 * 4096))
saved_inventory = subject._buffer_inventory((saved,))
checkpoint = saved.state_dict()
restored, new_tracker = make_model_and_tracker()
restored.load_state_dict(checkpoint)
restored_inventory = subject._buffer_inventory((restored,))
assert "aux_loss_scale" not in checkpoint
assert saved_inventory != restored_inventory, "The current after_load comparison must reject this valid restore"
assert float(saved.aux_loss_scale) == 1 / 32
assert float(restored.aux_loss_scale) == 1
# Native forward_backward_step weighs the current microbatch before consuming it.
new_tracker.weigh(valid_tokens=torch.tensor(8 * 4096), global_valid_tokens=torch.tensor(256 * 4096))
assert subject._buffer_inventory((restored,)) == saved_inventory
assert torch.equal(aux_loss_args(saved, "router_aux_loss")[0], aux_loss_args(restored, "router_aux_loss")[0])
print(json.dumps({"native_state_dict_keys": list(checkpoint), "saved_scale": float(saved.aux_loss_scale), "restored_initial_scale": 1, "current_after_load_equality": False, "equality_after_native_weigh": True}))
