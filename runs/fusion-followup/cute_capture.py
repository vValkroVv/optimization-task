"""Capture rank-zero native CuTe gated norm inputs and corresponding tangents."""

import importlib.util
import os
from pathlib import Path

import torch


def install(output: Path) -> None:
    if int(os.environ["RANK"]) != 0:
        return

    from torchtitan.models.common.norms.cute_gated_norm import op
    from torchtitan.trainer import Trainer

    helper = Path(__file__).resolve().parents[1] / "backward-components" / "capture.py"
    spec = importlib.util.spec_from_file_location("_cute_capture_helpers", helper)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    snapshot = helpers._snapshot
    layout = helpers._layout
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "cute-rank0.pt"
    if fixture.exists():
        raise FileExistsError(fixture)

    active = False
    forward_records = {}
    first_tangent = None
    shape = (2, 4096, 4096)
    original_backward = op._backward._backend_fns["cuda"]

    def capture_forward(x, weight, up, down, eps, gating_weight, residual=None):
        if residual is not None:
            op._validate_residual(x, residual)
        result = op._run_forward(x, weight, up, down, eps, gating_weight, residual)
        role = "plain" if residual is None else "residual"
        if active and x.shape == shape and role not in forward_records:
            tensors = dict(x=x, weight=weight, up=up, down=down, y=result[0])
            if residual is not None:
                tensors["residual"] = residual
            forward_records[role] = {
                "tensors": {name: snapshot(tensor) for name, tensor in tensors.items()},
                "layouts": {name: layout(tensor) for name, tensor in tensors.items()},
                "eps": eps,
                "gating_weight": gating_weight,
                "input_address": x.data_ptr(),
            }
        return result

    op._forward.register_kernel("cuda", capture_forward)
    op._forward_residual.register_kernel("cuda", capture_forward)

    @op._backward.register_kernel("cuda")
    def capture_backward(dy, x, *args):
        nonlocal first_tangent
        if active and x.shape == shape:
            if first_tangent is None:
                first_tangent = {"dy": snapshot(dy), "layout": layout(dy)}
            for record in forward_records.values():
                if "dy" in record["tensors"] or record["input_address"] != x.data_ptr():
                    continue
                captured = snapshot(x)
                if torch.equal(captured.view(torch.int16), record["tensors"]["x"].view(torch.int16)):
                    record["tensors"]["dy"] = snapshot(dy)
                    record["layouts"]["dy"] = layout(dy)
        return original_backward(dy, x, *args)

    original_step = Trainer.forward_backward_step

    def capture_step(trainer, *args, **kwargs):
        nonlocal active
        Trainer.forward_backward_step = original_step
        active = True
        try:
            result = original_step(trainer, *args, **kwargs)
        finally:
            active = False
        if set(forward_records) != {"plain", "residual"} or first_tangent is None:
            raise RuntimeError("Native microbatch did not exercise both CuTe norm variants")
        for record in forward_records.values():
            del record["input_address"]
        payload = {
            "update": trainer.step,
            "rank": 0,
            "microbatch": 1,
            "torch_revision": torch.version.git_version,
            "records": forward_records,
            "first_tangent": first_tangent,
        }
        temporary = fixture.with_suffix(".tmp")
        with temporary.open("xb") as writer:
            torch.save(payload, writer)
        temporary.rename(fixture)
        print(f"cute_native_fixture={fixture}", flush=True)
        return result

    Trainer.forward_backward_step = capture_step
