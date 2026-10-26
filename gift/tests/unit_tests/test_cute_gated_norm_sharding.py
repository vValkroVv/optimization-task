"""Exercise DTensor dispatch for CuTe ops without launching a CUDA kernel."""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from torchtitan.models.common.norms import GatedRMSNorm
from torchtitan.models.common.norms.cute_gated_norm import op


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_meta_dtensor_sharding_and_parameter_reductions(tmp_path, monkeypatch):
    if dist.is_initialized():
        pytest.skip("requires an isolated process group")
    validate = op._validate_inputs

    def validate_metadata(x, weight, up, down, eps):
        # A CPU mesh synthesizes CPU FakeTensors for metadata propagation.
        # Move only those descriptions to meta for the CUDA-only op's guard.
        if x.device.type == "cpu":
            x, weight, up, down = (t.to("meta") for t in (x, weight, up, down))
        return validate(x, weight, up, down, eps)

    monkeypatch.setattr(op, "_validate_inputs", validate_metadata)
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1
    )
    try:
        mesh = DeviceMesh(
            "cpu", torch.tensor([[0]]), mesh_dim_names=("batch", "sequence")
        )
        with torch.device("meta"):
            norm = GatedRMSNorm.Config(
                normalized_shape=128, norm_backend="cute", zero_centered=True
            ).build()
            for name, param in list(norm.named_parameters()):
                setattr(
                    norm,
                    name,
                    torch.nn.Parameter(
                        DTensor.from_local(
                            param,
                            mesh,
                            (Replicate(), Replicate()),
                        )
                    ),
                )
            for placements in ((Replicate(), Replicate()), (Shard(0), Shard(1))):
                x = DTensor.from_local(
                    torch.empty(2, 65, 128, dtype=torch.bfloat16),
                    mesh,
                    placements,
                ).requires_grad_()
                output = norm(x)
                grads = torch.autograd.grad(output.sum(), (x, *norm.parameters()))
                assert output.placements == placements
                assert grads[0].placements == placements
                expected = tuple(
                    Partial() if p.is_shard() else Replicate() for p in placements
                )
                assert all(grad.placements == expected for grad in grads[1:])
    finally:
        dist.destroy_process_group()
