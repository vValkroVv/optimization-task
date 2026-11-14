"""Probe the installed Torch wrapper-subclass conversion and swap behavior."""

import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

from torchtitan.distributed.gfsdp._tensor import RaggedShard, RaggedShardDTensor


def main() -> None:
    with tempfile.TemporaryDirectory(dir="runs/bf16-expert-cache") as temp_dir:
        store_path = Path(temp_dir) / "gloo-store"
        dist.init_process_group(
            "gloo", rank=0, world_size=1, init_method=f"file://{store_path}"
        )
        try:
            mesh = DeviceMesh("cpu", [0])
            local = torch.arange(4, dtype=torch.float32)
            spec = DTensorSpec(
                mesh,
                (RaggedShard(local_numel=4, global_offset=0, shard_numels=(4,)),),
                tensor_meta=TensorMeta(torch.Size((4,)), (1,), torch.float32),
            )
            param = nn.Parameter(
                RaggedShardDTensor(local, spec, requires_grad=True),
                requires_grad=True,
            )
            module = nn.Module()
            module.register_parameter("weight", param)
            identity = id(module.weight)

            module.to(dtype=torch.bfloat16)
            assert id(module.weight) == identity
            assert module.weight._local_tensor.dtype == torch.bfloat16

            restored = torch.arange(4, dtype=torch.float32) + 10
            restored_spec = DTensorSpec(
                mesh,
                (RaggedShard(local_numel=4, global_offset=0, shard_numels=(4,)),),
                tensor_meta=TensorMeta(torch.Size((4,)), (1,), torch.float32),
            )
            replacement = nn.Parameter(
                RaggedShardDTensor(restored, restored_spec, requires_grad=True),
                requires_grad=True,
            )
            torch.utils.swap_tensors(module.weight, replacement)
            assert id(module.weight) == identity
            assert module.weight._local_tensor.data_ptr() == restored.data_ptr()
            assert module.weight._local_tensor.dtype == torch.float32
            assert module.weight._spec.tensor_meta.dtype == torch.float32
        finally:
            dist.destroy_process_group()

    print(f"Torch {torch.__version__}: wrapper conversion/swap passed")


if __name__ == "__main__":
    main()
