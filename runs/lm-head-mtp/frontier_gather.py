"""Compare the stacked head replay with and without a private NCCL input alias."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap

import frontier_stack
import replay


def install_head_probe(head, alias):
    from torchtitan.distributed.gfsdp import _param_group

    torch = replay.torch
    dist = replay.dist
    (group,) = head._get_fsdp_state()._fsdp_param_groups
    (param,) = group._all_gather_dtype_to_fsdp_params[None]
    assert set(group._all_gather_dtype_to_fsdp_params) == {None}
    assert tuple(param._orig_size) == (128256, 4096)
    assert group.device.type == "cuda"
    assert group.orig_dtype == torch.float32
    assert group.param_dtype == torch.bfloat16
    assert group.reduce_dtype == torch.float32
    assert group._fp8_cache_layout is None and group._fp8_cache is None
    assert not group._bf16_cache_requested and not group._bf16_cache_enabled
    assert not group._direct_expert_dw_requested
    assert not hasattr(param._sharded_local_tensor, "fsdp_pre_all_gather")
    assert not hasattr(param._sharded_local_tensor, "fsdp_post_all_gather")
    comm = group._all_gather_process_group
    rank, size = comm.rank(), comm.size()
    assert dist.get_backend(comm) == "nccl"
    assert (rank, size) == (group.mesh_info.shard_mesh_rank, group.mesh_info.shard_mesh_size)

    def rank_local_input(owner, dtype):
        assert owner is group and dtype is None
        assert group.data_buffer_ctx.yccl_handles[dtype] is None
        output = group._all_gather_output[dtype]
        shard = group._padded_sharded_param_data[dtype]
        assert output.is_contiguous() and shard.is_contiguous()
        assert output.device == shard.device and output.numel() == size * shard.numel()
        return output.narrow(0, rank * shard.numel(), shard.numel())

    native_lazy_init = type(group).lazy_init
    native_gather = _param_group.all_gather
    native_nccl_api = dist.all_gather_into_tensor
    receipt = {
        "mode": "alias" if alias else "stack",
        "nccl_version": torch.cuda.nccl.version(),
        "group_rank": rank,
        "group_size": size,
        "gather_calls": 0,
        "native_all_gather_into_tensor_calls": 0,
        "master_to_bf16_casts": 0,
    }

    lazy_init = native_lazy_init
    if alias:
        # Keep the native setup. Replace just the allocation branch in this instance.
        body = textwrap.dedent(inspect.getsource(native_lazy_init))
        anchor = "        if yccl_handle is None:\n"
        assert body.count(anchor) == 1
        body = body.replace(
            anchor,
            "        if self is _benchmark_head_group:\n"
            "            self._all_gather_input[all_gather_dtype] = "
            "_benchmark_rank_local_input(self, all_gather_dtype)\n"
            "        elif yccl_handle is None:\n",
        )
        namespace = {
            **vars(_param_group),
            "_benchmark_head_group": group,
            "_benchmark_rank_local_input": rank_local_input,
        }
        exec(compile(body, __file__, "exec"), namespace)
        lazy_init = namespace["lazy_init"]

    def checked_lazy_init():
        before = torch.cuda.memory_allocated()
        lazy_init(group)
        receipt["lazy_init_allocation_delta_bytes"] = torch.cuda.memory_allocated() - before
        input_tensor = group._all_gather_input[None]
        output_tensor = group._all_gather_output[None]
        in_storage = input_tensor.untyped_storage().data_ptr()
        out_storage = output_tensor.untyped_storage().data_ptr()
        assert alias == (in_storage == out_storage)
        if alias:
            assert input_tensor.data_ptr() == (
                output_tensor.data_ptr() + rank * input_tensor.numel() * input_tensor.element_size()
            )
        receipt.update({
            "input_data_ptr": input_tensor.data_ptr(),
            "output_data_ptr": output_tensor.data_ptr(),
            "input_storage_ptr": in_storage,
            "output_storage_ptr": out_storage,
            "distinct_bf16_gather_storages": len({in_storage, out_storage}),
            "input_bytes": input_tensor.numel() * input_tensor.element_size(),
            "output_bytes": output_tensor.numel() * output_tensor.element_size(),
        })

    def checked_gather(*args, **kwargs):
        if args[0] is group:
            # GFSDP has established the owner and ordered the gather stream
            # behind overlapping release events before invoking all_gather.
            assert group.data_buffer_ctx.owner is group
            assert args[5] == comm and args[-1] is None
            assert args[3] is group._all_gather_input[None]
            assert args[4] is group._all_gather_output[None]
            if alias:
                group._all_gather_input_set_dtypes.discard(None)
            receipt["gather_calls"] += 1
            receipt["master_to_bf16_casts"] += None not in group._all_gather_input_set_dtypes
        return native_gather(*args, **kwargs)

    def counted_nccl(*args, **kwargs):
        if kwargs.get("output_tensor") is getattr(group, "_all_gather_output", {}).get(None):
            assert kwargs["group"] == comm
            assert kwargs["input_tensor"] is group._all_gather_input[None]
            receipt["native_all_gather_into_tensor_calls"] += 1
        return native_nccl_api(*args, **kwargs)

    group.lazy_init = checked_lazy_init
    _param_group.all_gather = checked_gather
    dist.all_gather_into_tensor = counted_nccl
    return receipt


def install_memory_probe():
    memory = []

    def capture_memory(stats):
        memory.append({
            "allocated_peak_bytes": stats["allocated_bytes.all.peak"],
            "reserved_peak_bytes": stats["reserved_bytes.all.peak"],
        })
        return stats

    # Preserve the native sample boundaries and memory-stat read count.
    body = textwrap.dedent(inspect.getsource(replay.main))
    anchor = "device_memory_usage(torch.cuda.memory_stats())"
    assert body.count(anchor) == 1
    body = body.replace(anchor, "device_memory_usage(_benchmark_capture_memory(torch.cuda.memory_stats()))")
    replay._benchmark_capture_memory = capture_memory
    methods = {}
    exec(compile(body, __file__, "exec"), vars(replay), methods)
    replay.main = methods["main"]
    return memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("stack", "alias"))
    parser.add_argument("chunks", type=int, choices=frontier_stack.PAIRS)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:True"
    make_head = replay.make_head
    receipt = None

    def make_probed_head(weight):
        nonlocal receipt
        head = make_head(weight)
        receipt = install_head_probe(head, args.arm == "alias")
        return head

    replay.make_head = make_probed_head
    sys.argv[1:] = ["stack", str(args.chunks), str(args.output)]
    memory = install_memory_probe()
    frontier_stack.main()
    result = json.loads(args.output.read_text())
    assert len(memory) == len(result["elapsed_ms"])
    receipt["memory"] = memory
    assert receipt["gather_calls"] == receipt["native_all_gather_into_tensor_calls"] > 0
    result["gather_probe"] = receipt
    result["gather_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
