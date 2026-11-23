"""Observe native eight-rank GFSDP head gathers and optionally alias the send slot."""

import inspect
import textwrap

import torch
import torch.distributed as dist


def install_gather_probe(head, alias):
    """Install before the first head use in a single-arm replay process."""
    from torchtitan.distributed.gfsdp import _param_group

    (head_group,) = head._get_fsdp_state()._fsdp_param_groups
    assert set(head_group._all_gather_dtype_to_fsdp_params) == {None}
    (param,) = head_group._all_gather_dtype_to_fsdp_params[None]
    assert tuple(param._orig_size) == (128256, 4096)
    assert head_group.device.type == "cuda"
    assert head_group.orig_dtype == torch.float32
    assert head_group.param_dtype == torch.bfloat16
    assert head_group.reduce_dtype == torch.float32
    assert not getattr(head_group, "_bf16_cache_requested", False)
    assert not getattr(head_group, "_bf16_cache_enabled", False)
    assert getattr(head_group, "_fp8_cache_layout", None) is None
    assert getattr(head_group, "_fp8_cache", None) is None
    assert not getattr(head_group, "_direct_expert_dw_requested", False)
    assert not hasattr(param._sharded_local_tensor, "fsdp_pre_all_gather")
    assert not hasattr(param._sharded_local_tensor, "fsdp_post_all_gather")

    comm = head_group._all_gather_process_group
    rank, size = comm.rank(), comm.size()
    ranks = dist.get_process_group_ranks(comm)
    assert dist.get_backend(comm) == "nccl"
    assert dist.get_world_size() == size == 8
    assert ranks == list(range(size)) and ranks[rank] == dist.get_rank()
    assert (rank, size) == (
        head_group.mesh_info.shard_mesh_rank,
        head_group.mesh_info.shard_mesh_size,
    )

    def rank_local_input(dtype):
        assert dtype is None
        output = head_group._all_gather_output[dtype]
        shard = head_group._padded_sharded_param_data[dtype]
        return output.narrow(0, rank * shard.numel(), shard.numel())

    receipt = {
        "mode": "alias" if alias else "stack",
        "nccl_version": torch.cuda.nccl.version(),
        "group_rank": rank,
        "group_size": size,
        "group_ranks": ranks,
        "gather_calls": 0,
        "native_all_gather_into_tensor_calls": 0,
        "master_to_bf16_casts": 0,
    }
    native_gather = _param_group.all_gather
    native_nccl = dist.all_gather_into_tensor
    lazy_init = type(head_group).lazy_init
    if alias:
        # Replace the allocation before native setup binds any parameter views.
        body = textwrap.dedent(inspect.getsource(lazy_init))
        anchor = "        if yccl_handle is None:\n"
        assert body.count(anchor) == 1, "Unsupported GFSDP lazy_init allocation"
        body = body.replace(
            anchor,
            "        if yccl_handle is None and self is _replay_head_group:\n"
            "            self._all_gather_input[all_gather_dtype] = "
            "_replay_rank_local_input(all_gather_dtype)\n"
            "        elif yccl_handle is None:\n",
        )
        namespace = {
            **vars(_param_group),
            "_replay_head_group": head_group,
            "_replay_rank_local_input": rank_local_input,
        }
        exec(compile(body, __file__, "exec"), namespace)
        lazy_init = namespace["lazy_init"]

    def checked_lazy_init():
        before = torch.cuda.memory_allocated()
        lazy_init(head_group)
        receipt["lazy_init_allocation_delta_bytes"] = (
            torch.cuda.memory_allocated() - before
        )
        input_tensor = head_group._all_gather_input[None]
        output_tensor = head_group._all_gather_output[None]
        master = head_group._padded_sharded_param_data[None]
        assert head_group.data_buffer_ctx.yccl_handles.get(None) is None
        assert input_tensor.is_contiguous() and output_tensor.is_contiguous()
        assert master.is_contiguous() and master.dtype == torch.float32
        assert input_tensor.dtype == output_tensor.dtype == torch.bfloat16
        assert input_tensor.device == output_tensor.device == master.device
        assert input_tensor.numel() == master.numel()
        assert output_tensor.numel() == size * master.numel()
        input_ptr, output_ptr = input_tensor.data_ptr(), output_tensor.data_ptr()
        input_storage = input_tensor.untyped_storage().data_ptr()
        output_storage = output_tensor.untyped_storage().data_ptr()
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        shared_storage = input_storage == output_storage
        assert shared_storage == alias
        if alias:
            assert input_ptr == output_ptr + rank * input_bytes
        receipt.update(
            input_data_ptr=input_ptr,
            output_data_ptr=output_ptr,
            input_storage_ptr=input_storage,
            output_storage_ptr=output_storage,
            distinct_bf16_gather_storages=len({input_storage, output_storage}),
            input_bytes=input_bytes,
            output_bytes=size * input_bytes,
            input_offset_bytes=input_ptr - output_ptr if shared_storage else None,
            rank_slot_offset_bytes=rank * input_bytes,
        )

    def checked_gather(owner, params, master, send, recv, pg, *args):
        if owner is not head_group:
            return native_gather(owner, params, master, send, recv, pg, *args)
        assert head_group.data_buffer_ctx.owner is head_group
        assert master is head_group._padded_sharded_param_data[None]
        assert (
            send is head_group._all_gather_input[None]
            and recv is head_group._all_gather_output[None]
        )
        assert pg is comm and args[-2:] == (None, None)
        if alias:
            # GFSDP has acquired the buffer and ordered this stream after release.
            head_group._all_gather_input_set_dtypes.discard(None)
        needs_cast = None not in head_group._all_gather_input_set_dtypes
        result = native_gather(owner, params, master, send, recv, pg, *args)
        receipt["master_to_bf16_casts"] += needs_cast
        assert (
            receipt["native_all_gather_into_tensor_calls"]
            == receipt["gather_calls"] + 1
        )
        receipt["gather_calls"] += 1
        return result

    def counted_nccl(output_tensor, input_tensor, group=None, async_op=False):
        outputs = getattr(head_group, "_all_gather_output", None)
        is_head_output = outputs is not None and output_tensor is outputs.get(None)
        if is_head_output:
            assert group is comm and input_tensor is head_group._all_gather_input[None]
            # Native all_gather marks the input valid after any required copy.
            assert None in head_group._all_gather_input_set_dtypes
        work = native_nccl(output_tensor, input_tensor, group=group, async_op=async_op)
        if is_head_output:
            receipt["native_all_gather_into_tensor_calls"] += 1
        return work

    head_group.lazy_init = checked_lazy_init
    _param_group.all_gather = checked_gather
    dist.all_gather_into_tensor = counted_nccl
    return receipt
