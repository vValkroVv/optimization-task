"""Compact BF16 staging for the pinned EP4 direct expert gradients."""

import logging
import os
from collections.abc import Callable

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.profiler import record_function

TORCH_REVISION = "8b353065ce260336c5fdfa74d754e8bd9b91b062"
NCCL_REVISION = (2, 30, 7)
OWNER_SHAPES = ((32, 3072, 4096), (32, 4096, 1536))
OWNER_ELEMENTS = 301_989_888
FQNS = tuple(f"layers.{index}[efsdp]" for index in range(7, 0, -1))
logger = logging.getLogger("gfsdp")


@triton.jit
def _write_owner(
    local, peer, destination, n, accumulate: tl.constexpr, block: tl.constexpr
):
    # One chunk contains 150,994,944 elements in the pinned recipe.
    index = tl.program_id(0) * block + tl.arange(0, block)
    mask = index < n
    local_value = tl.load(local + index, mask, other=0).to(tl.float32)
    peer_value = tl.load(peer + index, mask, other=0).to(tl.float32)
    reduced = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [local_value, peer_value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    if accumulate:
        previous = tl.load(destination + index, mask, other=0)
        reduced = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [previous, reduced],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(destination + index, reduced, mask)


def require_recipe(dims, parallelism, training, compile_config, ac_config) -> None:
    if (
        (dims.world_size, dims.ep, dims.dp_shard, dims.dp_replicate) != (8, 4, 8, 1)
        or (dims.tp, dims.pp, dims.cp) != (1, 1, 1)
        or parallelism.fsdp_backend != "gfsdp2"
        or not (
            parallelism.gfsdp_fp8_expert_weights
            and parallelism.gfsdp_direct_expert_dw
            and parallelism.gfsdp_retain_tail_weights
        )
        or parallelism.enable_fsdp_symm_mem
        or (training.mixed_precision_param, training.mixed_precision_reduce)
        != ("bfloat16", "float32")
        or training.enable_cpu_offload
        or (training.local_batch_size, training.global_batch_size, training.seq_len)
        != (2, 512, 4096)
        or not (compile_config.enable and "model" in compile_config.components)
        or compile_config.backend != "inductor"
        or ac_config.mode != "selective"
    ):
        raise RuntimeError(
            "Compact BF16 owner transport requires the pinned native EP4 recipe"
        )


def require_runtime(device: torch.device, group: dist.ProcessGroup) -> None:
    if (
        device.type != "cuda"
        or torch.version.git_version != TORCH_REVISION
        or torch.cuda.nccl.version() != NCCL_REVISION
        or torch.cuda.get_device_capability(device) != (9, 0)
        or "H100" not in torch.cuda.get_device_name(device)
        or group.size() != 2
        or dist.get_backend(group) != "nccl"
        or os.environ.get("TORCH_NCCL_NAN_CHECK", "").strip().lower()
        not in ("", "0", "false")
    ):
        raise RuntimeError(
            "Compact BF16 owner transport requires pinned H100/NCCL eFSDP2"
        )


class ExpertOwnerTransport:
    """One disjoint receive buffer on the existing shared communication stream."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        stream: torch.Stream,
        is_last_backward: Callable[[], bool],
    ) -> None:
        self.group = group
        self.rank = group.rank()
        self.stream = stream
        self.is_last_backward = is_last_backward
        self.index = 0
        self.chunk = OWNER_ELEMENTS // 2
        self.splits = [0, self.chunk] if self.rank == 0 else [self.chunk, 0]
        self.slot_ptr = None
        with torch.cuda.stream(stream):
            self.receive = torch.empty(self.chunk, device=device, dtype=torch.bfloat16)

    def bind_slot(self, slot: torch.Tensor) -> None:
        if (
            slot.dtype != torch.bfloat16
            or slot.shape != (2 * OWNER_ELEMENTS,)
            or not slot.is_contiguous()
            or slot.device != self.receive.device
            or slot.untyped_storage().nbytes()
            != 2 * OWNER_ELEMENTS * slot.element_size()
            or slot.data_ptr() == self.receive.data_ptr()
        ):
            raise RuntimeError(
                "Compact BF16 owner transport received invalid shared storage"
            )
        if self.slot_ptr is not None and slot.data_ptr() != self.slot_ptr:
            raise RuntimeError(
                "Compact BF16 owner transport requires one shared gradient slot"
            )
        self.slot_ptr = slot.data_ptr()

    def finish_backward(self) -> None:
        expected = len(FQNS) if self.is_last_backward() else 0
        if self.index != expected:
            raise RuntimeError(
                f"Compact BF16 owner transport completed {self.index} of {expected} expert reductions"
            )
        self.index = 0

    def reduce(
        self,
        fqn: str,
        slot: torch.Tensor,
        output: torch.Tensor,
        group: dist.ProcessGroup,
        stream: torch.Stream,
        *,
        accumulate: bool,
    ) -> None:
        if (
            not self.is_last_backward()
            or self.index >= len(FQNS)
            or fqn != FQNS[self.index]
        ):
            raise RuntimeError(
                f"Unexpected compact BF16 owner transport order at {fqn}, index {self.index}"
            )
        if (
            group is not self.group
            or stream != self.stream
            or slot.data_ptr() != self.slot_ptr
            or slot.shape != (2 * OWNER_ELEMENTS,)
            or slot.dtype != torch.bfloat16
            or output.shape != (OWNER_ELEMENTS,)
            or output.dtype != torch.float32
            or output.device != slot.device
            or not output.is_contiguous()
            or output.untyped_storage().data_ptr() == slot.untyped_storage().data_ptr()
        ):
            raise RuntimeError(f"Invalid compact BF16 owner storage at {fqn}")

        own = slot.narrow(0, self.rank * OWNER_ELEMENTS, OWNER_ELEMENTS)
        other = slot.narrow(0, (1 - self.rank) * OWNER_ELEMENTS, OWNER_ELEMENTS)
        with record_function(f"GFSDP::bf16_slot_owner_reduce ({fqn})"):
            for offset in (0, self.chunk):
                dist.all_to_all_single(
                    self.receive,
                    other.narrow(0, offset, self.chunk),
                    output_split_sizes=self.splits,
                    input_split_sizes=self.splits,
                    group=group,
                    async_op=False,
                )
                _write_owner[(triton.cdiv(self.chunk, 1024),)](
                    own.narrow(0, offset, self.chunk),
                    self.receive,
                    output.narrow(0, offset, self.chunk),
                    self.chunk,
                    accumulate,
                    1024,
                )
        self.index += 1


def admit_expert_owner_transport(
    groups,
    contexts,
    handles,
    region_groups,
    comm,
    is_last_backward,
):
    """Validate all members before giving any group the compact allocation."""
    by_name = {
        group._module_fqn: group
        for group in groups
        if group._direct_expert_dw_requested
    }
    chosen = set(by_name.values())
    if set(by_name) != set(FQNS) or sum(
        group._direct_expert_dw_requested for group in groups
    ) != len(FQNS):
        raise RuntimeError(
            "Compact BF16 owner transport requires the seven pinned expert groups"
        )
    matching = [
        ctx for ctx, members in contexts.items() if not chosen.isdisjoint(members)
    ]
    if (
        len(matching) != 1
        or set(contexts[matching[0]]) != chosen
        or handles[matching[0]] is not None
        or not chosen.isdisjoint(region_groups)
    ):
        raise RuntimeError(
            "Compact BF16 owner transport requires one complete native gradient context"
        )
    first = by_name[FQNS[0]]
    pair = first._reduce_scatter_process_group
    base_rank = dist.get_rank() % 4
    if (
        dist.get_process_group_ranks(pair) != [base_rank, base_rank + 4]
        or comm.reduce_scatter_stream != comm.all_gather_stream
        or not comm.retain_tail_weights
    ):
        raise RuntimeError(
            "Compact BF16 owner transport requires the pinned mesh and stream"
        )
    require_runtime(first.device, pair)
    for group in chosen:
        layout = group._fp8_cache_layout
        if (
            group.comm_ctx is not comm
            or group.device != first.device
            or group._reduce_scatter_process_group is not pair
            or group._all_reduce_process_group is not None
            or (group.orig_dtype, group.param_dtype, group.reduce_dtype)
            != (torch.float32, torch.bfloat16, torch.float32)
            or not group.reduce_grads
            or group.gradient_divide_factor != 1.0
            or group.force_sum_reduction_for_comms
            or group.mp_policy.bit32_acc_for_bit16_reduce_scatter
            or layout is None
            or tuple(part.shape for part in layout.global_slices) != OWNER_SHAPES
            or layout.global_numel != 2 * OWNER_ELEMENTS
            or layout.local_numel != OWNER_ELEMENTS
            or group.padded_unsharded_param_numel != {None: 2 * OWNER_ELEMENTS}
            or set(group._all_gather_dtype_to_fsdp_params) != {None}
            or len(group.fsdp_params) != 2
            or not all(param.sharded_param.requires_grad for param in group.fsdp_params)
        ):
            raise RuntimeError(f"Invalid compact BF16 owner group {group._module_fqn}")
    transport = ExpertOwnerTransport(
        pair, first.device, comm.reduce_scatter_stream, is_last_backward
    )
    for group in chosen:
        group._owner_transport = transport
    return transport
