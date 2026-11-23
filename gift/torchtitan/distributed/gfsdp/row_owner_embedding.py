"""Eager input embeddings routed to their dense-GFSDP row owners.

The parameter remains a flat-backed FP32 RaggedShardDTensor, including for
optimizer, clipping and DCP. Only selected BF16 rows cross the owner group.
The routing pattern follows NeMo Automodel's Engram owner embedding (commit
3ea1de8d0aa902c3329b585d194c2e5274a85f3f); gradient coalescing deliberately
uses native embedding backward to retain its source-local BF16 boundary.
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import cache
from typing import Any
import logging

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.function import once_differentiable
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

from torchtitan.protocols.module import Module

from ._common import FSDPMeshInfo
from ._tensor import RaggedShard, RaggedShardDTensor, install

logger = logging.getLogger("gfsdp")


@cache
def _compact_min_ordered_rows(device: torch.device, width: int) -> int | None:
    # The dispatch bound mirrors EmbeddingBackwardKernel.cu in this CUDA build.
    if (
        device.type != "cuda"
        or torch.__version__ != "2.15.0.dev20260822+cu130"
        or torch.version.git_version != "8b353065ce260336c5fdfa74d754e8bd9b91b062"
        or torch.version.cuda != "13.0"
    ):
        return None
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    stride_warped = (width + 31) // 32 * 32
    block_size = min(stride_warped, 1024)
    # Keep ceil(min(num_ids, num_weights) * stride_warped / block_size)
    # at least 4 * num_sms to exclude unordered atomic partial summation.
    return ((4 * num_sms - 1) * block_size) // stride_warped + 1


class RowOwnerEmbedding(Module):
    """A complete-row, untied input table owned by a one-axis GFSDP mesh.

    Install before GFSDP wrapping and pass the final ``weight`` as an ignored
    parameter to every enclosing wrapper. Call ``attach_comm_context`` after
    wrapping the root. The stem must remain eager and outside SAC replay.
    ``stage_observer`` is an optional diagnostic context factory; by default
    there is no stage instrumentation or synchronization.

    Scratch borrows existing GFSDP gradient bytes. Backward returns an owned
    FP32 local gradient for ordinary autograd accumulation. Optimizer updates
    stay external.
    """

    stage_observer: Callable[[str], AbstractContextManager] | None = None

    @classmethod
    @torch.no_grad()
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        dense_mesh: DeviceMesh,
        *,
        shard_alignment: int = 8,
    ) -> "RowOwnerEmbedding":
        if dense_mesh.ndim != 1:
            raise ValueError(
                "row-owner embeddings require one GFSDP shard axis (no HSDP)"
            )
        if any(
            (
                embedding.padding_idx is not None,
                embedding.max_norm is not None,
                embedding.scale_grad_by_freq,
                embedding.sparse,
            )
        ):
            raise ValueError(
                "row-owner embeddings require dense, unscaled lookup without padding/max_norm"
            )
        weight = embedding.weight
        if type(weight) is not nn.Parameter or weight.dtype != torch.float32:
            raise ValueError(
                "row-owner embeddings require an ordinary FP32 master before wrapping"
            )
        if not weight.is_contiguous() or weight.ndim != 2:
            raise ValueError(
                "row-owner embeddings require a contiguous [vocabulary, hidden] table"
            )
        info = FSDPMeshInfo(dense_mesh, shard_mesh_dim=0)
        vocab, width = weight.shape
        if vocab <= 0 or width <= 0:
            raise ValueError(
                "row-owner embeddings require nonempty vocabulary and hidden dimensions"
            )
        local_numel = weight.numel() // info.shard_mesh_size
        if (
            vocab % info.shard_mesh_size
            or shard_alignment <= 0
            or weight.numel() % (info.shard_mesh_size * shard_alignment)
            or local_numel % width
        ):
            raise ValueError(
                "row-owner embeddings require complete aligned rows with no table padding"
            )
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result._param_init = (
            embedding._param_init if isinstance(embedding, Module) else None
        )
        result.num_embeddings = vocab
        result.embedding_dim = width
        result.mesh_info = info
        result._route_process_group = info.shard_process_group
        result.rows_per_owner = vocab // info.shard_mesh_size
        result.row_start = result.rows_per_owner * info.shard_mesh_rank
        result._comm_ctx = None
        result._sync_groups = ()
        result._using_ema_weights = False
        result._weight_fqn = "weight"
        result._compact_source_logged = False
        result.register_buffer("_ema_weight", None, persistent=False)
        placement = RaggedShard(
            local_numel,
            local_numel * info.shard_mesh_rank,
            (local_numel,) * info.shard_mesh_size,
        )
        spec = DTensorSpec(
            dense_mesh,
            info.dp_placements(placement),
            tensor_meta=TensorMeta(weight.shape, weight.stride(), weight.dtype),
        )
        local = (
            weight.detach()
            .reshape(-1)
            .narrow(0, placement.global_offset, local_numel)
            .clone()
        )
        install()
        result.weight = nn.Parameter(
            RaggedShardDTensor(local, spec, requires_grad=False),
            requires_grad=weight.requires_grad,
        )
        result.train(embedding.training)
        return result

    def enable_dedicated_nccl_process_group(self, max_ctas: int) -> None:
        """Create a separate, rank-identical communicator for the row protocol."""
        dense_group = self.mesh_info.shard_process_group
        if self._route_process_group is not dense_group:
            raise RuntimeError("row-owner already has a dedicated process group")
        ranks = dist.get_process_group_ranks(dense_group)
        if ranks != list(range(dist.get_world_size())):
            raise ValueError("dedicated row-owner routing requires the ordered world mesh")
        if self.mesh_info.mesh.device_type != "cuda" or dist.group.WORLD.bound_device_id is not None:
            raise ValueError("dedicated row-owner routing requires a lazily initialized CUDA default group")

        options = dist.ProcessGroupNCCL.Options()
        options.config.max_ctas = max_ctas
        route_group = dist.new_group(
            ranks=ranks,
            backend="nccl",
            pg_options=options,
            sort_ranks=False,
            group_desc="row_owner_embedding",
        )
        if route_group is dense_group or route_group.rank() != self.mesh_info.shard_mesh_rank:
            raise RuntimeError("dedicated row-owner routing changed the shard rank mapping")
        self._route_process_group = route_group
        logger.info("row_owner_embedding dedicated NCCL communicator active max_ctas=%d", max_ctas)

    def reset_parameters(self) -> None:
        # Configurable embedding initialization normally uses _param_init.
        nn.init.normal_(self.weight)

    def _apply(self, fn: Callable, recurse: bool = True) -> "RowOwnerEmbedding":
        del recurse
        # GFSDP ignored-parameter sets retain this exact Parameter identity
        # across the trainer's meta -> to_empty materialization.
        with torch.no_grad():
            converted = fn(self.weight)
            if converted.dtype != torch.float32:
                raise ValueError("row-owner master parameters must remain FP32")
            replacement = nn.Parameter(
                converted, requires_grad=self.weight.requires_grad
            )
            torch.utils.swap_tensors(self.weight, replacement)
            if self._ema_weight is not None:
                self._ema_weight = fn(self._ema_weight)
        return self

    def attach_comm_context(self, root: nn.Module) -> None:
        from ._state import _get_module_fsdp_state
        from .fully_shard import _iter_fsdp_param_groups

        state = _get_module_fsdp_state(root)
        if state is None:
            raise ValueError("row-owner embeddings require a GFSDP root")
        self._comm_ctx = state._comm_ctx
        self._sync_groups = tuple(_iter_fsdp_param_groups(root, recurse=True))
        if any(
            param.sharded_param is self.weight
            for group in self._sync_groups
            for param in group.fsdp_params
        ):
            raise ValueError(
                "row-owner weight must be excluded from every GFSDP wrapper"
            )
        fqn = next(
            (name for name, value in root.named_parameters() if value is self.weight),
            "weight",
        )
        self._weight_fqn = fqn
        logger.info(
            "row_owner_embedding active fqn=%s global_shape=%s local_rows=%d "
            "owner_rank=%d owner_world=%d gfsdp_gather_reduce=excluded",
            fqn,
            tuple(self.weight.shape),
            self.rows_per_owner,
            self.mesh_info.shard_mesh_rank,
            self.mesh_info.shard_mesh_size,
        )

    def _stage(self, name: str) -> AbstractContextManager:
        return (
            self.stage_observer(name)
            if self.stage_observer is not None
            else nullcontext()
        )

    @contextmanager
    def _communication(self, *inputs: torch.Tensor) -> Iterator[None]:
        if self.weight.device.type != "cuda":
            yield
            return
        if self._comm_ctx is None or not hasattr(self._comm_ctx, "all_gather_stream"):
            raise RuntimeError(
                "attach the initialized GFSDP root communication context before lookup"
            )
        current = torch.cuda.current_stream(self.weight.device)
        # Synchronous c10d calls join this same origin stream for both
        # communicators, preserving the order with dense GFSDP collectives.
        stream = self._comm_ctx.all_gather_stream
        stream.wait_event(current.record_event())
        for tensor in inputs:
            if tensor.device == self.weight.device:
                tensor.record_stream(stream)
        with torch.cuda.stream(stream):
            yield
            complete = stream.record_event()
        # This also supplies the excluded gradient's optimizer/clipping
        # readiness, which the GFSDP root callback cannot supply for us.
        current.wait_event(complete)

    def _validate_sync_mode(self) -> None:
        if any(not group.reduce_grads for group in self._sync_groups):
            raise RuntimeError(
                "row-owner embeddings do not support deferred/no-sync gradient reduction"
            )

    @contextmanager
    def _scratch(self, size: int) -> Iterator[torch.Tensor]:
        # The crop's private head gradient bucket already exists for chunked
        # CE. Never recreate the excluded embedding's buffer or grow a pool.
        contexts = {
            group.grad_buffer_ctx._storage_ctx
            for group in self._sync_groups
            if group.grad_buffer_ctx is not None
            and hasattr(group.grad_buffer_ctx, "buffers")
        }
        candidates = [
            (buffer.numel(), context, buffer)
            for context in contexts
            for buffer in context.buffers.values()
        ]
        if not candidates:
            raise RuntimeError(
                "row-owner lookup requires an existing GFSDP gradient buffer"
            )
        capacity, context, buffer = max(candidates, key=lambda item: item[0])
        if size > capacity:
            raise RuntimeError(
                f"row-owner scratch needs {size} bytes; existing GFSDP capacity is {capacity} bytes"
            )
        stream = (
            torch.cuda.current_stream(self.weight.device)
            if self.weight.is_cuda
            else None
        )
        with context.borrow(stream):
            yield buffer.narrow(0, 0, size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._validate_sync_mode()
        return _OwnerLookup.apply(self.weight, input, self)

    def _route(self, ids: torch.Tensor) -> tuple:
        world = self.mesh_info.shard_mesh_size
        rank = self.mesh_info.shard_mesh_rank
        device = self.weight.device
        with self._stage("partition_ids"):
            valid_type = ids.dtype == torch.int64 and ids.device == device
            flat = (
                ids.reshape(-1)
                if valid_type
                else torch.empty(0, dtype=torch.int64, device=device)
            )
            valid = ((flat >= 0) & (flat < self.num_embeddings)).all().to(torch.int64)
            if not valid_type:
                valid.zero_()
            # Clamping makes metadata safe even for invalid IDs; all ranks
            # reject the shared validity matrix before any payload exchange.
            unique, inverse = torch.unique(
                flat.clamp(0, self.num_embeddings - 1), sorted=True, return_inverse=True
            )
            owners = torch.div(unique, self.rows_per_owner, rounding_mode="floor")
            metadata = torch.zeros(world + 1, dtype=torch.int64, device=device)
            metadata[:world].scatter_add_(0, owners, torch.ones_like(owners))
            metadata[world] = valid
        with self._stage("counts"):
            matrix = metadata.new_empty(world * (world + 1))
            dist.all_gather_into_tensor(
                matrix, metadata, group=self._route_process_group
            )
            # One explicit host read supplies splits and coordinated validity.
            host = matrix.view(world, world + 1).cpu()
            if not bool(host[:, world].all()):
                raise ValueError(
                    f"row-owner IDs must be signed int64 on the weight device and in [0, {self.num_embeddings}) on every rank"
                )
            send = tuple(host[rank, :world].tolist())
            receive = tuple(host[:, rank].tolist())
        with self._stage("request_exchange"):
            requested = unique.new_empty(sum(receive))
            dist.all_to_all_single(
                requested,
                unique,
                list(receive),
                list(send),
                group=self._route_process_group,
            )
        return unique, inverse, requested, send, receive

    @torch.no_grad()
    def setup_ema(self) -> None:
        if self.weight.requires_grad and self._ema_weight is None:
            self._ema_weight = torch.empty_like(self.weight._local_tensor)

    def _state_local(self, *, ema: bool) -> torch.Tensor | None:
        return (
            self._ema_weight
            if ema != self._using_ema_weights
            else self.weight._local_tensor
        )

    @torch.no_grad()
    def seed_ema(self) -> None:
        if self._ema_weight is not None:
            self._state_local(ema=True).copy_(self._state_local(ema=False))

    @torch.no_grad()
    def update_ema(self, alpha: float) -> None:
        if self._ema_weight is not None:
            self._state_local(ema=True).lerp_(self._state_local(ema=False), alpha)

    @torch.no_grad()
    def swap_ema(self) -> None:
        if self._ema_weight is not None:
            # Swap backing tensors, preserving the registered Parameter/FQN.
            self.weight._local_tensor, self._ema_weight = (
                self._ema_weight,
                self.weight._local_tensor,
            )
            self._using_ema_weights = not self._using_ema_weights

    def checkpoint_tensor(self, *, ema: bool) -> RaggedShardDTensor:
        local = self._state_local(ema=ema)
        if local is None:
            raise RuntimeError("row-owner EMA has not been allocated")
        return RaggedShardDTensor(local, self.weight._spec, requires_grad=False)


class _OwnerLookup(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        weight: RaggedShardDTensor,
        ids: torch.Tensor,
        owner: RowOwnerEmbedding,
    ) -> torch.Tensor:
        with owner._communication(weight._local_tensor, ids):
            unique, inverse, requested, send, receive = owner._route(ids)
            local_rows = requested.sub_(owner.row_start)
            width, count = owner.embedding_dim, requested.numel()
            send_offset = max(4 * count * width, 2 * unique.numel() * width)
            with owner._scratch(send_offset + 2 * count * width) as scratch:
                with owner._stage("owner_lookup"):
                    selected = _view(scratch, 0, count, width, torch.float32)
                    torch.index_select(
                        weight._local_tensor.view(owner.rows_per_owner, width),
                        0,
                        local_rows,
                        out=selected,
                    )
                    values = _view(scratch, send_offset, count, width, torch.bfloat16)
                    values.copy_(selected)
                with owner._stage("value_exchange"):
                    returned = _view(scratch, 0, unique.numel(), width, torch.bfloat16)
                    dist.all_to_all_single(
                        returned,
                        values,
                        list(send),
                        list(receive),
                        group=owner._route_process_group,
                    )
                with owner._stage("reconstruct"):
                    output = returned.index_select(0, inverse).view(*ids.shape, width)
        if output.is_cuda:
            # The allocation belongs to the communication stream; subsequent
            # trunk/MTP consumers run on the caller's compute stream.
            output.record_stream(torch.cuda.current_stream(output.device))
        ctx.owner = owner
        ctx.splits = (send, receive)
        num_unique = unique.numel()
        compact_rows = _compact_min_ordered_rows(weight.device, width)
        if compact_rows is not None:
            num_ids = ids.numel()
            if num_ids <= 3072:
                # ATen uses the feature kernel for these unscaled gradients.
                compact_rows = num_unique
            elif min(num_ids, owner.num_embeddings) >= compact_rows:
                compact_rows = max(num_unique, compact_rows)
            else:
                compact_rows = None
        ctx.compact_rows = compact_rows
        if compact_rows is None:
            ctx.save_for_backward(ids, unique, local_rows)
        else:
            ctx.num_unique = num_unique
            ctx.save_for_backward(inverse, local_rows)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        owner = ctx.owner
        compact_rows = ctx.compact_rows
        saved_route = ctx.saved_tensors
        local_rows = saved_route[-1]
        send, receive = ctx.splits
        with owner._communication(grad_output, *saved_route):
            width = owner.embedding_dim
            if compact_rows is None:
                ids, unique, _ = saved_route
                with owner._stage("source_coalescing"):
                    dense = torch.ops.aten.embedding_dense_backward.default(
                        grad_output.to(torch.bfloat16).contiguous(),
                        ids,
                        owner.num_embeddings,
                        -1,
                        False,
                    )
                received_offset = unique.numel() * width * 2
            else:
                inverse, _ = saved_route
                with owner._stage("compact_source_coalescing"):
                    # The inverse preserves token order and duplicate groups.
                    # Send only the BF16 rows consumed by the existing splits.
                    coalesced = torch.ops.aten.embedding_dense_backward.default(
                        grad_output.to(torch.bfloat16).contiguous().view(-1, width),
                        inverse,
                        compact_rows,
                        -1,
                        False,
                    ).narrow(0, 0, ctx.num_unique)
                if not owner._compact_source_logged:
                    owner._compact_source_logged = True
                    logger.info(
                        "row_owner_embedding compact source coalescing active fqn=%s "
                        "unique_rows=%d padded_rows=%d",
                        owner._weight_fqn,
                        ctx.num_unique,
                        compact_rows,
                    )
                received_offset = 0
            receive_count = sum(receive)
            wide_offset = received_offset + receive_count * width * 2
            # Align FP32 views even when an odd hidden dimension is supported.
            wide_offset = (wide_offset + 3) // 4 * 4
            gradient_offset = wide_offset + max(receive, default=0) * width * 4
            grad_weight_local = torch.zeros(
                (owner.rows_per_owner, width),
                dtype=torch.float32,
                device=owner.weight.device,
            )
            with owner._scratch(gradient_offset) as scratch:
                if compact_rows is None:
                    with owner._stage("source_coalescing"):
                        coalesced = _view(
                            scratch, 0, unique.numel(), width, torch.bfloat16
                        )
                        torch.index_select(dense, 0, unique, out=coalesced)
                        del dense
                with owner._stage("gradient_exchange"):
                    received = _view(
                        scratch, received_offset, receive_count, width, torch.bfloat16
                    )
                    dist.all_to_all_single(
                        received,
                        coalesced,
                        list(receive),
                        list(send),
                        group=owner._route_process_group,
                    )
                    # The communication stream now orders the compact buffer's read.
                    del coalesced
                with owner._stage("owner_reduction"):
                    offset = 0
                    for count in receive:
                        if count:
                            source = _view(
                                scratch, wide_offset, count, width, torch.float32
                            )
                            source.copy_(received[offset : offset + count])
                            grad_weight_local.index_add_(
                                0, local_rows[offset : offset + count], source
                            )
                        offset += count
        grad_weight = RaggedShardDTensor(
            grad_weight_local.reshape(-1), owner.weight._spec, requires_grad=False
        )
        if grad_weight.is_cuda:
            grad_weight._local_tensor.record_stream(
                torch.cuda.current_stream(owner.weight.device)
            )
        return grad_weight, None, None


def _view(
    storage: torch.Tensor,
    offset: int,
    rows: int,
    width: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (
        storage.narrow(0, offset, rows * width * dtype.itemsize)
        .view(dtype)
        .view(rows, width)
    )


def iter_row_owner_embeddings(
    model_parts: Iterable[nn.Module],
) -> Iterator[RowOwnerEmbedding]:
    """Enumerate each explicitly registered owner module once across roots."""
    seen: set[int] = set()
    for model in model_parts:
        for module in model.modules():
            if isinstance(module, RowOwnerEmbedding) and id(module) not in seen:
                seen.add(id(module))
                yield module
