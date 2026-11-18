"""Eager input embeddings routed to their dense-GFSDP row owners.

The parameter remains a flat-backed FP32 RaggedShardDTensor, including for
optimizer, clipping and DCP. Only selected BF16 rows cross the owner group.
The routing pattern follows NeMo Automodel's Engram owner embedding (commit
3ea1de8d0aa902c3329b585d194c2e5274a85f3f); gradient coalescing deliberately
uses native embedding backward to retain its source-local BF16 boundary.
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
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
_ROW_OWNER_OPTIMIZATIONS = ("compact", "owner_add", "owner_lookup", "routing")


def _resolve_optimization_stages(
    optimize: bool, optimization_stages: tuple[str, ...] | None
) -> tuple[str, ...]:
    if optimize and optimization_stages is not None:
        raise ValueError("pass either optimize or diagnostic optimization_stages")
    requested = _ROW_OWNER_OPTIMIZATIONS if optimize else (optimization_stages or ())
    if len(set(requested)) != len(requested):
        raise ValueError("row-owner optimization stages must not contain duplicates")
    unknown = set(requested).difference(_ROW_OWNER_OPTIMIZATIONS)
    if unknown:
        raise ValueError(f"unknown row-owner optimization stages: {sorted(unknown)}")
    return tuple(stage for stage in _ROW_OWNER_OPTIMIZATIONS if stage in requested)


def _compact_native_is_safe(
    num_ids: int, num_unique: int, width: int, num_sms: int
) -> bool:
    """Keep compact native dispatch away from its unordered BF16 atomic path."""
    if num_ids <= 3072:
        return True
    max_segments = min(num_ids, num_unique)
    stride_warped = ((width + 31) // 32) * 32
    block = min(stride_warped, 1024)
    sum_scatter_grid = (max_segments * stride_warped + block - 1) // block
    return sum_scatter_grid >= 4 * num_sms


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
        optimize: bool = False,
        optimization_stages: tuple[str, ...] | None = None,
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
        result._optimization_stages = _resolve_optimization_stages(
            optimize, optimization_stages
        )
        result.mesh_info = info
        result.rows_per_owner = vocab // info.shard_mesh_size
        result.row_start = result.rows_per_owner * info.shard_mesh_rank
        result._comm_ctx = None
        result._sync_groups = ()
        result._using_ema_weights = False
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
        if result.optimization_stages:
            logger.info(
                "row_owner_embedding optimizations enabled stages=%s "
                "compact_native_guard='N<=3072 or native sum_scatter_grid>=4*num_sms'",
                result.optimization_stages,
            )
        return result

    @property
    def optimization_stages(self) -> tuple[str, ...]:
        """Enabled opt-in CUDA stages, exposed as immutable construction state."""
        return self._optimization_stages

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
            optimized_routing = (
                "routing" in self.optimization_stages
                and device.type == "cuda"
                and valid_type
                and flat.numel() <= 8192
            )
            if optimized_routing:
                from . import _row_owner_embedding_kernels as kernels

                metadata = torch.empty(world + 1, dtype=torch.int64, device=device)
                safe_ids = torch.empty_like(flat)
                kernels.validate_and_clamp_ids(
                    flat, safe_ids, metadata[world], self.num_embeddings
                )
                route_ids = safe_ids
            else:
                valid = (
                    ((flat >= 0) & (flat < self.num_embeddings)).all().to(torch.int64)
                )
                if not valid_type:
                    valid.zero_()
                # Clamp before owner division so INT64_MIN/MAX never overflow
                # into an invalid or negative owner index.
                route_ids = flat.clamp(0, self.num_embeddings - 1)

            unique, inverse = torch.unique(route_ids, sorted=True, return_inverse=True)
            if optimized_routing:
                kernels.build_routing_metadata(
                    unique,
                    metadata[world],
                    metadata,
                    self.rows_per_owner,
                    world,
                )
            else:
                owners = torch.div(unique, self.rows_per_owner, rounding_mode="floor")
                metadata = torch.zeros(world + 1, dtype=torch.int64, device=device)
                metadata[:world].scatter_add_(0, owners, torch.ones_like(owners))
                metadata[world] = valid
        with self._stage("counts"):
            matrix = metadata.new_empty(world * (world + 1))
            dist.all_gather_into_tensor(
                matrix, metadata, group=self.mesh_info.shard_process_group
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
                group=self.mesh_info.shard_process_group,
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


def _native_source_coalesce(
    grad_bf16: torch.Tensor,
    indices: torch.Tensor,
    unique: torch.Tensor | None,
    num_weights: int,
    *,
    compact_rows: int | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use native ordered duplicate reduction and keep its BF16 boundary."""
    if compact_rows is not None:
        return torch.ops.aten.embedding_dense_backward.default(
            grad_bf16, indices, compact_rows, -1, False
        )
    dense = torch.ops.aten.embedding_dense_backward.default(
        grad_bf16, indices, num_weights, -1, False
    )
    if output is None:
        return dense.index_select(0, unique)
    torch.index_select(dense, 0, unique, out=output)
    del dense
    return output


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
            unique_count = unique.numel()
            compact_coalescing = False
            if "compact" in owner.optimization_stages and weight.device.type == "cuda":
                num_sms = torch.cuda.get_device_properties(
                    weight.device
                ).multi_processor_count
                compact_coalescing = _compact_native_is_safe(
                    ids.numel(), unique_count, width, num_sms
                )
            direct_lookup = (
                "owner_lookup" in owner.optimization_stages
                and weight.device.type == "cuda"
                and weight._local_tensor.dtype == torch.float32
                and weight._local_tensor.is_contiguous()
                and local_rows.dtype == torch.int64
                and local_rows.is_contiguous()
            )
            send_offset = (
                2 * unique_count * width
                if direct_lookup
                else max(4 * count * width, 2 * unique_count * width)
            )
            with owner._scratch(send_offset + 2 * count * width) as scratch:
                with owner._stage("owner_lookup"):
                    values = _view(scratch, send_offset, count, width, torch.bfloat16)
                    if direct_lookup:
                        if count:
                            from . import _row_owner_embedding_kernels as kernels

                            kernels.owner_lookup_bf16(
                                weight._local_tensor.view(owner.rows_per_owner, width),
                                local_rows,
                                values,
                            )
                    else:
                        selected = _view(scratch, 0, count, width, torch.float32)
                        torch.index_select(
                            weight._local_tensor.view(owner.rows_per_owner, width),
                            0,
                            local_rows,
                            out=selected,
                        )
                        values.copy_(selected)
                with owner._stage("value_exchange"):
                    returned = _view(scratch, 0, unique_count, width, torch.bfloat16)
                    dist.all_to_all_single(
                        returned,
                        values,
                        list(send),
                        list(receive),
                        group=owner.mesh_info.shard_process_group,
                    )
                with owner._stage("reconstruct"):
                    output = returned.index_select(0, inverse).view(*ids.shape, width)
        if output.is_cuda:
            # The allocation belongs to the communication stream; subsequent
            # trunk/MTP consumers run on the caller's compute stream.
            output.record_stream(torch.cuda.current_stream(output.device))
        ctx.owner = owner
        ctx.splits = (send, receive)
        ctx.input_shape = tuple(ids.shape)
        ctx.input_count = ids.numel()
        ctx.unique_count = unique_count
        ctx.compact_coalescing = compact_coalescing
        if compact_coalescing:
            ctx.save_for_backward(inverse, local_rows)
        else:
            ctx.save_for_backward(ids, unique, local_rows)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        owner = ctx.owner
        if ctx.compact_coalescing:
            inverse, local_rows = ctx.saved_tensors
            ids = unique = None
            saved_route = (inverse, local_rows)
        else:
            ids, unique, local_rows = ctx.saved_tensors
            inverse = None
            saved_route = (ids, unique, local_rows)
        send, receive = ctx.splits
        with owner._communication(grad_output, *saved_route):
            width = owner.embedding_dim
            unique_count = ctx.unique_count
            receive_count = sum(receive)
            compact_coalescing = ctx.compact_coalescing
            grad_bf16 = (
                grad_output.to(torch.bfloat16).contiguous().view(ctx.input_count, width)
            )
            grad_weight_local = torch.zeros(
                (owner.rows_per_owner, width),
                dtype=torch.float32,
                device=owner.weight.device,
            )
            direct_owner_add = (
                "owner_add" in owner.optimization_stages
                and grad_weight_local.is_cuda
                and grad_weight_local.dtype == torch.float32
                and grad_weight_local.is_contiguous()
                and local_rows.dtype == torch.int64
                and local_rows.is_contiguous()
            )
            if compact_coalescing:
                with owner._stage("compact_source_coalescing"):
                    coalesced = _native_source_coalesce(
                        grad_bf16,
                        inverse,
                        None,
                        owner.num_embeddings,
                        compact_rows=unique_count,
                    )
                del grad_bf16
                received_offset = 0
            else:
                received_offset = unique_count * width * 2
            received_end = received_offset + receive_count * width * 2
            scratch_size = received_end
            if not direct_owner_add:
                # Preserve the native index_add path's FP32 source conversion
                # for the disabled/unsupported fused-owner fallback.
                wide_offset = (received_end + 3) // 4 * 4
                scratch_size = wide_offset + max(receive, default=0) * width * 4
            with owner._scratch(scratch_size) as scratch:
                if not compact_coalescing:
                    coalesced = _view(scratch, 0, unique_count, width, torch.bfloat16)
                    with owner._stage("source_coalescing"):
                        # Preserve the full-vocabulary native reduction when
                        # compact dispatch is disabled or its guard fails.
                        _native_source_coalesce(
                            grad_bf16,
                            ids,
                            unique,
                            owner.num_embeddings,
                            output=coalesced,
                        )
                    del grad_bf16
                with owner._stage("gradient_exchange"):
                    received = _view(
                        scratch,
                        received_offset,
                        receive_count,
                        width,
                        torch.bfloat16,
                    )
                    dist.all_to_all_single(
                        received,
                        coalesced,
                        list(receive),
                        list(send),
                        group=owner.mesh_info.shard_process_group,
                    )
                    # Compact coalescing owns this tensor; NCCL has now queued
                    # its same-stream read, so release it before owner adds.
                    if compact_coalescing:
                        del coalesced
                with owner._stage("owner_reduction"):
                    # _route builds a sorted unique list per requester and
                    # keeps each requester's split intact. Thus each source
                    # segment has one writer per destination row; the loop
                    # also preserves source-rank addition order.
                    offset = 0
                    for count in receive:
                        if count:
                            source_rows = local_rows[offset : offset + count]
                            source_values = received[offset : offset + count]
                            if direct_owner_add:
                                from . import _row_owner_embedding_kernels as kernels

                                kernels.owner_add_bf16_(
                                    grad_weight_local, source_rows, source_values
                                )
                            else:
                                source = _view(
                                    scratch, wide_offset, count, width, torch.float32
                                )
                                source.copy_(source_values)
                                grad_weight_local.index_add_(0, source_rows, source)
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
