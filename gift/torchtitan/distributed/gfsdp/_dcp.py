import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Set
from math import prod
from typing import Any, TypeAlias

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.protocols.sharding_backend import EMA_MODEL

from ._param import GFSDPParam
from ._param_group import GFSDPParamGroup
from ._state import _get_module_fsdp_state
from ._tensor import (
    get_ragged_shard_placement,
    RaggedShardDTensor,
    replace_ragged_shard,
)

from .fully_shard import _iter_fsdp_param_groups, GFSDPModule

logger = logging.getLogger("gfsdp")

# NOTE (fedorovgv): where the saved checkpoint carries each parameter's real shape
PARAM_SHAPES = "_titan_param_shapes"


def param_shapes_key(rank: int | None = None) -> str:
    """The ``PARAM_SHAPES`` entry this rank owns."""
    if rank is None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    return f"{PARAM_SHAPES}.{rank}"


_ManagedBinding: TypeAlias = tuple[GFSDPParam, torch.Tensor]


_AC_WRAPPER_SEGMENT = "_checkpoint_wrapped_module"


def _strip_ac_wrapper(fqn: str) -> str:
    if _AC_WRAPPER_SEGMENT not in fqn:
        return fqn
    return ".".join(part for part in fqn.split(".") if part != _AC_WRAPPER_SEGMENT)


class _FlatCheckpointTensor(torch.Tensor):
    """Expose a flat DCP tensor while aliasing one local GFSDP buffer slice."""

    _local_tensor: torch.Tensor
    _chunk: ChunkStorageMetadata | None
    __slots__ = ["_local_tensor", "_chunk"]

    @staticmethod
    def _partition_offset(tensor: RaggedShardDTensor) -> int:
        """Return the flat offset of the tensor's fixed local TP or EP partition."""
        spec = tensor._spec
        # Replace the ragged dp_shard axis with Replicate so only the fixed
        # TP/EP partition structure remains, then reuse DTensor's per-axis offset computation.
        replicated_placements = replace_ragged_shard(spec, Replicate())
        shard_placements = [p for p in replicated_placements if p.__class__ is Shard]
        if not shard_placements:
            return 0
        assert len(shard_placements) == 1

        _, global_offsets = compute_local_shape_and_global_offset(
            tensor.shape,
            spec.mesh,
            replicated_placements,
        )
        shard_placement = shard_placements[0]
        assert isinstance(shard_placement, Shard)
        shard_dim = shard_placement.dim
        if shard_dim < 0:
            shard_dim += tensor.ndim
        # Fixed TP/EP partitions are concatenated in shard-coordinate order
        partition_stride = prod(s for d, s in enumerate(tensor.shape) if d != shard_dim)
        return global_offsets[shard_dim] * partition_stride

    @staticmethod
    def _chunk_from_ragged(tensor: RaggedShardDTensor) -> ChunkStorageMetadata | None:
        """Describe the tensor's non-empty local shard in flat DCP coordinates."""
        ragged_shard_placement = get_ragged_shard_placement(tensor._spec)
        shard_offset = ragged_shard_placement.global_offset
        local_partition_numel = sum(ragged_shard_placement.shard_numels)
        local_chunk_end = shard_offset + ragged_shard_placement.local_numel

        assert shard_offset >= 0 and local_chunk_end <= local_partition_numel

        partition_offset = _FlatCheckpointTensor._partition_offset(tensor)
        if ragged_shard_placement.local_numel == 0:
            return None

        chunk_offset = partition_offset + shard_offset
        return ChunkStorageMetadata(
            offsets=torch.Size((chunk_offset,)),
            sizes=torch.Size((ragged_shard_placement.local_numel,)),
        )

    @staticmethod
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        global_numel: int,
        chunk: ChunkStorageMetadata | None,
    ) -> "_FlatCheckpointTensor":
        """Create a flat logical wrapper over an explicit local backing view."""
        assert local_tensor.ndim == 1 and local_tensor.is_contiguous(), (
            "flat tensor must be contiguous 1-D"
        )
        if chunk is None:
            assert local_tensor.numel() == 0, "missing chunk needs an empty tensor"
        else:
            chunk_offset = int(chunk.offsets[0])
            chunk_size = int(chunk.sizes[0])
            assert chunk_size == local_tensor.numel(), "chunk size must match tensor"
            assert (
                chunk_offset >= 0
                and chunk_size > 0
                and chunk_offset + chunk_size <= global_numel
            ), "chunk out of flat tensor bounds"

        wrapper = torch.Tensor._make_wrapper_subclass(
            cls,
            (global_numel,),
            strides=(1,),
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=local_tensor.requires_grad,
        )
        wrapper._local_tensor = local_tensor
        wrapper._chunk = chunk
        return wrapper

    @classmethod
    def from_ragged_shard(cls, tensor: RaggedShardDTensor) -> "_FlatCheckpointTensor":
        """Create a flat DCP view over a RaggedShardDTensor backing tensor."""
        return cls(tensor._local_tensor, tensor.numel(), cls._chunk_from_ragged(tensor))

    @classmethod
    def __torch_dispatch__(
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        """Preserve flat checkpoint metadata for the few supported copies."""
        kwargs = kwargs or {}
        if func in (
            torch.ops.aten.detach.default,
            torch.ops.aten.clone.default,
            torch.ops.aten._to_copy.default,
        ):
            (arg,) = args
            return cls(func(arg._local_tensor, **kwargs), arg.numel(), arg._chunk)

        if func == torch.ops.aten.new_empty.default:
            arg, requested_size = args
            if torch.Size(requested_size) != arg.size():
                raise NotImplementedError(
                    "Flat GFSDP checkpoint tensors only support new_empty "
                    "with their existing logical size."
                )
            local_tensor = func(arg._local_tensor, arg._local_tensor.size(), **kwargs)
            return cls(local_tensor, arg.numel(), arg._chunk)

        raise NotImplementedError(
            f"Flat GFSDP checkpoint tensor dispatch for {func} is not implemented."
        )

    def __create_write_items__(self, fqn: str, object: Any) -> list[WriteItem]:
        """Create the DCP write item for this rank's flat local shard."""
        assert object is self
        if self._chunk is None:
            return []
        return [
            WriteItem(
                index=MetadataIndex(fqn, self._chunk.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=self._chunk,
                    properties=TensorProperties.create_from_tensor(self._local_tensor),
                    size=self.size(),
                ),
            )
        ]

    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        """Create the DCP load target chunks for this rank."""
        return [] if self._chunk is None else [self._chunk]

    def __get_tensor_shard__(self, index: MetadataIndex) -> torch.Tensor:
        """Return this rank's local backing tensor for a DCP read."""
        return self._local_tensor


class GFSDPOptimizerCheckpointState(Stateful):
    """Translate RaggedShardDTensor optimizer leaves into flat DCP views."""

    def __init__(self, optimizer: Stateful) -> None:
        self._optimizer = optimizer
        self._ragged_load_targets: dict[str, RaggedShardDTensor] = {}

    def state_dict(self) -> dict[str, Any]:
        """Build an optimizer state with flat views over ragged tensor leaves."""
        state_dict = self._optimizer.state_dict()
        ragged_targets: dict[str, RaggedShardDTensor] = {}
        checkpoint_state: dict[str, Any] = {}
        for key, value in state_dict.items():
            if isinstance(value, RaggedShardDTensor):
                ragged_targets[key] = value
                checkpoint_state[key] = _FlatCheckpointTensor.from_ragged_shard(value)
            else:
                checkpoint_state[key] = value
        self._ragged_load_targets = ragged_targets
        return checkpoint_state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore ragged views and apply the populated optimizer state."""
        optimizer_state = dict(state_dict)
        for key, ragged_target in self._ragged_load_targets.items():
            optimizer_state[key] = ragged_target
        self._optimizer.load_state_dict(optimizer_state)
        self._ragged_load_targets = {}


def _param_shapes(state_dict: Mapping[str, Any]) -> dict[str, list[int]]:
    """Real global shapes, read off the parameters before they are flattened."""
    return {
        fqn: list(value.shape)
        for fqn, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }


class GFSDPCheckpointAdapter:
    """Translate GFSDP model and EMA buffers into transient DCP states.

    Implements ``torchtitan.protocols.checkpoint_adapter.CheckpointAdapter``.
    The constructor keeps model roots without caching FQN bindings or tensor
    views.
    """

    def __init__(self, model_parts: Iterable[nn.Module]) -> None:
        self._model_parts = tuple(model_parts)
        if not any(
            _get_module_fsdp_state(module) is not None
            for model in self._model_parts
            for module in model.modules()
        ):
            raise ValueError(
                "fsdp_backend='gfsdp2' but the model has no GFSDP modules "
                "(e.g. checkpoint.create_seed_checkpoint skips parallelization), "
                "so it cannot be checkpointed in the gfsdp2 format."
            )

    def model_save_state_dict(self) -> dict[str, Any]:
        return self.build_state_dict(load_target=False)

    def wrap_optimizer_state(self, optimizer: Stateful) -> Stateful:
        return GFSDPOptimizerCheckpointState(optimizer)

    def load(
        self,
        states: dict[str, Any],
        *,
        checkpoint_id: str,
        process_group: dist.ProcessGroup | None,
        load_model: bool,
    ) -> None:
        if load_model:
            load_gfsdp_dcp(self, states, checkpoint_id, process_group)
        else:
            dcp.load(states, checkpoint_id=checkpoint_id)

    def _build_ordinary_state_dict(self) -> dict[str, Any]:
        """Return state dict with normalized FQNs as flat dict"""
        _lazy_init_gfsdp_roots(self._model_parts)
        state_dict: dict[str, Any] = {}
        for model in self._model_parts:
            for fqn, value in get_model_state_dict(model).items():
                assert fqn not in state_dict, f"duplicate model FQN across parts: {fqn}"
                state_dict[fqn] = value
        return state_dict

    def _get_param_data_map(self, *, ema: bool) -> dict[GFSDPParam, torch.Tensor]:
        """Build map GFSDPParam: local_data"""
        param_data: dict[GFSDPParam, torch.Tensor] = {}
        param_groups = (
            _iter_ema_param_groups(self._model_parts)
            if ema
            else (
                param_group
                for model in self._model_parts
                for param_group in _iter_fsdp_param_groups(model, recurse=True)
            )
        )
        for param_group in param_groups:
            group_data = list(
                param_group.iter_ema_param_data()
                if ema
                else param_group.iter_main_param_data()
            )
            assert len(group_data) == len(param_group.fsdp_params), (
                "GFSDP buffers do not cover every parameter in the group"
            )
            for fsdp_param, local_data in group_data:
                param_data[fsdp_param] = local_data
        return param_data

    def _manage_fqns_to_params(
        self,
        ordinary_state_dict: Mapping[str, Any],
        *,
        ema: bool,
    ) -> dict[str, _ManagedBinding]:
        """Map ordinary FQNs, including aliases, to managed buffer slices."""

        param_data = self._get_param_data_map(ema=ema)
        registered_param_to_fsdp_param: dict[int, GFSDPParam] = {}
        for fsdp_param in param_data:
            registered_param_to_fsdp_param[id(fsdp_param.sharded_param)] = fsdp_param
            registered_param_to_fsdp_param[id(fsdp_param.unsharded_param)] = fsdp_param

        bindings: dict[str, _ManagedBinding] = {}
        matched_params: set[GFSDPParam] = set()
        for model in self._model_parts:
            for fqn, param in model.named_parameters(remove_duplicate=False):
                fsdp_param = registered_param_to_fsdp_param.get(id(param))
                if fsdp_param is None:
                    continue
                bindings[_strip_ac_wrapper(fqn)] = (
                    fsdp_param,
                    param_data[fsdp_param],
                )
                matched_params.add(fsdp_param)

        assert matched_params == set(param_data), (
            "some managed GFSDP parameters have no ordinary model FQN"
        )
        if not ema:
            unmapped_ragged_fqns = [
                fqn
                for fqn, value in ordinary_state_dict.items()
                if isinstance(value, RaggedShardDTensor) and fqn not in bindings
            ]
            assert not unmapped_ragged_fqns, (
                f"unmapped RaggedShardDTensor FQNs: {sorted(unmapped_ragged_fqns)}"
            )
        return bindings

    @staticmethod
    def _make_checkpoint_view(
        fsdp_param: GFSDPParam, local_data: torch.Tensor
    ) -> _FlatCheckpointTensor:
        """Create a flat DCP view over an explicit GFSDP slice."""
        model_view = fsdp_param.to_sharded_dtensor(local_data)
        return _FlatCheckpointTensor.from_ragged_shard(model_view)

    def _build_managed_state_dict(
        self,
        ordinary_state_dict: Mapping[str, Any],
        *,
        ema: bool,
        load_target: bool,
    ) -> dict[str, Any]:
        """Build a managed DCP state, parameterized by buffer and direction."""
        bindings = self._manage_fqns_to_params(ordinary_state_dict, ema=ema)
        canonical_fqns = _canonical_load_fqns(bindings) if load_target else None
        if ema:
            selected = (
                set(canonical_fqns) if canonical_fqns is not None else set(bindings)
            )
            return {
                fqn: self._make_checkpoint_view(*bindings[fqn])
                for fqn in sorted(selected)
            }

        state_dict = dict(ordinary_state_dict)
        for fqn, (fsdp_param, local_data) in bindings.items():
            if canonical_fqns is not None and fqn not in canonical_fqns:
                del state_dict[fqn]
                continue
            state_dict[fqn] = self._make_checkpoint_view(fsdp_param, local_data)
        return state_dict

    def build_state_dict(self, *, load_target: bool) -> dict[str, Any]:
        """Build the model DCP state: main entries plus an EMA namespace."""
        ordinary_state_dict = self._build_ordinary_state_dict()
        state_dict = self._build_managed_state_dict(
            ordinary_state_dict, ema=False, load_target=load_target
        )
        ema_state_dict = self._build_managed_state_dict(
            ordinary_state_dict, ema=True, load_target=load_target
        )
        if ema_state_dict:
            state_dict[EMA_MODEL] = ema_state_dict
        if not load_target:
            state_dict[param_shapes_key()] = json.dumps(
                _param_shapes(ordinary_state_dict)
            )
        return state_dict

    def apply_main_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Run model load hooks after DCP has populated main target views."""
        ordinary_state_dict = self._build_ordinary_state_dict()
        bindings = self._manage_fqns_to_params(ordinary_state_dict, ema=False)

        loaded_model_state_dict: dict[str, Any] = {}
        for fqn, ordinary_value in ordinary_state_dict.items():
            if fqn not in state_dict:
                continue
            loaded_model_state_dict[fqn] = (
                ordinary_value if fqn in bindings else state_dict[fqn]
            )

        for model in self._model_parts:
            set_model_state_dict(
                model,
                model_state_dict=dict(loaded_model_state_dict),
                options=StateDictOptions(strict=False),
            )

    @torch.no_grad()
    def initialize_ema_from_main(self, fqns: Set[str]) -> None:
        """Copy main-params to EMA groups for the given FQNs."""
        fqns = frozenset(fqns)
        if not fqns:
            return
        ordinary_state_dict = self._build_ordinary_state_dict()
        ema_bindings = self._manage_fqns_to_params(ordinary_state_dict, ema=True)
        gap_params = {ema_bindings[fqn][0] for fqn in fqns}
        for param_group in _iter_ema_param_groups(self._model_parts):
            if gap_params.intersection(param_group.fsdp_params):
                param_group.initialize_ema_from_main()


class GFSDPLoadPlanner(DefaultLoadPlanner):
    """Validate the checkpoint against the target and select EMA FQNs to load."""

    def __init__(self) -> None:
        super().__init__()
        self.ema_init_fqns: frozenset[str] = frozenset()

    def set_up_planner(
        self,
        state_dict: dict[str, Any],
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        super().set_up_planner(state_dict, metadata, is_coordinator)
        self._validate_and_prune()

    def _validate_and_prune(self) -> None:
        assert self.metadata is not None
        state_metadata = self.metadata.state_dict_metadata
        ema_prefix = f"{EMA_MODEL}."

        ema_targets: set[str] = set()
        for key, value in self.state_dict.items():
            if not isinstance(value, _FlatCheckpointTensor):
                continue
            if key.startswith(ema_prefix):
                ema_targets.add(key[len(ema_prefix) :])
                continue
            saved = state_metadata.get(key)
            assert isinstance(saved, TensorStorageMetadata), (
                f"checkpoint missing required flat target {key!r}"
            )

        payload_fqns = frozenset(
            key[len(ema_prefix) :]
            for key in state_metadata
            if key.startswith(ema_prefix)
        )

        extra_fqns = payload_fqns - ema_targets
        if extra_fqns:
            logger.warning(
                "Ignoring %d EMA checkpoint entries with no load target"
                " (parameter group frozen or not EMA-tracked): %s",
                len(extra_fqns),
                sorted(extra_fqns),
            )

        gap_fqns = frozenset(ema_targets) - payload_fqns
        if gap_fqns:
            logger.warning(
                "The checkpoint carries no EMA weights for %d parameter(s), so"
                " theirs are seeded from the loaded LIVE weights and their"
                " average starts over from this step: %s",
                len(gap_fqns),
                sorted(gap_fqns),
            )
        for fqn in gap_fqns:
            key = f"{ema_prefix}{fqn}"
            self.state_dict.pop(key, None)
            mappings = getattr(self, "mappings", None)
            if isinstance(mappings, dict):
                mappings.pop(key, None)
        self.ema_init_fqns = frozenset(gap_fqns)


def load_gfsdp_dcp(
    adapter: GFSDPCheckpointAdapter,
    state_dict: dict[str, Any],
    checkpoint_id: str,
    process_group: dist.ProcessGroup | None,
) -> None:
    target = dict(state_dict)
    model_target = adapter.build_state_dict(load_target=True)
    ema_target = model_target.pop(EMA_MODEL, None)
    target.update(model_target)
    if ema_target:
        target[EMA_MODEL] = ema_target

    planner = GFSDPLoadPlanner()
    dcp.load(
        target,
        checkpoint_id=checkpoint_id,
        planner=planner,
        process_group=process_group,
    )

    adapter.apply_main_state_dict(target)
    adapter.initialize_ema_from_main(planner.ema_init_fqns)


def _iter_ema_param_groups(
    model_parts: Iterable[nn.Module],
) -> Iterator[GFSDPParamGroup]:
    """Yield whole parameter groups registered for EMA across model parts."""
    tracked_param_groups: set[GFSDPParamGroup] = set()
    for model in model_parts:
        for module in model.modules():
            if not isinstance(module, GFSDPModule):
                continue
            tracked_param_groups.update(
                module._get_ema_tracked_param_groups(recurse=False)
            )
    for model in model_parts:
        for param_group in _iter_fsdp_param_groups(model, recurse=True):
            if param_group in tracked_param_groups:
                yield param_group


def _lazy_init_gfsdp_roots(model_parts: Iterable[nn.Module]) -> None:
    """Lazily initialize each independent GFSDP root before checkpointing."""
    for model in model_parts:
        for module in model.modules():
            state = _get_module_fsdp_state(module)
            if state is None or state._is_root is not None:
                continue
            state._lazy_init()


def _canonical_load_fqns(bindings: Mapping[str, _ManagedBinding]) -> frozenset[str]:
    """Select one deterministic FQN for every tied parameter backing."""
    fqns_by_param: dict[GFSDPParam, list[str]] = defaultdict(list)
    for fqn in bindings:
        fqns_by_param[bindings[fqn][0]].append(fqn)
    return frozenset(min(param_fqns) for param_fqns in fqns_by_param.values())
