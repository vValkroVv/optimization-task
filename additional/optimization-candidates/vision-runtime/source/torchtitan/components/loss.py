# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast, Literal, TypeAlias

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.experimental import local_map
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import CompileConfig, Configurable
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]
SampleDenominator: TypeAlias = torch.Tensor | float


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss with sum reduction for token-based normalization."""
    if isinstance(pred, DTensor) and isinstance(labels, DTensor):
        return _cross_entropy_via_local_map(pred, labels)

    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def cross_entropy_loss_per_sample(
    pred: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Return the summed cross-entropy for each batch row.

    Sequence-sharded DTensors keep a ``Partial`` placement so the caller can
    sum chunk contributions before the normal DTensor reduction. Vocabulary
    sharding follows ``cross_entropy_loss`` and reduces its internal
    log-sum-exp before returning the per-token losses.
    """
    if not isinstance(pred, DTensor) or not isinstance(labels, DTensor):
        token_losses = torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(),
            labels.flatten(),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).unflatten(0, labels.shape)
        return token_losses.sum(dim=1)

    mesh = pred.device_mesh
    expected_labels_placements = tuple(
        Replicate() if isinstance(p, Shard) and p.dim == 2 else p
        for p in pred.placements
    )
    if labels.placements != expected_labels_placements:
        raise ValueError(
            "cross_entropy_loss_per_sample: expected labels placements "
            f"{expected_labels_placements}, got {labels.placements}"
        )

    def _flatten_placement(p):
        if isinstance(p, Shard):
            return Shard(0 if p.dim == 0 else p.dim - 1)
        return p

    vocab_sharded = any(isinstance(p, Shard) and p.dim == 2 for p in pred.placements)
    out_placements = [
        Shard(0)
        if isinstance(p, Shard) and p.dim == 0
        else Partial()
        if isinstance(p, Shard) and p.dim == 1
        else Replicate()
        for p in pred.placements
    ]

    pred_placements = pred.placements
    labels_placements = labels.placements

    @local_map(
        out_placements=out_placements,
        in_placements=(pred_placements, labels_placements),
        in_grad_placements=(pred_placements, labels_placements),
        device_mesh=mesh,
    )
    def _local_cross_entropy_per_sample(
        pred_local: torch.Tensor, labels_local: torch.Tensor
    ) -> torch.Tensor:
        flat_pred = pred_local.flatten(0, 1).float()
        flat_labels = labels_local.flatten()
        if not vocab_sharded:
            token_losses = torch.nn.functional.cross_entropy(
                flat_pred,
                flat_labels,
                reduction="none",
                ignore_index=IGNORE_INDEX,
            )
        else:
            flat_pred_dtensor = DTensor.from_local(
                flat_pred,
                mesh,
                tuple(_flatten_placement(p) for p in pred_placements),
                run_check=False,
            )
            flat_labels_dtensor = DTensor.from_local(
                flat_labels,
                mesh,
                tuple(_flatten_placement(p) for p in labels_placements),
                run_check=False,
            )
            token_losses_dtensor = torch.nn.functional.cross_entropy(
                flat_pred_dtensor,
                flat_labels_dtensor,
                reduction="none",
                ignore_index=IGNORE_INDEX,
            )
            assert isinstance(token_losses_dtensor, DTensor)
            token_losses = token_losses_dtensor.to_local()
        return token_losses.unflatten(0, labels_local.shape).sum(dim=1)

    return _local_cross_entropy_per_sample(pred, labels)


def valid_tokens_per_sample(labels: torch.Tensor) -> torch.Tensor:
    """Count valid labels per batch row, reducing sequence-sharded DTensors."""
    if not isinstance(labels, DTensor):
        return (labels != IGNORE_INDEX).sum(dim=1)

    out_placements = [
        Shard(0)
        if isinstance(p, Shard) and p.dim == 0
        else Partial()
        if isinstance(p, Shard) and p.dim == 1
        else Replicate()
        for p in labels.placements
    ]

    @local_map(
        out_placements=out_placements,
        in_placements=(labels.placements,),
        device_mesh=labels.device_mesh,
    )
    def _local_valid_tokens_per_sample(labels_local: torch.Tensor) -> torch.Tensor:
        return (labels_local != IGNORE_INDEX).sum(dim=1)

    return _local_valid_tokens_per_sample(labels)


def replicate_per_sample_counts(counts: torch.Tensor) -> torch.Tensor:
    """Reduce per-row counts across non-batch mesh axes.

    A sequence-sharded count is a ``Partial`` sum. It must be reduced before
    it divides each chunk's per-row CE; otherwise every CP rank would use its
    local token count as the denominator.
    """
    if not isinstance(counts, DTensor):
        return counts
    placements = tuple(
        Shard(0) if isinstance(p, Shard) and p.dim == 0 else Replicate()
        for p in counts.placements
    )
    if counts.placements == placements:
        return counts
    return counts.redistribute(counts.device_mesh, placements)


def clamp_sample_denominator(denominator: SampleDenominator) -> SampleDenominator:
    """Keep an all-ignored global batch finite without changing nonempty loss."""
    if isinstance(denominator, torch.Tensor):
        return denominator.clamp_min(1)
    return max(denominator, 1)


def _cross_entropy_via_local_map(pred: DTensor, labels: DTensor) -> torch.Tensor:
    mesh = pred.device_mesh
    # Labels don't have a vocab dim.
    expected_labels_placements = tuple(
        Replicate() if isinstance(p, Shard) and p.dim == 2 else p
        for p in pred.placements
    )
    if labels.placements != expected_labels_placements:
        raise ValueError(
            f"cross_entropy_loss: expected labels placements {expected_labels_placements}, "
            f"got {labels.placements}"
        )

    # After local flatten(0, 1), tensor dims are [batch*seq, vocab].
    # Per-axis placement:
    #   Shard on batch/seq -> Shard(0) (valid because reduction is sum)
    #   Shard on vocab -> Shard(1)
    def _flatten_placement(p):
        if isinstance(p, Shard):
            return Shard(0 if p.dim == 0 else p.dim - 1)
        return p

    vocab_sharded = any(isinstance(p, Shard) and p.dim == 2 for p in pred.placements)

    # Per-axis output placement for sum reduction:
    #   Shard on non-vocab-dim -> Partial
    #   Shard on vocab-dim -> Replicate
    out_placements = [
        Partial() if isinstance(p, Shard) and p.dim != 2 else Replicate()
        for p in pred.placements
    ]

    @local_map(
        out_placements=out_placements,
        in_placements=(pred.placements, labels.placements),
        in_grad_placements=(pred.placements, labels.placements),
        device_mesh=mesh,
    )
    def _local_cross_entropy(
        pred_local: torch.Tensor, labels_local: torch.Tensor
    ) -> torch.Tensor:
        flat_pred = pred_local.flatten(0, 1).float()
        flat_labels = labels_local.flatten(0, 1)
        if not vocab_sharded:
            return torch.nn.functional.cross_entropy(
                flat_pred,
                flat_labels,
                reduction="sum",
                ignore_index=IGNORE_INDEX,
            )

        # vocab_sharded == True => loss parallel case
        # TODO: rewrite the entire loss parallel using megatron style.
        flat_pred_placements = tuple(_flatten_placement(p) for p in pred.placements)
        flat_labels_placements = tuple(_flatten_placement(p) for p in labels.placements)
        pred_dtensor = DTensor.from_local(
            flat_pred, mesh, flat_pred_placements, run_check=False
        )
        labels_dtensor = DTensor.from_local(
            flat_labels, mesh, flat_labels_placements, run_check=False
        )
        loss_dtensor = torch.nn.functional.cross_entropy(
            pred_dtensor,
            labels_dtensor,
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )
        assert isinstance(loss_dtensor, DTensor)
        return loss_dtensor.to_local()

    return _local_cross_entropy(pred, labels)


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MSE loss with sum reduction for Transformer models training."""
    return torch.nn.functional.mse_loss(
        pred.float(), labels.float().detach(), reduction="sum"
    )


class BaseLoss(ABC, Configurable):
    """Abstract base class for all loss functions.

    Provides compile support and a unified ``__call__`` signature:
    ``(pred, labels, global_valid_tokens) -> scaled_loss``.
    Subclasses must call the base ``__init__`` before setting ``self.fn``.
    """

    fn: LossFunction

    component_name: str = "loss_ce"
    _components: dict[str, torch.Tensor]

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __init__(
        self, config: Config, *, compile_config: CompileConfig | None = None
    ) -> None:
        del config, compile_config
        self.reset_components()
        self._step_targets: list[torch.Tensor] = []
        self._step_target_storages: set[int] = set()
        # ``schedule.step`` calls that registered targets, so far this run.
        self._armed_steps = 0
        self._saw_shape_probe = False

    def _maybe_compile(self, compile_config: CompileConfig | None) -> None:
        if (
            compile_config is not None
            and compile_config.enable
            and "loss" in compile_config.components
        ):
            logger.info("Compiling the loss function with torch.compile")
            self.fn = torch.compile(
                self.fn,
                backend=compile_config.backend,
                dynamic=compile_config.dynamic,
                fullgraph=True,
            )

    def pack_targets(
        self, labels: torch.Tensor, extra_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        """Fold what this loss needs out of the batch into ``labels``.

        A pipeline schedule microbatches the target tensor and nothing else.
        """
        del extra_kwargs
        return labels

    def prepare_targets(
        self, labels: torch.Tensor, *, positions: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Optionally prepare loss-only targets before the batch moves to CUDA.

        The trainer keeps the original labels for masks, token accounting and
        auxiliary losses. A returned tensor is passed only to this loss.
        """
        return None

    def prepare_batch(
        self,
        labels: torch.Tensor,
        sample_token_counts: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
        packing_mode: str = "none",
    ) -> tuple[
        torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        """Return targets, loss units, sample units and row denominators."""
        targets = self.prepare_targets(labels, positions=positions)
        loss_units = self.normalization_count(labels, positions=positions)
        return targets, loss_units, None, None

    def normalization_count(
        self, labels: torch.Tensor, *, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Count this batch's loss units; clamp only after the global reduction."""
        return (labels != IGNORE_INDEX).sum()

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        sample_token_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._begin_call()
        loss = self.fn(pred, labels)
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens
        self.add_component(self.component_name, loss.detach())
        return self._finish_call(loss)

    def _begin_call(self) -> None:
        """Open a slot for this call's components.

        Per call, not summed on the spot, because not every call to the loss is
        a microbatch: on its first ``schedule.step`` the pipeline runs the loss
        once more on the last stage to learn the gradient shapes, and that
        value never enters the schedule's ``losses``. Keeping calls apart is
        what lets ``keep_microbatches`` drop the probe.
        """
        self._calls.append({})

    def _finish_call(self, loss: torch.Tensor) -> torch.Tensor:
        """Close the current call, stamped with the identity of its return.

        The stamp is what ``keep_microbatches`` matches against the schedule's
        ``losses``. Closing also separates calls for losses that never open a
        slot themselves: ``add_component`` folds into the last slot only while
        it is unfinished, so the next call starts a fresh one. A weak
        reference, twice over: a strong one would keep every microbatch's
        autograd graph alive until reset on the non-PP path, and a bare
        ``id()`` could alias a freed probe tensor's address with a later real
        loss.
        """
        if len(self._calls) <= len(self._call_outputs):
            self._begin_call()  # a call that recorded no components still counts
        self._call_outputs.append(weakref.ref(loss))
        return loss

    def declared_components(self) -> tuple[str, ...]:
        """The component names this loss can produce, in a fixed order.

        Declared and not discovered because the reduction is positional: under
        pipeline parallelism only the last stage ever calls the loss, so a
        vector built from what happened to accumulate would have a different
        length on every stage. ``AuxLossTracker`` lays out its slots by this.
        """
        return (self.component_name,)

    def add_component(self, name: str, value: torch.Tensor) -> None:
        # A new slot when there is none, or when the last one was already
        # closed by _finish_call: without this, a loss that does not call
        # _begin_call itself (ChunkedCELoss) folded every microbatch of the
        # step into one dict, and nothing could take the pipeline's shape
        # probe back out of it.
        if len(self._calls) <= len(self._call_outputs):
            self._begin_call()
        current = self._calls[-1]
        previous = current.get(name)
        current[name] = value if previous is None else previous + value

    def reset_components(self) -> None:
        """Start a new optimizer step's accounting. Called by the trainer."""
        self._calls: list[dict[str, torch.Tensor]] = []
        self._call_outputs: list[weakref.ref] = []
        self._committed_calls = 0

    def keep_microbatches(self, losses: list[torch.Tensor]) -> None:
        """Keep only the calls whose return entered the schedule's ``losses``.

        The pipeline's first ``schedule.step`` runs the loss once more for
        gradient-shape metadata, and WHERE that call lands among the real
        microbatches depends on the torch release — currently after the first
        forward, so trimming by count from either end keeps the probe and
        drops a real microbatch. Membership in ``losses`` (the very tensors
        the schedule collected) is the release-independent test. Calls kept by
        a previous invocation are behind the watermark and stay: the trainer
        calls this once per ``schedule.step`` under gradient accumulation.
        """
        wanted = {id(loss) for loss in losses}
        mark = self._committed_calls
        calls = self._calls
        outputs = self._call_outputs
        # The schedule's losses are alive here, so a live weakref compares by
        # real identity; the probe's tensor is dead by now and its ref
        # resolves to None instead of a reused address.
        kept = [
            (call, out)
            for call, out in zip(calls[mark:], outputs[mark:])
            if (target := out()) is not None and id(target) in wanted
        ]
        if len(kept) != len(losses):
            raise RuntimeError(
                f"loss bookkeeping out of sync: the schedule returned "
                f"{len(losses)} microbatch losses but {len(kept)} of the "
                f"{len(calls) - mark} loss calls since the last step match "
                "them; components would be wrong."
            )
        self._calls = calls[:mark] + [call for call, _ in kept]
        self._call_outputs = outputs[:mark] + [out for _, out in kept]
        self._committed_calls = len(self._calls)

    def set_step_targets(self, targets: Any) -> None:
        """Register the targets of this ``schedule.step``, by storage.

        Microbatch targets are views of what the trainer passed, while the
        schedule's metadata probe brings a target of its own -- so an unknown
        storage is the probe. Registering arms the test, ``None`` disarms it.
        The tensors are held, not just their addresses: a freed storage is
        handed straight back out and would read as a microbatch.
        """
        pinned: list[torch.Tensor] = []
        storages: set[int] = set()
        stack: list[Any] = [targets]
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple)):
                stack.extend(item)
            elif isinstance(item, torch.Tensor):
                local = item.to_local() if isinstance(item, DTensor) else item
                pinned.append(local)
                storages.add(local.untyped_storage().data_ptr())
        self._step_targets = pinned
        self._step_target_storages = storages
        if storages:
            self._armed_steps += 1

    def _is_step_target(self, labels: torch.Tensor) -> bool:
        """Whether ``labels`` is a view of a target registered for this step."""
        local = labels.to_local() if isinstance(labels, DTensor) else labels
        return local.untyped_storage().data_ptr() in self._step_target_storages

    def is_shape_probe_target(self, labels: torch.Tensor) -> bool:
        """Whether ``labels`` came from the schedule's metadata probe.

        Consumes the one probe this run is allowed. Taking a real microbatch
        for the probe would silently drop its parameter gradients, so an
        unknown target outside the one window where a probe is possible is an
        error rather than an assumption.
        """
        if not self._step_target_storages or self._is_step_target(labels):
            return False
        # The schedule infers gradient metadata while it initializes its
        # stages, which happens inside the first ``schedule.step`` of the run.
        if self._armed_steps > 1:
            raise RuntimeError(
                "loss called with a target belonging to no microbatch, on "
                f"step {self._armed_steps} of this run. The schedule runs its "
                "metadata probe only while it initializes its stages, on the "
                "first step, so microbatch targets have stopped being views "
                "of the target passed to step() and the probe test is unsound."
            )
        if self._saw_shape_probe:
            raise RuntimeError(
                "loss called with a target belonging to no microbatch of this "
                "step, after the schedule's one metadata probe was already "
                "seen: the schedule probes more than once now and the probe "
                "test is unsound."
            )
        self._saw_shape_probe = True
        return True

    def components(self) -> dict[str, torch.Tensor]:
        """Terms accumulated since the last reset. Must sum to the total loss."""
        totals: dict[str, torch.Tensor] = {}
        for call in self._calls:
            for name, value in call.items():
                previous = totals.get(name)
                totals[name] = value if previous is None else previous + value
        return totals


class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss with sum reduction for token-based normalization."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        super().__init__(config, compile_config=compile_config)
        self.fn: LossFunction = cross_entropy_loss
        self._maybe_compile(compile_config)


class MSELoss(BaseLoss):
    """MSE loss with sum reduction for Transformer models training (e.g. Flux)."""

    component_name = "mse"

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        super().__init__(config, compile_config=compile_config)
        self.fn: LossFunction = mse_loss
        self._maybe_compile(compile_config)


class GradAccumulator:
    """Accumulates chunk gradients into a pre-allocated buffer.

    Instead of collecting chunk gradients in a list and concatenating at the end,
    this uses a pre-allocated buffer with in-place copies for better memory efficiency.

    Args:
        reference: Reference tensor for shape, device, and DTensor-ness. If a
            DTensor, only its device mesh is reused; the placement of the
            returned DTensor is taken from the first added chunk (see add()),
            not from this reference, so the buffer is labeled with the actual
            gradient placement (e.g. Partial(sum) on the TP axis when the
            forward used a Replicate input with a Shard(0) weight, as in
            ColwiseParallel lm_head) rather than the activation placement.
        num_chunks: Number of chunks that will be added.
        seq_dim: The sequence dimension along which chunks are accumulated.
        dtype: Dtype for the buffer.

    Usage:
        accumulator = GradAccumulator(hidden_states, num_chunks=4, dtype=torch.float32)
        for chunk_grad in chunk_grads:
            accumulator.add(chunk_grad)
        full_grad = accumulator.result()
    """

    def __init__(
        self,
        reference: torch.Tensor,
        *,
        num_chunks: int,
        seq_dim: int = 1,
        dtype: torch.dtype,
    ):
        from torch.distributed.tensor import DTensor, Placement

        self.num_chunks = num_chunks
        self.seq_dim = seq_dim
        self._next_idx = 0
        self._next_offset = 0
        self._device_mesh: DeviceMesh | None = None
        # Captured from the first added chunk; see __init__ docstring.
        self._placements: tuple[Placement, ...] | None = None

        if isinstance(reference, DTensor):
            self._device_mesh = reference.device_mesh
            local = reference.to_local()
        else:
            local = reference

        self._buffer = torch.zeros_like(local, dtype=dtype)

    def add(self, chunk_grad: torch.Tensor) -> None:
        """Add the next chunk gradient sequentially.

        Chunks must be added in order (0, 1, 2, ..., num_chunks - 1).
        """
        from torch.distributed.tensor import DTensor

        if self._next_idx >= self.num_chunks:
            raise ValueError(f"Already added {self.num_chunks} chunks, cannot add more")

        if isinstance(chunk_grad, DTensor):
            if self._placements is None:
                self._placements = chunk_grad.placements
            elif chunk_grad.placements != self._placements:
                # All chunks come from the same op chain and must share a
                # placement. Otherwise the buffer mixes frames and result()
                # would mislabel them.
                raise ValueError(
                    f"chunk_grad placement {chunk_grad.placements} does not "
                    f"match first chunk's placement {self._placements}"
                )
            chunk_grad = chunk_grad.to_local()
        elif self._placements is not None:
            # Earlier chunks were DTensor but this one is a plain tensor;
            # mixing the two would silently drop the implied reduction.
            raise ValueError(
                "chunk_grad is a plain tensor but earlier chunks were "
                f"DTensor with placement {self._placements}"
            )

        if chunk_grad.dtype != self._buffer.dtype:
            chunk_grad = chunk_grad.to(self._buffer.dtype)

        chunk_seq_len = chunk_grad.shape[self.seq_dim]
        start = self._next_offset
        end = start + chunk_seq_len
        if end > self._buffer.shape[self.seq_dim]:
            raise ValueError("Chunk gradients exceed the reference sequence length")

        slices = [slice(None)] * self._buffer.ndim
        slices[self.seq_dim] = slice(start, end)
        self._buffer[tuple(slices)] = chunk_grad

        self._next_idx += 1
        self._next_offset = end

    def result(self) -> torch.Tensor:
        """Return the accumulated gradient tensor, wrapped as DTensor if needed.

        When the chunks were Partial(sum), the returned DTensor is also
        Partial(sum); autograd performs the implied reduction once when this
        gradient lands on the decoder-side leaf.
        """
        from torch.distributed.tensor import DTensor

        if self._device_mesh is not None:
            if self._placements is None:
                raise ValueError(
                    "No DTensor chunk was added; cannot wrap the buffer as "
                    "DTensor without a known placement. Either pass DTensor "
                    "chunks to add(), or use a plain reference tensor."
                )
            return DTensor.from_local(
                self._buffer,
                device_mesh=self._device_mesh,
                placements=self._placements,
            )
        return self._buffer


def _get_fsdp_param_groups(
    module: FSDPModule, *, initialize: bool = True
) -> tuple[Any, ...]:
    """Isolate the FSDP2 state access needed to preserve runtime policies.

    PyTorch has policy setters but no getters; the forward setter also cannot
    restore integer reshard policies. Newer builds store multiple groups per
    state; earlier builds use the singular attribute.
    """
    state = module._get_fsdp_state()
    if initialize:
        state._lazy_init()
    if hasattr(state, "_fsdp_param_groups"):
        return tuple(state._fsdp_param_groups)
    group = state._fsdp_param_group
    return () if group is None else (group,)


def _shares_fsdp_group_outside_head(group: Any, head: nn.Module) -> bool:
    # Parameter sets alone cannot detect a tied embedding/head: both own the
    # same weight, but the embedding still needs it in the decoder backward.
    head_modules = set(head.modules())
    return any(module not in head_modules for module in group.modules)


@dataclass
class _ChunkedFSDPGroupState:
    group: Any
    post_forward_mesh_info: Any
    reshard_after_backward: bool
    reduce_grads: bool
    defer_reshard: bool

    def last_chunk(self) -> None:
        self.group.reduce_grads = self.reduce_grads
        # A standalone head has no later decoder backward to release its
        # weights. Shared groups with RAF=False must keep them for the norm or
        # tied embedding; RAF=True retains its earlier release policy.
        self.group.reshard_after_backward = (
            self.reshard_after_backward and not self.defer_reshard
        )

    def restore(self) -> None:
        self.group.post_forward_mesh_info = self.post_forward_mesh_info
        self.group.reshard_after_backward = self.reshard_after_backward
        self.group.reduce_grads = self.reduce_grads


@contextmanager
def _chunked_fsdp_head(
    head: nn.Module, *, requires_grad: bool
) -> Iterator[list[_ChunkedFSDPGroupState]]:
    """Keep weights across chunks without overwriting the caller's policy."""
    from torchtitan.distributed.gfsdp import GFSDPModule

    previous_last_backward = None
    if requires_grad and isinstance(head, GFSDPModule):
        previous_last_backward = head._get_fsdp_state()._state_ctx.is_last_backward
    states = []
    if isinstance(head, FSDPModule):
        for group in _get_fsdp_param_groups(head):
            states.append(
                _ChunkedFSDPGroupState(
                    group=group,
                    post_forward_mesh_info=group.post_forward_mesh_info,
                    reshard_after_backward=group.reshard_after_backward,
                    reduce_grads=group.reduce_grads,
                    defer_reshard=(
                        requires_grad
                        and group.post_forward_mesh_info is None
                        and _shares_fsdp_group_outside_head(group, head)
                    ),
                )
            )
    try:
        if previous_last_backward is not None:
            # The decoder backward happens after these detached head chunks.
            # Ranks may have different chunk counts with compact targets.
            head.set_is_last_backward(False)
        for state in states:
            state.group.post_forward_mesh_info = None
            state.group.reshard_after_backward = False
            state.group.reduce_grads = False
            # Do not use set_requires_gradient_sync(): it also overwrites
            # all_reduce_grads, undoing the trainer's deferred HSDP all-reduce.
        yield states
    finally:
        for state in states:
            state.restore()
        if previous_last_backward is not None:
            head.set_is_last_backward(previous_last_backward)
    if not requires_grad:
        # Validation has no backward to release weights. Apply the original
        # forward policy after the last chunk, retaining RAF=False as requested.
        for state in states:
            if state.post_forward_mesh_info is not None:
                state.group.reshard()


class ChunkedCELoss(BaseLoss):
    """Chunked cross-entropy loss that splits the sequence dimension to reduce peak memory.

    Instead of materializing the full [B, L, V] logits tensor at once, this splits
    the hidden states into N chunks along the sequence dimension and computes
    lm_head + cross_entropy_loss on each chunk sequentially. This reduces peak memory
    from O(B*L*V) to O(B*L/N*V).

    The flow:
    1. Model forward with _skip_lm_head=True to get hidden states [B, L, D]
    2. Detach hidden states at the boundary
    3. Split detached hidden states into N chunks along seq dim
    4. Disable FSDP reshard on lm_head to keep weight unsharded across chunks
    5. For each chunk: lm_head(chunk) -> ce_loss -> backward()
    6. Assemble chunk gradients into a full gradient [B, L, D] via GradAccumulator
    7. Backward through the decoder via hidden_states.backward(accumulated_grad)

    FSDP2 composability:
        The lm_head's FSDP reshard-after-forward and reshard-after-backward are
        temporarily disabled during the chunked loop so that the weight stays
        unsharded across all chunks (avoiding repeated all-gathers). Gradient
        synchronization is disabled until the last chunk, so all local chunks
        share one final reduce-scatter even when ranks have different counts,
        unless the caller disabled sync. Original policies are restored even on
        failure. With RAF=False, a shared norm/head group stays unsharded until
        the decoder backward; a standalone head follows its original backward
        reshard policy on the last chunk.

    TP / SP composability:
        Hidden states are redistributed to ``Replicate()`` on the TP mesh
        before chunking, so each chunk enters the lm_head as ``Replicate()``
        input regardless of whether SP is enabled. With SP, this is an
        all-gather from ``Shard(1)``; without SP, it's a no-op.

        When loss parallel is applied, each TP rank
        computes partial CE on its ``V/tp`` slice, with an internal
        all-reduce for the correct log-sum-exp.

    CP: Further chunks the local sequence dimension. Works out of the box.

    Compile: ce_loss can be compiled independently; lm_head is not compiled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        num_chunks: int = 8
        """Number of chunks to split the sequence into."""

        normalization: Literal["tokens", "samples"] = "tokens"
        """Normalize CE by tokens, or mean per nonempty batch row.

        ``samples`` first divides each row's summed valid-token CE by that
        row's required pre-shift valid-token count, then divides the local row sum by the global
        number of nonempty rows. Fully ignored rows contribute zero.
        """

        def __post_init__(self) -> None:
            if self.normalization not in ("tokens", "samples"):
                raise ValueError("CE normalization must be 'tokens' or 'samples'")

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        super().__init__(config, compile_config=compile_config)
        self.fn: LossFunction = cross_entropy_loss
        self.sample_fn: LossFunction = cross_entropy_loss_per_sample
        self._maybe_compile(compile_config)
        if (
            compile_config is not None
            and compile_config.enable
            and "loss" in compile_config.components
        ):
            self.sample_fn = torch.compile(
                self.sample_fn,
                backend=compile_config.backend,
                dynamic=compile_config.dynamic,
                fullgraph=True,
            )
        self.num_chunks = config.num_chunks
        self.normalization = config.normalization
        self.lm_head: nn.Module | None = None

    def set_lm_head(self, lm_head: nn.Module) -> None:
        """Set the lm_head module. Must be called before the first __call__."""
        self.lm_head = lm_head
        if isinstance(lm_head, FSDPModule):
            # Binding happens before the model's first forward; do not initialize
            # this inner FSDP state as a root. HSDP partial-reduce buffers require
            # the same parameter layout in head chunks and the norm backward,
            # and across microbatches. Keep this policy for the loss's lifetime.
            for group in _get_fsdp_param_groups(lm_head, initialize=False):
                if getattr(
                    group.mesh_info, "replicate_process_group", None
                ) is not None and _shares_fsdp_group_outside_head(group, lm_head):
                    group.reduce_scatter_unused_params = True

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        global_valid_samples: SampleDenominator | None = None,
        sample_token_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute chunked cross-entropy loss.

        ``pred`` should be hidden states from model forward with
        ``_skip_lm_head=True``.

        When ``pred`` does not require grad (e.g. validation), runs chunked
        forward only — no per-chunk backward or gradient accumulation.

        Returns a differentiable loss. When ``.backward()`` is called on it
        (either by the trainer or the PP schedule), it triggers backward
        through the decoder via a custom autograd Function.
        """
        pred = self._replicate_on_tp(pred)
        total_loss, accumulated_grad = self._chunked_loss_and_grad(
            pred,
            labels,
            global_valid_tokens,
            sample_token_counts,
            global_valid_samples=global_valid_samples,
            component=self.component_name,
            shape_probe=self.is_shape_probe_target(labels),
        )
        if accumulated_grad is None:
            return self._finish_call(total_loss)

        return self._finish_call(self._finish(pred, accumulated_grad, total_loss))

    @staticmethod
    def _replicate_on_tp(pred: torch.Tensor) -> torch.Tensor:
        """Gather the TP axis so ``lm_head`` sees Replicate input.

        Called from ``__call__``, because the gradient is grafted onto what this
        returns: done on a local copy instead, the redistribute is missing from
        the caller's graph and a sequence-sharded input gets a ``Replicate`` grad.
        Keyed on the mesh axis rather than a tensor dim, so it also serves the
        stacked ``[num_mtp + 1, B, S, D]`` input, where the sequence axis is 2.
        """
        if not isinstance(pred, DTensor):
            return pred
        names = pred.device_mesh.mesh_dim_names
        if names is None or "tp" not in names:
            return pred
        tp = names.index("tp")
        if isinstance(pred.placements[tp], Replicate):
            return pred
        placements = list(pred.placements)
        placements[tp] = Replicate()
        return pred.redistribute(pred.device_mesh, tuple(placements))

    def prepare_batch(
        self,
        labels: torch.Tensor,
        sample_token_counts: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
        packing_mode: str = "none",
    ) -> tuple[
        torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        """Resolve row denominators for chunked CE."""
        if self.normalization != "samples":
            prepared_batch = super().prepare_batch(labels, positions=positions)
            return prepared_batch
        sample_token_counts = (
            valid_tokens_per_sample(labels)
            if packing_mode != "none"
            else cast(torch.Tensor, sample_token_counts)
        )
        targets = self.prepare_targets(labels, positions=positions)
        loss_units = self.normalization_count(labels, positions=positions)
        valid_samples = (sample_token_counts > 0).sum().reshape(1)
        return targets, loss_units, valid_samples, sample_token_counts

    def _sample_counts(
        self,
        labels: torch.Tensor,
        sample_token_counts: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Clamp the required row denominators for sample-normalized CE."""
        if self.normalization != "samples":
            return None
        return cast(torch.Tensor, sample_token_counts).clamp_min(1)

    def _chunked_loss_and_grad(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        sample_token_counts: torch.Tensor | None = None,
        *,
        global_valid_samples: SampleDenominator | list[SampleDenominator] | None = None,
        num_heads: int = 1,
        component: str | None = None,
        argmax_sink: list[torch.Tensor] | None = None,
        shape_probe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run chunked ``lm_head`` + cross-entropy over the sequence.

        Returns ``(total_loss, grad w.r.t. hidden_states)``, the gradient
        ``None`` in validation. ``num_heads`` are the heads ``MTPLoss`` folded
        into the batch axis, split back out in ``_chunk_loss``; ``component``
        names the metrics term, ``None`` leaving that to the caller.
        """
        num_chunks = self.num_chunks
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedCELoss"

        # Check if it's training model or validation mode
        requires_grad = hidden_states.requires_grad

        # Chunking always operates on the *local* view: when ``t`` is a
        # Shard(1) DTensor, chunking the global view would distribute whole
        # chunks across ranks (e.g. size=2, num_chunks=8: chunks 0-3 on
        # rank 0, 4-7 on rank 1), leaving half the per-chunk DTensors with
        # local seq=0 and breaking GradAccumulator's slice writes.
        # ``local_map`` runs the chunking body on plain tensors; under the
        # non-DTensor (eager) path we call ``_chunk_local`` directly.
        # ``.contiguous()`` breaks shared storage from ``torch.chunk``.
        def _chunk_local(t):
            return tuple(c.contiguous() for c in torch.chunk(t, num_chunks, dim=1))

        def _chunk(t):
            if not isinstance(t, DTensor):
                return _chunk_local(t)
            p = t.placements
            wrapped = local_map(
                _chunk_local,
                out_placements=(p,) * num_chunks,
                in_placements=(p,),
                device_mesh=t.device_mesh,
            )
            return wrapped(t)

        # ``detach`` + ``requires_grad_`` makes each chunk a leaf so it
        # accumulates ``.grad`` for ``GradAccumulator``.
        h_chunks = [
            c.detach().requires_grad_(requires_grad) for c in _chunk(hidden_states)
        ]
        label_chunks = list(_chunk(labels))

        grad_accumulator = GradAccumulator(
            hidden_states,
            num_chunks=num_chunks,
            dtype=torch.float32,
        )

        total_loss = hidden_states.new_zeros((), dtype=torch.float32)
        sample_counts = self._sample_counts(labels, sample_token_counts)

        with _chunked_fsdp_head(
            lm_head, requires_grad=requires_grad and not shape_probe
        ) as fsdp_states:
            last_idx = len(h_chunks) - 1
            for i, (h_chunk, label_chunk) in enumerate(zip(h_chunks, label_chunks)):
                if i == last_idx:
                    for state in fsdp_states:
                        state.last_chunk()

                logits = lm_head(h_chunk)
                if argmax_sink is not None:
                    # Reuse these logits for per-head greedy acceptance metrics.
                    with torch.no_grad():
                        argmax_sink.append(logits.detach().argmax(dim=-1))

                chunk_loss = self._chunk_loss(
                    logits,
                    label_chunk,
                    global_valid_tokens,
                    num_heads=num_heads,
                    sample_counts=sample_counts,
                    global_valid_samples=global_valid_samples,
                )
                total_loss = total_loss + chunk_loss.detach()
                if component is not None:
                    self.add_component(component, chunk_loss.detach())

                if requires_grad and shape_probe:
                    (chunk_grad,) = torch.autograd.grad(chunk_loss, h_chunk)
                    grad_accumulator.add(chunk_grad)
                elif requires_grad:
                    chunk_loss.backward()
                    assert h_chunk.grad is not None
                    grad_accumulator.add(h_chunk.grad)
                    h_chunk.grad = None
        if not requires_grad:
            return total_loss, None

        accumulated_grad = grad_accumulator.result().to(hidden_states.dtype)
        return total_loss, accumulated_grad

    def _chunk_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None,
        *,
        num_heads: int,
        sample_counts: torch.Tensor | None = None,
        global_valid_samples: SampleDenominator | None = None,
    ) -> torch.Tensor:
        """Loss for one chunk of ``lm_head`` output."""
        if sample_counts is None:
            chunk_loss = self.fn(logits, labels)
            if global_valid_tokens is not None:
                chunk_loss = chunk_loss / global_valid_tokens
            return chunk_loss

        chunk_loss = (self.sample_fn(logits, labels) / sample_counts).sum()
        if global_valid_samples is not None:
            chunk_loss = chunk_loss / clamp_sample_denominator(global_valid_samples)
        return chunk_loss

    def _finish(
        self,
        hidden_states: torch.Tensor,
        accumulated_grad: torch.Tensor,
        total_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Both ``__call__`` paths end here: fill in this loss's ``lm_head``."""
        assert self.lm_head is not None, "Set lm_head before calling ChunkedCELoss"
        return self._gradient_backprop(
            hidden_states,
            accumulated_grad,
            total_loss,
            self.lm_head,
            isinstance(self.lm_head, FSDPModule),
        )

    @staticmethod
    def _gradient_backprop(
        hidden_states: torch.Tensor,
        accumulated_grad: torch.Tensor,
        total_loss: torch.Tensor,
        lm_head: nn.Module,
        fsdp_enabled: bool,
    ) -> torch.Tensor:
        """Return a differentiable loss via _DecoderOutputGradientBackProp.
        When ``.backward()`` is called (by the trainer or PP schedule),
        autograd calls ``_DecoderOutputGradientBackProp.backward`` which
        returns ``accumulated_grad`` as the gradient for ``hidden_states``,
        propagating through the decoder. Subclasses override to swap in a
        different autograd Function.
        """
        deferred_groups = (
            tuple(
                group
                for group in _get_fsdp_param_groups(cast(FSDPModule, lm_head))
                if group.reshard_after_backward
                and group.post_forward_mesh_info is None
                and _shares_fsdp_group_outside_head(group, lm_head)
            )
            if fsdp_enabled
            else ()
        )
        return _DecoderOutputGradientBackProp.apply(
            hidden_states, accumulated_grad, total_loss, deferred_groups
        )


def _sample_mean_cross_entropy_loss(
    pred: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Sum CE after dividing each token by its sample's valid count."""
    targets, counts = labels.unbind(dim=-1)
    token_loss = torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        targets.flatten(),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    return (token_loss / counts.flatten().to(token_loss.dtype).clamp_min(1)).sum()


class CompactCELoss(ChunkedCELoss):
    """Apply the chunked LM head only at supervised, already-aligned targets.

    CPU preparation returns ``[1, 3, N]`` containing target labels, flattened
    decoder positions and per-target normalization counts. No CUDA ``nonzero``
    or scalar read is needed. The decoder and its norm still see all tokens;
    selection happens in the loss before the vocabulary projection.

    Uses the existing chunked backward/FSDP machinery. An empty local batch
    selects one ignored position so the head and decoder keep their backward
    graph on every rank. TP, CP, PP, full DTensor and MTP require the ordinary
    loss path and must be rejected when selecting this loss in the trainer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedCELoss.Config):
        normalization: Literal["tokens", "samples"] = "tokens"
        """Token mean, or mean of each nonempty sample's token-mean CE."""

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        if config.num_chunks < 1:
            raise ValueError("CompactCELoss.num_chunks must be positive")
        super().__init__(config, compile_config=None)
        self.fn = (
            cross_entropy_loss
            if self.normalization == "tokens"
            else _sample_mean_cross_entropy_loss
        )
        self._maybe_compile(compile_config)

    def _sample_counts(
        self,
        labels: torch.Tensor,
        sample_token_counts: torch.Tensor | None = None,
    ) -> None:
        """Compact CE already carries its denominators inside prepared targets."""
        return None

    def prepare_batch(
        self,
        labels: torch.Tensor,
        sample_token_counts: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
        packing_mode: str = "none",
    ) -> tuple[
        torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        """Prepare selected targets from explicit row or document metadata."""
        units = self.normalization_count(labels, positions=positions)
        samples = None
        if self.normalization == "samples":
            if packing_mode == "none":
                sample_token_counts = cast(torch.Tensor, sample_token_counts)
                samples = (sample_token_counts > 0).sum().reshape(1)
            else:
                samples = units.reshape(1)
        targets = self.prepare_targets(
            labels,
            sample_token_counts,
            positions=positions,
            packing_mode=packing_mode,
        )
        return targets, units, samples, None

    @staticmethod
    def _validate_cpu_labels(labels: torch.Tensor) -> None:
        if isinstance(labels, DTensor) or labels.device.type != "cpu":
            raise ValueError("Compact loss targets must be prepared from CPU labels")
        if labels.ndim != 2 or not all(labels.shape):
            raise ValueError("Compact loss requires nonempty [batch, sequence] labels")
        if labels.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("Compact loss labels must have integer dtype")

    @staticmethod
    def _sample_groups(
        labels: torch.Tensor, positions: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map aligned target slots to rows or packed documents on CPU.

        Packed labels must already be shifted within each document, with the
        cross-document target ignored. Position resets identify documents, not
        a request to shift targets a second time.
        """
        if positions is None:
            rows = torch.arange(labels.shape[0], device="cpu")
            group_ids = rows.repeat_interleave(labels.shape[1])
            num_groups = labels.shape[0]
        else:
            if (
                isinstance(positions, DTensor)
                or positions.device.type != "cpu"
                or positions.shape != labels.shape
                or positions.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("Compact loss positions must match CPU integer labels")
            invalid_step = (positions[:, 1:] != 0) & (
                positions[:, 1:] != positions[:, :-1] + 1
            )
            if bool((positions < 0).any()) or bool(invalid_step.any()):
                raise ValueError(
                    "Sample loss positions must advance by one or reset to zero"
                )
            starts = positions == 0
            starts[:, 0] = True
            group_ids = starts.flatten().long().cumsum(0) - 1
            num_groups = int(starts.sum())
        counts = torch.bincount(
            group_ids[labels.flatten() != IGNORE_INDEX], minlength=num_groups
        )
        return group_ids, counts

    def prepare_targets(
        self,
        labels: torch.Tensor,
        sample_token_counts: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
        packing_mode: str = "none",
    ) -> torch.Tensor:
        """Select supervised targets with row or packed document denominators."""
        self._validate_cpu_labels(labels)
        flat_labels = labels.flatten()
        selected = (flat_labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        targets = flat_labels.index_select(0, selected).long()
        counts = torch.ones_like(selected)
        if self.normalization == "samples":
            if packing_mode != "none":
                groups, group_counts = self._sample_groups(labels, positions)
                counts = group_counts.index_select(0, groups.index_select(0, selected))
            else:
                sample_token_counts = cast(torch.Tensor, sample_token_counts)
                counts = sample_token_counts.index_select(
                    0, selected // labels.shape[1]
                ).clamp_min(1)
        if selected.numel() == 0:
            selected = torch.zeros(1, dtype=torch.long, device="cpu")
            targets = torch.full_like(selected, IGNORE_INDEX)
            counts = torch.ones_like(selected)
        return torch.stack((targets, selected, counts)).unsqueeze(0)

    def normalization_count(
        self, labels: torch.Tensor, *, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._validate_cpu_labels(labels)
        if self.normalization == "tokens":
            return super().normalization_count(labels, positions=positions)
        _, counts = self._sample_groups(labels, positions)
        return (counts > 0).sum()

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        sample_token_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(pred, DTensor) or isinstance(labels, DTensor):
            raise ValueError(
                "CompactCELoss requires local tensors without TP/CP/full_dtensor"
            )
        if pred.ndim != 3:
            raise ValueError(
                "CompactCELoss requires [batch, sequence, hidden] states without MTP"
            )
        if labels.ndim == 2:
            # Validation callers may not use the training CPU preparation hook.
            if self.normalization == "samples":
                raise ValueError(
                    "Sample-normalized CompactCELoss requires prepared CPU targets"
                )
            return super().__call__(pred, labels, global_valid_tokens)
        if (
            labels.ndim != 3
            or labels.shape[:2] != (1, 3)
            or labels.shape[2] == 0
            or labels.dtype != torch.long
        ):
            raise ValueError("CompactCELoss requires prepared int64 targets [1, 3, N]")
        hidden = pred.flatten(0, 1).index_select(0, labels[0, 1]).unsqueeze(0)
        targets = (
            labels[:, 0, :]
            if self.normalization == "tokens"
            else labels[:, (0, 2), :].transpose(1, 2).contiguous()
        )
        return super().__call__(hidden, targets, global_valid_tokens)


class _DecoderOutputGradientBackProp(torch.autograd.Function):
    """Bridges chunked lm_head backward with decoder backward via autograd.

    Forward takes hidden_states (connected to decoder graph), the accumulated
    gradient from chunked lm_head backward, and the loss value. Returns a
    detached loss with this Function as its grad_fn.

    Backward returns accumulated_grad as the gradient for hidden_states.
    Autograd then propagates this through the decoder layers automatically —
    no explicit hidden_states.backward() needed.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        hidden_states: torch.Tensor,
        accumulated_grad: torch.Tensor,
        loss: torch.Tensor,
        deferred_groups: tuple[Any, ...],
    ) -> torch.Tensor:
        ctx.save_for_backward(accumulated_grad)
        ctx.deferred_groups = deferred_groups
        return loss.detach()

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        (accumulated_grad,) = ctx.saved_tensors
        if ctx.deferred_groups:
            groups = ctx.deferred_groups

            def reshard_after_decoder_backward():
                # Usually the norm's FSDP post-backward has already done this.
                # Also handle unused siblings: a group left POST_BACKWARD by
                # the head chunks may be skipped by FSDP's final callback.
                for group in groups:
                    if group.reshard_after_backward:
                        group.reshard()

            torch.autograd.Variable._execution_engine.queue_callback(
                reshard_after_decoder_backward
            )
        # Return accumulated_grad as the gradient for hidden_states.
        # Autograd then propagates this through hidden_states' existing
        # decoder graph — equivalent to hidden_states.backward(accumulated_grad)
        # but expressed as a return value so autograd handles the traversal
        # in a single pass (no "backward through graph twice" error).
        # Note: this is not safe if downstream accidentally runs tensor ops after
        # the loss returns, which would produce a non-trivial grad_output that we need
        # to properly handle. The complicated part is that grad_output might not be
        # on the same device mesh as accumlated_grad.
        return accumulated_grad, None, None, None


class MTPLoss(ChunkedCELoss):
    """Chunked cross-entropy loss for Multi-Token Prediction (MTP).

    ``pred`` is the stacked hidden states ``[num_mtp + 1, B, S, D]`` produced by
    a ``Decoder`` with MTP enabled (in training): index 0 is the main
    next-token head (post the model's final norm); indices ``1..num_mtp`` are the
    MTP heads (each post its own final norm).

    The heads are folded into the batch axis (``[H, B, S, D]`` -> ``[H*B, S, D]``),
    so the shared ``lm_head`` runs once per chunk for all heads; ``_chunk_loss``
    splits the logits back and applies per-head weights. Chunks are ``H`` times
    bigger: scale ``num_chunks`` accordingly.

    Labels for MTP head ``k`` are the main labels rolled left by ``k`` with the
    last ``k`` positions ignored (no valid future target). The total loss is
    ``main_CE + sum_k loss_weights[k] * CE_k``. Token normalization uses the
    main head's ``global_valid_tokens`` for every head. Sample normalization
    uses one global nonempty-row count per shifted head, because a shift may
    remove all targets from a row (including at a packed-document boundary).

    The trainer wires ``lm_head`` / ``_skip_lm_head`` for any ``ChunkedCELoss``,
    so MTPLoss is set up automatically.
    """

    component_name = "loss_ce"

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedCELoss.Config):
        # One weight per MTP head; ``len`` must equal the number of MTP heads
        # the model produces.
        loss_weights: list[float]

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        super().__init__(config, compile_config=compile_config)
        if not config.loss_weights:
            raise ValueError(
                "MTPLoss.loss_weights is empty; use ChunkedCELoss without MTP heads."
            )
        self.loss_weights = list(config.loss_weights)
        self.acceptance_sinks: list[list[dict[str, torch.Tensor]]] = []

    def prepare_batch(
        self,
        labels: torch.Tensor,
        sample_token_counts: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
        packing_mode: str = "none",
    ) -> tuple[
        torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        """Count each MTP head using its shifted targets."""
        samples = None
        if self.normalization == "samples":
            head_labels = [labels] + [
                self._shift_labels(labels, index, positions)
                for index in range(1, len(self.loss_weights) + 1)
            ]
            samples = torch.stack(
                [(targets != IGNORE_INDEX).any(dim=1).sum() for targets in head_labels]
            )
        targets = self.prepare_targets(labels, positions=positions)
        loss_units = self.normalization_count(labels, positions=positions)
        return targets, loss_units, samples, None

    def declared_components(self) -> tuple[str, ...]:
        """``loss_ce``, the weighted MTP mean, and per-head terms when N > 1.

        The conditions are the ones ``__call__`` accumulates under: a head whose
        weight is zero contributes nothing and is not named, so no tag is a
        constant zero.
        """
        names = [self.component_name]
        if sum(self.loss_weights) != 0.0:
            names.append("loss_mtp")
        if len(self.loss_weights) > 1:
            names += [
                f"loss_mtp{head}"
                for head, weight in enumerate(self.loss_weights, start=1)
                if weight != 0.0
            ]
        return tuple(names)

    def pack_targets(
        self, labels: torch.Tensor, extra_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        """Carry ``positions`` beside the labels as ``[B, 2, S]``; unpacked in
        ``__call__``."""
        positions = extra_kwargs.get("positions")
        if positions is None:
            return labels
        return torch.stack([labels, positions.to(labels.dtype)], dim=1)

    def _sample_counts(
        self,
        labels: torch.Tensor,
        sample_token_counts: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Count each MTP head's shifted targets across non-batch mesh axes."""
        if self.normalization != "samples":
            return None
        counts = valid_tokens_per_sample(labels)
        counts = replicate_per_sample_counts(counts)
        return counts.clamp_min(1)

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        global_valid_samples: SampleDenominator | list[SampleDenominator] | None = None,
        sample_token_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._begin_call()
        # Positions ride inside ``labels`` as [B, 2, S]: a pipeline schedule
        # microbatches the target tensor and nothing else.
        positions = None
        if labels.dim() == 3:
            labels, positions = labels[:, 0], labels[:, 1]

        # MTP runs only in training: the model returns stacked hidden states
        # [num_mtp+1, B, S, D]. In eval (or MTP disabled) it returns the usual
        # [B, S, D] hidden states, so fall back to the main-only chunked CE.
        if pred.dim() == 3:
            main_global_valid_samples = (
                global_valid_samples[0]
                if isinstance(global_valid_samples, list)
                else global_valid_samples
            )
            return super().__call__(
                pred,
                labels,
                global_valid_tokens,
                main_global_valid_samples,
            )
        if pred.dim() != 4:
            raise ValueError(
                "MTPLoss expects stacked hidden states [num_mtp+1, B, S, D]; got "
                f"shape {tuple(pred.shape)}."
            )

        num_heads = pred.shape[0]
        if num_heads - 1 != len(self.loss_weights):
            raise ValueError(
                f"MTPLoss.loss_weights has length {len(self.loss_weights)} but the "
                f"model produced {num_heads - 1} MTP head(s)."
            )

        # Once, on the stack: the folded view inherits it.
        pred = self._replicate_on_tp(pred)
        self._reject_sharded_fold(pred, dims=(0, 1), name="the stacked hidden states")
        self._reject_sharded_fold(labels, dims=(0,), name="the labels")
        head_labels = [labels] + [
            self._shift_labels(labels, head_idx, positions)
            for head_idx in range(1, num_heads)
        ]

        sink: list[torch.Tensor] | None = [] if self.acceptance_sinks else None

        # Before the fold: torch.cat makes a new storage that looks like the PP probe.
        shape_probe = self.is_shape_probe_target(labels)
        combined_labels = torch.cat(head_labels, dim=0)
        total_loss, accumulated_grad = self._chunked_loss_and_grad(
            pred.flatten(0, 1),
            combined_labels,
            global_valid_tokens,
            global_valid_samples=global_valid_samples,
            num_heads=num_heads,
            argmax_sink=sink,
            shape_probe=shape_probe,
        )

        if sink is not None:
            # [H*B, S] -> per-head [B, S].
            head_argmaxes = list(torch.chunk(torch.cat(sink, dim=1), num_heads, dim=0))
            counters = self._acceptance_counters(head_argmaxes, labels, positions)
            for subscriber in self.acceptance_sinks:
                subscriber.append(counters)

        if accumulated_grad is None:
            return self._finish_call(total_loss)

        # [H*B, S, D] -> [H, B, S, D], matching the decoder's stacked output.
        return self._finish_call(
            self._finish(
                pred, self._unfold_heads(accumulated_grad, num_heads), total_loss
            )
        )

    def _chunk_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None,
        *,
        num_heads: int,
        sample_counts: torch.Tensor | None = None,
        global_valid_samples: SampleDenominator | list[SampleDenominator] | None = None,
    ) -> torch.Tensor:
        """Weighted sum of per-head CEs for one folded chunk; records per-head metrics.

        Main head weight is 1.0, MTP heads use ``loss_weights``.
        """
        if num_heads == 1:
            main_valid_samples = (
                global_valid_samples[0]
                if isinstance(global_valid_samples, list)
                else global_valid_samples
            )
            return super()._chunk_loss(
                logits,
                labels,
                global_valid_tokens,
                num_heads=num_heads,
                sample_counts=sample_counts,
                global_valid_samples=main_valid_samples,
            )
        head_weights = [1.0, *self.loss_weights]
        assert len(head_weights) == num_heads, (
            f"{num_heads} head(s) folded into the batch, {len(head_weights)} weight(s)"
        )
        num_mtp = len(self.loss_weights)
        mtp_weight_total = sum(self.loss_weights)
        head_sample_counts = (
            torch.chunk(sample_counts, num_heads, dim=0)
            if sample_counts is not None
            else (None,) * num_heads
        )
        chunk_loss: torch.Tensor | None = None
        for head_idx, (weight, head_logits, head_labels) in enumerate(
            zip(
                head_weights,
                torch.chunk(logits, num_heads, dim=0),
                torch.chunk(labels, num_heads, dim=0),
            )
        ):
            head_valid_samples = (
                global_valid_samples[head_idx]
                if isinstance(global_valid_samples, list)
                else global_valid_samples
            )
            head_loss = (
                super()._chunk_loss(
                    head_logits,
                    head_labels,
                    global_valid_tokens,
                    num_heads=1,
                    sample_counts=head_sample_counts[head_idx],
                    global_valid_samples=head_valid_samples,
                )
                * weight
            )
            detached = head_loss.detach()
            if head_idx == 0:
                self.add_component(self.component_name, detached)
            else:
                # The conditions are the ones ``declared_components`` names
                # under: a zero-weight head contributes nothing and gets no tag.
                if mtp_weight_total != 0.0:
                    self.add_component("loss_mtp", detached / mtp_weight_total)
                if num_mtp > 1 and weight != 0.0:
                    self.add_component(f"loss_mtp{head_idx}", detached / weight)
            chunk_loss = head_loss if chunk_loss is None else chunk_loss + head_loss
        assert chunk_loss is not None
        return chunk_loss

    @staticmethod
    def _reject_sharded_fold(
        t: torch.Tensor, *, dims: tuple[int, ...], name: str
    ) -> None:
        """Refuse a fold that would not be a view of the local data."""
        if not isinstance(t, DTensor):
            return
        mesh_axis_names = t.device_mesh.mesh_dim_names
        sharded = {
            (mesh_axis_names[axis] if mesh_axis_names is not None else axis): p.dim
            for axis, p in enumerate(t.placements)
            if isinstance(p, Shard) and p.dim in dims
        }
        if sharded:
            raise ValueError(
                f"MTPLoss folds the heads into the batch, which needs {name} "
                f"unsharded on tensor dim(s) {dims}; mesh axes {sharded} shard them."
            )

    @staticmethod
    def _unfold_heads(folded: torch.Tensor, num_heads: int) -> torch.Tensor:
        """``[num_heads * B, ...]`` -> ``[num_heads, B, ...]``, undoing the fold."""
        if not isinstance(folded, DTensor):
            return folded.unflatten(0, (num_heads, -1))

        return DTensor.from_local(
            folded.to_local().unflatten(0, (num_heads, -1)),
            folded.device_mesh,
            tuple(
                Shard(p.dim + 1) if isinstance(p, Shard) else p
                for p in folded.placements
            ),
            run_check=False,
        )

    @staticmethod
    def _shift_labels(
        labels: torch.Tensor, k: int, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Main labels rolled left by ``k``; targets with no valid future dropped.

        Beyond the last ``k`` positions, ``positions`` (per-document, restarting
        at 0) also drops targets reached across a document boundary: attention is
        reset there, so the head cannot see the document it would predict.
        """
        shifted = torch.roll(labels, shifts=-k, dims=1)
        shifted[:, -k:] = IGNORE_INDEX
        if positions is None:
            return shifted
        # positions step by one inside a document and restart at 0 across a
        # boundary, so landing k ahead stayed inside iff it advanced by k.
        same_doc = torch.roll(positions, shifts=-k, dims=1) == positions + k
        same_doc[:, -k:] = True  # the wrapped tail is already IGNORE
        return shifted.masked_fill(~same_doc, IGNORE_INDEX)

    def _acceptance_counters(
        self,
        head_argmaxes: list[torch.Tensor],
        labels: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """One microbatch's acceptance counters from this forward's argmaxes.

        ``head_argmaxes[0]`` is the main head; entry ``h`` is MTP head ``h``.
        ``_vs_gt`` compares a head against its own shifted data labels,
        ``_vs_base`` against the main head's argmax rolled to the same target
        slot (the training-time proxy for spec-decode acceptance). Chains are
        cumulative ANDs over heads 1..h. Counters are rank-local sums; the
        monitor owns their accumulation over microbatches and the reduce over
        the loss mesh.
        """
        with torch.no_grad():
            base = head_argmaxes[0]
            per_gt, per_base, per_valid = [], [], []
            chain_gt, chain_base = [], []
            cum_gt: torch.Tensor | None = None
            cum_base: torch.Tensor | None = None
            for h in range(1, len(head_argmaxes)):
                label_h = self._shift_labels(labels, h, positions)
                valid_h = label_h != IGNORE_INDEX
                base_h = torch.roll(base, shifts=-h, dims=1)
                match_gt = (head_argmaxes[h] == label_h) & valid_h
                match_base = (head_argmaxes[h] == base_h) & valid_h
                per_gt.append(match_gt.sum())
                per_base.append(match_base.sum())
                per_valid.append(valid_h.sum())
                cum_gt = match_gt if cum_gt is None else cum_gt & match_gt
                cum_base = match_base if cum_base is None else cum_base & match_base
                chain_gt.append(cum_gt.sum())
                chain_base.append(cum_base.sum())
            return {
                "per_head_correct_gt": torch.stack(per_gt).float(),
                "per_head_correct_base": torch.stack(per_base).float(),
                "per_head_valid": torch.stack(per_valid).float(),
                "chain_correct_gt": torch.stack(chain_gt).float(),
                "chain_correct_base": torch.stack(chain_base).float(),
                "main_valid": (labels != IGNORE_INDEX).sum().float(),
            }
