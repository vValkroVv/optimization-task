# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeAlias

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.experimental import local_map

from torchtitan.config import CompileConfig, Configurable
from torchtitan.config.schema.components.loss import (
    BaseLossConfig,
    ChunkedCELossConfig,
    CrossEntropyLossConfig,
    MSELossConfig,
    MTPLossConfig,
)
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]


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
    Subclasses must implement ``__init__``, call this one first and set
    ``self.fn``.
    """

    fn: LossFunction

    component_name: str = "loss_ce"
    _components: dict[str, torch.Tensor]

    Config: TypeAlias = BaseLossConfig

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
            self.fn = torch.compile(self.fn, backend=compile_config.backend)

    def pack_targets(
        self, labels: torch.Tensor, extra_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        """Fold what this loss needs out of the batch into ``labels``.

        A pipeline schedule microbatches the target tensor and nothing else.
        """
        del extra_kwargs
        return labels

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
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

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = CrossEntropyLossConfig

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        super().__init__(config, compile_config=compile_config)
        self.fn: LossFunction = cross_entropy_loss
        self._maybe_compile(compile_config)


class MSELoss(BaseLoss):
    """MSE loss with sum reduction for Transformer models training (e.g. Flux)."""

    component_name = "mse"

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = MSELossConfig

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
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor, Placement

        self.num_chunks = num_chunks
        self.seq_dim = seq_dim
        self._next_idx = 0
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
        start = self._next_idx * chunk_seq_len
        end = start + chunk_seq_len

        slices = [slice(None)] * self._buffer.ndim
        slices[self.seq_dim] = slice(start, end)
        self._buffer[tuple(slices)] = chunk_grad

        self._next_idx += 1

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


def _fsdp_flags(module: nn.Module) -> tuple[bool, bool]:
    state = module._get_fsdp_state()  # type: ignore[operator]
    groups = getattr(state, "_fsdp_param_groups", None)
    if groups is None:
        group = getattr(state, "_fsdp_param_group", None)
        groups = [group] if group is not None else []
    return state._state_ctx.is_last_backward, all(g.reduce_grads for g in groups)


def _is_plain_fsdp(module: nn.Module) -> bool:
    """Torch FSDP2, excluding the gfsdp backend: GFSDPModule subclasses
    FSDPModule on purpose, and its buffer lifecycle must stay untouched by
    the head-reduction coalescing below."""
    from torchtitan.distributed.gfsdp import GFSDPModule

    return isinstance(module, FSDPModule) and not isinstance(module, GFSDPModule)


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
        unsharded across all chunks (avoiding repeated all-gathers). Reduce-scatter
        fires per-chunk, and FSDP2 accumulates the sharded gradients correctly.

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

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = ChunkedCELossConfig

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        super().__init__(config, compile_config=compile_config)
        self.fn: LossFunction = cross_entropy_loss
        self._maybe_compile(compile_config)
        self.num_chunks = config.num_chunks
        self.lm_head: nn.Module | None = None
        # set by the trainer; only real PP losses use it (probes stay synced)
        self.is_last_accumulation_step = True
        self._head_has_own_fsdp_group = False

    def set_lm_head(self, lm_head: nn.Module) -> None:
        """Set the lm_head module. Must be called before the first __call__."""
        self.lm_head = lm_head
        if _is_plain_fsdp(lm_head):
            # available before lazy init; tied embeddings share the head's group
            self._head_has_own_fsdp_group = lm_head._get_fsdp_state()._modules == (
                lm_head,
            )
            if self._head_has_own_fsdp_group:
                # pipeline_grad_reduce must not drain the loss-deferred partial
                lm_head._loss_managed_grad_reduce = True
            else:
                logger.warning(
                    "Chunked loss head shares its FSDP group with other modules; "
                    "keeping per-microbatch all-reduce synchronization."
                )

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        sync_all_reduce: bool = True,
    ) -> torch.Tensor:
        """Compute chunked cross-entropy loss.

        ``pred`` should be hidden states from model forward with
        ``_skip_lm_head=True``.

        When ``pred`` does not require grad (e.g. validation), runs chunked
        forward only — no per-chunk backward or gradient accumulation.

        ``sync_all_reduce=False`` defers the head's HSDP replica all-reduce
        (torch FSDP only); the PP schedule passes ``False`` to every loss but
        the optimizer step's last real one, which gets ``True`` and drains
        the deferred accumulator.

        Returns a differentiable loss. When ``.backward()`` is called on it
        (either by the trainer or the PP schedule), it triggers backward
        through the decoder via a custom autograd Function.
        """
        pred = self._replicate_on_tp(pred)
        total_loss, accumulated_grad = self._chunked_loss_and_grad(
            pred,
            labels,
            global_valid_tokens,
            component=self.component_name,
            shape_probe=self.is_shape_probe_target(labels),
            sync_all_reduce=sync_all_reduce,
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
        names = pred.device_mesh.mesh_dim_names if isinstance(pred, DTensor) else None
        if names is None or "tp" not in names:
            return pred
        tp = names.index("tp")
        if isinstance(pred.placements[tp], Replicate):
            return pred
        placements = list(pred.placements)
        placements[tp] = Replicate()
        return pred.redistribute(pred.device_mesh, tuple(placements))

    def _chunked_loss_and_grad(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        *,
        num_heads: int = 1,
        component: str | None = None,
        argmax_sink: list[torch.Tensor] | None = None,
        shape_probe: bool = False,
        sync_all_reduce: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run chunked ``lm_head`` + cross-entropy over the sequence.

        Returns ``(total_loss, grad w.r.t. hidden_states)``, the gradient
        ``None`` in validation. ``num_heads`` are the heads ``MTPLoss`` folded
        into the batch axis, split back out in ``_chunk_loss``; ``component``
        names the metrics term, ``None`` leaving that to the caller. For a
        torch-FSDP head in its own group, ``sync_all_reduce=False`` defers
        the HSDP replica all-reduce across gradient-accumulation steps.
        """
        num_chunks = self.num_chunks
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedCELoss"
        fsdp_enabled = isinstance(lm_head, FSDPModule)
        # gfsdp heads keep main's flag-transparent path untouched (its buffer
        # lifecycle is managed outside; see GFSDPModule).
        plain_fsdp = fsdp_enabled and _is_plain_fsdp(lm_head)
        if not self._head_has_own_fsdp_group:
            sync_all_reduce = True

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

        # Keep the weight unsharded across chunks, coalesce the per-chunk grad
        # syncs into one reduce-scatter, and give back what the caller set.
        was_last_backward = was_sync = True
        if fsdp_enabled:
            was_last_backward, was_sync = _fsdp_flags(lm_head)
            lm_head.set_reshard_after_forward(False)
            lm_head.set_reshard_after_backward(False)
            lm_head.set_requires_gradient_sync(False, recurse=False)
            lm_head.set_is_last_backward(False)

        try:
            last_idx = len(h_chunks) - 1
            for i, (h_chunk, label_chunk) in enumerate(zip(h_chunks, label_chunks)):
                # torch FSDP: sync unconditionally, as before the gfsdp
                # rework — the stock no-sync left by the PP backward would
                # otherwise accumulate an unsharded fp32 head gradient.
                if fsdp_enabled and (was_sync or plain_fsdp) and i == last_idx:
                    lm_head.set_requires_gradient_sync(True, recurse=False)
                    if plain_fsdp:
                        # the sync setter also arms AR, so override it last
                        lm_head.set_requires_all_reduce(sync_all_reduce, recurse=False)

                logits = lm_head(h_chunk)
                if argmax_sink is not None:
                    # Greedy predictions for the acceptance counters, taken from
                    # the logits this loop already materializes; cat over the
                    # sequence restores the full [B, S] argmax per head.
                    with torch.no_grad():
                        argmax_sink.append(logits.detach().argmax(dim=-1))

                chunk_loss = self._chunk_loss(
                    logits, label_chunk, global_valid_tokens, num_heads=num_heads
                )
                total_loss = total_loss + chunk_loss.detach()
                if component is not None:
                    self.add_component(component, chunk_loss.detach())

                if requires_grad and shape_probe:
                    # PP shape probe runs on garbage activations: take the grad
                    # w.r.t. the input only, so nothing lands in lm_head's grads.
                    (chunk_grad,) = torch.autograd.grad(chunk_loss, h_chunk)
                    grad_accumulator.add(chunk_grad)
                elif requires_grad:
                    chunk_loss.backward()
                    assert h_chunk.grad is not None
                    grad_accumulator.add(h_chunk.grad)
                    h_chunk.grad = None
        finally:
            if fsdp_enabled:
                lm_head.set_is_last_backward(was_last_backward)
                lm_head.set_reshard_after_forward(True)
                lm_head.set_reshard_after_backward(True)
                lm_head.set_requires_gradient_sync(
                    True if plain_fsdp else was_sync, recurse=False
                )
                if plain_fsdp:
                    lm_head.set_requires_all_reduce(sync_all_reduce, recurse=False)
                lm_head.reshard()

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
    ) -> torch.Tensor:
        """Loss for one chunk of ``lm_head`` output."""
        chunk_loss = self.fn(logits, labels)
        if global_valid_tokens is not None:
            chunk_loss = chunk_loss / global_valid_tokens
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
        return _DecoderOutputGradientBackProp.apply(
            hidden_states, accumulated_grad, total_loss
        )


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
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        accumulated_grad: torch.Tensor,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(accumulated_grad)
        return loss.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (accumulated_grad,) = ctx.saved_tensors
        # Return accumulated_grad as the gradient for hidden_states.
        # Autograd then propagates this through hidden_states' existing
        # decoder graph — equivalent to hidden_states.backward(accumulated_grad)
        # but expressed as a return value so autograd handles the traversal
        # in a single pass (no "backward through graph twice" error).
        # Note: this is not safe if downstream accidentally runs tensor ops after
        # the loss returns, which would produce a non-trivial grad_output that we need
        # to properly handle. The complicated part is that grad_output might not be
        # on the same device mesh as accumlated_grad.
        return accumulated_grad, None, None


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
    ``main_CE + sum_k loss_weights[k] * CE_k``. Every head is normalized by
    ``global_valid_tokens`` (the main head's valid-token count); the per-head
    difference of ``k`` tokens is absorbed by ``loss_weights``.

    The trainer wires ``lm_head`` / ``_skip_lm_head`` for any ``ChunkedCELoss``,
    so MTPLoss is set up automatically.
    """

    component_name = "loss_ce"

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = MTPLossConfig

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        super().__init__(config, compile_config=compile_config)
        self.loss_weights = list(config.loss_weights)
        self.acceptance_sinks: list[list[dict[str, torch.Tensor]]] = []

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

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        sync_all_reduce: bool = True,
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
            return super().__call__(
                pred, labels, global_valid_tokens, sync_all_reduce=sync_all_reduce
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
        total_loss, accumulated_grad = self._chunked_loss_and_grad(
            pred.flatten(0, 1),
            torch.cat(head_labels, dim=0),
            global_valid_tokens,
            num_heads=num_heads,
            argmax_sink=sink,
            shape_probe=shape_probe,
            sync_all_reduce=sync_all_reduce,
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
    ) -> torch.Tensor:
        """Weighted sum of per-head CEs for one folded chunk; records per-head metrics.

        Main head weight is 1.0, MTP heads use ``loss_weights``.
        """
        if num_heads == 1:
            return super()._chunk_loss(
                logits, labels, global_valid_tokens, num_heads=num_heads
            )
        head_weights = [1.0, *self.loss_weights]
        assert len(head_weights) == num_heads, (
            f"{num_heads} head(s) folded into the batch, {len(head_weights)} weight(s)"
        )
        num_mtp = len(self.loss_weights)
        mtp_weight_total = sum(self.loss_weights)
        chunk_loss: torch.Tensor | None = None
        for head_idx, (weight, head_logits, head_labels) in enumerate(
            zip(
                head_weights,
                torch.chunk(logits, num_heads, dim=0),
                torch.chunk(labels, num_heads, dim=0),
            )
        ):
            head_loss = (
                super()._chunk_loss(
                    head_logits, head_labels, global_valid_tokens, num_heads=1
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
        # functional padding: slice assignment lowers to DTensor-unsupported
        # in-place ops under loss parallel
        tail = torch.full_like(labels[:, -k:], IGNORE_INDEX)
        shifted = torch.cat([labels[:, k:], tail], dim=1)
        if positions is None:
            return shifted
        # positions step by one inside a document and restart at 0 across a
        # boundary, so landing k ahead stayed inside iff it advanced by k.
        # The wrapped tail of the mask is garbage, but the tail is already
        # IGNORE, so the extra masked_fill there is a no-op.
        same_doc = torch.roll(positions, shifts=-k, dims=1) == positions + k
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
