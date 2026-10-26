from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias

import torch
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.config.schema.callbacks.base import CallbackConfig
from torchtitan.callbacks.metrics.emitters import MetricEmitter
from torchtitan.config.schema.protocols.types import StrEnum


class Event(StrEnum):
    """Every lifecycle point the trainer emits. Nothing else is dispatched."""

    fit_start = "fit_start"
    after_load = "after_load"
    batch_start = "batch_start"
    before_dataloader = "before_dataloader"
    after_dataloader = "after_dataloader"
    microbatch_start = "microbatch_start"
    after_forward = "after_forward"
    before_optimizer_step = "before_optimizer_step"
    after_optimizer_step = "after_optimizer_step"
    batch_end = "batch_end"
    validation_start = "validation_start"
    validation_end = "validation_end"
    after_checkpoint_save = "after_checkpoint_save"
    fit_end = "fit_end"


# Where the step's records are published. Records accumulate across the
# events of a step and leave at its end, so the publishing collectives run
# exactly once per step and at the same point on every pipeline stage --
# publishing per event would multiply the pp gather by the event count, and
# publishing only when a rank happens to have records would desynchronise it.
_FLUSH_EVENTS: frozenset[Event] = frozenset({Event.batch_end, Event.fit_end})

_PAYLOAD: Mapping[Event, frozenset[str]] = {
    Event.after_dataloader: frozenset({"input_dict", "labels"}),
    Event.microbatch_start: frozenset({"microbatch_idx"}),
    Event.after_forward: frozenset(
        {"input_dict", "labels", "loss", "global_valid_tokens", "microbatch_idx"}
    ),
    Event.before_optimizer_step: frozenset({"global_valid_tokens"}),
    Event.after_optimizer_step: frozenset({"grad_norm", "global_valid_tokens"}),
    Event.batch_end: frozenset({"loss", "grad_norm", "global_valid_tokens"}),
}


# Events emitted from inside ``train_step``. A trainer whose train_step does
# not emit them (Flux) refuses callbacks that implement any of these.
STEP_EVENTS: frozenset[Event] = frozenset(
    {
        Event.batch_start,
        Event.before_dataloader,
        Event.after_dataloader,
        Event.microbatch_start,
        Event.after_forward,
        Event.before_optimizer_step,
        Event.after_optimizer_step,
        Event.batch_end,
    }
)


class Mutation(StrEnum):
    """The axes of a training step a :class:`Mutator` may change."""

    batch = "batch"
    lr = "lr"
    weights = "weights"
    grads = "grads"
    optimizer_state = "optimizer_state"
    # Gating / routing mode of MoE modules (no tensor write; not auditable).
    routing = "routing"
    # ``ResidualScale.alpha`` (non-persistent buffer).
    residual_scale = "residual_scale"


# --------------------------------------------------------------------------- #
# State views
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class MutableState:
    """What a :class:`Mutator` sees: the trainer, plus the event's payload.

    ``input_dict``/``labels`` are read back by the trainer after
    ``after_dataloader`` — replacing them here IS the batch mutation channel.
    """

    trainer: Any
    event: Event | None = None
    input_dict: dict[str, torch.Tensor] | None = None
    labels: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    grad_norm: torch.Tensor | None = None
    global_valid_tokens: torch.Tensor | None = None
    microbatch_idx: int | None = None
    metrics: dict[str, Any] | None = None
    # Filled by the callbacks of one event, emitted once when the event ends.
    _records: list = field(default_factory=list, repr=False)

    @property
    def step(self) -> int:
        return self.trainer.step

    def record(self, name: str, value: Any, reduce: Any = None) -> None:
        """Hand one metric to the step's emitter, as an observer does.

        A mutator holds the trainer and could reach the metrics processor
        itself; then its numbers would miss the stacking, the host copy and
        the pipeline-stage gather the emitter does once for the whole step.
        """
        from torchtitan.callbacks.metrics.sample import MetricRecord
        from torchtitan.config.schema.callbacks.metrics.sample import ReduceConfig

        self._records.append(
            MetricRecord(
                name=name,
                value=value,
                step=self.step,
                reduce=reduce if reduce is not None else ReduceConfig(),
            )
        )

    def record_all(self, metrics: dict[str, Any]) -> None:
        for name, value in metrics.items():
            self.record(name, value)


@dataclass(frozen=True, slots=True)
class OptimizerGroupView:
    """One optimizer param group, read-only: hyperparameters as a copy."""

    optimizer_name: str
    optimizer_type: type
    part_index: int
    group_index: int
    hparams: Mapping[str, Any]
    params: tuple[torch.Tensor, ...]


def _read_only(mapping: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """A view on the trainer's dict that cannot be rebound through."""
    return None if mapping is None else MappingProxyType(mapping)


@dataclass(frozen=True, slots=True)
class ObserverState:
    """What an :class:`Observer` sees.

    Built by the handler once per event, after every mutator of that event
    ran. There is no ``trainer`` field, no optimizer or scheduler object, no
    metrics processor: hyperparameters come back as copies, and the only way
    out is :meth:`record`. Modules are exposed as such -- hooks and
    weight reads need them -- so an observer determined to write through one
    still can; what the type says is that it must not.
    """

    event: Event | None
    step: int
    model_parts: tuple[torch.nn.Module, ...]
    parallel_dims: Any
    device: Any
    config: Any
    input_dict: Mapping[str, torch.Tensor] | None = None
    labels: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    grad_norm: torch.Tensor | None = None
    global_valid_tokens: torch.Tensor | None = None
    microbatch_idx: int | None = None
    metrics: Mapping[str, Any] | None = None
    last_saved_checkpoint: str | None = None
    # The trainer's device counter itself (or an int): converting it here would
    # be a host sync on every event, so readers call int() when they need it.
    ntokens_seen: Any = 0
    gradient_accumulation_steps: int = 1
    loss_fn_type: type | None = None
    # MTP only: the head weights the loss mixes with. None for other losses.
    mtp_loss_weights: tuple[float, ...] | None = None
    _optimizers: tuple[Any, ...] = field(default=(), repr=False)
    _emit_scalars: Callable[[int, Mapping[str, Any]], None] | None = field(
        default=None, repr=False
    )
    # The loss object, when it publishes MTP acceptance counters.
    _acceptance_source: Any = field(default=None, repr=False)
    # Filled by the callbacks of one event, emitted once when the event ends.
    _records: list = field(default_factory=list, repr=False)

    # -- reads ------------------------------------------------------------- #

    def optimizer_groups(self) -> tuple[OptimizerGroupView, ...]:
        """Every param group of every optimizer, hyperparameters copied."""
        views: list[OptimizerGroupView] = []
        for part_index, optimizer in enumerate(self._optimizers):
            for group_index, group in enumerate(optimizer.param_groups):
                hparams = MappingProxyType(
                    {key: value for key, value in group.items() if key != "params"}
                )
                views.append(
                    OptimizerGroupView(
                        optimizer_name=type(optimizer).__name__,
                        optimizer_type=type(optimizer),
                        part_index=part_index,
                        group_index=group_index,
                        hparams=hparams,
                        params=tuple(group.get("params", ())),
                    )
                )
        return tuple(views)

    def optimizer_state(self, param: torch.Tensor) -> Mapping[str, Any]:
        """The optimizer's state entries for ``param`` (moments, step), read-only."""
        for optimizer in self._optimizers:
            state = optimizer.state.get(param)
            if state:
                return MappingProxyType(state)
        return MappingProxyType({})

    # -- the only ways to act ---------------------------------------------- #

    def record(self, name: str, value: Any, reduce: Any = None) -> None:
        """Hand one metric to the step's emitter.

        ``value`` may stay a tensor: the emitter stacks everything the step
        produced and copies it to the host once, and gathers over pipeline
        stages once. A callback that converts and publishes on its own pays
        for both again, per metric.
        """
        from torchtitan.callbacks.metrics.sample import MetricRecord
        from torchtitan.config.schema.callbacks.metrics.sample import ReduceConfig

        self._records.append(
            MetricRecord(
                name=name,
                value=value,
                step=self.step,
                reduce=reduce if reduce is not None else ReduceConfig(),
            )
        )

    def record_all(self, metrics: Mapping[str, Any]) -> None:
        """``record`` for a whole mapping."""
        for name, value in metrics.items():
            self.record(name, value)

    def publish(self, step: int, metrics: Mapping[str, Any]) -> None:
        """The sink, for the emitter only: reduced scalars go to the logger."""

        if self._emit_scalars is None:
            raise RuntimeError("this ObserverState has no metrics sink")
        if metrics:
            self._emit_scalars(step, dict(metrics))

    def subscribe_acceptance(self, sink: list) -> None:
        """Have the MTP loss append its acceptance counters into ``sink``."""
        if self._acceptance_source is None:
            raise RuntimeError("this loss publishes no acceptance counters")
        sinks = self._acceptance_source.acceptance_sinks
        if not any(existing is sink for existing in sinks):
            sinks.append(sink)

    def unsubscribe_acceptance(self, sink: list) -> None:
        if self._acceptance_source is not None:
            self._acceptance_source.acceptance_sinks = [
                s for s in self._acceptance_source.acceptance_sinks if s is not sink
            ]

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_mutable(cls, mutable: MutableState) -> ObserverState:
        """Snapshot the trainer after the event's mutators ran."""
        trainer = mutable.trainer
        loss_fn = trainer.loss_fn
        loss_weights = getattr(loss_fn, "loss_weights", None)
        return cls(
            event=mutable.event,
            step=int(trainer.step),
            model_parts=tuple(trainer.model_parts),
            parallel_dims=trainer.parallel_dims,
            device=trainer.device,
            config=trainer.config,
            # Proxied, not copied: an observer reads the batch it trains on, but
            # replacing an entry here is the mutator's channel, not its own.
            input_dict=_read_only(mutable.input_dict),
            labels=mutable.labels,
            loss=mutable.loss,
            grad_norm=mutable.grad_norm,
            global_valid_tokens=mutable.global_valid_tokens,
            microbatch_idx=mutable.microbatch_idx,
            metrics=_read_only(mutable.metrics),
            last_saved_checkpoint=trainer.checkpointer.last_saved_path,
            ntokens_seen=trainer.ntokens_seen,
            gradient_accumulation_steps=int(trainer.gradient_accumulation_steps),
            loss_fn_type=type(loss_fn) if loss_fn is not None else None,
            mtp_loss_weights=(
                tuple(float(w) for w in loss_weights)
                if loss_weights is not None
                else None
            ),
            _optimizers=tuple(trainer.optimizers.optimizers),
            _emit_scalars=trainer.metrics_processor.log_extra_metrics,
            _acceptance_source=(
                loss_fn if hasattr(loss_fn, "acceptance_sinks") else None
            ),
            # The same buffer the mutators wrote into: one event, one emission.
            _records=mutable._records,
        )


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #


class Callback:
    """Base of both kinds. Not instantiated directly: subclass Observer or Mutator."""

    Config: TypeAlias = CallbackConfig

    def stateful_objects(self) -> dict[str, Stateful]:
        return {}

    def described_children(self) -> Mapping[str, str]:
        return {}

    def fit_start(self, state: Any) -> None:
        pass

    def after_load(self, state: Any) -> None:
        pass

    def batch_start(self, state: Any) -> None:
        pass

    def before_dataloader(self, state: Any) -> None:
        pass

    def after_dataloader(self, state: Any) -> None:
        pass

    def microbatch_start(self, state: Any) -> None:
        pass

    def after_forward(self, state: Any) -> None:
        pass

    def before_optimizer_step(self, state: Any) -> None:
        pass

    def after_optimizer_step(self, state: Any) -> None:
        pass

    def batch_end(self, state: Any) -> None:
        pass

    def validation_start(self, state: Any) -> None:
        pass

    def validation_end(self, state: Any) -> None:
        pass

    def after_checkpoint_save(self, state: Any) -> None:
        pass

    def fit_end(self, state: Any) -> None:
        pass

    def close(self) -> None:
        pass


class Observer(Callback):
    """Reads the step, publishes metrics, owns only its own state.

    Every hook receives an :class:`ObserverState`. Writing into the training
    state is not a rule here but an impossibility: the view carries no
    trainer, no optimizer, no scheduler.
    """


class Aggregator(Observer):
    """Observes what the other callbacks of the event produced, not the step.

    The point of the type is that the dependency stops living in the order of a
    preset's dict: an aggregator listed first still runs last.

    Same contract between aggregators as between observers -- they read what
    the event produced globally, never each other's output.
    """


class Mutator(Callback):
    """Changes the step. Declares which axes, and gets the trainer to do it.

    ``mutates`` on the class is the declaration, and it is conservative:
    every hook the class overrides counts as a writer of every declared
    axis. A mutator whose hooks differ -- one writes, the others only keep
    books -- may instead mark the writing hooks with :func:`mutates`; then
    only the marked hooks count (in the conflict check and in
    ``describe()``) and the class attribute is derived as their union.
    """

    mutates: ClassVar[frozenset[Mutation]] = frozenset()
    _axes_on_hooks: ClassVar[bool] = False

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if abstract:
            # A framework base (PostMutator) declares no axes of its own;
            # every concrete subclass still must.
            return
        on_hooks: dict[Event, frozenset[Mutation]] = {}
        for event in Event:
            axes = getattr(getattr(cls, event.value), "__mutates__", None)
            if axes:
                on_hooks[event] = axes
        on_class = cls.__dict__.get("mutates") or frozenset()
        if on_class and on_hooks:
            raise TypeError(
                f"{cls.__name__}: mutation axes are declared both on the "
                f"class and on hooks ({', '.join(e.value for e in on_hooks)});"
                " one style per class -- with marked hooks the class "
                "attribute is derived."
            )
        if on_hooks:
            # The batch channel exists at exactly one event: the trainer
            # reads the mutated fields back only after ``after_dataloader``.
            for event, axes in on_hooks.items():
                if Mutation.batch in axes and event is not Event.after_dataloader:
                    raise TypeError(
                        f"{cls.__name__}.{event.value} declares "
                        "Mutation.batch, but the trainer reads the batch "
                        "back only after after_dataloader — anywhere else "
                        "the mutation is a silent no-op."
                    )
            declared = frozenset().union(*on_hooks.values())
            cls._axes_on_hooks = True
        elif on_class:
            declared = frozenset(on_class)
            cls._axes_on_hooks = False
        elif getattr(cls, "mutates", frozenset()):
            # A subclass that redeclares nothing inherits the parent's
            # declaration, whichever style, wholesale.
            return
        else:
            raise TypeError(
                f"{cls.__name__}: a Mutator must declare a non-empty "
                f"`mutates` (subset of {[m.value for m in Mutation]}), or "
                "mark its writing hooks with @mutates(...); a callback that "
                "changes nothing is an Observer."
            )
        bad = {axis for axis in declared if not isinstance(axis, Mutation)}
        if bad:
            raise TypeError(f"{cls.__name__}: unknown mutation axes {bad}")
        cls.mutates = declared


def mutates(*axes: Mutation):
    """Mark the Mutator hook that writes ``axes``, e.g. ``@mutates(Mutation.lr)``"""

    def mark(hook):
        hook.__mutates__ = frozenset(axes)
        return hook

    return mark


class PostMutator(Mutator, abstract=True):
    """A mutator that steers from what the observers of the event computed.

    Same contract as :class:`Mutator` -- full ``MutableState``, declared axes
    -- but the handler runs it after the observers and before the aggregators.
    The observers of the event see the step BEFORE this mutator acted: its
    change is the event's last word, visible from the next event on.
    """


def _overrides(callback: Callback, event: Event) -> bool:
    return getattr(type(callback), event.value) is not getattr(Callback, event.value)


def _hook_mutates(callback: Callback, event: Event) -> frozenset[Mutation]:
    """The axes ``callback`` writes at ``event``.

    A marked hook speaks for itself. Otherwise the class declaration decides:
    a class that marks its hooks makes the unmarked ones bookkeeping, a
    classic class makes every overridden hook a conservative writer of all
    its axes.
    """
    if not isinstance(callback, Mutator):
        return frozenset()
    marked = getattr(getattr(type(callback), event.value), "__mutates__", None)
    if marked is not None:
        return marked
    if type(callback)._axes_on_hooks:
        return frozenset()
    return callback.mutates if _overrides(callback, event) else frozenset()


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #


class CallbackHandler:
    """Resolves the schedule once, then dispatches events along it."""

    def __init__(self, callbacks: Sequence[Callback]) -> None:
        stray = [
            type(c).__name__
            for c in callbacks
            if not isinstance(c, (Observer, Mutator))
        ]
        if stray:
            raise TypeError(f"every callback is an Observer or a Mutator; not: {stray}")

        both = [
            type(c).__name__
            for c in callbacks
            if isinstance(c, Mutator) and isinstance(c, Observer)
        ]
        if both:
            raise TypeError(
                f"{both}: a callback is either a Mutator or an Observer, never "
                "both -- it would run twice per event, once per group. A "
                "mutator that also publishes metrics uses MutableState.record()."
            )
        self.callbacks: list[Callback] = list(callbacks)
        self._emitter = MetricEmitter()
        # The step's records, from every event of it, drained at batch_end.
        self._pending_records: list = []
        # Per event: mutators first, then observers, then aggregators.
        self.schedule: dict[Event, tuple[tuple[Callback, ...], ...]] = {
            event: self._order(self.callbacks, event) for event in Event
        }
        self._check_mutation_conflicts()

    @staticmethod
    def _order(
        callbacks: Sequence[Callback], event: Event
    ) -> tuple[tuple[Callback, ...], ...]:
        """Four groups by type; inside a group, the order they were built in.

        Nothing is sorted: the type decides which group a callback is in, and
        ``CallbackManager`` decides the order within one. So the only thing a
        preset's key order can change is the order of callbacks that are
        interchangeable by contract.
        """
        running = [c for c in callbacks if _overrides(c, event)]
        return (
            tuple(
                c
                for c in running
                if isinstance(c, Mutator) and not isinstance(c, PostMutator)
            ),
            tuple(
                c
                for c in running
                if isinstance(c, Observer) and not isinstance(c, Aggregator)
            ),
            tuple(c for c in running if isinstance(c, PostMutator)),
            tuple(c for c in running if isinstance(c, Aggregator)),
        )

    def _check_mutation_conflicts(self) -> None:
        for event, (mutators, _, post_mutators, _aggregators) in self.schedule.items():
            owner: dict[Mutation, str] = {}
            for callback in (*mutators, *post_mutators):
                for axis in _hook_mutates(callback, event):
                    if axis in owner:
                        raise ValueError(
                            f"{event}: {owner[axis]} and {type(callback).__name__} "
                            f"both mutate '{axis}'; the order between them would "
                            "decide the training trajectory. Keep one, or make the "
                            "dependency explicit in one callback."
                        )
                    owner[axis] = type(callback).__name__

        for callback in self.callbacks:
            if (
                isinstance(callback, Mutator)
                and Mutation.batch in callback.mutates
                and not _overrides(callback, Event.after_dataloader)
            ):
                raise ValueError(
                    f"{type(callback).__name__} declares Mutation.batch but does "
                    "not implement after_dataloader — the only event whose batch "
                    "the trainer reads back. Anywhere else the mutation would be "
                    "a silent no-op."
                )

    @staticmethod
    def _check_batch(state: MutableState, mutators: tuple[Callback, ...]) -> None:
        """The event's batch mutators must leave a complete, consistent batch.

        The trainer trains on what ``after_dataloader`` returns, so a mutator
        that dropped a field, or replaced only one of the two (labels cut to
        one sequence length, inputs left at another), is caught here with the
        suspects named — not later as a bare AssertionError or a shape error
        inside the forward.
        """
        names = [type(m).__name__ for m in mutators]
        if state.input_dict is None or state.labels is None:
            raise ValueError(
                "after_dataloader returned "
                f"input_dict={type(state.input_dict).__name__}, "
                f"labels={type(state.labels).__name__}; a batch mutator "
                f"dropped a field. Mutators of the event: {names}."
            )
        seq_len = state.labels.shape[-1]
        for key, tensor in state.input_dict.items():
            if tensor.dim() >= 2 and tensor.shape[1] != seq_len:
                raise ValueError(
                    f"after_dataloader returned input_dict[{key!r}] with "
                    f"sequence length {tensor.shape[1]} but labels with "
                    f"{seq_len}; batch mutators ({names}) must replace the "
                    "fields consistently."
                )

    # -- dispatch ---------------------------------------------------------- #

    def emit(self, event: Event, trainer: Any, **fields: Any) -> MutableState:
        """Run ``event`` along the schedule.

        Returns the mutable state so the trainer can read back what the
        mutators changed (the batch).
        """
        undeclared = fields.keys() - _PAYLOAD.get(event, frozenset())
        if undeclared:
            raise ValueError(
                f"{event}: {sorted(undeclared)} is not this event's payload "
                f"(it carries {sorted(_PAYLOAD.get(event, ())) or 'nothing'}). "
                "A field an event should carry is declared in _PAYLOAD, not "
                "passed ad hoc."
            )
        records = self._pending_records
        mutable = MutableState(trainer=trainer, event=event, _records=records, **fields)
        mutators, observers, post_mutators, aggregators = self.schedule[event]
        for callback in mutators:
            getattr(callback, event.value)(mutable)
        if event is Event.after_dataloader and mutators:
            self._check_batch(mutable, mutators)
        view: ObserverState | None = None
        if observers or aggregators:
            view = ObserverState.from_mutable(mutable)
        for callback in observers:
            getattr(callback, event.value)(view)
        for callback in post_mutators:
            getattr(callback, event.value)(mutable)
        for callback in aggregators:
            getattr(callback, event.value)(view)
        if event in _FLUSH_EVENTS and self.callbacks:
            if view is None:
                view = ObserverState.from_mutable(mutable)
            self._emitter.emit(
                records, parallel_dims=view.parallel_dims, publish=view.publish
            )
            records.clear()
        return mutable

    # -- introspection ----------------------------------------------------- #

    def describe(self) -> str:
        """The resolved order per event, as the trainer logs it."""
        lines = [
            "callbacks schedule (mutators -> observers -> post-mutators -> "
            "aggregators, in CallbackManager order)"
        ]
        width = max(len(event.value) for event in Event)
        for event in Event:
            groups = self.schedule[event]
            if not any(groups):
                continue
            parts = []
            for callback in (c for group in groups for c in group):
                name = type(callback).__name__
                # The axes THIS hook writes; a mutator's bookkeeping hook
                # shows bare, which is the truth of it.
                axes = _hook_mutates(callback, event)
                if axes:
                    name += "[" + ",".join(sorted(m.value for m in axes)) + "]"
                parts.append(name)
            lines.append(f"  {event.value:<{width}}  " + " -> ".join(parts))
        for callback in self.callbacks:
            children = callback.described_children()
            if children:
                inner = ", ".join(f"{key}={name}" for key, name in children.items())
                lines.append(f"  {type(callback).__name__} runs: {inner}")
        return "\n".join(lines)

    def step_event_hooks(self) -> list[str]:
        """The hooks living on step events, as "Class.event".

        What a trainer whose train_step does not emit those events checks
        itself against: every name here would silently never run there.
        """
        hooks = []
        for callback in self.callbacks:
            for event in sorted(STEP_EVENTS, key=lambda e: e.value):
                if _overrides(callback, event):
                    hooks.append(f"{type(callback).__name__}.{event.value}")
        return hooks

    def close(self) -> None:
        for callback in self.callbacks:
            callback.close()

    def stateful_objects(self) -> dict[str, Stateful]:
        states: dict[str, Stateful] = {}
        for callback in self.callbacks:
            for key, value in callback.stateful_objects().items():
                if not isinstance(value, Stateful):
                    raise TypeError(
                        f"{type(callback).__name__} put a "
                        f"{type(value).__name__} under checkpoint key '{key}'; "
                        "stateful_objects() must return Stateful objects "
                        "(state_dict/load_state_dict)."
                    )
                if key in states:
                    raise ValueError(
                        f"Checkpoint key '{key}' is claimed by two callbacks: "
                        f"{type(states[key]).__name__} and "
                        f"{type(value).__name__}."
                    )
                states[key] = value
        return states

    def __iter__(self) -> Iterator[Callback]:
        return iter(self.callbacks)
