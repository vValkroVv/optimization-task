# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Observer/Mutator contract and the handler's schedule.

Mutators run before observers on every event, post-mutators after them and
before the aggregators; inside a group the order is the one CallbackManager
built them in; two mutators of one axis on one event are rejected at
construction; an observer never sees the trainer.
"""

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchtitan.trainer as trainer_mod
from torchtitan.distributed.backends.fsdp2 import FSDP2Backend
from torchtitan.callbacks.builtin.monitoring import MonitoringCallback
from torchtitan.callbacks.base import (
    Aggregator,
    mutates,
    Callback,
    CallbackHandler,
    Event,
    MutableState,
    Mutation,
    Mutator,
    Observer,
    ObserverState,
    PostMutator,
    STEP_EVENTS,
)
from tests.unit_tests.callbacks.trainer_stub import snapshot


class _Log:
    def __init__(self):
        self.calls: list[str] = []


class _Reader(Observer):
    def __init__(self, log, name):
        self.log, self.name = log, name

    def batch_end(self, state):
        assert isinstance(state, ObserverState)
        self.log.calls.append(self.name)


class _Drainer(Aggregator):
    """Reads what the callbacks of the event produced, like TimersMonitor."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def batch_end(self, state):
        assert isinstance(state, ObserverState)
        self.log.calls.append(self.name)


class _LrMutator(Mutator):
    mutates = frozenset({Mutation.lr})

    def __init__(self, log, name):
        self.log, self.name = log, name

    def batch_end(self, state):
        assert isinstance(state, MutableState)
        self.log.calls.append(self.name)


class _BatchMutator(Mutator):
    mutates = frozenset({Mutation.batch})

    def __init__(self, log, name):
        self.log, self.name = log, name

    def after_dataloader(self, state):
        # The one event whose batch the trainer reads back; a {batch} mutator
        # without it is rejected at handler construction.
        self.log.calls.append(f"{self.name}@after_dataloader")

    def batch_end(self, state):
        self.log.calls.append(self.name)


class _SteeringMutator(PostMutator):
    """Steers from what an observer computed, like DynamicBlockLRRescaler."""

    mutates = frozenset({Mutation.lr})

    def __init__(self, log, name):
        self.log, self.name = log, name

    def batch_end(self, state):
        assert isinstance(state, MutableState)
        self.log.calls.append(self.name)


def _trainer(**extra):
    return SimpleNamespace(
        step=3, model_parts=[], parallel_dims=None, device="cpu", config=None, **extra
    )


def test_mutators_run_before_observers_whatever_the_registration_order():
    log = _Log()
    handler = CallbackHandler(
        [
            _Reader(log, "r1"),
            _LrMutator(log, "m_lr"),
            _Reader(log, "r2"),
            _BatchMutator(log, "m_batch"),
        ]
    )
    handler.emit(Event.batch_end, _trainer())
    assert log.calls == ["m_lr", "m_batch", "r1", "r2"]


def test_post_mutators_run_after_observers_and_before_aggregators():
    # A post-mutator reads an observer's finished per-step result, so it runs
    # after the observers; what it records still leaves with this event's
    # aggregator (emitter) drain, so it runs before those — wherever the
    # preset listed it.
    log = _Log()
    handler = CallbackHandler(
        [
            _SteeringMutator(log, "post"),
            _Drainer(log, "agg"),
            _Reader(log, "obs"),
            _BatchMutator(log, "mut"),
        ]
    )
    handler.emit(Event.batch_end, _trainer())
    assert log.calls == ["mut", "obs", "post", "agg"]


def test_a_mutator_and_a_post_mutator_of_one_axis_still_conflict():
    log = _Log()
    with pytest.raises(ValueError, match="both mutate 'lr'"):
        CallbackHandler([_LrMutator(log, "early"), _SteeringMutator(log, "late")])


def test_a_group_runs_in_the_order_it_was_built():
    log = _Log()
    handler = CallbackHandler(
        [_Reader(log, "a"), _Reader(log, "b"), _Reader(log, "c"), _Reader(log, "d")]
    )
    handler.emit(Event.batch_end, _trainer())
    assert log.calls == ["a", "b", "c", "d"]


def test_two_mutators_of_one_axis_on_one_event_are_rejected():
    log = _Log()
    with pytest.raises(ValueError, match="both mutate 'lr'"):
        CallbackHandler([_LrMutator(log, "a"), _LrMutator(log, "b")])


def test_a_mutator_must_declare_what_it_mutates():
    with pytest.raises(TypeError, match="non-empty `mutates`"):

        class _Silent(Mutator):  # noqa: F811
            pass


def test_a_callback_must_be_an_observer_or_a_mutator():
    class _Bare(Callback):
        pass

    with pytest.raises(TypeError, match="Observer or a Mutator"):
        CallbackHandler([_Bare()])


def test_a_batch_mutator_must_implement_after_dataloader():
    # The trainer reads the mutated batch back at exactly one event; a {batch}
    # mutator hooked anywhere else would be a silent no-op that describe()
    # still lists as Foo[batch].
    class _Elsewhere(Mutator):
        mutates = frozenset({Mutation.batch})

        def batch_end(self, state):
            pass

    with pytest.raises(ValueError, match="does not implement after_dataloader"):
        CallbackHandler([_Elsewhere()])


def test_batch_mutators_must_leave_a_complete_consistent_batch():
    # The guard lives in emit, at the one event whose batch the trainer reads
    # back, and raises ValueError (an assert vanishes under python -O and
    # names nobody): a dropped field and a one-sided replacement both name
    # the event's mutators.
    class _Dropper(Mutator):
        mutates = frozenset({Mutation.batch})

        def after_dataloader(self, state):
            state.labels = None

    with pytest.raises(ValueError, match="_Dropper"):
        CallbackHandler([_Dropper()]).emit(
            Event.after_dataloader,
            _trainer(),
            input_dict={"input": torch.zeros(2, 8)},
            labels=torch.zeros(2, 8),
        )

    class _Shrinker(Mutator):
        mutates = frozenset({Mutation.batch})

        def after_dataloader(self, state):
            state.labels = state.labels[:, :4]  # input stays at 8

    with pytest.raises(ValueError, match="sequence length 8 but labels with 4"):
        CallbackHandler([_Shrinker()]).emit(
            Event.after_dataloader,
            _trainer(),
            input_dict={"input": torch.zeros(2, 8)},
            labels=torch.zeros(2, 8),
        )


def test_an_event_rejects_payload_it_does_not_carry():
    # _PAYLOAD is the single declaration of what each event carries; before
    # it, the trainer's leftover loop variables (the last microbatch's
    # input_dict/labels/loss) leaked into optimizer events and batch_end.
    with pytest.raises(ValueError, match="not this event's payload"):
        CallbackHandler([]).emit(Event.batch_end, _trainer(), labels=torch.zeros(2))


def test_marked_hooks_narrow_the_writers():
    # Optional precision: a class that marks its writing hooks makes the
    # unmarked ones bookkeeping -- the exact false-conflict case of the
    # conservative class-level declaration, where every overridden hook
    # counts as a writer.
    class _EarlyLr(Mutator):
        @mutates(Mutation.lr)
        def after_optimizer_step(self, state):
            pass

        def batch_end(self, state):  # bookkeeping only
            pass

    log = _Log()
    handler = CallbackHandler([_EarlyLr(), _LrMutator(log, "late")])  # no conflict
    line = next(
        x for x in handler.describe().splitlines() if x.strip().startswith("batch_end")
    )
    # describe() is per hook: the bookkeeping hook shows bare.
    assert "_EarlyLr -" in line and "_LrMutator[lr]" in line


def test_one_declaration_style_per_class():
    with pytest.raises(TypeError, match="one style per class"):

        class _Both(Mutator):  # noqa: F811
            @mutates(Mutation.lr)
            def batch_end(self, state):
                pass

            mutates = frozenset({Mutation.lr})


def test_the_batch_axis_may_be_marked_on_after_dataloader_only():
    with pytest.raises(TypeError, match="reads the batch back only after"):

        class _Elsewhere(Mutator):  # noqa: F811
            @mutates(Mutation.batch)
            def batch_end(self, state):
                pass


def test_a_callback_may_not_be_both_a_mutator_and_an_observer():
    # _order fills each group with an independent isinstance, so a hybrid
    # would land in both and run twice per event -- once with MutableState,
    # once with ObserverState, silently. The handler rejects it and points
    # the "mutate and publish" case at MutableState.record().
    class _Hybrid(Mutator, Observer):
        mutates = frozenset({Mutation.lr})

    with pytest.raises(TypeError, match="never.*both|both.*never"):
        CallbackHandler([_Hybrid()])


def test_observer_state_carries_no_trainer():
    state = snapshot(_trainer())
    assert not hasattr(state, "trainer")
    assert "trainer" not in ObserverState.__slots__
    with pytest.raises(AttributeError):
        object.__getattribute__(state, "optimizers")


def test_observer_state_is_a_snapshot_taken_after_the_mutators():
    log = _Log()
    seen = {}

    class _Batch(Mutator):
        mutates = frozenset({Mutation.batch})

        def after_dataloader(self, state):
            state.labels = state.labels * 2

    class _Peek(Observer):
        def after_dataloader(self, state):
            seen["labels"] = state.labels.clone()

    handler = CallbackHandler([_Peek(), _Batch()])
    out = handler.emit(
        Event.after_dataloader,
        _trainer(),
        input_dict={"input": torch.zeros(2, 2)},
        labels=torch.tensor([1.0, 2.0]),
    )
    assert torch.equal(seen["labels"], torch.tensor([2.0, 4.0]))
    assert torch.equal(
        out.labels, torch.tensor([2.0, 4.0])
    )  # the trainer reads this back
    del log


def test_describe_lists_mutators_with_their_axes_before_the_observers():
    log = _Log()
    handler = CallbackHandler([_Reader(log, "r"), _LrMutator(log, "m")])
    text = handler.describe()
    line = next(x for x in text.splitlines() if x.strip().startswith("batch_end"))
    assert "_LrMutator[lr] -> _Reader" in line


def test_an_aggregator_runs_last_wherever_it_is_listed():
    # The point of the type: the run's dict lists the drainer first, and it
    # still runs after every observer. Before, this was a convention about
    # where to put the "timers" key, and nothing checked it.
    log = _Log()
    handler = CallbackHandler(
        [
            _Drainer(log, "drain"),
            _Reader(log, "r1"),
            _LrMutator(log, "m"),
            _Reader(log, "r2"),
        ]
    )
    handler.emit(
        Event.batch_end,
        SimpleNamespace(
            step=1, model_parts=[], parallel_dims=None, device=None, config=None
        ),
    )
    assert log.calls == ["m", "r1", "r2", "drain"]


def test_describe_shows_the_aggregator_after_the_observers():
    log = _Log()
    handler = CallbackHandler([_Drainer(log, "drain"), _Reader(log, "r")])
    line = next(
        x for x in handler.describe().splitlines() if x.strip().startswith("batch_end")
    )
    assert line.split(None, 1)[1].strip() == "_Reader -> _Drainer"


def test_every_event_the_trainer_emits_is_an_event_and_vice_versa():
    source = pathlib.Path(trainer_mod.__file__).read_text()
    tree = ast.parse(source)
    emitted = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "emit"
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and isinstance(node.args[0].value, ast.Name)
            and node.args[0].value.id == "Event"
        ):
            emitted.add(node.args[0].attr)
    assert emitted == {event.value for event in Event}
    assert STEP_EVENTS < set(Event)


def test_monitoring_forwards_every_event_to_its_sources():
    for event in Event:
        assert getattr(MonitoringCallback, event.value) is not getattr(
            Callback, event.value
        ), f"MonitoringCallback does not forward {event.value} to its sources"


def test_no_observer_reaches_for_the_trainer():
    # By subclass and bytecode, not a file-level substring scan: that scan
    # excused any file merely containing the word "Mutator" (an import was
    # enough) and would have gone blind the moment an observer moved into a
    # mutator's file. ``state.trainer`` compiles to a LOAD_ATTR, so "trainer"
    # shows up in co_names of the hook (or of a closure inside it).
    import torchtitan.callbacks.manager  # noqa: F401  (imports every builtin)

    def co_names(code) -> set[str]:
        names = set(code.co_names)
        for const in code.co_consts:
            if hasattr(const, "co_names"):
                names |= co_names(const)
        return names

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    offenders = []
    for cls in set(subclasses(Observer)):
        if not cls.__module__.startswith("torchtitan."):
            continue  # test doubles
        for event in Event:
            hook = cls.__dict__.get(event.value)
            code = getattr(hook, "__code__", None)
            if code is not None and "trainer" in co_names(code):
                offenders.append(f"{cls.__name__}.{event.value}")
    assert offenders == [], offenders


# --- the real list ------------------------------------------------------- #


def test_the_built_schedule_is_the_order_in_callback_manager():
    """Pins the resolved order of the real callbacks.

    The order a run lists its callbacks in is how the order of interchangeable
    callbacks is changed, so it has to show up here rather than in a run three
    weeks later. What key order does NOT decide is where an ``Aggregator``
    runs: ``timers`` is listed FIRST here and still comes out last, because it
    drains the registry the others fill while they run.
    """
    from torchtitan.callbacks import CallbackManager
    from torchtitan.callbacks.builtin.ema import EMA
    from torchtitan.callbacks.builtin.mtp_acc_len_monitor import MtpAccLenMonitor
    from torchtitan.callbacks.builtin.seq_length_warmup import SeqLengthWarmup
    from torchtitan.callbacks.builtin.monitoring import MonitoringCallback
    from torchtitan.callbacks.builtin.system_monitor import SystemMetricsMonitor
    from torchtitan.callbacks.builtin.timers_monitor import TimersMonitor

    manager = CallbackManager(
        {
            "timers": TimersMonitor.Config(),
            "monitoring": MonitoringCallback.Config(),
            "seq_length_warmup": SeqLengthWarmup.Config(),
            # host metrics need psutil
            "system_metrics": SystemMetricsMonitor.Config(host_log_interval=0),
            "ema": EMA.Config(),
            "mtp_acc_len": MtpAccLenMonitor.Config(),
        },
        model_parts=[nn.Linear(4, 4)],
        sharding_backend=FSDP2Backend(),
    )
    line = next(
        x for x in manager.describe().splitlines() if x.strip().startswith("batch_end")
    )
    assert line.split(None, 1)[1].strip() == (
        "SeqLengthWarmup[batch] -> MonitoringCallback -> SystemMetricsMonitor"
        " -> MtpAccLenMonitor -> TimersMonitor"
    )
