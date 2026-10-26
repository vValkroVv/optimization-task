"""A complete trainer double for callback tests.

``ObserverState.from_mutable`` reads the trainer's attributes directly --
production diagnostics must not be softened for test doubles -- so a test's
double carries every attribute a real trainer has. This is the one place
that knows the full shape.
"""

from dataclasses import replace
from types import SimpleNamespace

from torchtitan.callbacks.base import MutableState, ObserverState


def make_trainer(**overrides) -> SimpleNamespace:
    """A trainer double with every attribute the state snapshot reads."""
    logged: list = []
    base: dict = dict(
        step=3,
        model_parts=[],
        parallel_dims=None,
        device="cpu",
        config=None,
        optimizers=SimpleNamespace(optimizers=()),
        loss_fn=None,
        checkpointer=SimpleNamespace(last_saved_path=None),
        metrics_processor=SimpleNamespace(
            log_extra_metrics=lambda step, metrics: logged.append((step, metrics))
        ),
        ntokens_seen=0,
        gradient_accumulation_steps=1,
    )
    base.update(overrides)
    trainer = SimpleNamespace(**base)
    trainer.logged = logged
    return trainer


def snapshot(trainer=None, *, step=None, **fields) -> ObserverState:
    """An ObserverState off a double -- what ``from_trainer`` used to be.

    Fills the attributes a partial double left out, so an old-style
    ``SimpleNamespace(step=..)`` still snapshots; production code gets no
    such tolerance.
    """
    if trainer is None:
        trainer = make_trainer()
    for name, value in vars(make_trainer()).items():
        if not hasattr(trainer, name):
            setattr(trainer, name, value)
    state = ObserverState.from_mutable(MutableState(trainer=trainer, **fields))
    return state if step is None else replace(state, step=int(step))
