"""Completes partial trainer doubles for the callback tests.

``ObserverState.from_mutable`` reads the trainer's attributes directly:
production must fail loudly on a broken trainer instead of handing silent
``None``s to the reducers. The tests' tolerance lives HERE, not there — a
``SimpleNamespace`` double is filled up to the full trainer shape (from
``trainer_stub.make_trainer``) before the strict snapshot runs.
"""

import pytest

from tests.unit_tests.callbacks.trainer_stub import make_trainer
from torchtitan.callbacks import base

_strict_from_mutable = base.ObserverState.from_mutable.__func__


@pytest.fixture(autouse=True)
def _complete_trainer_doubles(monkeypatch):
    reference = vars(make_trainer())

    def filling(cls, mutable):
        trainer = mutable.trainer
        for name, value in reference.items():
            if not hasattr(trainer, name):
                setattr(trainer, name, value)
        return _strict_from_mutable(cls, mutable)

    monkeypatch.setattr(base.ObserverState, "from_mutable", classmethod(filling))
