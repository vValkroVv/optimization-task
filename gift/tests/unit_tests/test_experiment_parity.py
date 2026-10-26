"""Every experiment matches its readable typed-config and model snapshot.

See ``config_golden.py`` to inspect a change or re-pin an intended one.
"""

import pytest

from tests.unit_tests.config_golden import check_golden, load_golden, resolve
from tests.unit_tests.config_helpers import EXPERIMENT_ROOT

EXPERIMENTS = sorted(path.name for path in EXPERIMENT_ROOT.glob("*.yaml"))


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_matches_golden(experiment):
    key = f"experiments/{experiment}"
    check_golden(key, resolve(key))


def test_no_stale_experiment_fingerprints():
    """A snapshot whose YAML is gone is a leftover, not a regression anchor."""
    pinned = {
        key.removeprefix("experiments/")
        for key in load_golden()
        if key.startswith("experiments/")
    }
    assert pinned <= set(EXPERIMENTS), sorted(pinned - set(EXPERIMENTS))
