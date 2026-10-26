"""Every model preset matches its readable written and actual model snapshot.

A preset lives under ``<family>/``, or ``<family>/<variant>/``; a variant's
presets carry it in their label (the ``ModelSpec.flavor``).
See ``config_golden.py`` to inspect or re-pin.
"""

import pytest

from tests.unit_tests.config_golden import check_golden, load_golden, resolve
from tests.unit_tests.config_helpers import PRESET_ROOT
from torchtitan.config.loader import load_raw_document

PRESETS = sorted(
    str(path.relative_to(PRESET_ROOT)) for path in PRESET_ROOT.rglob("*.yaml")
)


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_matches_golden(preset):
    family, *variant_dirs, _ = preset.split("/")
    label = load_raw_document(str(PRESET_ROOT / preset)).data["model"]["label"]
    if variant_dirs:
        assert label.startswith(f"{variant_dirs[0]}/"), preset

    key = f"presets/{preset}"
    snapshot = resolve(key)
    assert snapshot["model"].config.family == family
    check_golden(key, snapshot)


def test_no_stale_preset_fingerprints():
    """A pin whose preset is gone is a leftover, not a regression anchor."""
    pinned = {
        key.removeprefix("presets/")
        for key in load_golden()
        if key.startswith("presets/")
    }
    assert pinned <= set(PRESETS), sorted(pinned - set(PRESETS))
