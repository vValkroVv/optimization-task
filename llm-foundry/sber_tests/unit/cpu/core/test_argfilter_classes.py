"""`TEST_MODALITY=<name>` runs `test_local_jobs.py::<class>` for the classes
`get_modality_mapping` lists. A renamed class leaves a node id that selects
nothing, as `TestVision` / `TestVideo` did after the vision and video tests
became `TestOmniVision*` / `TestOmniVideo*`."""

import re
from pathlib import Path

import pytest

from sber_tests.test_utils.argfilter import get_modality_mapping

TEST_LOCAL_JOBS = Path(__file__).parents[3] / "integration" / "test_local_jobs.py"


@pytest.mark.unit
@pytest.mark.fast
def test_every_mapped_class_exists():
    classes = set(re.findall(r"^class (\w+)", TEST_LOCAL_JOBS.read_text(), re.M))
    missing = {
        modality: sorted(set(names) - classes)
        for modality, names in get_modality_mapping().items()
        if set(names) - classes
    }

    assert not missing, f"{missing} are not classes in {TEST_LOCAL_JOBS.name}"
