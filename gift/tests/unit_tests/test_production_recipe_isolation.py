"""A change to one production run must not alter another run through imports."""

import shutil
from pathlib import Path

import pytest
import yaml

from torchtitan.config.loader import load_raw_document


@pytest.mark.parametrize(
    "override", [{"optimizer": {"lr": 0.001}}, {"training": {"steps": 10}}]
)
def test_gift4_does_not_inherit_gift35_run_edits(tmp_path, override):
    configs = tmp_path / "configs"
    shutil.copytree(Path(__file__).resolve().parents[2] / "configs", configs)
    gift4 = configs / "experiments" / "gift_4_700b.yaml"
    original = load_raw_document(str(gift4)).data
    gift35 = configs / "experiments" / "gift_3_5_700b.yaml"
    with gift35.open("a") as destination:
        destination.write("\n")
        yaml.safe_dump(override, stream=destination)
    changed = load_raw_document(str(gift35)).data
    for section, values in override.items():
        for name, value in values.items():
            assert changed[section][name] == value
    assert load_raw_document(str(gift4)).data == original
