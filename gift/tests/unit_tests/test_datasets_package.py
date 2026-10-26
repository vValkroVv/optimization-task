"""``python torchtitan/train.py`` puts ``torchtitan/`` first on ``sys.path``, so a
regular ``torchtitan/datasets`` package would shadow HF ``datasets`` there"""

import sys
from importlib.machinery import PathFinder
from pathlib import Path

import torchtitan


def test_script_mode_does_not_shadow_datasets_library():
    package_dir = Path(torchtitan.__file__).parent
    spec = PathFinder.find_spec("datasets", [str(package_dir), *sys.path])
    assert spec is not None and spec.origin is not None, "HF datasets is not installed"
    assert package_dir not in Path(spec.origin).parents, spec.origin
