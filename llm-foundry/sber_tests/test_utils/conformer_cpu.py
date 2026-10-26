"""Import the gigaomni conformer on a CPU-only host.

``llmfoundry/__init__.py`` and the ``gigaomni`` package inits pull composer,
flash-attn and the vision encoders; none of that is needed to run the
conformer with ``attention_backend="sdpa"``. When the regular import fails,
the intermediate packages are seeded as bare namespaces (their ``__init__``
is skipped, submodules still resolve by path), the conformer package is
imported, and every entry the seeding added to ``sys.modules`` is removed
again so other test modules in the same session see an unchanged import
state. The returned module objects stay fully usable.
"""

import importlib
import importlib.machinery
import os
import sys
import types
from typing import Tuple

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFORMER_PKG = "llmfoundry.models.gigaomni.modalities.encoders.conformer"
_NAMESPACES = (
    "llmfoundry",
    "llmfoundry.models",
    "llmfoundry.models.gigaomni",
    "llmfoundry.models.gigaomni.modalities",
    "llmfoundry.models.gigaomni.modalities.encoders",
)


def _seed_namespace(name: str) -> None:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = [os.path.join(REPO_ROOT, *name.split("."))]
    sys.modules[name] = mod


def import_conformer_modules() -> Tuple[types.ModuleType, types.ModuleType]:
    """Return ``(modeling_conformer, configuration_conformer)`` modules."""
    try:
        modeling = importlib.import_module(f"{CONFORMER_PKG}.modeling_conformer")
        configuration = importlib.import_module(f"{CONFORMER_PKG}.configuration_conformer")
        return modeling, configuration
    except ImportError:
        pass
    before = set(sys.modules)
    for name in _NAMESPACES:
        if name in sys.modules:
            raise RuntimeError(f"{name} is importable but the conformer is not")
        _seed_namespace(name)
    try:
        modeling = importlib.import_module(f"{CONFORMER_PKG}.modeling_conformer")
        configuration = importlib.import_module(f"{CONFORMER_PKG}.configuration_conformer")
    finally:
        for name in set(sys.modules) - before:
            if name.startswith("llmfoundry"):
                del sys.modules[name]
    return modeling, configuration
