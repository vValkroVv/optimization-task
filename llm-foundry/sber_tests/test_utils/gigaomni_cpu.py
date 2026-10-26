"""Import gigaomni data and model modules on a CPU-only host.

Same approach as ``conformer_cpu.py``: when the regular import fails (no
composer, streaming, flash-attn or audioaug on the host), every package prefix
of the requested modules whose ``__init__`` cannot be imported is seeded as a
bare namespace, the modules are imported by path, and every ``sys.modules``
entry the seeding added is removed again. ``audioaug`` is stubbed (the video
processor only touches it while decoding a soundtrack); ``omni_data``
additionally needs ``streaming.Stream`` and ``composer.core`` stand-ins plus
the two ``llmfoundry.data`` modules it imports for the dataset class, none of
which the placeholder-id check uses. On a host with the real dependencies the
first import succeeds and nothing is seeded or stubbed.
"""

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Tuple

from sber_tests.test_utils.conformer_cpu import _seed_namespace

MODULES = {
    "video_processor": "llmfoundry.data.processors.gigaomni_video_processor",
    "audio_processor": "llmfoundry.data.processors.gigaomni_audio_processor",
    "video_collator": "llmfoundry.data.collators.video_collator",
    "audio_collator": "llmfoundry.data.collators.audio_collator",
    "video_audio_collator": "llmfoundry.data.collators.video_audio_collator",
    "omni_data": "llmfoundry.data.omni_data",
    "modelling": "llmfoundry.models.gigaomni.modelling_gigaomni",
    "modalities": "llmfoundry.models.gigaomni.modalities",
    "audio_modality": "llmfoundry.models.gigaomni.modalities.audio_modality",
    "video_only_modality": "llmfoundry.models.gigaomni.modalities.video_only_modality",
    "vision_modality": "llmfoundry.models.gigaomni.modalities.vision_modality",
    "conformer": "llmfoundry.models.gigaomni.modalities.encoders.conformer.modeling_conformer",
    "conformer_config": "llmfoundry.models.gigaomni.modalities.encoders.conformer.configuration_conformer",
    "audio_adapter": "llmfoundry.models.gigaomni.modalities.adapters.audio_linear_adapter",
    "mm_utils": "llmfoundry.data.processors.mm_utils",
}
_STUB_PREFIXES = ("llmfoundry", "audioaug", "streaming", "composer")


def _stub_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


def _stub_missing_third_party() -> None:
    if "audioaug" not in sys.modules:
        # annotations in the audio processor resolve these at import time
        _stub_module("audioaug", FeatureParams=object, AudioInfo=object)

    if "streaming" not in sys.modules:
        _stub_module("streaming", Stream=type("Stream", (), {}))

    if "composer" not in sys.modules:
        composer = _stub_module("composer")
        composer.__path__ = []
        _stub_module("composer.core", Evaluator=object, DataSpec=object)
        _stub_module("composer.utils", dist=None)

    _stub_module(
        "llmfoundry.data.text_data",
        StreamingTextDataset=object,
        FlashAttnCollatorWrapper=object,
    )
    _stub_module("llmfoundry.data.data", read_binary_tokenized_sample=None)


def _package_prefixes(module_names) -> list:
    prefixes = []

    for name in module_names:
        parts = name.split(".")

        for depth in range(1, len(parts)):
            prefix = ".".join(parts[:depth])

            if prefix not in prefixes:
                prefixes.append(prefix)

    return sorted(prefixes, key=lambda prefix: prefix.count("."))


def _seed_unimportable_prefixes(module_names) -> None:
    """Seed the package prefixes whose ``__init__`` does not import here; a
    package that imports with the stubs in place keeps its real ``__init__``."""
    for prefix in _package_prefixes(module_names):
        if prefix in sys.modules:
            continue

        try:
            importlib.import_module(prefix)
        except ImportError:
            _seed_namespace(prefix)


def import_modules(*module_names: str) -> Tuple[types.ModuleType, ...]:
    """Return the modules named by ``module_names`` (dotted), in order."""
    try:
        return tuple(importlib.import_module(name) for name in module_names)
    except ImportError:
        pass

    if "llmfoundry" in sys.modules:
        raise RuntimeError("llmfoundry is importable but the requested modules are not")

    before = set(sys.modules)
    _stub_missing_third_party()
    _seed_unimportable_prefixes(module_names)

    try:
        return tuple(importlib.import_module(name) for name in module_names)
    finally:
        for name in set(sys.modules) - before:
            if name.startswith(_STUB_PREFIXES):
                del sys.modules[name]


def import_gigaomni_modules() -> SimpleNamespace:
    """Return the modules in ``MODULES`` keyed by alias."""
    return SimpleNamespace(**dict(zip(MODULES, import_modules(*MODULES.values()))))
