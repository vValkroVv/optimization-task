"""A written config records only what YAML wrote: completion happens at
materialize, so a resolved config reloads with the same behavior.

A value that ``__post_init__`` derives into a written config is baked into
``resolved_config.yaml``: an edit of its input in the bundle would then leave
the derived value stale. Hidden state that steers behavior would make two
equal configs behave differently.
"""

import dataclasses
from typing import Any

import pytest

from tests.unit_tests.config_helpers import EXPERIMENT_ROOT, PRESET_ROOT
from torchtitan.config import load_document, load_raw_document
from torchtitan.config.configurable import Configurable
from torchtitan.config.decode import decode
from torchtitan.config.sections import ModelSection

_DOCUMENTS = sorted(
    [("presets", p) for p in PRESET_ROOT.rglob("*.yaml")]
    + [("experiments", p) for p in EXPERIMENT_ROOT.glob("*.yaml")],
    key=lambda item: str(item[1]),
)
_IDS = [str(p.relative_to(p.parents[1])) for _, p in _DOCUMENTS]

# Provenance (diagnostics only) and the materialized flag (a phase, not a
# setting); anything else steering behavior must be a YAML field.
_ALLOWED_METADATA = {"path", "fields", "origins", "init_names", "materialized"}


def _written_model(kind: str, path) -> tuple[Any, dict]:
    if kind == "experiments":
        document, _ = load_document(str(path))
    else:
        document = load_raw_document(str(path))
    raw = document.data["model"]
    return decode(ModelSection, raw, path="model").config, raw["config"]


def _default(field: dataclasses.Field) -> Any:
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING


def _derived_values(node: Any, raw: Any, path: str) -> list[str]:
    """Paths whose value YAML did not write and the schema default leaves
    unset: values a written config derived for itself."""
    if isinstance(node, list):
        raw_items = raw if isinstance(raw, list) else [None] * len(node)
        return [
            found
            for index, (item, raw_item) in enumerate(zip(node, raw_items))
            for found in _derived_values(item, raw_item, f"{path}.{index}")
        ]
    if isinstance(node, dict):
        raw_map = raw if isinstance(raw, dict) else {}
        return [
            found
            for key, item in node.items()
            for found in _derived_values(item, raw_map.get(key), f"{path}.{key}")
        ]
    if not (dataclasses.is_dataclass(node) and not isinstance(node, type)):
        return []
    raw_map = raw if isinstance(raw, dict) else {}
    found = []
    for field in dataclasses.fields(node):
        if not field.init:
            continue
        value = getattr(node, field.name)
        where = f"{path}.{field.name}"
        if field.name in raw_map:
            found += _derived_values(value, raw_map[field.name], where)
            continue
        default = _default(field)
        if default is None and value is not None:
            found.append(where)
        elif value != default:
            # A default the decoder layered settings on; recurse with nothing
            # written beneath it.
            found += _derived_values(value, {}, where)
    return found


def _metadata_keys(node: Any, seen: set[int]) -> set[str]:
    if id(node) in seen:
        return set()
    seen.add(id(node))
    keys: set[str] = set()
    if isinstance(node, Configurable.Config):
        try:
            keys |= set(object.__getattribute__(node, "_metadata"))
        except AttributeError:
            pass
    if isinstance(node, (list, tuple)):
        for item in node:
            keys |= _metadata_keys(item, seen)
    elif isinstance(node, dict):
        for item in node.values():
            keys |= _metadata_keys(item, seen)
    elif dataclasses.is_dataclass(node) and not isinstance(node, type):
        for field in dataclasses.fields(node):
            keys |= _metadata_keys(getattr(node, field.name), seen)
    return keys


@pytest.mark.parametrize(("kind", "path"), _DOCUMENTS, ids=_IDS)
def test_written_config_holds_no_derived_values(kind, path):
    written, raw = _written_model(kind, path)
    assert _derived_values(written, raw, "model.config") == []


@pytest.mark.parametrize(("kind", "path"), _DOCUMENTS, ids=_IDS)
def test_only_provenance_rides_along_as_metadata(kind, path):
    written, _ = _written_model(kind, path)
    tree = written.materialize()
    unexpected = (_metadata_keys(written, set()) | _metadata_keys(tree, set())) - (
        _ALLOWED_METADATA
    )
    assert unexpected == set()
