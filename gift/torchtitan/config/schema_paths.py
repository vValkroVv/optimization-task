"""Resolve field schemas for raw documents without constructing runtime objects."""

import dataclasses
import re
from collections.abc import Mapping
from typing import Annotated, get_args, get_origin

from .configurable import family_of, import_spec
from .decode import _get_field_types
from .registry import TRAINER_TYPES, family_for_base
from .sections import ModelSection

_PATH_TOKEN = re.compile(r"(?:^|\.)([^.\[\]]+)|\[(\d+)\]")


def _path_tokens(path: str) -> tuple[str | int, ...]:
    return tuple(
        int(match[2]) if match[2] is not None else match[1]
        for match in _PATH_TOKEN.finditer(path)
    )


def concrete_schema(schema):
    if get_origin(schema) is Annotated:  # OwnerFilled / Derived markers
        schema = get_args(schema)[0]
    args = get_args(schema)
    if args and type(None) in args:
        candidates = [arg for arg in args if arg is not type(None)]
        return candidates[0] if len(candidates) == 1 else schema
    return schema


def _selected_schema(schema, raw):
    schema = concrete_schema(schema)
    if not isinstance(schema, type) or not dataclasses.is_dataclass(schema):
        return schema
    family = family_for_base(schema)
    family_base = family is not None
    if family is None:
        family = next((f for f in map(family_of, schema.__mro__) if f), None)
    if family is not None and isinstance(raw, dict):
        type_name = raw.get("type", family.default if family_base else None)
        if isinstance(type_name, str):
            selected = family.types.get(type_name)
            if isinstance(selected, type) and issubclass(selected, schema):
                return selected
    return schema


def schema_at(data, path):
    """Resolve the selected schema at a dotted config path; invalid input returns None."""
    raw_trainer = data.get("trainer", {})
    name = (
        raw_trainer.get("type", "default")
        if isinstance(raw_trainer, dict)
        else "default"
    )
    if not isinstance(name, str):
        return None
    entry = TRAINER_TYPES.get(name)
    if entry is None:
        return None
    schema, raw = import_spec(entry.spec), data
    for index, key in enumerate(_path_tokens(path)):
        if index == 0 and key == "model":
            schema, raw = ModelSection, raw.get("model", {})
            continue
        schema = _selected_schema(schema, raw)
        origin = get_origin(schema)
        if origin in (dict, Mapping):
            schema = get_args(schema)[1]
        elif origin in (list, tuple):
            schema = get_args(schema)[0]
        elif isinstance(schema, type) and dataclasses.is_dataclass(schema):
            if not isinstance(key, str):
                return None
            schema = _get_field_types(schema).get(key)
        else:
            return None
        if schema is None:
            return None
        if isinstance(raw, dict):
            raw = raw.get(key, {})
        elif isinstance(raw, (list, tuple)) and isinstance(key, int) and key < len(raw):
            raw = raw[key]
        else:
            raw = {}
    return _selected_schema(schema, raw)
