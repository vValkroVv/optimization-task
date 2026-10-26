"""Structural dump of config trees, for readable golden snapshots."""

import dataclasses
import enum
import functools

from torchtitan.config.configurable import family_of, type_name_of


def structural_dump(obj, *, semantic_types: bool = False):
    """A config tree as comparable plain data.

    Callables compare by identity-relevant structure (``partial`` by function
    and arguments), dataclasses record their class so that a swapped component
    type (``RMSNorm`` vs ``GatedRMSNorm``, ``Linear`` vs ``Unembedding``) is a
    difference, not a coincidence of equal fields.
    """

    def dump(value):
        return structural_dump(value, semantic_types=semantic_types)

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        if semantic_types:
            type_name = type_name_of(cls)
            family = next(
                (family for base in cls.__mro__ if (family := family_of(base))), None
            )
            identity = (
                f"{family.name}:{type_name}"
                if family is not None and type_name is not None
                else f"{cls.__module__}.{cls.__qualname__}"
            )
            type_key = "__type__"
        else:
            identity = f"{cls.__module__}.{cls.__qualname__}"
            type_key = "__class__"
        return {
            type_key: identity,
            **{
                f.name: dump(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
                if f.compare
            },
        }
    if isinstance(obj, functools.partial):
        return {
            "__partial__": dump(obj.func),
            "args": [dump(a) for a in obj.args],
            "kwargs": {k: dump(v) for k, v in obj.keywords.items()},
        }
    if isinstance(obj, enum.Enum):
        return f"{type(obj).__qualname__}.{obj.name}"
    if callable(obj):
        return f"{obj.__module__}.{obj.__qualname__}"
    if isinstance(obj, (list, tuple)):
        return [dump(v) for v in obj]
    if isinstance(obj, dict):
        return {k: dump(v) for k, v in obj.items()}
    return obj
