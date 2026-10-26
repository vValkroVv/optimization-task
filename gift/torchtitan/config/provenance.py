"""Immutable source metadata shared by compact and derived config trees."""

from collections.abc import Iterator, Mapping
from types import MappingProxyType


class SourceMap(Mapping[str, str]):
    """Read-only string metadata that can be shared through config deep copies."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, str] | None = None) -> None:
        self._values = MappingProxyType(dict(values or {}))

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (dict(self._values),)

    def __repr__(self) -> str:
        return repr(dict(self._values))


class ConfigMetadata:
    """Service state kept outside the dataclass's public configuration fields."""

    __slots__ = ("_metadata",)

    def copy_metadata_from(self, source: "ConfigMetadata") -> None:
        """Preserve service state when copying or replacing a config node.

        Immutable source maps are shared; the metadata container is independent.
        Runtime converters use this interface rather than its storage details.
        """
        try:
            metadata = object.__getattribute__(source, "_metadata")
        except AttributeError:
            metadata = {}
        object.__setattr__(self, "_metadata", dict(metadata))

    def _get_metadata(self, key, default):
        try:
            return self._metadata.get(key, default)
        except AttributeError:
            return default

    def _set_metadata(self, key, value):
        try:
            metadata = dict(object.__getattribute__(self, "_metadata"))
        except AttributeError:
            metadata = {}
        metadata[key] = value
        object.__setattr__(self, "_metadata", metadata)

    _source_path = property(
        lambda self: self._get_metadata("path", ""),
        lambda self, value: self._set_metadata("path", value),
    )
    _source_fields = property(
        lambda self: self._get_metadata("fields", SourceMap()),
        lambda self, value: self._set_metadata("fields", value),
    )
    _source_origins = property(
        lambda self: self._get_metadata("origins", SourceMap()),
        lambda self, value: self._set_metadata("origins", value),
    )
    _explicit_param_init_names = property(
        lambda self: self._get_metadata("init_names", None),
        lambda self, value: self._set_metadata("init_names", value),
    )
