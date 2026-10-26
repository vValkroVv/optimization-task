"""Config error types; every message carries the full config path."""

import difflib


class ConfigError(Exception):
    """Base class for every experiment-config error."""


class ConfigFileError(ConfigError):
    """Raised for YAML/file-level problems: missing file, bad syntax, import cycles."""


class ConfigSchemaError(ConfigError):
    """Raised when a document does not match the schema."""

    def __init__(self, message: str, *, source: str | None = None):
        self.source = source
        super().__init__(message)


def join_path(prefix: str, key: str | int) -> str:
    """Append *key* to a dotted config path."""
    if isinstance(key, int):
        return f"{prefix}[{key}]" if prefix else f"[{key}]"
    return f"{prefix}.{key}" if prefix else str(key)


def suggest(value: str | int, candidates) -> str:
    """Return a `` Did you mean 'x'?`` fragment, or an empty string."""
    matches = difflib.get_close_matches(
        str(value), [str(c) for c in candidates], n=1, cutoff=0.6
    )
    return f" Did you mean {matches[0]!r}?" if matches else ""


class ConfigFieldError(ValueError):
    """A local invariant associated with one configuration field."""

    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(message)
