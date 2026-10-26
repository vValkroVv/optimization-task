"""Dot-path command-line overrides: each ``--a.b value`` is one YAML layer.

See docs/config.md, "CLI overrides".
"""

import re
from typing import Any

from .errors import ConfigFileError, ConfigSchemaError
from .loader import parse_yaml_value


def parse_argv(argv: list[str]) -> tuple[str, list[tuple[str, str]]]:
    """Split ``[config.yaml, --a.b, value, --c.d=value, --flag]``.

    Returns the config path and the raw ``(path, value)`` override pairs; a
    flag with no value is ``"true"``.
    """
    config_path: str | None = None
    raw_overrides: list[tuple[str, str]] = []

    arg_index = 0
    while arg_index < len(argv):
        arg = argv[arg_index]
        if arg == "--":
            arg_index += 1
            continue
        if arg.startswith("--"):
            if config_path is None:
                raise ConfigSchemaError(
                    "the config file path must precede CLI overrides"
                )
            override_text = arg[2:]
            if "=" in override_text:
                override_path, raw_value = override_text.split("=", 1)
            else:
                override_path = override_text
                next_arg = argv[arg_index + 1] if arg_index + 1 < len(argv) else None
                if next_arg is None or next_arg.startswith("--"):
                    raw_value = "true"
                else:
                    raw_value = next_arg
                    arg_index += 1
            # Empty, or whitespace: "--a.b 1" passed as one token.
            if override_path.split() != [override_path]:
                raise ConfigSchemaError(
                    f"malformed override argument {arg!r}: write `--a.b value` "
                    "or `--a.b=value`"
                )
            raw_overrides.append((override_path, raw_value))
        elif config_path is None:
            config_path = arg
        else:
            raise ConfigSchemaError(
                f"unexpected positional argument {arg!r}; exactly one config file "
                f"is accepted (already got {config_path!r})"
            )
        arg_index += 1

    if config_path is None:
        raise ConfigSchemaError(
            "a config file path is required as the first positional argument"
        )
    return config_path, raw_overrides


def build_override_layers(
    raw_overrides: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one YAML layer per override, in argv order: ``[("a.b", "1"), ...]``
    -> ``[{"a": {"b": 1}}, ...]``.

    One layer each, so ``--x 5 --x null`` resets ``x`` as two files would
    (merged first, the ``null`` would only delete the ``5``).

    Returns the layers and the applied values as a flat ``path -> value``
    mapping for the run metadata (flat keeps ``null`` resets visible), a
    repeated path at its last position.

    Raises:
        ConfigSchemaError: For a malformed path or value.
    """
    override_layers: list[dict[str, Any]] = []
    applied_overrides: dict[str, Any] = {}
    for override_path, raw_value in raw_overrides:
        path_keys = override_path.split(".")
        if not all(path_keys) or "[" in override_path or "]" in override_path:
            raise ConfigSchemaError(
                f"malformed override path {override_path!r}: use dotted keys; a "
                "list item cannot be addressed, set the whole list"
            )
        parsed_value = _parse_override_value(raw_value, override_path)
        override_layer: Any = parsed_value
        for key in reversed(path_keys):
            override_layer = {key: override_layer}
        override_layers.append(override_layer)
        applied_overrides.pop(override_path, None)
        applied_overrides[override_path] = parsed_value
    return override_layers, applied_overrides


_YAML_NULL_SPELLINGS = frozenset({"null", "Null", "NULL", "~"})


def _parse_override_value(raw_value: str, override_path: str) -> Any:
    """Parse *raw_value* as a YAML value; a bare ``a,b`` as the flow list
    ``[a, b]``."""
    if not raw_value.strip():
        raise ConfigSchemaError(
            f"{override_path}: empty command-line value; use null to reset the "
            "key to its default"
        )
    # Shell quotes are already removed: a plain CLI string keeps its hash.
    if re.search(r"\s#", raw_value) and raw_value.lstrip()[:1] not in (
        "[",
        "{",
        "'",
        '"',
        "!",
    ):
        return raw_value
    yaml_text = raw_value
    if "," in raw_value and raw_value.lstrip()[:1] not in ("[", "{", "'", '"', "!"):
        yaml_text = f"[{raw_value}]"
    try:
        parsed_value = parse_yaml_value(yaml_text, source=override_path)
    except ConfigFileError as exc:
        raise ConfigSchemaError(f"{exc}; quote it to pass a string") from None
    if parsed_value is None and raw_value.strip() not in _YAML_NULL_SPELLINGS:
        # `#tag` is a YAML comment: a silent reset is never what was meant.
        raise ConfigSchemaError(
            f"{override_path}: {raw_value!r} reads as null in YAML; quote it to "
            "pass a string, or write null to reset the key"
        )
    return parsed_value
