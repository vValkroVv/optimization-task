"""Portable values accepted by open mappings in declarative configs."""

from typing import TypeAlias

ConfigValue: TypeAlias = (
    None | bool | int | float | str | list["ConfigValue"] | dict[str, "ConfigValue"]
)
