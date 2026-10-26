from collections.abc import Iterable
from typing import TypeVar, cast

from torchtitan.config.configurable import Configurable, config


_PlacedConfig = TypeVar("_PlacedConfig", bound=Configurable.Config)


def placed_configs(
    entries: Iterable[object], kind: type[_PlacedConfig], *, path: str
) -> tuple[_PlacedConfig, ...]:
    """Check placed positions without dropping transparent freeze wrappers.

    Runtime owners consume this view rather than the written layer-group union.
    A FrozenConfig proxies its inner config's fields but must retain its own
    build method, so validate the inner type and return the original wrapper.
    """
    from torchtitan.config.schema.components.lora import FrozenConfig

    result = []
    for index, entry in enumerate(entries):
        inner = entry
        while isinstance(inner, FrozenConfig):
            inner = inner.inner
        if not isinstance(inner, kind):
            raise ValueError(
                f"{path}.{index}: expected placed {kind.__name__}, "
                f"got {type(inner).__name__}; materialize the model first"
            )
        result.append(cast(_PlacedConfig, entry))
    return tuple(result)


@config
class LayerGroupConfig(Configurable.Config):
    """A run of a decoder's named ``blocks``: one block or a cycle of them,
    repeated."""

    block: str | None = None
    cycle: list[str] | None = None
    repeat: int = 1

    def __post_init__(self) -> None:
        if (self.block is None) == (self.cycle is None):
            raise ValueError("a layer group sets exactly one of `block` or `cycle`")
        if self.cycle is not None and not self.cycle:
            raise ValueError("`cycle` needs at least one block name")
        if self.repeat < 1:
            raise ValueError(f"`repeat` must be at least 1, got {self.repeat}")

    def names(self) -> list[str]:
        names = [self.block] if self.block is not None else list(self.cycle or ())
        return names * self.repeat


def count_layers(entries: list) -> int:
    """Positions *entries* fill: a group its names, anything else one."""
    return sum(
        len(entry.names()) if isinstance(entry, LayerGroupConfig) else 1
        for entry in entries
    )


def group_names(entries: list) -> list[str]:
    return [
        name
        for entry in entries
        if isinstance(entry, LayerGroupConfig)
        for name in entry.names()
    ]


def check_block_names(entries: list, blocks: dict, *, path: str = "layers") -> None:
    from torchtitan.config.errors import suggest

    for name in group_names(entries):
        if name not in blocks:
            raise ValueError(
                f"{path}: unknown block {name!r}.{suggest(name, blocks)} "
                f"Defined blocks: {', '.join(sorted(blocks)) or 'none'}"
            )


def place_blocks(entries: list, blocks: dict) -> list:
    """An independent config per position: groups expand to copies of their
    named blocks, explicit entries are copied."""
    from torchtitan.config.configurable import independent_config_positions

    placed = []
    for entry in entries:
        if isinstance(entry, LayerGroupConfig):
            placed += [independent_config_positions(blocks[n]) for n in entry.names()]
        else:
            placed.append(independent_config_positions(entry))
    return placed


def reject_positions(blocks: dict, *, path: str = "blocks") -> None:
    """``layer_id`` is where a block is placed, never part of a named block."""
    for name, block in blocks.items():
        if getattr(block, "layer_id", None) is not None:
            raise ValueError(
                f"{path}.{name}.layer_id: a block's position is set where "
                "`layers` places it; remove it"
            )
