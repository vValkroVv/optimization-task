"""Fill or check child config fields the owner also knows."""

from typing import TypeVar

from torchtitan.config.configurable import fields_with_role, OWNER_FILLED

__all__ = [
    "require_completed",
    "resolve_child",
    "validate_linear_shapes",
    "validate_values",
]

_Completed = TypeVar("_Completed")


def require_completed(value: _Completed | None, field: str) -> _Completed:
    """Read a child or dimension after its owning config has completed it."""
    if value is None:
        raise ValueError(f"{field} must be completed by its owner before use")
    return value


def _raise_mismatch(owner: str, wrong: list[str]) -> None:
    if wrong:
        raise ValueError(f"{owner} mismatches: {'; '.join(wrong)}")


def validate_linear_shapes(owner: str, *specs) -> None:
    """Reject projections of the wrong shape; specs are ``(config, name, in, out)``.

    Shape, not type: a quantized ``Linear`` subclass is fine, another size is not.
    """
    _raise_mismatch(
        owner,
        [
            f"{name}: expected ({want_in}, {want_out}), got "
            f"({cfg.in_features}, {cfg.out_features})"
            for cfg, name, want_in, want_out in specs
            if (cfg.in_features, cfg.out_features) != (want_in, want_out)
        ],
    )


def validate_values(owner: str, *specs) -> None:
    """Reject values disagreeing with the owner's; specs are ``(name, got, want)``."""
    _raise_mismatch(
        owner,
        [
            f"{name}: expected {want}, got {got}"
            for name, got, want in specs
            if got != want
        ],
    )


def resolve_child(
    child, factory, /, *, name, dims, require=None, extra=None, init=None
):
    """Build ``child`` from ``dims`` and ``require`` when absent, else check it.

    ``dims`` are the child's OwnerFilled fields: ``None`` ones on a present
    child are filled. ``require`` values (e.g. a ``bias`` bool, which has no
    unset) are only checked. ``None`` ``extra`` values are filled too.
    ``init`` is the owner's ``param_init``; a present child's own entries
    overlay it. A complete child keeps its type and identity; completing a
    partial child preserves its non-init metadata too.
    """
    filled = fields_with_role(factory, OWNER_FILLED)
    for key in dims:
        assert key in filled, f"{factory.__qualname__}.{key} is not OwnerFilled"
    require = require or {}
    if child is None:
        extra = {k: v for k, v in (extra or {}).items() if v is not None}
        if init:
            extra["param_init"] = dict(init)
        return factory(**dims, **require, **extra)
    unset = {
        k: v
        for k, v in (*dims.items(), *(extra or {}).items())
        if getattr(child, k) is None and v is not None
    }
    if init and child.param_init != {}:
        merged = {**init, **(child.param_init or {})}
        if merged != child.param_init:
            unset["param_init"] = merged
    if unset:
        child = child._copy(**unset)

    def yaml(value):
        return str(value).lower() if isinstance(value, bool) else value

    _raise_mismatch(
        name,
        [
            f"{k}: expected {v}, got {getattr(child, k)}"
            for k, v in dims.items()
            if getattr(child, k) != v
        ]
        + [
            f"requires {k}: {yaml(v)} (got {yaml(getattr(child, k))})"
            for k, v in require.items()
            if getattr(child, k) != v
        ],
    )
    return child
