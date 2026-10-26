"""Identify pipeline metadata forwards for runtime observers."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_METADATA_INFERENCE: ContextVar[bool] = ContextVar(
    "pipeline_metadata_inference", default=False
)


def is_pipeline_metadata_inference() -> bool:
    """Whether the current forward is a pipeline setup probe."""
    return _METADATA_INFERENCE.get()


@contextmanager
def pipeline_metadata_inference() -> Iterator[None]:
    """Mark a metadata forward without changing model execution."""
    token = _METADATA_INFERENCE.set(True)
    try:
        yield
    finally:
        _METADATA_INFERENCE.reset(token)
