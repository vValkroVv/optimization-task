"""The document-level sections the builder consumes itself: ``config_version``,
``trainer`` and ``model``."""

from dataclasses import dataclass

from .schema.protocols.model import BaseModelConfig

CONFIG_VERSION = 1


@dataclass(kw_only=True, slots=True)
class TrainerSection:
    """The optional ``trainer:`` section: which trainer runs the job."""

    type: str = "default"


@dataclass(kw_only=True, slots=True)
class ModelSection:
    """The ``model:`` section: the model's own Config plus a label."""

    config: BaseModelConfig
    """The model Config in compact form, its class picked by ``type``."""

    label: str = "custom"
    """Size/preset label (e.g. ``16B``), the ModelSpec flavor; not used for
    dispatch."""
