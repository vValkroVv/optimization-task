from typing import ClassVar

from torchtitan.config.configurable import config, Configurable


@config(family="loss")
class BaseLossConfig(Configurable.Config):
    pass


@config(type="cross_entropy")
class CrossEntropyLossConfig(BaseLossConfig):
    _owner_spec: ClassVar[str] = "torchtitan.components.loss:CrossEntropyLoss"


@config(type="mse")
class MSELossConfig(BaseLossConfig):
    _owner_spec: ClassVar[str] = "torchtitan.components.loss:MSELoss"


@config(type="chunked_cross_entropy")
class ChunkedCELossConfig(BaseLossConfig):
    _owner_spec: ClassVar[str] = "torchtitan.components.loss:ChunkedCELoss"

    num_chunks: int = 8
    """Number of chunks to split the sequence into."""

    num_chunks_on_monitoring_steps: int | None = None
    """Chunk count on active monitoring steps; None keeps num_chunks."""

    def __post_init__(self) -> None:
        for name in ("num_chunks", "num_chunks_on_monitoring_steps"):
            value = getattr(self, name)
            if value is None and name != "num_chunks":
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@config(type="mtp")
class MTPLossConfig(ChunkedCELossConfig):
    _owner_spec: ClassVar[str] = "torchtitan.components.loss:MTPLoss"

    # One weight per MTP head; ``len`` must equal the number of MTP heads
    # the model produces.
    loss_weights: list[float]

    def __post_init__(self):
        ChunkedCELossConfig.__post_init__(self)
        if not self.loss_weights:
            raise ValueError(
                "loss_weights is empty; use ChunkedCELoss without MTP heads"
            )
