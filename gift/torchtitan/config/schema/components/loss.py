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


@config(type="mtp")
class MTPLossConfig(ChunkedCELossConfig):
    _owner_spec: ClassVar[str] = "torchtitan.components.loss:MTPLoss"

    # One weight per MTP head; ``len`` must equal the number of MTP heads
    # the model produces.
    loss_weights: list[float]

    def __post_init__(self):
        if not self.loss_weights:
            raise ValueError(
                "loss_weights is empty; use ChunkedCELoss without MTP heads"
            )
