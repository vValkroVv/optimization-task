import math
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.nn_modules import LinearConfig


@config(type="unembedding")
class UnembeddingConfig(LinearConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.embeddings.token_debeddings:Unembedding"
    )

    # 0.0 disables the cap (bitwise-identical to a plain Linear).
    logit_softcap: float = 0.0

    def __post_init__(self) -> None:
        LinearConfig.__post_init__(self)
        if not math.isfinite(self.logit_softcap) or self.logit_softcap < 0:
            raise ValueError("logit_softcap must be finite and >= 0")
