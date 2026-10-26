from typing import ClassVar

from torchtitan.config.configurable import config, OwnerFilled
from torchtitan.config.schema.protocols.module import ModuleConfig
from torchtitan.config.schema.models.common.validation import require_completed


@config
class EmbeddingConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.embeddings.token_embeddings:Embedding"
    )
    default_init_role: ClassVar[str] = "embedding"

    # Filled by the owning model when unset.
    num_embeddings: OwnerFilled[int] = None
    embedding_dim: OwnerFilled[int] = None

    def require_dimensions(self) -> tuple[int, int]:
        return (
            require_completed(self.num_embeddings, "Embedding.num_embeddings"),
            require_completed(self.embedding_dim, "Embedding.embedding_dim"),
        )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        return 0
