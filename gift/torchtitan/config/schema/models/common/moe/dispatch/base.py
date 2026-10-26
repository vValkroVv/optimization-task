from typing import Self

from torchtitan.config.moe import TokenDropConfig
from torchtitan.config.configurable import config, Configurable


@config(family="token_dispatcher")
class BaseTokenDispatcherConfig(Configurable.Config):
    """Token dispatcher; top_k and num_experts arrive with the ``dispatch`` call."""

    token_drop: TokenDropConfig | None = None

    @property
    def _capacity_num_tokens(self) -> int | None:
        return self._get_metadata("capacity_num_tokens", None)

    @_capacity_num_tokens.setter
    def _capacity_num_tokens(self, value: int | None) -> None:
        self._set_metadata("capacity_num_tokens", value)

    def with_capacity_context(
        self, *, num_tokens: int, top_k: int, num_experts: int, ep_size: int
    ) -> Self:
        """Bind one-forward token bounds before constructing the backend."""
        config = self._copy()
        config._capacity_num_tokens = num_tokens
        policy = config.token_drop
        if policy is not None:
            policy.source_capacity(
                num_tokens=num_tokens,
                top_k=top_k,
                num_experts=num_experts,
                ep_size=ep_size,
                max_num_tokens=num_tokens,
            )
        return config

    def __post_init__(self) -> None:
        if self.token_drop is not None:
            self.token_drop.validate()
