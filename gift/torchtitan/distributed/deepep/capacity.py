"""Host-side bounds shared with lightweight HybridEP configuration."""

from torchtitan.config.schema.models.common.moe.dispatch.capacity import (
    non_blocking_buffer_capacity,
    retained_receive_capacity,
)

__all__ = ["non_blocking_buffer_capacity", "retained_receive_capacity"]
