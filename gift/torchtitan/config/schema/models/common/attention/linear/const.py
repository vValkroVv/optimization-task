"""Shared pieces of the linear-attention families (GDN, KDA)."""

from torchtitan.config.schema.models.common.norms.const import GatedNormActivation

__all__ = ["GatedNormActivation", "linear_delta_core_flops"]


def linear_delta_core_flops(
    *, num_v_heads: int, head_k_dim: int, head_v_dim: int
) -> int:
    """Approximate forward FLOPs per token of the delta-rule recurrence core.

    Per value head, three ``head_k_dim × head_v_dim`` matmuls over the recurrent
    state (readout, delta correction, gated update) = ``6·head_k_dim·head_v_dim``
    FLOPs. Linear in seq_len (no ``s²`` term). Omits elementwise gating/decay.
    """
    return num_v_heads * 6 * head_k_dim * head_v_dim
