"""GIFT 3.5 model: hybrid Gated DeltaNet (GDN) + MLA MoE.

Most layers are GDN linear attention; every ``full_attn_period``-th layer is full
MLA. Linear layers are non-RoPE; with ``varlen=False`` and an sdpa MLA the whole
model is NONE mask-kind.

Every block norm (pre+post attention, pre+post FFN) and the final norm is a
zero-centered self-gated RMSNorm (:class:`GatedRMSNorm`). Every MoE router gate
takes the row-renormalizing init.

The MLA layers carry a sigmoid output gate (:class:`SigmoidGate`, config
``gate``): a ``dim -> n_heads*v_head_dim`` projection of the layer input,
``sigmoid``-gated into the attention output before ``wo``.
"""

from .model import GiftHybridModel
from .state_dict_adapter import GiftHybridStateDictAdapter

__all__ = [
    "GiftHybridModel",
    "GiftHybridStateDictAdapter",
]
