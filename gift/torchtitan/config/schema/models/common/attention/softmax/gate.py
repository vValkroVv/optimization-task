from typing import ClassVar

from torchtitan.config.configurable import config, Derived
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.protocols.module import ModuleConfig


@config
class SigmoidGateConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.softmax.gate:SigmoidGate"
    )

    # ``out_features`` is the attention output width before ``wo``
    # (n_heads*head_dim for GQA, n_heads*v_head_dim for MLA); None: the owner sizes it.
    gate_proj: Derived[LinearConfig] = None

    def fwd_flops_per_token(self, seq_len: int) -> int:
        assert self.gate_proj is not None
        return self.gate_proj.fwd_flops_per_token(seq_len)
