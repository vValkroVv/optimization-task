"""Pure restrictions of the runtime's batched expert-bias update."""

from torchtitan.config.schema.models.common.moe.moe import MoEConfig
from torchtitan.config.schema.protocols.module import iter_module_configs


def validate_batched_load_balancing(model) -> None:
    moes = [
        (path, module)
        for path, module in iter_module_configs(model)
        if isinstance(module, MoEConfig)
    ]
    enabled = [(path, moe) for path, moe in moes if moe.load_balance_coeff]
    if not enabled:
        return
    if len(enabled) != len(moes):
        disabled = ", ".join(path for path, moe in moes if not moe.load_balance_coeff)
        raise ValueError(
            "batched MoE load balancing requires every placed MoE to enable it; "
            f"disabled at: {disabled}"
        )
    counts = {moe.num_experts for _, moe in enabled}
    if len(counts) > 1:
        sites = ", ".join(f"{path}={moe.num_experts}" for path, moe in enabled)
        raise ValueError(
            "batched MoE load balancing requires equal expert counts across "
            f"placed MoEs; got {sites}"
        )
