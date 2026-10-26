"""Parameter initialization helpers for torchtitan models.

``param_init`` maps local parameter names to ``InitSpec`` values
(``torchtitan.config.schema.protocols.module``); ``depth_scaled_std`` and
``FusedW13Init`` live in the schema so ``materialize()`` stays torch-free.
"""

from collections.abc import Callable

import torch

from torchtitan.config.schema.models.common.param_init import (
    depth_scaled_std,
    DepthScaled,
    DimScaled,
    FusedW13Init,
)
from torchtitan.config.schema.protocols.module import InitSpec
from torchtitan.protocols.module import register_param_init, resolve_param_init

__all__ = ["depth_scaled_std", "FusedW13Init"]


def _init_fused_w13(
    param: torch.Tensor,
    *,
    gate_init: Callable[[torch.Tensor], None],
    up_init: Callable[[torch.Tensor], None],
) -> None:
    assert param.ndim == 4 and param.size(1) == 2, (
        f"w13_E2FD must have shape (E, 2, F, D), but received {tuple(param.shape)}"
    )
    with torch.no_grad():
        gate_init(param[:, 0])
        up_init(param[:, 1])


def _unmaterialized(spec: InitSpec):
    raise ValueError(
        f"a {type(spec).__name__} init reached the runtime; the model builder "
        "materializes it (materialize_init) before build"
    )


register_param_init(DepthScaled)(_unmaterialized)
register_param_init(DimScaled)(_unmaterialized)
register_param_init(FusedW13Init)(
    lambda s: lambda param: _init_fused_w13(
        param,
        gate_init=resolve_param_init(s.gate),
        up_init=resolve_param_init(s.up),
    )
)
