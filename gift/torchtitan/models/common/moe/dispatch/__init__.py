from torch.distributed.device_mesh import DeviceMesh

from .all_to_all import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    permute_to_expert_major,
    unpermute_from_expert_major,
)
from .base import BaseEPTokenDispatcher, BaseTokenDispatcher
from .deepep import DeepEPDispatchMetadata, DeepEPTokenDispatcher
from .hybridep import HybridEPDispatchMetadata, HybridEPTokenDispatcher
from .local import (
    local_reorder,
    LocalDispatchMetadata,
    LocalTokenDispatcher,
    score_and_scatter_add,
)

__all__ = [
    "AllToAllDispatchMetadata",
    "AllToAllTokenDispatcher",
    "BaseEPTokenDispatcher",
    "BaseTokenDispatcher",
    "DeepEPDispatchMetadata",
    "DeepEPTokenDispatcher",
    "HybridEPDispatchMetadata",
    "HybridEPTokenDispatcher",
    "LocalDispatchMetadata",
    "LocalTokenDispatcher",
    "dispatcher_for",
    "local_reorder",
    "permute_to_expert_major",
    "unpermute_from_expert_major",
    "score_and_scatter_add",
]


def dispatcher_for(
    dispatcher: BaseTokenDispatcher, *, ep_mesh: DeviceMesh | None
) -> BaseTokenDispatcher:
    """The dispatcher to use now that the EP mesh is known, meshes wired.

    The choice lives here rather than as a runtime branch inside a backend: an
    EP=1 fallback made the all-to-all one's metadata type a union of two.

    Both directions, so neither the default nor an explicit config keeps a
    backend the mesh cannot support. ``DeepEP``/``HybridEP`` are never
    substituted: they require EP, validated in ``prepare_config``.
    """
    config = None
    if ep_mesh is None:
        if isinstance(dispatcher, AllToAllTokenDispatcher):
            config = LocalTokenDispatcher.Config(token_drop=dispatcher.token_drop)
    elif isinstance(dispatcher, LocalTokenDispatcher):
        config = AllToAllTokenDispatcher.Config(token_drop=dispatcher.token_drop)
    if config is not None:
        config._capacity_num_tokens = dispatcher._capacity_num_tokens
        dispatcher = config.build()
    dispatcher.wire_meshes(ep_mesh=ep_mesh)
    return dispatcher


# A backend is selected by naming its config on the experts
# (``token_dispatcher=DeepEPTokenDispatcher.Config()``):
# - ``AllToAllTokenDispatcher`` -- PyTorch all-to-all collectives. What the
#   default ``LocalTokenDispatcher`` becomes once an EP mesh exists; at EP=1
#   ``dispatcher_for`` keeps ``LocalTokenDispatcher``.
# - ``DeepEPTokenDispatcher`` -- DeepEP custom kernels for H100/NVLink Switch.
# - ``HybridEPTokenDispatcher`` -- HybridEP with TMA optimization for
#   GB200/NVLink72; its ``non_blocking_capacity_factor`` knob lives on its own
#   config. SM counts come from HYBRIDEP_NUM_SMS_{DISPATCH,COMBINE}.
#
# DeepEP/HybridEP need https://github.com/deepseek-ai/DeepEP installed, and both
# require expert parallelism (validated in ``Decoder.prepare_config``).
