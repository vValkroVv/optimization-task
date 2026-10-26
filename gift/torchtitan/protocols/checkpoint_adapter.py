from typing import Any, Protocol

import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful


class CheckpointAdapter(Protocol):
    """How a sharding backend's model and optimizer state reach DCP.

    A backend without an adapter is checkpointed the upstream way: the model
    through ``get_model_state_dict``, everything through plain ``dcp.load``. A
    backend with one owns the model part of every save and load, and the
    in-repo HF conversion, which reads the plain model state dict, is refused.
    """

    def model_save_state_dict(self) -> dict[str, Any]:
        """The model entries DCP writes."""
        ...

    def wrap_optimizer_state(self, optimizer: Stateful) -> Stateful:
        """The optimizer as DCP should read and write it."""
        ...

    def load(
        self,
        states: dict[str, Any],
        *,
        checkpoint_id: str,
        process_group: dist.ProcessGroup | None,
        load_model: bool,
    ) -> None:
        """Load ``states`` and, when ``load_model``, the model itself."""
        ...
