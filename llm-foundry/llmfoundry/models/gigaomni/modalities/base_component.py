import torch

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BaseOmniComponent(ABC):
    _top_level_training: bool = False

    def set_top_level_training(self, training: bool) -> None:
        """Record whether the TOP-LEVEL model is in training mode.

        Called by ``GigaOmniForCausalLM._forward_components`` before every
        component forward. Components whose behaviour must follow the run
        phase rather than their own ``.training`` (which a frozen-encoder eval
        wrap may flip) read ``self._top_level_training``.
        """
        self._top_level_training = bool(training)

    @abstractmethod
    def unwrap_tuple_output(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        outputs: Tuple[torch.Tensor, ...],
    ) -> List[Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def _concat_inputs(self, inputs: List[Dict[str, torch.Tensor]]) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def _split_outputs(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        pass
