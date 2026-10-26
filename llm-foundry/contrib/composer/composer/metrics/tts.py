from typing import Mapping, Any

import torch
from torch import Tensor
from torchmetrics import Metric

from .nlp import (
    LanguageCrossEntropy
)


###
### Cross Entropy
###

class TextCrossEntropy(LanguageCrossEntropy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def update(self, output: Mapping, target: Mapping):
        output = output['text_logits']
        target = target['text_labels'].reshape(-1)
        super().update(output=output, target=target)

    def compute(self) -> Tensor:
        return self.sum_loss / (self.total_items + 1e-8)


class SpeechCrossEntropy(LanguageCrossEntropy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def update(self, output: Mapping, target: Mapping):
        output = output['speech_logits']
        target = target['speech_labels'].reshape(-1)
        super().update(output=output, target=target)

    def compute(self) -> Tensor:
        return self.sum_loss / (self.total_items + 1e-8)



###
### Accuracy
###

class NextTokenAccuracy(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    correct: Tensor
    total: Tensor

    def __init__(
        self,
        out_key: str,
        label_key: str,
        ignore_token_id: int = -100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.out_key = out_key
        self.label_key = label_key
        self.ignore_token_id = ignore_token_id
        self.add_state("correct", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, output: Mapping, target: Mapping):
        output = output[self.out_key].argmax(-1)
        target = target[self.label_key]
        mask = target != self.ignore_token_id
        self.correct += ((output == target) & mask).long().sum()
        self.total += mask.long().sum()

    def compute(self):
        return self.correct / (self.total + 1e-8)


class TextNextTokenAccuracy(NextTokenAccuracy):
    def __init__(
        self,
        out_key: str = "text_logits",
        label_key: str = "text_labels",
        ignore_token_id: int = -100,
        **kwargs: Any,
    ) -> None:
        super().__init__(out_key, label_key, ignore_token_id, **kwargs)


class SpeechNextTokenAccuracy(NextTokenAccuracy):
    def __init__(
        self,
        out_key: str = "speech_logits",
        label_key: str = "speech_labels",
        ignore_token_id: int = -100,
        **kwargs: Any,
    ) -> None:
        super().__init__(out_key, label_key, ignore_token_id, **kwargs)
