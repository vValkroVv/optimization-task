from typing import Dict, List

import torch

from torchmetrics import Metric
from discrete_speech_metrics import UTMOS as utmos

from composer.utils import dist


class UTMOS(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    score: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        sr: int = 24000,
        output_key: str = "wav_gen",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_key = output_key
        if dist.is_initialized():
            if dist.get_local_rank() == 0:
                utmos(sr)
            dist.barrier()
        self.utmos = utmos(sr)
        self.sr = sr

        self.add_state("score", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    @torch.inference_mode()
    @torch.autocast("cuda", enabled=False)
    def update(self, outputs: Dict, labels: List[str]) -> Dict:
        wavs = outputs[self.output_key]
        if f"{self.output_key}_lengths" not in outputs:
            wav_lengths = torch.tensor([wavs.shape[-1]] * len(wavs), device=wavs.device)
        else:
            wav_lengths = outputs[f"{self.output_key}_lengths"]
        for wav, wav_len in zip(wavs, wav_lengths):
            self.score += self.utmos.predictor(wav[..., :wav_len], self.sr).sum()
            self.total += 1

    def compute(self) -> float:
        return self.score / self.total
