import os
import re
from typing import Dict, List

import gigaam
import torch
from gigaam.preprocess import SAMPLE_RATE as GIGAAM_SAMPLE_RATE
from gigaam.vad_utils import segment_audio
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import Resample
from torchmetrics import Metric
from torchmetrics.text import CharErrorRate, WordErrorRate

from composer.utils import dist


class TextErrorRateMixin(Metric):
    def update(self, outputs: Dict, labels: List[str]):
        output_list_copy = outputs["text"].copy()
        labels_copy = labels.copy()
        assert len(output_list_copy) == len(labels_copy)

        for i in range(len(output_list_copy)):
            output_list_copy[i] = re.sub(r"[^\w\s]", "", output_list_copy[i]).lower()
            labels_copy[i] = re.sub(r"[^\w\s]", "", labels_copy[i]).lower()

        super().update(output_list_copy, labels_copy)


class WordErrorRateComposer(TextErrorRateMixin, WordErrorRate): ...


class ASRMetricMixin(Metric):
    MAX_WAV_VALUE = 32768.0

    def __init__(
        self,
        orig_sample_rate: int = 24000,
        asr_model: str = "ctc",
        output_key: str = "wav_gen",
        label_key: str = "wav_gt",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_key = output_key
        self.label_key = label_key
        self.orig_sample_rate = orig_sample_rate
        self.resample = Resample(orig_sample_rate, GIGAAM_SAMPLE_RATE)
        if dist.is_initialized():
            if dist.get_local_rank() == 0:
                gigaam.load_model(asr_model, weights_only=False)
            dist.barrier()
        self.asr_model = gigaam.load_model(asr_model, weights_only=False)
        self._dtype = self.asr_model._dtype

    @torch.inference_mode()
    @torch.autocast("cuda", enabled=False)
    def transcribe(self, wavs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        wavs = self.resample(wavs.view(wavs.shape[0], 1, -1))
        wavs_int = (wavs * self.MAX_WAV_VALUE).to(torch.int16)
        num_seg_in_wavs, batch_segments, seg_lengths = [], [], []
        for wav, leng in zip(wavs_int, lengths):
            segments, _ = segment_audio(
                wav[None, :, :leng].cpu(), GIGAAM_SAMPLE_RATE, device=wavs.device
            )
            batch_segments.extend(segments)
            seg_lengths.extend([seg.shape[-1] for seg in segments])
            num_seg_in_wavs.append(len(segments))
        seg_lengths = torch.tensor(seg_lengths, device=wavs.device)
        if len(batch_segments) > 0:
            batch_segments = pad_sequence(batch_segments, batch_first=True).to(wavs.device)
        else:
            batch_segments = wavs
            num_seg_in_wavs = [1] * len(wavs)
            seg_lengths = lengths

        encoded, encoded_len = self.asr_model.forward(batch_segments.squeeze(1), seg_lengths)
        transcriptions = self.asr_model.decoding.decode(self.asr_model.head, encoded, encoded_len)
        cat_transcriptions = []
        for n in num_seg_in_wavs:
            cat_transcriptions.append(" ".join(transcriptions[:n]))
            transcriptions = transcriptions[n:]

        return cat_transcriptions

    def update(self, outputs: Dict, labels: List[str]) -> Dict:
        if "transcrib_gen" not in outputs:
            outputs["transcrib_gen"] = self.transcribe(
                outputs[self.output_key], outputs[f"{self.output_key}_lengths"]
            )
        if "transcrib_gt" not in outputs:
            outputs["transcrib_gt"] = self.transcribe(
                outputs[self.label_key], outputs[f"{self.label_key}_lengths"]
            )
        return super().update({"text": outputs["transcrib_gen"]}, outputs["transcrib_gt"])


class SpeechWER(ASRMetricMixin, TextErrorRateMixin, WordErrorRate): ...


class SpeechCER(ASRMetricMixin, TextErrorRateMixin, CharErrorRate): ...
