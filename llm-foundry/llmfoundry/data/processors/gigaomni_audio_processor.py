import torch
import logging
import audioaug
import librosa
import numpy as np

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Dict, Tuple, Any, Optional
from transformers import AutoProcessor

from .base_processor import BaseProcessor, DataModality
from .mm_utils import (
    predict_spec_length,
    insert_tokens_sequences_on_positions,
    duplicate_tokens_on_positions,
    seconds_to_hms,
    parse_inter_timings_and_create_times,
)


logger = logging.getLogger(__name__)

_LIBROSA_AUDIO_SUFFIXES = frozenset({".opus", ".ogg", ".mp3", ".flac", ".m4a", ".aac"})


class GigaOmniAudioProcessor(BaseProcessor):
    data_modality = DataModality.AUDIO
    attributes = []
    valid_kwargs = [
        "dither_val",
        "high_freq_filterbank",
        "normalize_gain",
        "num_filterbank_chans",
        "sample_rate",
        "wav_scaling",
        "encoder_subsampling_factor",
        "projector_subsampling_factor",
        "add_duration",
        "add_timestamps",
        "inter_timings",
        "framerate_ms",
        "audio_token_id",
        "tokenizer",
    ]

    def __init__(
        self,
        dither_val: float = 1e-5,
        high_freq_filterbank: int = 8000,
        normalize_gain: bool = True,
        num_filterbank_chans: int = 64,
        sample_rate: int = 16000,
        wav_scaling: bool = True,
        add_duration: bool = False,
        add_timestamps: bool = False,
        inter_timings: Optional[List[str]] = None,
        framerate_ms: Optional[int] = None,
        encoder_subsampling_factor: Optional[int] = None,
        projector_subsampling_factor: int = 4,
        audio_token_id: int = 128006,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dither_val = dither_val
        self.high_freq_filterbank = high_freq_filterbank
        self.normalize_gain = normalize_gain
        self.num_filterbank_chans = num_filterbank_chans
        self.sample_rate = sample_rate
        self.wav_scaling = wav_scaling
        self.encoder_subsampling_factor = encoder_subsampling_factor
        self.projector_subsampling_factor = projector_subsampling_factor

        self.add_duration = add_duration
        self.add_timestamps = add_timestamps
        self.inter_timings = inter_timings
        self.framerate_ms = framerate_ms

        if self.inter_timings and framerate_ms is None:
            assert encoder_subsampling_factor is not None, \
                "inter_timings requires either framerate_ms or encoder_subsampling_factor!"

            self.framerate_ms = 10 * int(encoder_subsampling_factor) * int(projector_subsampling_factor)

        self.audio_token_id = audio_token_id

        self.auto_map = {
            "AutoProcessor": "gigaomni_speech_processor.GigaOmniSpeechProcessor",
        }

    def __call__(
        self,
        files: Optional[List[Dict[str, Any]]] = None,
        dataset_sample: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess audio files and insert audio tokens into the text input_ids and labels.
        You can provide dataset_sample in the form of:
        {
            "text": {
                "input_ids": torch.LongTensor,
                "labels": torch.LongTensor
            }
        } or input_ids and labels as separate arguments.
        """
        self._processor_fallback_count = 0

        file_contents = self._prepare_file_contents(files)

        dataset_sample = self._prepare_dataset_sample(
            dataset_sample, input_ids, labels, attention_mask
        )

        if len(file_contents) == 0:
            return dataset_sample

        dataset_sample = self._insert_audio_tokens(dataset_sample, file_contents)

        dataset_sample[self.data_modality.value] = {
            "spectrograms": torch.cat([spectrogram for spectrogram, _ in file_contents], dim=0),
            "spectrogram_lengths": torch.tensor(
                [spectrogram.shape[0] for spectrogram, _ in file_contents],
                dtype=torch.long,
            ),
            "processor_fallback_count": self._processor_fallback_count,
        }

        return dataset_sample

    def get_num_tokens(self, files: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Number of tokens that `__call__` will insert in place of the audio placeholders,
        including inter-timing (`add_timestamps`) and duration (`add_duration`) tokens.
        """

        file_contents = self._prepare_file_contents(files)

        if len(file_contents) == 0:
            return 0

        return sum(
            len(token_sequence)
            for token_sequence in self._build_tokens_sequences(file_contents)
        )

    def _build_tokens_sequences(
        self, file_contents: List[Tuple[torch.Tensor, float]]
    ) -> List[List[int]]:
        """
        Build the full token sequence for every audio file: audio placeholders,
        optionally interleaved with inter-timing tokens and followed by duration tokens.
        """
        tokens_sequences = [
            [self.audio_token_id]
            * predict_spec_length(spectrogram.shape[0], self.projector_subsampling_factor)
            for spectrogram, _ in file_contents
        ]

        if self.add_timestamps:
            tokens_sequences = self._add_timestamps(tokens_sequences, file_contents)

        if self.add_duration:
            tokens_sequences = self._add_duration(tokens_sequences, file_contents)

        return tokens_sequences

    def _insert_audio_tokens(
        self,
        dataset_sample: Dict[str, Dict[str, torch.Tensor]],
        file_contents: List[Tuple[torch.Tensor, float]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        tokens_sequences = [
            torch.tensor(token_sequence, dtype=torch.long)
            for token_sequence in self._build_tokens_sequences(file_contents)
        ]

        modality_token_positions = (
            (dataset_sample["text"]["input_ids"] == self.audio_token_id)
            .nonzero(as_tuple=True)[0]
            .tolist()
        )

        dataset_sample["text"]["input_ids"] = insert_tokens_sequences_on_positions(
            dataset_sample["text"]["input_ids"],
            tokens_sequences,
            modality_token_positions,
        )

        if "labels" in dataset_sample["text"]:
            dataset_sample["text"]["labels"] = duplicate_tokens_on_positions(
                dataset_sample["text"]["labels"],
                [len(token_sequence) for token_sequence in tokens_sequences],
                modality_token_positions,
            )

        return dataset_sample

    def _add_timestamps(self, tokens_sequences: List[List[int]], file_contents: List[Tuple[torch.Tensor, float]]) -> List[List[int]]:
        updated_tokens_sequences = []
        for token_sequence, (_, duration) in zip(tokens_sequences, file_contents):
            times = parse_inter_timings_and_create_times(self.inter_timings, int(duration))
            timstamp_positions = [int(time * 1000 // self.framerate_ms) for time in times]

            updated_tokens_sequences.append(
                insert_tokens_sequences_on_positions(
                    torch.tensor(token_sequence, dtype=torch.long),
                    [torch.tensor(self.tokenizer.encode(seconds_to_hms(time), add_special_tokens=False), dtype=torch.long) for time in times],
                    timstamp_positions,
                    remove_tokens_on_positions=False,
                ).tolist()
            )

        return updated_tokens_sequences

    def _add_duration(self, tokens_sequences: List[List[int]], file_contents: List[Tuple[torch.Tensor, float]]) -> List[List[int]]:
        for token_sequence, (_, duration) in zip(tokens_sequences, file_contents):
            timing_string = seconds_to_hms(duration)
            token_sequence.extend(self.tokenizer.encode(timing_string, add_special_tokens=False))
        return tokens_sequences

    def _prepare_file_contents(
        self, files: Optional[List[Dict[str, Any]]] = None
    ) -> List[Tuple[torch.Tensor, float]]:
        if not files:
            return []

        file_contents = []
        for file in files:
            if file["type"] != self.data_modality.value:
                continue

            content = file.get("content")
            if isinstance(content, bytes):
                file_contents.append(
                    self._read_audio_bytes(content, file.get("name", "audio.wav"))
                )
            elif isinstance(content, (np.ndarray, torch.Tensor)):
                file_contents.append(self._read_audio_waveform(content))
            elif content is not None:
                file_contents.append(content)
            else:
                file_contents.append(self._read_audio(file["path"]))

        return file_contents

    def _read_audio(self, audio_path: str) -> Tuple[torch.Tensor, float]:
        try:
            if not Path(audio_path).exists():
                raise ValueError(f"Audio file {audio_path} does not exist!")

            if Path(audio_path).suffix.lower() in _LIBROSA_AUDIO_SUFFIXES:
                try:
                    wav, _ = librosa.load(
                        audio_path, sr=self.sample_rate, mono=True
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Error reading audio file {audio_path}! Error:\n{exc}"
                    )
                return self._read_audio_waveform(wav)

            read_dtype = np.float32 if self.wav_scaling else np.int16

            try:
                wav, info = audioaug.read_waveform(audio_path, dtype=read_dtype)
            except Exception as exc:
                raise ValueError(f"Error reading audio file {audio_path}! Error:\n{exc}")

            if info.sample_rate != self.sample_rate:
                aug = audioaug.Resample(self.sample_rate)
                wav, info = audioaug.apply_augmentation(
                    wav, info, [aug], return_dtype=wav.dtype
                )

            duration = info.frames / info.sample_rate

            features_params = self._get_audioaug_features_params()
            spectrogram = audioaug.extract_features(wav, info, features_params)
            spectrogram = torch.from_numpy(spectrogram).view(-1, self.num_filterbank_chans)

            assert spectrogram.shape[0] > 0, (
                f"Got zero-sized spectrogram from {audio_path}!"
            )
            return spectrogram, duration
        except Exception as exc:
            logger.error(
                f"Error reading audio spectrograms from `{audio_path}` with audioaug, returning zero-sized spectrograms! Error:\n{exc}"
            )

            self._processor_fallback_count += 1
            return torch.zeros(20, self.num_filterbank_chans, dtype=torch.float32), 1

    def _read_audio_bytes(
        self, content: bytes, filename: str
    ) -> Tuple[torch.Tensor, float]:
        """Decode an audio file fetched from an object-store payload."""
        suffix = Path(filename).suffix or ".wav"
        with NamedTemporaryFile(suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            return self._read_audio(temp_file.name)

    def _read_audio_waveform(
        self, waveform: np.ndarray | torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Extract model features from a decoded mono waveform."""
        try:
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.detach().cpu().numpy()
            wav = np.asarray(waveform)

            if wav.ndim == 2:
                channel_axis = 0 if wav.shape[0] <= wav.shape[1] else 1
                wav = wav.mean(axis=channel_axis)
            if wav.ndim != 1:
                raise ValueError(f"Expected mono audio waveform, got shape {wav.shape}")

            if np.issubdtype(wav.dtype, np.integer):
                if self.wav_scaling:
                    wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
                else:
                    wav = wav.astype(np.int16)
            elif self.wav_scaling:
                wav = wav.astype(np.float32)
            else:
                wav = (np.clip(wav, -1.0, 1.0) * np.iinfo(np.int16).max).astype(
                    np.int16
                )

            info = audioaug.AudioInfo(
                sample_rate=self.sample_rate,
                channels=1,
                frames=wav.shape[0],
            )
            spectrogram = audioaug.extract_features(
                wav, info, self._get_audioaug_features_params()
            )
            spectrogram = torch.from_numpy(spectrogram).view(
                -1, self.num_filterbank_chans
            )
            assert spectrogram.shape[0] > 0, "Got zero-sized spectrogram from waveform!"
            return spectrogram, wav.shape[0] / self.sample_rate
        except Exception as exc:
            logger.error(
                f"Error reading audio spectrograms from waveform, returning zero-sized spectrograms! Error:\n{exc}"
            )

            self._processor_fallback_count += 1
            return torch.zeros(20, self.num_filterbank_chans, dtype=torch.float32), 1

    def _get_audioaug_features_params(self) -> audioaug.FeatureParams:
        return audioaug.FeatureParams(
            sampling_freq=self.sample_rate,
            high_freq_filterbank=self.high_freq_filterbank,
            dither_val=self.dither_val,
            normalize_gain=self.normalize_gain,
            num_filterbank_chans=self.num_filterbank_chans,
        )


GigaOmniAudioProcessor.register_for_auto_class(AutoProcessor)
