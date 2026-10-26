import os
import sys
import pytest
import torch

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib", "composer"))
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib", "gigafsdp"))
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib", "streaming"))
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib", "flash-linear-attention"))

from llmfoundry.data.processors.gigaomni_audio_processor import GigaOmniAudioProcessor  # noqa: E402

AUDIO_TOKEN_ID = 128006
NUM_CHANS = 64
# 10 ms per spectrogram frame, so `duration` seconds -> 100 * duration frames.
FRAMES_PER_SECOND = 100


class _CharTokenizer:
    """Deterministic stand-in for a text tokenizer: one token per character."""

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) for char in text]


def _audio_file(duration_seconds: float):
    spectrogram = torch.zeros(
        int(duration_seconds * FRAMES_PER_SECOND), NUM_CHANS, dtype=torch.float32
    )
    return {"type": "audio", "content": (spectrogram, duration_seconds)}


def _make_processor(**kwargs):
    processor = GigaOmniAudioProcessor(
        audio_token_id=AUDIO_TOKEN_ID,
        num_filterbank_chans=NUM_CHANS,
        encoder_subsampling_factor=2,
        projector_subsampling_factor=4,
        **kwargs,
    )
    processor.tokenizer = _CharTokenizer()
    return processor


def _num_inserted_tokens(processor, files):
    """Count what `__call__` really inserts in place of the audio placeholders."""
    prefix = [1, 2, 3]
    suffix = [4, 5]
    placeholders = [AUDIO_TOKEN_ID] * len(files)
    input_ids = torch.tensor(prefix + placeholders + suffix, dtype=torch.long)
    result = processor(files=files, input_ids=input_ids, labels=input_ids.clone())
    return result["text"]["input_ids"].numel() - len(prefix) - len(suffix)


@pytest.mark.unit
class TestGigaOmniAudioGetNumTokens:
    def test_empty_files_returns_zero(self):
        processor = _make_processor(add_duration=True, add_timestamps=True, inter_timings=["1s"])
        assert processor.get_num_tokens(None) == 0
        assert processor.get_num_tokens([]) == 0
        assert processor.get_num_tokens([{"type": "image", "path": "x.png"}]) == 0

    @pytest.mark.parametrize(
        "processor_kwargs",
        [
            {},
            {"add_duration": True},
            {"add_timestamps": True, "inter_timings": ["1s"]},
            {"add_timestamps": True, "inter_timings": ["2s"]},
            {"add_timestamps": True, "inter_timings": ["1s", "5s"]},
            {"add_timestamps": True, "add_duration": True, "inter_timings": ["2s"]},
        ],
    )
    def test_matches_tokens_inserted_by_call(self, processor_kwargs):
        processor = _make_processor(**processor_kwargs)
        files = [_audio_file(1.0), _audio_file(6.0), _audio_file(65.0)]

        assert processor.get_num_tokens(files) == _num_inserted_tokens(processor, files)

    def test_timestamps_and_duration_add_tokens(self):
        files = [_audio_file(6.0)]
        plain = _make_processor().get_num_tokens(files)
        extended = _make_processor(
            add_timestamps=True, add_duration=True, inter_timings=["2s"]
        ).get_num_tokens(files)

        assert extended > plain
