"""Video-audio (128007) route: processor -> VideoAudioCollator (`video_audio`
bundle) -> second conformer route -> scatter onto the 128007 placeholders."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sber_tests.test_utils.gigaomni_cpu import import_gigaomni_modules

M = import_gigaomni_modules()

IMAGE_ID, VIDEO_IMAGE_ID, AUDIO_ID, VIDEO_AUDIO_ID = 128004, 128005, 128006, 128007
TEXT_ID, TIMESTAMP_ID = 7, 9
FEAT_IN, D_MODEL, LLM_HIDDEN, ADAPTER_FACTOR = 64, 32, 24, 4
ENCODER, ADAPTER = (
    "ConformerForAudioLanguageModeling",
    "AudioLinearAdapterWithStackingSubsampler",
)
BF16_TOL = dict(atol=2**-6, rtol=2**-6)


def predicted_rows(length: int) -> int:
    return M.mm_utils.predict_spec_length(length, ADAPTER_FACTOR)


# ---------------------------------------------------------------- processor


def _processor(order: str = "interleaved"):
    return M.video_processor.GigaOmniVideoProcessor(
        vision_processor_params={"num_image_tokens_per_crop": 4},
        video_image_token_id=VIDEO_IMAGE_ID,
        video_audio_token_id=VIDEO_AUDIO_ID,
        video_audio_tokens_order=order,
    )


def _video_content(num_frames: int, spectrogram_length: int = None, mark: float = 1.0):
    """Pre-decoded `content` in the shape `_read_video` returns (no ffmpeg needed)."""
    content = {
        "pixel_values": torch.zeros(num_frames, 3, 8, 8),
        "frames_lengths": [4] * num_frames,
        "timestamps": [float(i) for i in range(num_frames)],
    }

    if spectrogram_length is not None:
        content["spectrogram"] = torch.full((spectrogram_length, FEAT_IN), mark)
        content["predicted_spectrogram_length"] = predicted_rows(spectrogram_length)

    return content


def _sample(num_files: int):
    tokens = [TEXT_ID, TEXT_ID]

    for _ in range(num_files):
        tokens += [VIDEO_IMAGE_ID, TEXT_ID]

    return {"text": {"input_ids": torch.tensor(tokens), "labels": torch.tensor(tokens)}}


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize("order", ["interleaved", "sequential"])
def test_processor_emits_per_clip_lengths_matching_placeholders(order):
    lengths = [40, 30]
    files = [
        {"type": "video_audio", "content": _video_content(3, lengths[0], mark=1.0)},
        {"type": "video_audio", "content": _video_content(2, lengths[1], mark=2.0)},
    ]

    sample = _processor(order)(files=files, dataset_sample=_sample(len(files)))

    soundtracks = sample["video_audio"]
    assert set(soundtracks) == {
        "spectrograms",
        "spectrogram_lengths",
        "processor_fallback_count",
    }
    assert torch.equal(soundtracks["spectrogram_lengths"], torch.tensor(lengths))
    assert soundtracks["spectrograms"].shape == (sum(lengths), FEAT_IN)
    assert torch.all(soundtracks["spectrograms"][: lengths[0]] == 1.0)
    assert torch.all(soundtracks["spectrograms"][lengths[0] :] == 2.0)
    assert sample["video"]["pixel_values"].shape[0] == 5
    assert (sample["text"]["input_ids"] == VIDEO_AUDIO_ID).sum() == sum(
        predicted_rows(length) for length in lengths
    )
    assert (sample["text"]["input_ids"] == VIDEO_IMAGE_ID).sum() == 3 * 4 + 2 * 4


@pytest.mark.unit
@pytest.mark.fast
def test_processor_mixed_video_and_video_audio_files():
    files = [
        {"type": "video", "content": _video_content(2)},
        {"type": "video_audio", "content": _video_content(2, 30)},
    ]

    sample = _processor()(files=files, dataset_sample=_sample(len(files)))

    assert torch.equal(sample["video_audio"]["spectrogram_lengths"], torch.tensor([30]))
    assert sample["video"]["pixel_values"].shape[0] == 4
    assert (sample["text"]["input_ids"] == VIDEO_AUDIO_ID).sum() == predicted_rows(30)


@pytest.mark.unit
@pytest.mark.fast
def test_processor_video_without_soundtrack_has_no_spectrogram_keys():
    files = [{"type": "video", "content": _video_content(2)}]

    sample = _processor()(files=files, dataset_sample=_sample(1))

    assert "video_audio" not in sample
    assert set(sample["video"]) == {
        "pixel_values",
        "processor_fallback_count",
        "num_crops_per_frame",
    }
    assert (sample["text"]["input_ids"] == VIDEO_AUDIO_ID).sum() == 0


@pytest.mark.unit
@pytest.mark.fast
def test_processor_read_soundtracks_false_treats_video_audio_as_video(monkeypatch):
    """The opt-out for models without an audio encoder: no decode, no placeholders."""
    processor = _processor()
    processor.read_soundtracks = False
    read_audio_flags = []

    def fake_read_video(path, read_audio=False):
        read_audio_flags.append(read_audio)
        return _video_content(2)

    monkeypatch.setattr(processor, "_read_video", fake_read_video)
    files = [{"type": "video_audio", "path": "clip.mp4"}]

    sample = processor(files=files, dataset_sample=_sample(1))

    assert read_audio_flags == [False]
    assert "video_audio" not in sample
    assert (sample["text"]["input_ids"] == VIDEO_AUDIO_ID).sum() == 0
    assert (sample["text"]["input_ids"] == VIDEO_IMAGE_ID).sum() == 2 * 4


def _raise(*args, **kwargs):
    raise OSError("no audio stream")


@pytest.mark.unit
@pytest.mark.fast
def test_processor_unreadable_soundtrack_falls_back_and_counts(monkeypatch):
    """Like an unreadable wav: a 20-row zero block, counted under `video_audio`
    (the `processor_fallback_counter` callback enforces the run-level limit)."""
    processor = _processor()
    processor.video_decode_backend = "pyav"
    monkeypatch.setattr(
        processor, "_read_video_frames_pyav", lambda path: ([object()] * 2, [0.0, 1.0])
    )
    monkeypatch.setattr(
        processor.vision_processor,
        "_preprocess_images",
        lambda frames, num_threads=1: {
            "pixel_values": torch.zeros(len(frames), 3, 8, 8),
            "num_crops_per_image": [1] * len(frames),
        },
    )
    monkeypatch.setattr(M.video_processor.av, "open", _raise)
    files = [{"type": "video_audio", "path": "clip.mp4"}]

    sample = processor(files=files, dataset_sample=_sample(1))

    soundtracks = sample["video_audio"]
    assert soundtracks["processor_fallback_count"] == 1
    assert sample["video"]["processor_fallback_count"] == 0
    assert torch.equal(soundtracks["spectrogram_lengths"], torch.tensor([20]))
    assert torch.all(soundtracks["spectrograms"] == 0)
    assert (sample["text"]["input_ids"] == VIDEO_AUDIO_ID).sum() == predicted_rows(20)


class _StubTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": [TIMESTAMP_ID]}


@pytest.mark.unit
@pytest.mark.fast
def test_processor_sequential_timestamps_keep_every_audio_placeholder():
    """The audio run is split per frame; the remainder must not be dropped."""
    num_frames, length = 4, 130
    assert predicted_rows(length) % num_frames != 0
    processor = _processor("sequential")
    processor.add_timestamps = True
    processor.tokenizer = _StubTokenizer()
    files = [{"type": "video_audio", "content": _video_content(num_frames, length)}]

    input_ids = processor(files=files, dataset_sample=_sample(1))["text"]["input_ids"]

    assert (input_ids == VIDEO_AUDIO_ID).sum() == predicted_rows(length)
    assert (input_ids == VIDEO_IMAGE_ID).sum() == num_frames * 4
    assert (input_ids == TIMESTAMP_ID).sum() == 2 * num_frames


# ----------------------------------------------------------------- collator


SPECTROGRAM_COLLATORS = [
    ("audio_collator", "AudioCollator"),
    ("video_audio_collator", "VideoAudioCollator"),
]


def _collator(module: str, name: str):
    return getattr(getattr(M, module), name)()


def _soundtrack_example(lengths, mark: float = 1.0):
    """What the processor writes for one sample's clips, either as the `audio`
    slot or as the `video_audio` slot."""
    return {
        "spectrograms": torch.full((sum(lengths), FEAT_IN), mark),
        "spectrogram_lengths": torch.tensor(lengths),
        "processor_fallback_count": 0,
    }


def _batch(collator, examples, microbatch_size: int):
    modality_batch = collator(examples)
    bundle_name = collator.data_modality.value
    batch = {
        "input_ids": torch.zeros(len(examples), 4, dtype=torch.long),
        "modality_inputs": {bundle_name: modality_batch} if modality_batch else {},
    }
    return modality_batch, collator.split_batch(batch, microbatch_size)


@pytest.mark.unit
@pytest.mark.fast
def test_video_audio_collator_round_trip_at_microbatch_size_two():
    examples = [
        _soundtrack_example((40, 30), mark=1.0),
        None,
        None,
        _soundtrack_example((20,), mark=4.0),
    ]

    modality_batch, microbatches = _batch(
        _collator("video_audio_collator", "VideoAudioCollator"),
        examples,
        microbatch_size=2,
    )

    assert torch.equal(
        modality_batch["num_spectrograms_per_sample"], torch.tensor([2, 0, 0, 1])
    )
    assert torch.equal(
        modality_batch["spectrogram_lengths"], torch.tensor([40, 30, 20])
    )
    assert modality_batch["spectrograms"].shape == (90, FEAT_IN)

    first, second = microbatches
    assert set(first) == set(second) == {"video_audio"}
    assert torch.equal(
        first["video_audio"]["spectrogram_lengths"], torch.tensor([40, 30])
    )
    assert torch.all(first["video_audio"]["spectrograms"] == 1.0)
    assert torch.equal(second["video_audio"]["spectrogram_lengths"], torch.tensor([20]))
    assert torch.all(second["video_audio"]["spectrograms"] == 4.0)


@pytest.mark.unit
@pytest.mark.fast
def test_video_audio_collator_microbatch_without_clips_has_no_bundle():
    examples = [_soundtrack_example((40,)), None, None, None]

    _, microbatches = _batch(
        _collator("video_audio_collator", "VideoAudioCollator"),
        examples,
        microbatch_size=2,
    )

    assert torch.equal(
        microbatches[0]["video_audio"]["spectrogram_lengths"], torch.tensor([40])
    )
    assert microbatches[1] == {}


def _split_again(collator, microbatches, microbatch_size: int):
    """Split every microbatch once more, as the trainer does under dynamic SP."""
    split_twice = []

    for microbatch in microbatches:
        batch = {
            "input_ids": torch.zeros(microbatch_size, 4, dtype=torch.long),
            "modality_inputs": microbatch,
        }
        split_twice += collator.split_batch(batch, 1)

    return split_twice


def _assert_same_microbatches(expected, actual):
    assert len(expected) == len(actual)

    for expected_item, actual_item in zip(expected, actual):
        assert set(expected_item) == set(actual_item)

        for bundle_name, bundle in expected_item.items():
            assert set(bundle) == set(actual_item[bundle_name])

            for key, value in bundle.items():
                assert torch.equal(value, actual_item[bundle_name][key])


@pytest.mark.unit
@pytest.mark.fast
def test_omni_collator_rejects_sample_slots_without_a_collator():
    """A config listing the video processor but not `VideoAudioCollator` would
    drop the soundtracks while their 128007 placeholders stay in input_ids."""
    text_collator = SimpleNamespace(data_modality=M.omni_data.DataModality.TEXT)
    omni_collator = M.omni_data.OmniCollator(
        collators=[text_collator, _collator("video_collator", "VideoCollator")]
    )
    examples = [
        {
            "text": {"input_ids": torch.zeros(4, dtype=torch.long)},
            "video": {
                "pixel_values": torch.zeros(1, 3, 8, 8),
                "processor_fallback_count": 0,
                "num_crops_per_frame": 1,
            },
            "video_audio": _soundtrack_example((40,)),
        }
    ]

    with pytest.raises(
        AssertionError, match=r"\['video_audio'\].*no collator.*video_audio_conformer"
    ):
        omni_collator(examples)


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize("module, name", SPECTROGRAM_COLLATORS)
def test_spectrogram_collator_microbatches_can_be_split_again(module, name):
    collator = _collator(module, name)
    examples = [
        _soundtrack_example((40, 30), mark=1.0),
        None,
        None,
        _soundtrack_example((20,), mark=4.0),
    ]
    _, split_once = _batch(collator, examples, microbatch_size=1)
    _, halves = _batch(collator, examples, microbatch_size=2)

    split_twice = _split_again(collator, halves, 2)

    _assert_same_microbatches(split_once, split_twice)


# -------------------------------------------------------------------- model


class _EmbeddingOnlyLanguageModel(nn.Module):
    """The decoder stand-in: owns the embedding table, records the merged
    ``inputs_embeds`` and returns them as the logits slot."""

    def __init__(self):
        super().__init__()
        # bf16 like the trained decoder, so the merged rows share one dtype
        self.embed_tokens = nn.Embedding(VIDEO_AUDIO_ID + 1, LLM_HIDDEN).bfloat16()
        self.config = SimpleNamespace(return_dict=False, pad_token_id=0)
        self.received = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds, **kwargs):
        self.received = inputs_embeds
        return (None, inputs_embeds)


def _build_components():
    torch.manual_seed(0)
    config = M.conformer_config.ConformerConfig(
        feat_in=FEAT_IN,
        n_layers=2,
        d_model=D_MODEL,
        n_heads=4,
        ff_expansion_factor=2,
        subsampling="striding_conv1d",
        subsampling_factor=4,
        subsampling_conv_channels=D_MODEL,
        attn_chunk_size=8,
        conv_kernel_size=5,
        conv_norm_type="layer_norm",
        dropout=0.0,
        dropout_emb=0.0,
        dropout_att=0.0,
        attention_backend="sdpa",
        max_position_seq_len=256,
    )
    encoder = M.conformer.ConformerForAudioLanguageModeling(config)
    adapter = M.audio_adapter.AudioLinearAdapterWithStackingSubsampler(
        ae_hidden_size=D_MODEL,
        llm_hidden_size=LLM_HIDDEN,
        subsampling_factor=ADAPTER_FACTOR,
    )
    return encoder.eval(), adapter.eval()


def _audio_modality(preset: str):
    return M.audio_modality.AudioModality(
        M.audio_modality.AudioModalityConfig(data_flow_preset=preset)
    )


def _video_only_modality():
    return M.video_only_modality.VideoOnlyModality(
        M.video_only_modality.VideoOnlyModalityConfig(
            data_flow_preset="video_only_intern_vit"
        )
    )


def _build_model(modalities):
    """Model shell around the real conformer + adapter and real modalities."""
    encoder, adapter = _build_components()
    model = object.__new__(M.modelling.GigaOmniForCausalLM)
    nn.Module.__init__(model)
    model.encoders = nn.ModuleDict({ENCODER: encoder})
    model.adapters = nn.ModuleDict({ADAPTER: adapter})
    model.heads = nn.ModuleDict()
    model.modalities = modalities
    model.language_model = _EmbeddingOnlyLanguageModel()
    return model


class _CallSpy:
    """Records the bundle sizes of every conformer call."""

    def __init__(self, component: nn.Module):
        self.calls = []
        component.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        (inputs,) = args
        self.calls.append([len(item["spectrogram_lengths"]) for item in inputs])


def _clip(length: int, mark: float) -> torch.Tensor:
    torch.manual_seed(int(mark * 10))
    return torch.randn(length, FEAT_IN) + mark


def _bundle(clips):
    return {
        "spectrograms": torch.cat(clips),
        "spectrogram_lengths": torch.tensor([clip.shape[0] for clip in clips]),
    }


def _input_ids(blocks):
    """`blocks` = sequence of (token_id, count); text tokens separate the blocks."""
    tokens = [TEXT_ID]

    for token_id, count in blocks:
        tokens += [token_id] * count + [TEXT_ID]

    return torch.tensor([tokens])


def _standalone_rows(model, clip: torch.Tensor) -> torch.Tensor:
    bundle = [_bundle([clip])]
    encoder, adapter = model.encoders[ENCODER], model.adapters[ADAPTER]

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        encoded = encoder.unwrap_tuple_output(bundle, encoder(bundle))
        return adapter.unwrap_tuple_output(encoded, adapter(encoded))[0][
            "audio_embeds"
        ].float()


def _forward(model, input_ids, modality_inputs):
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model.forward(input_ids=input_ids, modality_inputs=modality_inputs)

    return model.language_model.received.float()


@pytest.fixture
def cpu_dummies(monkeypatch):
    """The modality dummies allocate on cuda; redirect them to cpu for the test."""

    class _CpuTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        def zeros(self, *args, **kwargs):
            return torch.zeros(*args, **{**kwargs, "device": "cpu"})

        def tensor(self, *args, **kwargs):
            return torch.tensor(*args, **{**kwargs, "device": "cpu"})

    for module in (M.audio_modality, M.video_only_modality):
        monkeypatch.setattr(module, "torch", _CpuTorch())


@pytest.mark.unit
@pytest.mark.fast
def test_forward_scatters_audio_and_video_audio_rows_from_the_right_clips():
    model = _build_model(
        [_audio_modality("audio_conformer"), _audio_modality("video_audio_conformer")]
    )
    spy = _CallSpy(model.encoders[ENCODER])
    audio_clips = [_clip(40, mark=1.0), _clip(30, mark=2.0)]
    video_clip = _clip(20, mark=4.0)
    audio_rows = [predicted_rows(40), predicted_rows(30)]
    video_rows = predicted_rows(20)
    input_ids = _input_ids(
        [
            (AUDIO_ID, audio_rows[0]),
            (VIDEO_AUDIO_ID, video_rows),
            (AUDIO_ID, audio_rows[1]),
        ]
    )
    modality_inputs = {
        "audio": _bundle(audio_clips),
        "video_audio": _bundle([video_clip]),
    }

    merged = _forward(model, input_ids, modality_inputs)

    assert spy.calls == [[2, 1]]
    audio_positions = (input_ids[0] == AUDIO_ID).nonzero().flatten()
    video_positions = (input_ids[0] == VIDEO_AUDIO_ID).nonzero().flatten()
    expected_audio = torch.cat([_standalone_rows(model, clip) for clip in audio_clips])
    torch.testing.assert_close(merged[0, audio_positions], expected_audio, **BF16_TOL)
    torch.testing.assert_close(
        merged[0, video_positions], _standalone_rows(model, video_clip), **BF16_TOL
    )
    table = model.language_model.embed_tokens.weight.float()
    assert not torch.allclose(merged[0, video_positions[0]], table[VIDEO_AUDIO_ID])
    text_positions = (input_ids[0] == TEXT_ID).nonzero().flatten()
    torch.testing.assert_close(
        merged[0, text_positions], table[TEXT_ID].expand(len(text_positions), -1)
    )


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize(
    "with_audio, with_video_audio",
    [(True, False), (False, True), (False, False), (True, True)],
)
def test_dummy_matrix_runs_conformer_once_and_merges(
    cpu_dummies, with_audio, with_video_audio
):
    model = _build_model(
        [_audio_modality("audio_conformer"), _audio_modality("video_audio_conformer")]
    )
    spy = _CallSpy(model.encoders[ENCODER])
    blocks, modality_inputs = [], {}

    if with_audio:
        blocks.append((AUDIO_ID, predicted_rows(40)))
        modality_inputs["audio"] = _bundle([_clip(40, mark=1.0)])

    if with_video_audio:
        blocks.append((VIDEO_AUDIO_ID, predicted_rows(20)))
        modality_inputs["video_audio"] = _bundle([_clip(20, mark=4.0)])

    input_ids = _input_ids(blocks)

    merged = _forward(model, input_ids, modality_inputs)

    assert spy.calls == [[1, 1]]
    assert merged.shape == (1, input_ids.shape[1], LLM_HIDDEN)
    assert modality_inputs["audio"]["spectrogram_lengths"].shape == (1,)
    assert modality_inputs["video_audio"]["spectrogram_lengths"].shape == (1,)
    assert ("dummy_value" in modality_inputs["audio"]) is not with_audio
    assert ("dummy_value" in modality_inputs["video_audio"]) is not with_video_audio


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize("video_first", [True, False])
def test_dummy_injection_is_independent_of_modality_order(cpu_dummies, video_first):
    modalities = [
        _audio_modality("audio_conformer"),
        _audio_modality("video_audio_conformer"),
    ]
    video_only = _video_only_modality()
    modalities = [video_only] + modalities if video_first else modalities + [video_only]
    modality_inputs = {}

    for modality in modalities:
        modality.update_modality_inputs_with_dummy_values(modality_inputs)

    assert set(modality_inputs) == {"audio", "video", "video_audio"}

    for bundle_name in ("audio", "video_audio"):
        assert set(modality_inputs[bundle_name]) == {
            "spectrograms",
            "spectrogram_lengths",
            "dummy_value",
        }
        assert modality_inputs[bundle_name]["spectrograms"].shape == (20, FEAT_IN)

    assert set(modality_inputs["video"]) == {
        "pixel_values",
        "n_crops_per_sample",
        "num_crops_per_frame",
        "dummy_value",
    }
    assert modality_inputs["video"]["pixel_values"].shape == (1, 3, 448, 448)


@pytest.mark.unit
@pytest.mark.fast
def test_dummy_injection_keeps_real_inputs(cpu_dummies):
    real_inputs = {
        "video": {"pixel_values": torch.ones(2, 3, 8, 8)},
        "audio": _bundle([_clip(40, 1.0)]),
        "video_audio": _bundle([_clip(20, 4.0)]),
    }
    modality_inputs = {name: dict(bundle) for name, bundle in real_inputs.items()}

    for modality in [
        _video_only_modality(),
        _audio_modality("audio_conformer"),
        _audio_modality("video_audio_conformer"),
    ]:
        modality.update_modality_inputs_with_dummy_values(modality_inputs)

    for name, bundle in real_inputs.items():
        assert "dummy_value" not in modality_inputs[name]

        for key, value in bundle.items():
            assert modality_inputs[name][key] is value


# ------------------------------------------------- placeholder count check


@pytest.mark.unit
@pytest.mark.fast
def test_forward_rejects_real_clip_without_its_placeholders():
    """A processor inserting another id (e.g. the 128008 constructor default)
    leaves a real clip with zero placeholders: the merge must raise, not skip."""
    model = _build_model([_audio_modality("video_audio_conformer")])
    clip = _clip(20, mark=4.0)
    input_ids = _input_ids([(AUDIO_ID, predicted_rows(20))])

    with pytest.raises(AssertionError, match="got a real input"):
        _forward(model, input_ids, {"video_audio": _bundle([clip])})


@pytest.mark.unit
@pytest.mark.fast
def test_forward_twice_on_the_same_batch_keeps_the_dummies_dummy(cpu_dummies):
    """GigaFSDP's first `__call__` runs a fake forward on the batch before the
    real one; the dummies the first pass injected in place must not become real
    inputs with zero placeholders in the second."""
    model = _build_model(
        [_audio_modality("audio_conformer"), _audio_modality("video_audio_conformer")]
    )
    clip = _clip(20, mark=4.0)
    input_ids = _input_ids([(VIDEO_AUDIO_ID, predicted_rows(20))])
    modality_inputs = {"video_audio": _bundle([clip])}

    first = _forward(model, input_ids, modality_inputs)

    assert modality_inputs["audio"]["dummy_value"]

    second = _forward(model, input_ids, modality_inputs)

    assert torch.equal(first, second)
