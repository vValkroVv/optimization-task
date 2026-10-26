import os
import sys
import math
import pytest
import numpy as np
import torch

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib", "composer"))
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib", "gigafsdp"))
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib", "streaming"))
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib", "flash-linear-attention"))

av = pytest.importorskip("av")

from PIL import Image  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from transformers import CLIPImageProcessor, Qwen2VLImageProcessor  # noqa: E402

from llmfoundry.data import processors  # noqa: E402
from llmfoundry.data.datasets.omni_mcqa_dataset import OmniMCQADataset  # noqa: E402
from llmfoundry.data.processors import gigaomni_video_processor as video_module  # noqa: E402
from llmfoundry.data.processors.gigaomni_video_processor import (  # noqa: E402
    AV_TIME_BASE,
    GigaOmniVideoProcessor,
    available_cpu_cores,
)


FPS = 30
SECONDS = 10
GOP_SIZE = 30
FRAME_SIZE = 80
N_BANDS = 10
BAND_PX = FRAME_SIZE // N_BANDS
CROP_SIZE = 448


def _gray(idx: int) -> int:
    return idx ^ (idx >> 1)


def _ungray(code: int) -> int:
    idx = 0
    while code:
        idx ^= code
        code >>= 1
    return idx


def paint_frame(idx: int) -> np.ndarray:
    """Solid horizontal bands: band k is white iff bit k of gray(idx) is set.

    Adjacent frame indices differ in exactly one band, so the encoder sees a
    small change per frame and never inserts a scene-cut keyframe.
    """
    code = _gray(idx)
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    for band in range(N_BANDS):
        if code >> band & 1:
            frame[band * BAND_PX : (band + 1) * BAND_PX] = 255

    return frame


def frame_index(image: Image.Image) -> int:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    assert array.shape[:2] == (FRAME_SIZE, FRAME_SIZE), array.shape
    code = 0
    for band in range(N_BANDS):
        band_mean = array[band * BAND_PX : (band + 1) * BAND_PX].mean()
        assert band_mean < 64 or band_mean > 192, (
            f"band {band} mean {band_mean:.1f} is ambiguous"
        )
        if band_mean > 128:
            code |= 1 << band

    return _ungray(code)


def make_synthetic_mp4(
    path,
    seconds: int = SECONDS,
    fps: int = FPS,
    gop_size: int = GOP_SIZE,
    codec: str = "mpeg4",
    with_audio: bool = False,
    pts_offset_frames: int = 0,
) -> str:
    """Encode `seconds * fps` painted frames with a fixed GOP; return the path. `pts_offset_frames` > 0 starts the stream late (non-zero `start_time`)."""
    n_frames = seconds * fps
    with av.open(str(path), "w") as container:
        video = container.add_stream(codec, rate=fps)
        video.width = FRAME_SIZE
        video.height = FRAME_SIZE
        video.pix_fmt = "yuv420p"
        video.gop_size = gop_size
        video.options = {"bf": "0", "sc_threshold": "1000000000"}

        audio = None
        if with_audio:
            audio = container.add_stream("aac", rate=16000)
            audio.layout = "mono"

        for idx in range(n_frames):
            frame = av.VideoFrame.from_ndarray(paint_frame(idx), format="rgb24")
            if pts_offset_frames > 0:
                frame.pts = idx + pts_offset_frames

            for packet in video.encode(frame):
                container.mux(packet)

        for packet in video.encode():
            container.mux(packet)

        if audio is not None:
            samples_per_frame = 1024
            n_samples = seconds * 16000
            t = np.arange(n_samples) / 16000.0
            tone = (0.3 * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)
            for start in range(0, n_samples, samples_per_frame):
                chunk = tone[start : start + samples_per_frame][None, :]
                audio_frame = av.AudioFrame.from_ndarray(
                    chunk, format="fltp", layout="mono"
                )
                audio_frame.sample_rate = 16000
                audio_frame.pts = start
                for packet in audio.encode(audio_frame):
                    container.mux(packet)

            for packet in audio.encode():
                container.mux(packet)

    return str(path)


def keyframe_indices(path: str) -> list:
    with av.open(path) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        fps = float(stream.average_rate)
        return [
            round(float(packet.pts * time_base) * fps)
            for packet in container.demux(stream)
            if packet.is_keyframe and packet.pts is not None
        ]


def decode_first_frame_after_seek(path: str, indices, fps: float) -> list:
    """The pre-fix rule: seek, then take the first decoded frame (= the GOP keyframe)."""
    frames = []
    with av.open(path) as container:
        for idx in indices:
            container.seek(int(idx / fps * 1_000_000))
            for frame in container.decode(video=0):
                frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
                break

    return frames


_REAL_AV_OPEN = av.open


class _ContainerProxy:
    """`av.open` stand-in that forwards to a real container; subclasses override `seek` / `decode`."""

    def __init__(self, path):
        self.container = _REAL_AV_OPEN(path)
        self.streams = self.container.streams

    def seek(self, *args, **kwargs):
        return self.container.seek(*args, **kwargs)

    def decode(self, **kwargs):
        return self.container.decode(**kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.container.close()


def make_image_processor() -> CLIPImageProcessor:
    return CLIPImageProcessor(
        do_resize=True,
        size={"shortest_edge": CROP_SIZE},
        resample=3,
        do_center_crop=True,
        crop_size={"height": CROP_SIZE, "width": CROP_SIZE},
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_convert_rgb=True,
    )


def make_video_processor(num_frames: int = 8, **kwargs) -> GigaOmniVideoProcessor:
    vision_processor_params = {
        "image_processor": make_image_processor(),
        "image_aspect_ratio": "pad",
        "num_image_tokens_per_crop": 256,
    }
    vision_processor_params.update(kwargs.pop("vision_processor_params", {}))
    return GigaOmniVideoProcessor(
        vision_processor_params=vision_processor_params,
        num_frames=num_frames,
        video_decode_backend="pyav",
        video_image_token_id=128005,
        video_audio_token_id=128007,
        **kwargs,
    )


VISION_PARAMS = {
    "pad": {},
    "anyres": {
        "image_aspect_ratio": "anyres",
        "image_grid_pinpoints": "[[448, 448], [448, 896], [896, 448]]",
    },
    "qwen": {
        "image_processor": Qwen2VLImageProcessor(
            patch_size=14, merge_size=2, temporal_patch_size=2
        ),
        "image_aspect_ratio": "qwen",
        "num_image_tokens_per_crop": None,
        "qwen_max_image_tokens": 64,
    },
}


def photo_frames(n: int, size=(160, 96)) -> list:
    rng = np.random.default_rng(0)
    return [
        Image.fromarray(
            rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        )
        for _ in range(n)
    ]


@pytest.fixture(scope="module")
def synthetic_mp4(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("video") / "gop30.mp4"
    try:
        make_synthetic_mp4(path)
    except av.error.ValueError as exc:  # pragma: no cover - encoder-less pyav build
        pytest.skip(f"pyav cannot encode mpeg4 here: {exc}")

    return str(path)


class TestSyntheticFixture:
    def test_frame_code_roundtrip(self):
        for idx in [0, 1, 2, 15, 45, 255, 256, 299]:
            assert frame_index(Image.fromarray(paint_frame(idx))) == idx

    def test_keyframes_sit_on_the_gop_grid(self, synthetic_mp4):
        assert keyframe_indices(synthetic_mp4) == list(
            range(0, SECONDS * FPS, GOP_SIZE)
        )

    def test_sequential_decode_recovers_every_index(self, synthetic_mp4):
        with av.open(synthetic_mp4) as container:
            decoded = [
                frame_index(Image.fromarray(frame.to_ndarray(format="rgb24")))
                for frame in container.decode(video=0)
            ]

        assert decoded == list(range(SECONDS * FPS))


class TestExactFrameDecode:
    REQUESTED = np.array([0, 15, 45, 299])

    def test_pre_fix_rule_returns_gop_keyframes(self, synthetic_mp4):
        frames = decode_first_frame_after_seek(
            synthetic_mp4, self.REQUESTED, float(FPS)
        )

        assert [frame_index(frame) for frame in frames] == [0, 0, 30, 270]

    def test_decode_pyav_chunk_returns_requested_frames(self, synthetic_mp4):
        frames = GigaOmniVideoProcessor._decode_pyav_chunk(
            synthetic_mp4, self.REQUESTED, float(FPS)
        )

        assert [frame_index(frame) for frame in frames] == self.REQUESTED.tolist()

    @pytest.mark.parametrize(
        "indices",
        [
            list(range(0, 300)),  # every frame: forward decode after the first seeks
            list(range(3, 300, 7)),  # strides crossing GOP boundaries
            list(range(5, 300, 30)),  # one per GOP, 5 frames after each keyframe
            list(range(0, 300, 45)),  # sparser than a GOP: seeks
            [10, 10, 10, 11],  # repeated index reuses the frame
            [45, 15, 299, 0, 0, 1],  # backward jumps seek again
        ],
    )
    def test_forward_decode_returns_the_requested_frames(self, synthetic_mp4, indices):
        frames = GigaOmniVideoProcessor._decode_pyav_chunk(
            synthetic_mp4, np.array(indices), float(FPS)
        )

        assert [frame_index(frame) for frame in frames] == indices

    def test_past_eof_then_backward_stays_aligned(self, synthetic_mp4):
        frames = GigaOmniVideoProcessor._decode_pyav_chunk(
            synthetic_mp4, np.array([350, 10]), float(FPS)
        )

        assert [frame_index(frame) for frame in frames] == [299, 10]

    def test_seek_that_decodes_nothing_yields_black_frames_in_place(
        self, synthetic_mp4, monkeypatch
    ):
        class _NoFrames(_ContainerProxy):
            def decode(self, **kwargs):
                return iter(())

        monkeypatch.setattr(video_module.av, "open", _NoFrames)
        frames = GigaOmniVideoProcessor._decode_pyav_chunk(
            synthetic_mp4, np.array([0, 15, 45]), float(FPS)
        )

        assert len(frames) == 3
        assert all(frame.size == (FRAME_SIZE, FRAME_SIZE) for frame in frames)
        assert all(np.asarray(frame).max() == 0 for frame in frames)

    def test_decode_pyav_chunk_past_eof_returns_last_frame(self, synthetic_mp4):
        frames = GigaOmniVideoProcessor._decode_pyav_chunk(
            synthetic_mp4, np.array([299, 350]), float(FPS)
        )

        assert [frame_index(frame) for frame in frames] == [299, 299]

    def test_stream_start_time_offsets_the_targets(self, tmp_path):
        late = make_synthetic_mp4(tmp_path / "late.mp4", pts_offset_frames=45)
        with av.open(late) as container:
            stream = container.streams.video[0]
            start_s = float((stream.start_time or 0) * stream.time_base)

        if start_s <= 0:
            pytest.skip("the muxer normalised the start offset away")

        frames = GigaOmniVideoProcessor._decode_pyav_chunk(
            late, np.array([0, 15, 45, 299]), float(FPS)
        )

        assert [frame_index(frame) for frame in frames] == [0, 15, 45, 299]

    def test_read_video_frames_pyav_frames_match_timestamps(self, synthetic_mp4):
        processor = make_video_processor(num_frames=10)
        frames, timestamps = processor._read_video_frames_pyav(synthetic_mp4)
        expected = np.linspace(0, SECONDS * FPS - 1, 10, dtype=int)

        assert [frame_index(frame) for frame in frames] == expected.tolist()
        assert timestamps == [idx / FPS for idx in expected]
        assert processor._processor_fallback_count == 0

    def test_corrupt_file_falls_back_to_black_frames_and_counts(self, tmp_path):
        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_bytes(b"\x00" * 1024)
        processor = make_video_processor(num_frames=4)
        frames, timestamps = processor._read_video_frames_pyav(str(corrupt))

        assert len(frames) == 4 and len(timestamps) == 4
        assert all(np.asarray(frame).max() == 0 for frame in frames)
        assert processor._processor_fallback_count == 1


def count_seeks(monkeypatch) -> list:
    """Patch `av.open` so every `container.seek` target (microseconds) lands in the returned list."""
    seeks = []

    class _SeekCounter(_ContainerProxy):
        def seek(self, offset, *args, **kwargs):
            seeks.append(offset)
            return self.container.seek(offset, *args, **kwargs)

    monkeypatch.setattr(video_module.av, "open", _SeekCounter)
    return seeks


class TestForwardRoute:
    def test_forward_decode_takes_over_once_the_seeks_settle(
        self, synthetic_mp4, monkeypatch
    ):
        # seeks at 0, 7 and 14 decode 1, 8 and 15 frames: from a mean of 8 a
        # stride of 7 is decoded forward
        seeks = count_seeks(monkeypatch)
        indices = np.arange(0, 300, 7)
        frames = GigaOmniVideoProcessor._decode_pyav_chunk(
            synthetic_mp4, indices, float(FPS)
        )

        assert [frame_index(frame) for frame in frames] == indices.tolist()
        assert len(seeks) == 3

    def test_sparser_than_a_gop_always_seeks(self, synthetic_mp4, monkeypatch):
        seeks = count_seeks(monkeypatch)
        indices = np.arange(0, 300, 45)
        frames = GigaOmniVideoProcessor._decode_pyav_chunk(
            synthetic_mp4, indices, float(FPS)
        )

        assert [frame_index(frame) for frame in frames] == indices.tolist()
        assert len(seeks) == len(indices)

    def test_forward_run_finding_the_decoder_drained_seeks_instead(
        self, synthetic_mp4, monkeypatch
    ):
        # the first seek decodes two frames, so every following index is one
        # frame forward; the run to the past-EOF index finds the decoder
        # drained and must hand over to a seek that yields the last frame
        seeks = count_seeks(monkeypatch)
        indices = np.array(list(range(1, SECONDS * FPS)) + [SECONDS * FPS])
        frames = GigaOmniVideoProcessor._decode_pyav_chunk(
            synthetic_mp4, indices, float(FPS)
        )

        assert [frame_index(frame) for frame in frames] == list(
            range(1, SECONDS * FPS)
        ) + [SECONDS * FPS - 1]
        assert len(seeks) == 2

    def test_backward_jump_always_seeks(self, synthetic_mp4, monkeypatch):
        seeks = count_seeks(monkeypatch)
        indices = np.array([0, 7, 14, 21, 28, 35, 10])
        frames = GigaOmniVideoProcessor._decode_pyav_chunk(
            synthetic_mp4, indices, float(FPS)
        )

        assert [frame_index(frame) for frame in frames] == indices.tolist()
        assert seeks[-1] == int(10 / FPS * AV_TIME_BASE)


class TestParallelDecode:
    def test_pool_matches_single_container_in_order(self, synthetic_mp4):
        indices = np.linspace(0, SECONDS * FPS - 1, 37, dtype=int)
        processor = make_video_processor(num_frames=37)
        processor._decode_workers = 1
        serial = processor._decode_pyav_frames(synthetic_mp4, indices, float(FPS))
        processor._decode_workers = 4
        pooled = processor._decode_pyav_frames(synthetic_mp4, indices, float(FPS))

        assert [frame_index(frame) for frame in serial] == indices.tolist()
        assert [frame_index(frame) for frame in pooled] == indices.tolist()
        assert all(
            np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(serial, pooled)
        )

    def test_pool_never_exceeds_frame_count(self, synthetic_mp4):
        processor = make_video_processor(num_frames=2)
        processor._decode_workers = 16
        frames = processor._decode_pyav_frames(
            synthetic_mp4, np.array([3, 200]), float(FPS)
        )

        assert [frame_index(frame) for frame in frames] == [3, 200]


class TestParallelPreprocess:
    @pytest.mark.parametrize("aspect_ratio", ["pad", "anyres", "qwen"])
    def test_parallel_matches_serial_bit_for_bit(self, aspect_ratio):
        processor = make_video_processor(
            num_frames=6, vision_processor_params=VISION_PARAMS[aspect_ratio]
        )
        frames = photo_frames(6)
        serial = processor.vision_processor._preprocess_images(frames)
        parallel = processor.vision_processor._preprocess_images(frames, num_threads=4)

        assert torch.equal(serial["pixel_values"], parallel["pixel_values"])
        assert serial["num_crops_per_image"] == parallel["num_crops_per_image"]
        if aspect_ratio == "qwen":
            assert torch.equal(serial["grid_thw"], parallel["grid_thw"])
        else:
            assert serial["grid_thw"] is None and parallel["grid_thw"] is None

    def test_read_video_keeps_per_frame_crop_counts_in_order(self, synthetic_mp4):
        processor = make_video_processor(
            num_frames=5, vision_processor_params=VISION_PARAMS["anyres"]
        )
        processor._preprocess_workers = 4
        result = processor._read_video(synthetic_mp4)
        num_crops = processor.vision_processor._get_num_crops((FRAME_SIZE, FRAME_SIZE))

        assert result["frames_lengths"] == [num_crops * 256] * 5
        assert result["pixel_values"].shape[0] == num_crops * 5
        assert len(result["timestamps"]) == 5


class TestBf16Pixels:
    @pytest.mark.parametrize("aspect_ratio", ["pad", "anyres", "qwen"])
    def test_pixel_values_are_bf16_on_both_paths(self, aspect_ratio):
        processor = make_video_processor(
            num_frames=4, vision_processor_params=VISION_PARAMS[aspect_ratio]
        )
        frames = photo_frames(4)
        preprocess = processor.vision_processor._preprocess_images

        assert preprocess(frames)["pixel_values"].dtype == torch.bfloat16
        assert preprocess(frames, num_threads=2)["pixel_values"].dtype == torch.bfloat16

    def test_read_video_emits_bf16(self, synthetic_mp4):
        processor = make_video_processor(num_frames=3)

        assert (
            processor._read_video(synthetic_mp4)["pixel_values"].dtype == torch.bfloat16
        )

    def test_bf16_input_is_bit_identical_under_bf16_autocast(self):
        """The patch-embed conv under autocast rounds fp32 pixels to bf16 itself, so a
        bf16 tensor from the processor gives the same activations; the upcast that the
        ViT embeddings do (`.to(weight.dtype)`) is exact and changes nothing."""
        torch.manual_seed(0)
        conv = torch.nn.Conv2d(3, 8, kernel_size=14, stride=14)
        pixels_fp32 = torch.randn(2, 3, 56, 56)
        pixels_bf16 = pixels_fp32.to(torch.bfloat16)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            from_fp32 = conv(pixels_fp32)
            from_bf16 = conv(pixels_bf16)
            from_bf16_upcast = conv(pixels_bf16.to(conv.weight.dtype))

        assert from_fp32.dtype == torch.bfloat16
        assert torch.equal(from_fp32, from_bf16)
        assert torch.equal(from_fp32, from_bf16_upcast)
        assert not torch.equal(pixels_fp32, pixels_bf16.float())


class TestParallelBudget:
    def test_budget_follows_dataloader_workers(self, monkeypatch):
        monkeypatch.setattr(video_module, "available_cpu_cores", lambda: 96)
        one = make_video_processor(num_dataloader_workers=1)
        eight = make_video_processor(num_dataloader_workers=8)
        many = make_video_processor(num_dataloader_workers=64)

        assert (one._decode_workers, one._preprocess_workers) == (16, 32)
        assert (eight._decode_workers, eight._preprocess_workers) == (4, 8)
        assert (many._decode_workers, many._preprocess_workers) == (1, 2)

    def test_budget_is_logged_once(self, monkeypatch, caplog):
        monkeypatch.setattr(video_module, "available_cpu_cores", lambda: 96)
        with caplog.at_level("INFO", logger=video_module.logger.name):
            make_video_processor(num_dataloader_workers=8)

        budget_lines = [
            r.message for r in caplog.records if "parallel budget" in r.message
        ]
        assert budget_lines == [
            "Video processor parallel budget: cores=96, dataloader_workers=8, "
            "decode_workers=4, preprocess_workers=8"
        ]

    def test_available_cores_honours_cgroup_quota(self, tmp_path):
        affinity = len(os.sched_getaffinity(0))
        quota_v2 = tmp_path / "cpu.max"
        quota_v2.write_text("200000 100000\n")
        no_quota = tmp_path / "cpu.max.unlimited"
        no_quota.write_text("max 100000\n")
        quota_v1 = tmp_path / "cpu.cfs_quota_us"
        quota_v1.write_text("150000\n")
        period_v1 = tmp_path / "cpu.cfs_period_us"
        period_v1.write_text("100000\n")

        assert available_cpu_cores([(str(quota_v2), None)]) == min(affinity, 2)
        assert available_cpu_cores([(str(no_quota), None)]) == affinity
        assert available_cpu_cores([(str(quota_v1), str(period_v1))]) == min(
            affinity, 2
        )
        assert available_cpu_cores([(str(tmp_path / "missing"), None)]) == affinity
        assert 1 <= available_cpu_cores() <= affinity


class TestDataloaderHandoff:
    def test_num_workers_reaches_processors_that_accept_it(self, monkeypatch):
        seen = []

        class _Video:
            valid_kwargs = ["num_dataloader_workers", "num_frames"]

            def __init__(self, **kwargs):
                seen.append(("video", kwargs))

        class _Audio:
            valid_kwargs = ["sample_rate"]

            def __init__(self, **kwargs):
                seen.append(("audio", kwargs))

        monkeypatch.setattr(processors, "_Video", _Video, raising=False)
        monkeypatch.setattr(processors, "_Audio", _Audio, raising=False)
        configs = OmegaConf.create(
            [
                {"name": "_Video", "num_frames": 8},
                {"name": "_Audio", "sample_rate": 16000},
                {"name": "_Video", "num_frames": 4, "num_dataloader_workers": 2},
            ]
        )
        built = processors.build_processors(configs, num_dataloader_workers=8)

        assert [type(p).__name__ for p in built] == ["_Video", "_Audio", "_Video"]
        assert seen == [
            ("video", {"num_frames": 8, "num_dataloader_workers": 8}),
            ("audio", {"sample_rate": 16000}),
            ("video", {"num_frames": 4, "num_dataloader_workers": 2}),
        ]

    def test_num_workers_default_is_injected(self, monkeypatch):
        seen = []

        class _Video:
            valid_kwargs = ["num_dataloader_workers"]

            def __init__(self, **kwargs):
                seen.append(kwargs)

        monkeypatch.setattr(processors, "_Video", _Video, raising=False)
        processors.build_processors([{"name": "_Video"}], num_dataloader_workers=6)

        assert seen == [{"num_dataloader_workers": 6}]

    def test_mcqa_dataset_hands_over_its_loader_workers(self, monkeypatch):
        seen = []

        class _Video:
            valid_kwargs = ["num_dataloader_workers"]

            def __init__(self, **kwargs):
                seen.append(kwargs)

        monkeypatch.setattr(processors, "_Video", _Video, raising=False)
        dataset = object.__new__(OmniMCQADataset)
        dataset.num_dataloader_workers = 5
        built = dataset._build_processors(OmegaConf.create([{"name": "_Video"}]))

        assert [type(p).__name__ for p in built] == ["_Video"]
        assert seen == [{"num_dataloader_workers": 5}]
