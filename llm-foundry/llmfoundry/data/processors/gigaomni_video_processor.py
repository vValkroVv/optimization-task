import av
import os
import math
import torch
import random
import logging
import audioaug
import numpy as np

from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from decord import VideoReader, cpu
from omegaconf import ListConfig, OmegaConf
from av.audio.resampler import AudioResampler
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoProcessor, AutoImageProcessor

from .base_processor import BaseProcessor, DataModality
from .gigaomni_audio_processor import GigaOmniAudioProcessor
from .gigaomni_vision_processor import GigaOmniVisionProcessor
from .mm_utils import (
    get_crop_size,
    insert_tokens_sequences_on_positions,
    predict_spec_length,
    duplicate_tokens_on_positions,
)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s |  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# PyAV seeks and reports container durations in microseconds
AV_TIME_BASE = 1_000_000


# (quota file, period file) for cgroup v2 (`cpu.max` holds both) and v1
CGROUP_CPU_QUOTA_FILES = (
    ("/sys/fs/cgroup/cpu.max", None),
    ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
)


def available_cpu_cores(cgroup_files=CGROUP_CPU_QUOTA_FILES) -> int:
    """
    CPU cores this process may use: the cgroup quota when one is set, else the
    affinity mask. `os.cpu_count()` reports the host (a 96-core job sees 128).
    """
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    cores = cores or os.cpu_count() or 32
    for quota_path, period_path in cgroup_files:
        try:
            with open(quota_path) as quota_file:
                fields = quota_file.read().split()

            if period_path is None:
                quota, period = fields[0], fields[1]
            else:
                with open(period_path) as period_file:
                    quota, period = fields[0], period_file.read().split()[0]

            if quota != "max" and int(quota) > 0:
                cores = min(cores, max(1, math.ceil(int(quota) / int(period))))

            break
        except (OSError, ValueError, IndexError):
            continue

    return cores


class GigaOmniVideoProcessor(BaseProcessor):
    data_modality = DataModality.VIDEO
    attributes = []
    valid_kwargs = [
        "vision_processor_params",
        "audio_processor_params",
        "num_frames",
        "add_timestamps",
        "video_decode_backend",
        "video_image_token_id",
        "video_audio_token_id",
        "video_audio_tokens_order",
        "read_soundtracks",
        "tokenizer",
        "fps_strategy",
        "fps_schedule",
        "num_dataloader_workers",
    ]
    valid_fps_strategies = ["constant", "dynamic", "random"]
    FPS_FALLBACK = 1.0
    MAX_DECODE_WORKERS = 16
    MAX_PREPROCESS_WORKERS = 32

    def __init__(
        self,
        vision_processor_params: Dict[str, Any] = {},
        audio_processor_params: Dict[str, Any] = {},
        num_frames: int = 8,
        add_timestamps: bool = False,
        video_image_token_id: int = 128006,
        video_audio_token_id: int = 128008,
        video_decode_backend: str = "decord",
        video_audio_tokens_order: str = "interleaved",
        read_soundtracks: bool = True,
        fps_strategy: Optional[str] = None,
        fps_schedule: Optional[List[Dict[str, Any]]] = None,
        num_dataloader_workers: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.audio_processor_params = audio_processor_params
        self.audio_processor = GigaOmniAudioProcessor(
            **self.audio_processor_params,
            audio_token_id=video_audio_token_id,
        )

        self.vision_processor_params = vision_processor_params
        self.vision_processor = GigaOmniVisionProcessor(
            **self.vision_processor_params,
            image_token_id=video_image_token_id,
        )

        self.num_frames = num_frames
        self.add_timestamps = add_timestamps
        self.video_audio_tokens_order = video_audio_tokens_order
        # False trains a model without an audio encoder on `video_audio` data: the
        # soundtrack is neither decoded nor given placeholder tokens.
        self.read_soundtracks = read_soundtracks
        self.video_decode_backend = video_decode_backend
        self.video_image_token_id = video_image_token_id
        self.video_audio_token_id = video_audio_token_id

        self.auto_map = {
            "AutoProcessor": "gigaomni_video_processor.GigaOmniVideoProcessor",
        }

        self.fps_strategy = fps_strategy
        self.fps_schedule = fps_schedule or []
        if self.fps_strategy:
            self._validate_fps_config()

        self._init_parallel_budget(num_dataloader_workers)

    def _init_parallel_budget(self, num_dataloader_workers: int) -> None:
        """
        Threads per video for frame decode and CLIP preprocessing, sized so that
        `num_dataloader_workers` processes on this node's cores do not oversubscribe.
        """
        cores = available_cpu_cores()
        budget = max(cores // max(num_dataloader_workers, 1), 4)
        self._decode_workers = min(budget // 3, self.MAX_DECODE_WORKERS)
        self._preprocess_workers = min(budget * 2 // 3, self.MAX_PREPROCESS_WORKERS)
        logger.info(
            f"Video processor parallel budget: cores={cores}, "
            f"dataloader_workers={num_dataloader_workers}, "
            f"decode_workers={self._decode_workers}, "
            f"preprocess_workers={self._preprocess_workers}"
        )

    def __call__(
        self,
        files: Optional[List[Dict[str, Any]]] = None,
        dataset_sample: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess video files and insert video tokens into the text input_ids and labels.
        You can provide dataset_sample in the form of:
        {
            "text": {
                "input_ids": torch.LongTensor,
                "labels": torch.LongTensor
            }
        } or input_ids and labels as separate arguments.
        """
        self._processor_fallback_count = 0
        self._soundtrack_fallback_count = 0

        file_contents = self._prepare_file_contents(files)
        dataset_sample = self._prepare_dataset_sample(
            dataset_sample, input_ids, labels, attention_mask
        )

        if len(file_contents) == 0:
            return dataset_sample

        dataset_sample = self._insert_video_audio_tokens(dataset_sample, file_contents)

        dataset_sample[self.data_modality.value] = {
            "pixel_values": torch.cat(
                [file_content["pixel_values"] for file_content in file_contents]
            ),
            "num_crops_per_frame": self._get_num_crops_per_frame(file_contents),
            "processor_fallback_count": self._processor_fallback_count,
        }

        if self.vision_processor.image_aspect_ratio == "qwen":
            dataset_sample[self.data_modality.value]["grid_thw"] = torch.cat(
                [fc["grid_thw"] for fc in file_contents]
            )

        spectrograms = [
            file_content["spectrogram"]
            for file_content in file_contents
            if "spectrogram" in file_content
        ]

        if spectrograms:
            # The AudioCollator payload in its own bundle; frame fallbacks are
            # counted under `video`, soundtrack fallbacks here.
            dataset_sample[DataModality.VIDEO_AUDIO.value] = {
                "spectrograms": torch.cat(spectrograms),
                "spectrogram_lengths": torch.tensor(
                    [spectrogram.shape[0] for spectrogram in spectrograms],
                    dtype=torch.long,
                ),
                "processor_fallback_count": self._soundtrack_fallback_count,
            }

        return dataset_sample

    def _get_num_crops_per_frame(self, file_contents: List[Dict[str, Any]]) -> int:
        """Crops per frame shared by all frames of the sample, 0 if they differ.

        Read by the token reduction only; a decode fallback (square black frames)
        next to a healthy anyres video gives 0.
        """
        num_crops = {
            frame_length // self.vision_processor.num_image_tokens_per_crop
            for file_content in file_contents
            for frame_length in file_content["frames_lengths"]
        }

        return num_crops.pop() if len(num_crops) == 1 else 0

    def _validate_fps_config(self):
        assert self.fps_strategy in self.valid_fps_strategies, (
            f"Unsupported fps_strategy: {self.fps_strategy}, "
            f"expected one of {self.valid_fps_strategies}"
        )

        if isinstance(self.fps_schedule, ListConfig):
            self.fps_schedule: List[Dict[str, Any]] = OmegaConf.to_container(
                self.fps_schedule, resolve=True
            )  # type: ignore

        if self.fps_schedule:
            self.fps_schedule.sort(key=lambda x: x.get("max_duration", float("inf")))
        else:
            logger.warning(
                f"{self.fps_strategy=} is set, but fps_schedule is empty, "
                f"fallback to {self.FPS_FALLBACK} fps",
            )

        for i, rule in enumerate(self.fps_schedule):
            assert "fps" in rule, (
                f"fps_schedule[{i}] missing required 'fps' key: {rule}"
            )

            fps_val = rule["fps"]
            if self.fps_strategy == "random":
                assert isinstance(fps_val, (list, tuple)), (
                    f"fps_schedule[{i}]: fps must be a non-empty list "
                    f"for 'random' strategy, got {fps_val}"
                )
            else:
                assert isinstance(fps_val, (int, float)), (
                    f"fps_schedule[{i}]: fps must be a number "
                    f"for '{self.fps_strategy}' strategy, got {fps_val}"
                )

    def get_num_tokens(self, files: Optional[List[Dict[str, Any]]] = None) -> int:
        if self.fps_strategy and self.fps_strategy == "random":
            logger.warning(
                "Random fps_strategy is set, get_num_tokens may return "
                "different token count than __call__ due to random FPS selection"
            )

        file_contents = self._prepare_file_contents(files)
        tokens_sequences = [
            self._get_video_audio_tokens_for_file(file_content)
            for file_content in file_contents
        ]

        if len(tokens_sequences) == 0:
            return 0

        return sum(len(token_sequence) for token_sequence in tokens_sequences)

    def save_pretrained(self, save_directory: str, **kwargs):  # type: ignore
        # ImageProcessor is not serializable, so we need to save it separately
        self.vision_processor.image_processor.save_pretrained(save_directory, **kwargs)  # type: ignore
        self.vision_processor_params["image_processor"] = None
        output = super().save_pretrained(save_directory, **kwargs)
        return output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):  # type: ignore
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor.vision_processor.image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        processor.vision_processor_params["image_processor"] = str(
            Path(pretrained_model_name_or_path).absolute()
        )
        assert processor.tokenizer is not None, (
            "Tokenizer is required for GigaOmniVideoProcessor!"
        )
        return processor

    def _insert_video_audio_tokens(
        self,
        dataset_sample: Dict[str, Dict[str, Any]],
        file_contents: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        modality_token_positions = (
            (dataset_sample["text"]["input_ids"] == self.video_image_token_id)
            .nonzero(as_tuple=True)[0]
            .tolist()
        )

        err_msg = (
            f"Number of occurances of `modality_token_id={self.video_image_token_id}` in input_ids must be equal to "
            f"number of files but got {len(modality_token_positions)} and {len(file_contents)}!"
        )
        assert len(modality_token_positions) == len(file_contents), err_msg

        tokens_sequences = [
            self._get_video_audio_tokens_for_file(file_content)
            for file_content in file_contents
        ]

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

    def _get_video_audio_tokens_for_file(
        self, file_content: Dict[str, Any]
    ) -> torch.Tensor:
        if self.video_audio_tokens_order == "sequential":
            return self._get_video_audio_tokens_for_file_sequential(file_content)
        elif self.video_audio_tokens_order == "interleaved":
            return self._get_video_audio_tokens_for_file_interleaved(file_content)
        else:
            raise ValueError(
                f"Invalid order of frame and audio tokens: {self.video_audio_tokens_order}"
            )

    def _get_video_audio_tokens_for_file_sequential(
        self, file_content: Dict[str, Any]
    ) -> torch.Tensor:
        tokens = [self.video_image_token_id] * sum(file_content["frames_lengths"])

        if self.add_timestamps:
            tokens = self._insert_timestamps(tokens, file_content["timestamps"])

        if "predicted_spectrogram_length" in file_content:
            audio_tokens = [self.video_audio_token_id] * file_content[
                "predicted_spectrogram_length"
            ]

            # In case of sequential video and audio tokens order, we need to insert timestamps for audio tokens as well.
            if self.add_timestamps:
                audio_tokens = self._insert_timestamps(
                    audio_tokens, file_content["timestamps"]
                )

            tokens += audio_tokens

        return torch.tensor(tokens, dtype=torch.long)

    def _get_video_audio_tokens_for_file_interleaved(
        self, file_content: Dict[str, Any]
    ) -> torch.Tensor:
        if "predicted_spectrogram_length" not in file_content:
            # NOTE (Sbr): For videos without audio stream, sequential order is the same as interleaved.
            return self._get_video_audio_tokens_for_file_sequential(file_content)

        num_frames = len(file_content["frames_lengths"])
        num_audio_tokens_per_frame = (
            file_content["predicted_spectrogram_length"] // num_frames
        )
        audio_tokens_per_frame_remainder = (
            file_content["predicted_spectrogram_length"] % num_frames
        )

        tokens = []
        for frame_idx in range(num_frames):
            video_tokens = [self.video_image_token_id] * file_content["frames_lengths"][
                frame_idx
            ]
            if self.add_timestamps:
                video_tokens = self._insert_timestamps(
                    video_tokens, [file_content["timestamps"][frame_idx]]
                )

            tokens += video_tokens
            tokens += [self.video_audio_token_id] * num_audio_tokens_per_frame
            if frame_idx < audio_tokens_per_frame_remainder:
                tokens += [self.video_audio_token_id]

        return torch.tensor(tokens, dtype=torch.long)

    def _prepare_file_contents(
        self, files: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        if not files:
            return []

        file_contents = []
        for file in files:
            if file["type"] not in ["video", "video_audio"]:
                continue

            if "content" in file and file["content"] is not None:
                file_contents.append(file["content"])
            else:
                file_contents.append(
                    self._read_video(
                        file["path"],
                        read_audio=self.read_soundtracks
                        and file["type"] == "video_audio",
                    )
                )

        return file_contents

    def _get_target_fps(self, duration: float) -> float:
        """
        Select target FPS from schedule based on strategy and video duration.

        Strategies:
            constant: always uses FPS from the first rule (fixed FPS)
            dynamic:  FPS from the first rule whose max_duration covers the video
            random:   like dynamic, but randomly picks from a list of FPS values

        Falls back to FPS_FALLBACK when schedule is empty or no rule matches.
        """
        if not self.fps_schedule:
            return self.FPS_FALLBACK

        if self.fps_strategy == "constant":
            return float(self.fps_schedule[0]["fps"])

        for rule in self.fps_schedule:
            max_duration = rule.get("max_duration", float("inf"))
            if duration > max_duration:
                continue

            if self.fps_strategy == "random":
                return float(random.choice(rule["fps"]))
            else:
                return float(rule["fps"])

        return self.FPS_FALLBACK

    def _get_target_frames(self, total_frames: int, native_fps: float) -> int:
        """
        Calculate number of frames to sample based on target FPS.
        Clamped to [1, total_frames].
        """
        duration = total_frames / native_fps
        target_fps = self._get_target_fps(duration)
        target_frames = duration * target_fps

        return max(min(int(target_frames), total_frames), 1)

    def _get_sample_indices(self, total_frames: int, native_fps: float) -> np.ndarray:
        """
        Get frame indices to sample from video.

        Uses fps_strategy if set, otherwise samples num_frames uniformly.
        Output never exceeds num_frames.
        """
        if self.fps_strategy:
            target_frames = self._get_target_frames(total_frames, native_fps)
        else:
            target_frames = self.num_frames

        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        if len(frame_indices) > self.num_frames:
            selection_indices = np.linspace(
                0, len(frame_indices) - 1, self.num_frames, dtype=int
            )
            frame_indices = frame_indices[selection_indices]

        return frame_indices

    def _read_video(self, video_path: str, read_audio: bool = False) -> Dict[str, Any]:
        if self.video_decode_backend == "pyav":
            video_frames, timestamps = self._read_video_frames_pyav(video_path)
        elif self.video_decode_backend == "decord":
            video_frames, timestamps = self._read_video_frames_decord(video_path)
        else:
            raise ValueError(
                f"Unsupported video decode backend: {self.video_decode_backend}"
            )

        preprocessed = self.vision_processor._preprocess_images(
            video_frames, self._preprocess_workers
        )
        assert len(preprocessed["num_crops_per_image"]) == len(timestamps), (
            f"Number of frames and timestamps must be equal but got {len(preprocessed['num_crops_per_image'])} and {len(timestamps)}!"
        )

        frames_lengths = [
            num_crops * self.vision_processor.num_image_tokens_per_crop
            for num_crops in preprocessed["num_crops_per_image"]
        ]

        result = {
            "pixel_values": preprocessed["pixel_values"],
            "frames_lengths": frames_lengths,
            "timestamps": timestamps,
        }

        if self.vision_processor.image_aspect_ratio == "qwen":
            result["grid_thw"] = preprocessed["grid_thw"]

        if read_audio:
            spectrogram = self._read_video_audio_stream_pyav(video_path)
            result["spectrogram"] = spectrogram
            result["predicted_spectrogram_length"] = predict_spec_length(
                spectrogram.shape[0],
                self.audio_processor.projector_subsampling_factor,
            )

        return result

    def _read_video_frames_decord(
        self, video_path: str
    ) -> Tuple[List[Image.Image], List[float]]:
        try:
            decord_vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

            total_frames = len(decord_vr)
            native_fps = decord_vr.get_avg_fps() or 30.0
            frame_id_list = self._get_sample_indices(total_frames, native_fps)

            list_frames = decord_vr.get_batch(frame_id_list).asnumpy()
            list_frames = [Image.fromarray(frame) for frame in list_frames]

            timestamps = [idx / native_fps for idx in frame_id_list]
        except Exception as exc:
            logger.error(
                f"Error reading video frames from `{video_path}` with decord, returning black frames! Error:\n{exc}"
            )

            crop_size = get_crop_size(self.vision_processor.image_processor)
            list_frames = [
                Image.new("RGB", (crop_size, crop_size)) for _ in range(self.num_frames)
            ]
            timestamps = [float(i) for i in range(self.num_frames)]

            self._processor_fallback_count += 1

        return list_frames, timestamps

    @staticmethod
    def _decode_until(decoder, threshold_s: float, time_base) -> Tuple[Any, int]:
        """
        The first frame from `decoder` whose pts reaches `threshold_s` (the last
        decoded one when none does, None when the decoder is drained) and the
        number of frames decoded to get there.
        """
        frame, decoded = None, 0

        for frame in decoder:
            decoded += 1

            if (
                frame.pts is None
                or time_base is None
                or float(frame.pts * time_base) >= threshold_s
            ):
                break

        return frame, decoded

    @staticmethod
    def _decode_pyav_chunk(
        video_path: str, indices: np.ndarray, fps: float
    ) -> List[Image.Image]:
        """
        Decode the frames at `indices` (frame numbers at `fps`) from their own container.

        The frame for a target is the first one whose presentation time reaches
        the target (half a frame of tolerance for VFR/odd-rate clips); targets
        are offset by the stream's start time so index 0 is its first frame.
        `container.seek` lands on the keyframe at or before the target, so a
        seek costs decoding the GOP up to the target: the container keeps
        decoding forward to the next target when it is at most as many frames
        ahead as its seeks have had to decode on average, and seeks otherwise
        (always backwards, and when a forward run finds the decoder drained).
        A repeated index reuses the frame, a container without pts/time_base
        yields the first decoded frame, a target past EOF yields the last
        decoded frame, and a seek that decodes nothing repeats the previous
        frame (a black one when there is none) so frames stay aligned with the
        timestamps.
        """
        frames = []
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            time_base = stream.time_base

            # frame indices count from the stream's first frame, while pts and
            # seek positions are absolute: offset targets by the stream start
            start_s = 0.0
            if stream.start_time is not None and time_base is not None:
                start_s = float(stream.start_time * time_base)

            decoder = None
            last_pts_s = None
            seek_frames, n_seeks = 0, 0

            for idx in indices:
                target_s = start_s + idx / fps
                threshold_s = target_s - 0.5 / fps

                if (
                    last_pts_s is not None
                    and threshold_s <= last_pts_s <= target_s + 0.5 / fps
                ):
                    frames.append(frames[-1])
                    continue

                frame = None

                if (
                    last_pts_s is not None
                    and n_seeks > 0
                    and 0 <= (target_s - last_pts_s) * fps <= seek_frames / n_seeks
                ):
                    frame, _ = GigaOmniVideoProcessor._decode_until(
                        decoder, threshold_s, time_base
                    )

                if frame is None:
                    container.seek(int(target_s * AV_TIME_BASE))
                    decoder = container.decode(video=0)
                    frame, decoded = GigaOmniVideoProcessor._decode_until(
                        decoder, threshold_s, time_base
                    )
                    seek_frames += decoded
                    n_seeks += 1

                if frame is not None:
                    if frame.pts is not None and time_base is not None:
                        last_pts_s = float(frame.pts * time_base)

                    frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
                elif frames:
                    frames.append(frames[-1])
                else:
                    logger.warning(
                        f"PyAV decoded nothing at frame {idx} of `{video_path}`, using a black frame"
                    )
                    frames.append(Image.new("RGB", (stream.width, stream.height)))

        return frames

    def _decode_pyav_frames(
        self, video_path: str, frame_indices: np.ndarray, fps: float
    ) -> List[Image.Image]:
        """Split `frame_indices` over up to `_decode_workers` containers (pyav containers are not thread-safe)."""
        n_workers = max(1, min(self._decode_workers, len(frame_indices)))

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            chunks = pool.map(
                lambda chunk: self._decode_pyav_chunk(video_path, chunk, fps),
                np.array_split(frame_indices, n_workers),
            )
            return [frame for chunk in chunks for frame in chunk]

    def _read_video_frames_pyav(
        self, video_path: str
    ) -> Tuple[List[Image.Image], List[float]]:
        try:
            with av.open(video_path) as container:
                video_stream = container.streams.video[0]
                native_fps = float(
                    video_stream.average_rate or video_stream.guessed_rate or 30,
                )

                total_frames = video_stream.frames or 0
                if total_frames <= 0:
                    if (
                        video_stream.duration is not None
                        and video_stream.time_base is not None
                    ):
                        duration = float(video_stream.duration * video_stream.time_base)
                        total_frames = int(duration * native_fps)
                    elif container.duration is not None:
                        # microseconds -> seconds
                        duration = container.duration / AV_TIME_BASE
                        total_frames = int(duration * native_fps)
                    else:
                        raise RuntimeError(
                            f"Could not determine video duration for {video_path}",
                        )

            frame_indices = self._get_sample_indices(total_frames, native_fps)
            list_frames = self._decode_pyav_frames(
                video_path, frame_indices, native_fps
            )

            assert len(list_frames) > 0, (
                f"Number of frames decoded from `{video_path}` with PyAV is equal to 0"
            )

            target_len = len(frame_indices)
            if len(list_frames) < target_len:
                logger.warning(
                    f"PyAV decoded {len(list_frames)}/{target_len} frames from `{video_path}`, padding last frame"
                )
                list_frames += [list_frames[-1]] * (target_len - len(list_frames))

            timestamps = [idx / native_fps for idx in frame_indices]

        except Exception as exc:
            logger.error(
                f"Error reading video frames from `{video_path}` with PyAV, returning black frames! Error:\n{exc}"
            )

            crop_size = get_crop_size(self.vision_processor.image_processor)
            list_frames = [
                Image.new("RGB", (crop_size, crop_size)) for _ in range(self.num_frames)
            ]
            timestamps = [float(i) for i in range(self.num_frames)]

            self._processor_fallback_count += 1

        return list_frames, timestamps

    def _read_video_audio_stream_pyav(self, video_path: str) -> torch.Tensor:
        try:
            with av.open(video_path, mode="r") as container:
                audio_stream = container.streams.best("audio")
                assert audio_stream, (
                    f"Got empty audio stream from `{video_path}` with pyav!"
                )

                chunks = []
                resampler = AudioResampler(
                    format="s16", layout="mono", rate=self.audio_processor.sample_rate
                )
                for packet in container.demux(audio_stream):
                    if packet.dts is None and packet.pts is None:
                        continue

                    for frame in packet.decode():
                        chunks += [
                            rframe.to_ndarray().T
                            for rframe in resampler.resample(frame)
                        ]  # type: ignore

            assert chunks, f"Can't read audio stream from `{video_path}` with pyav!"

            audio_i16 = np.concatenate(chunks, axis=0).astype(np.int16, copy=False)
            audio = audio_i16.astype(np.float32) / np.iinfo(np.int16).max

            info = audioaug.AudioInfo(
                sample_rate=self.audio_processor.sample_rate,
                channels=1,
                frames=audio.shape[0],
            )
            spectrogram = audioaug.extract_features(
                audio, info, self.audio_processor._get_audioaug_features_params()
            )
            spectrogram = torch.from_numpy(spectrogram).view(
                -1, self.audio_processor.num_filterbank_chans
            )
            return spectrogram
        except Exception as exc:
            logger.error(
                f"Error reading audio stream from `{video_path}` with pyav, returning "
                f"zero-sized spectrograms! Error:\n{exc}"
            )

            self._soundtrack_fallback_count += 1
            return torch.zeros(
                20, self.audio_processor.num_filterbank_chans, dtype=torch.float32
            )

    def _insert_timestamps(
        self, video_token_ids: List[int], timestamps: List[float]
    ) -> List[int]:
        assert self.tokenizer is not None, "Tokenizer is needed to insert timestamps!"
        # A soundtrack run need not divide by the frame count; the remainder goes
        # to the first frames instead of being dropped.
        tokens_per_frame, remainder = divmod(len(video_token_ids), len(timestamps))

        ids_with_timestamps = []
        start = 0

        for idx, timestamp in enumerate(timestamps):
            end = start + tokens_per_frame + (1 if idx < remainder else 0)
            tokens_slice = video_token_ids[start:end]
            start = end
            timestamp_token_ids = self.tokenizer(
                self._format_timestamp_string(timestamp), add_special_tokens=False
            )["input_ids"]
            ids_with_timestamps.extend(timestamp_token_ids + tokens_slice)

        return ids_with_timestamps

    def _format_timestamp_string(self, timestamp: float) -> str:
        return f"{int(timestamp // 3600):02d}:{int((timestamp % 3600) // 60):02d}:{int(timestamp % 60):02d}"


GigaOmniVideoProcessor.register_for_auto_class(AutoProcessor)
