import io
import json
import inspect
import logging
import tarfile
import numpy as np
import torch
import torch.utils
import torch.utils.data

from collections.abc import Callable
from typing import List, Dict, Any, Optional, Tuple, TypeAlias
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from pathlib import Path
from transformers import PreTrainedTokenizer

from streaming import Stream
from composer.core import Evaluator, DataSpec
from llmfoundry.data import processors, collators, wtime_sample
from llmfoundry.data.text_data import StreamingTextDataset
from llmfoundry.data.data import read_binary_tokenized_sample
from llmfoundry.data.processors.base_processor import BaseProcessor
from llmfoundry.data.collators import TextCollator
from llmfoundry.data.processors.base_processor import DataModality


CollatorType: TypeAlias = Callable[
    [List[Dict[str, Dict[str, torch.Tensor]]]], Dict[str, Dict[str, torch.Tensor]]
]


logging.basicConfig(
    format="%(asctime)s | %(levelname)s |  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class OmniStream(Stream):
    def __init__(
        self,
        name: Optional[str] = None,
        modality: Optional[str] = None,
        files_folder_path: Optional[str] = None,
        **kwargs,
    ):
        self.name = name
        self.modality = modality
        self.files_folder_path: Optional[str] = files_folder_path

        self.stream_kwargs = self._kwargs_filter(kwargs)

        assert (files_folder_path is not None or self.stream_kwargs.get("remote")
                or self.stream_kwargs.get("local")), (
            f"Stream {name} requires `files_folder_path`, `remote` or `local` in stream config"
        )

        super().__init__(**self.stream_kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _kwargs_filter(self, kwargs):
        """
        Extracts from the passed `kwargs` dictionary only those keys that are valid
        arguments for the `Stream` constructor.

        Args:
            kwargs (Dict): Dictionary of keyword arguments.

        Returns:
            Dict: Dictionary containing only arguments valid for the `Stream` constructor.
        """

        stream_kwargs = list(inspect.signature(Stream.__init__).parameters.keys())[1:]
        stream_kwargs = {key: kwargs.pop(key, None) for key in stream_kwargs}
        return stream_kwargs

    def __repr__(self):
        repr = ""
        repr += f"Stream name: {self.name}\n"
        repr += f"Modality: {self.modality}\n"
        repr += f"Files folder path: {self.files_folder_path}\n"
        repr += "\n".join(
            [f"{key}: {value}" for key, value in self.stream_kwargs.items()]
        )
        return repr


class StreamingOmniDataset(StreamingTextDataset):
    def __init__(
        self,
        processors: List[BaseProcessor],
        allow_short_inputs: bool = False,
        timed_units: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Allow sequences of 64 tokens or fewer only when explicitly enabled."""
        super().__init__(**kwargs)
        self.processors = processors
        self.allow_short_inputs = allow_short_inputs
        self.timed_units = {
            i: om.to_container(om.merge(timed_units or {}, getattr(s, "timed_units", None) or {}), resolve=True)
            for i, s in enumerate(self.streams) if getattr(s, "sample_format", None) == "timed_units"
        }
        if self.timed_units:
            audio = next(p for p in processors if p.data_modality == DataModality.AUDIO)
            if audio.add_duration or audio.add_timestamps or audio.encoder_subsampling_factor != 4:
                raise ValueError("timed_units needs encoder_subsampling_factor 4 and no add_duration / add_timestamps")
            self._timed_units_audio = audio
            guard = (timed_units or {}).get("guard", {})
            self._timed_units_guard = wtime_sample.FallbackGuard(name="timed_units", **guard)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        if not 0 <= idx < self.num_samples:
            raise ValueError(f"Invalid sample index {idx}: expected 0 <= index < {self.num_samples}")
        max_attempts = min(10000, self.num_samples)
        last_error = None

        for offset in range(max_attempts):
            sample_idx = (idx + offset) % self.num_samples
            try:
                return self.get_sample(sample_idx)
            except Exception as exc:
                last_error = exc
                logger.warning(f"Can't get sample {idx}, trying {sample_idx}, got exception:\n{exc}")

        raise ValueError(f"Can't get sample {idx} after {max_attempts} attempts!") from last_error

    def get_sample(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Reads and processes samples from datasets with different modalities. Return example:
        {
            "text": {
                "input_ids": torch.Tensor([...]),
                "labels": torch.Tensor([...]),
            },
            "*modality*": {  # For all modalities in the sample (optional)
                ...
            },
            ...
        }

        Args:
            idx (int): index of the sample

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: dataset sample
        """
        sample = super(StreamingTextDataset, self).__getitem__(idx)

        timed_units = self.timed_units.get(self._get_stream_idx(idx))
        if timed_units is not None:
            input_ids, labels, files = self._read_timed_units(idx, sample, timed_units)
        else:
            input_ids = read_binary_tokenized_sample(self.max_seq_len, sample, "tokens")
            if "labels" in sample:
                labels = read_binary_tokenized_sample(self.max_seq_len, sample, "labels")
            else:
                if "files" in sample:
                    raise ValueError("Multimodal samples must provide labels")
                labels = input_ids.clone()
                if self.tokenizer.pad_token_id is not None:
                    labels[labels == self.tokenizer.pad_token_id] = -100
            files = self._read_files_from_sample(idx, sample)

        dataset_sample = {"text": {"input_ids": input_ids, "labels": labels}}

        for processor in self.processors:
            dataset_sample.update(processor(files=files, dataset_sample=dataset_sample))

        processed_length = len(dataset_sample["text"]["input_ids"])
        if processed_length > self.max_seq_len:
            raise ValueError(
                f"Sample {idx} has {processed_length} tokens after processing, "
                f"exceeding max_seq_len={self.max_seq_len}"
            )

        if not self.allow_short_inputs and processed_length <= 64:
            raise ValueError(f"Sample {idx} has too short sequence length: {processed_length}")

        return dataset_sample

    def _read_timed_units(self, idx: int, sample: Dict[str, Any], cfg: Dict[str, Any]):
        audio = self._timed_units_audio
        rng = np.random.default_rng([int(self.shuffle_seed), max(int(self.next_epoch) - 1, 0), int(idx)])
        input_ids, labels, waves = wtime_sample.to_omni_sample(
            sample, cfg, rng, guard=self._timed_units_guard, max_seq_len=self.max_seq_len,
            audio_token_id=audio.audio_token_id, sample_rate=audio.sample_rate,
            bos_token_id=self.tokenizer.bos_token_id, eos_token_id=self.tokenizer.eos_token_id,
            ms_per_token=10 * audio.encoder_subsampling_factor * audio.projector_subsampling_factor,
        )
        files = [{"type": "audio", "name": f"span{i}.wav", "content": w} for i, w in enumerate(waves)]
        return torch.from_numpy(input_ids), torch.from_numpy(labels), files

    def _read_files_from_sample(
        self, idx: int, sample: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        stream = self._get_stream(idx)
        if "files" not in sample:
            return []
        files = json.loads(sample["files"].decode("utf-8", errors="strict"))

        if stream.files_folder_path is not None:
            for file in files:
                file["path"] = str(Path(stream.files_folder_path) / file["path"])
            return files

        payload = sample.get("payloads")
        assert isinstance(payload, bytes), (
            f"Expected payloads in stream {stream.name} to be bytes, got {type(payload)}"
        )

        unpacked_files = self._unpack_tar_bytes(payload)

        if len(files) != len(unpacked_files):
            raise ValueError(
                f"Stream {stream.name}: metadata contains {len(files)} files, "
                f"but tar archive contains {len(unpacked_files)} files"
            )

        for file, (member_name, content) in zip(files, unpacked_files):
            assert file["type"] in {"image", "audio"}, (
                f"Stream {stream.name}: for S3 download only image and audio files are supported, "
                f"got type={file['type']}"
            )
            if file["name"] != member_name:
                raise ValueError(
                    f"Stream {stream.name}: file order mismatch, "
                    f"expected {file['name']}, got {member_name}"
                )
            file["content"] = content
        return files

    @staticmethod
    def _unpack_tar_bytes(payload: bytes) -> List[Tuple[str, bytes]]:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as tar:
            unpacked_files = [(member.name, tar.extractfile(member).read()) for member in tar]
        return unpacked_files

    def _get_stream(self, idx: int) -> OmniStream:
        stream_idx = self._get_stream_idx(idx)
        return self.streams[stream_idx]  # type: ignore

    def _get_stream_idx(self, idx: int) -> int:
        shard_idx = self.spanner[idx][0]
        stream_idx = self.stream_per_shard[shard_idx]
        return stream_idx


class OmniCollator:
    def __init__(
        self,
        collators: List[CollatorType],
        batch_mode: Optional[str] = None,
    ):
        self.collators = collators
        self.batch_mode = batch_mode

    def __call__(
        self, examples: List[Dict[str, Dict[str, torch.Tensor]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Collates samples from dataset with different modalities. Return example:
        {
            "input_ids": torch.Tensor([...]),
            "labels": torch.Tensor([...]),
            "modality_inputs": {
                "*modality*": { # For all modalities in the sample (optional)
                    ...
                },
                ...
            },
            ...
        }

        Args:
            examples (List[Dict[str, Dict[str, torch.Tensor]]]): list of samples from dataset

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: collated batch
        """

        collated_modalities = {
            collator.data_modality.value  # type: ignore
            for collator in self.collators
        }
        uncollated_modalities = sorted(set().union(*examples) - collated_modalities)
        assert not uncollated_modalities, (
            f"Samples carry {uncollated_modalities} but the dataset has no collator "
            "for them, so their tensors would be dropped while their placeholder "
            "tokens stay in input_ids. Add the matching collator together with the "
            "modality that reads its bundle (for `video_audio` soundtracks: "
            "`VideoAudioCollator` and an `AudioModality` with `data_flow_preset: "
            "video_audio_conformer`), or stop emitting that data (`read_soundtracks: "
            "false` on `GigaOmniVideoProcessor`)."
        )

        batch = {"modality_inputs": {}}
        for collator in self.collators:
            data_modality = collator.data_modality.value  # type: ignore
            collated_modality = collator(
                [example.get(data_modality, None) for example in examples]
            )  # type: ignore

            if collator.data_modality == DataModality.TEXT:  # type: ignore
                batch.update(collated_modality)
            elif collated_modality:
                batch["modality_inputs"][data_modality] = collated_modality  # type: ignore

        if self.batch_mode is not None:
            batch.update({"mode": self.batch_mode})  # type: ignore

        batch["local_max_seq_len"] = batch["input_ids"].shape[1]  # type: ignore
        return batch

    def split_batch(
        self, batch: Dict[str, Any], microbatch_size: int
    ) -> List[Dict[str, Dict[str, torch.Tensor]]]:
        assert batch["input_ids"].shape[0] % microbatch_size == 0, (
            f"Batch size {batch['input_ids'].shape[0]} is not divisible by microbatch size {microbatch_size}!"
        )

        num_microbatches = batch["input_ids"].shape[0] // microbatch_size

        splitted_batch: List[Dict[str, Any]] = [
            {"modality_inputs": {}} for _ in range(num_microbatches)
        ]
        for collator in self.collators:
            collator_batch_split = collator.split_batch(batch, microbatch_size)  # type: ignore
            for i in range(num_microbatches):
                if collator.data_modality == DataModality.TEXT:  # type: ignore
                    splitted_batch[i].update(collator_batch_split[i])
                else:
                    splitted_batch[i]["modality_inputs"].update(collator_batch_split[i])

        if self.batch_mode is not None:
            for i in range(num_microbatches):
                splitted_batch[i]["mode"] = self.batch_mode

        return splitted_batch


def wrap_dataloader(dataloader: torch.utils.data.DataLoader, collator: OmniCollator):
    def get_num_samples_in_batch(batch: Dict[str, Any]) -> int:
        assert isinstance(batch, dict)
        tokens_size = batch["input_ids"].shape[0]
        labels_size = batch["labels"].shape[0]
        assert tokens_size == labels_size
        return tokens_size

    def split_batch(
        batch: Dict[str, Dict[str, torch.Tensor]], microbatch_size: int
    ) -> List[Dict[str, Dict[str, torch.Tensor]]]:
        return collator.split_batch(batch, microbatch_size)

    return DataSpec(
        dataloader=dataloader,
        split_batch=split_batch,
        get_num_samples_in_batch=get_num_samples_in_batch,
    )


def build_omni_dataloader(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizer,
    device_batch_size: int,
    left_padding: bool = False,
    batch_mode: Optional[str] = None,
):
    processors_list = processors.build_processors(
        cfg.dataset.pop("processors", []), cfg.num_workers
    )

    collators_list = [
        getattr(collators, collator_config.pop("name"))(
            **om.to_container(collator_config, resolve=True)
        )  # type: ignore
        for collator_config in cfg.dataset.pop("collators", [])
    ]

    if not any(
        collator.data_modality == DataModality.TEXT for collator in collators_list
    ):
        collators_list = [
            TextCollator(
                pad_token_id=0,
                scale=2.0,
                left_padding=left_padding,
            )
        ] + collators_list

    streams = []
    streams_dict = cfg.dataset.pop("streams", [])
    for _, stream in streams_dict.items():
        streams.append(OmniStream(**stream))

    if not cfg.dataset.get("cache_limit"):
        cfg.dataset["cache_limit"] = "8gb"

    timed_units = cfg.dataset.pop("timed_units", None)
    dataset = StreamingOmniDataset(
        processors=processors_list,
        timed_units=timed_units,
        streams=streams,
        batch_size=device_batch_size,
        tokenizer=tokenizer,
        **cfg.dataset,
    )

    collate_fn = OmniCollator(collators=collators_list, batch_mode=batch_mode)
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        persistent_workers=cfg.get("persistent_workers", True),
        timeout=cfg.get("timeout", 0),
    )

    return wrap_dataloader(dataloader, collator=collate_fn)


def build_evaluator(
    cfg: DictConfig, dataset_name: str, tokenizer, model, *args, **kwargs
):
    eval_loader_cfg = cfg["eval_loaders"][dataset_name]
    eval_loader_cfg: DictConfig = om.merge(
        cfg.default_eval_loader_params, eval_loader_cfg
    )  # type: ignore
    eval_loader_cfg.dataset = om.merge(
        cfg.default_eval_dataset_params, eval_loader_cfg.dataset
    )

    eval_loader = build_omni_dataloader(
        cfg=eval_loader_cfg,
        tokenizer=tokenizer,
        device_batch_size=cfg.device_eval_batch_size,
        left_padding=eval_loader_cfg.pop("left_padding", False),
        batch_mode=eval_loader_cfg.pop("mode", None),
    )

    metric_names = list(eval_loader_cfg.pop("metric_names", model.val_metrics.keys()))
    eval_loader = Evaluator(
        label=dataset_name,
        dataloader=eval_loader,
        metric_names=metric_names,
        subset_num_batches=eval_loader_cfg.pop("subset_num_batches"),
    )

    return eval_loader
