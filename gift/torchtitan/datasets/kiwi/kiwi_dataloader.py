"""Multiprocess dataloader packing multimodal samples into token batches.

Workers still produce :class:`~torchtitan.datasets.kiwi.batch.ModelBatchData`.
The trainer-facing iterator yields the ``(dict[str, Tensor | None], Tensor)``
pair ``ReplicatedBatchDataLoader`` broadcasts. ``iter_model_batches`` is the
unpacked view.

Process layout, always ``spawn`` so workers never inherit a CUDA context:
  - one IO process: StreamMixer over DataStream iterators fills a mixed queue;
  - ``num_packing_workers`` processes that preprocess, pack to token_budget,
    and push ``ModelBatchData`` to the batch queue.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import random
import signal
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from queue import Empty, Full
from types import TracebackType
from typing import Any, Self, TypeAlias, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.config.schema.datasets.kiwi.kiwi_dataloader import (
    KiwiDataLoaderConfig,
    _count_paddings,
)
from torchtitan.tools.logging import logger

from .batch import ModalityData, ModelBatchData
from .configs import TextDataStreamConfig
from .constants import (
    DIFFUSION_GENERATION,
    ENCODER_MAPPING,
    ENCODER_ORDER,
    IPC_TIMEOUT,
    DataModality,
)
from .data_streams import DataStream, TextDataStream, VisionDataStream
from .packing import (
    BankRouter,
    BatchClass,
    StreamMixer,
    iter_with_warnings_on_slow,
    queue_weights,
)
from .magi_ranges import MagiRanges
from .preprocessing import Preprocessor, SampleClass, SampleRef, TargetMask
from .s3 import init_s3_client


@dataclass
class StreamCursor:
    """Out-of-order consume cursor: everything below ``watermark`` is done."""

    watermark: int = 0
    skip: set[int] = field(default_factory=set)

    def ack(self, idx: int | None) -> None:
        if idx is None or idx < self.watermark:
            return
        self.skip.add(idx)
        while self.watermark in self.skip:
            self.skip.remove(self.watermark)
            self.watermark += 1

    def to_dict(self) -> dict[str, Any]:
        return {"watermark": self.watermark, "skip": sorted(self.skip)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamCursor:
        if "watermark" in data:
            return cls(watermark=int(data["watermark"]), skip=set(data.get("skip", ())))
        return cls(watermark=int(data.get("step", 0)))


def _configure_worker(
    *,
    base_seed: int,
    dp_rank: int,
    worker_id: int,
    log_level: str,
) -> int:
    _prctl_pdeathsig()
    torch.set_num_threads(1)
    logger.setLevel(logging.getLevelNamesMapping()[log_level])
    seed = (base_seed * 1_000_003 + dp_rank * 10_007 + worker_id) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def _prctl_pdeathsig() -> None:
    parent = os.getppid()
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(1, signal.SIGKILL) != 0:
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
    except OSError:
        return
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGKILL)


def _padded_modalities(
    supported_modalities: Sequence[DataModality],
) -> list[DataModality]:
    return [modality for modality in supported_modalities if modality != "text"]


def _batch_has_modality(batch: BatchClass, modality: DataModality) -> bool:
    encoder = ENCODER_MAPPING[modality]
    return bool(getattr(batch, encoder))


def ensure_modalities(
    preprocessor: Preprocessor,
    batch: BatchClass,
    supported_modalities: Sequence[DataModality],
) -> None:
    for modality in _padded_modalities(supported_modalities):
        if _batch_has_modality(batch, modality):
            continue
        pad_data = preprocessor.prpr_mapper[modality].get_padding(1)
        pad_sample = SampleClass()
        if f"<{modality}>" in DIFFUSION_GENERATION:
            dtimestep = preprocessor._get_dtimestep()
            dtimestep.target_mask = TargetMask.empty(dtimestep.tokens)
            pad_sample.add(dtimestep, "text")
        pad_sample.add(pad_data, modality)
        batch.add(pad_sample)


def _pad_batch(
    preprocessor: Preprocessor,
    batch: BatchClass,
    token_budget: int,
    supported_modalities: Sequence[DataModality],
) -> None:
    ensure_modalities(preprocessor, batch, supported_modalities)
    pad_len = token_budget - batch.length
    if pad_len <= 0:
        return
    pad_data = preprocessor.prpr_mapper["text"].get_padding(pad_len)
    pad_sample = SampleClass(type_of_content="text")
    pad_sample.add(pad_data, "text")
    batch.add(pad_sample)


def _put_sample(queue: Any, sample: Any, stop_event: Any) -> bool:
    while not stop_event.is_set():
        try:
            queue.put(sample, timeout=IPC_TIMEOUT)
            return True
        except Full:
            continue
    return False


def iter_queue(
    queue: Any,
    *,
    stop_event: Any,
    timeout: float,
    name: str = "",
) -> Iterator[Any]:
    """Yield from ``queue`` until ``stop_event``, else TimeoutError."""
    label = name or "<unnamed>"
    while not stop_event.is_set():
        deadline = time.monotonic() + timeout
        item = None
        while not stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"queue {label} empty for {timeout:.0f}s")
            try:
                item = queue.get(timeout=min(IPC_TIMEOUT, remaining))
                break
            except Empty:
                continue
        if item is None:
            return
        yield item


def _build_stream(
    stream_cfg: Any,
    *,
    dp_rank: int,
    dp_world_size: int,
    s3_client: Any = None,
) -> DataStream:
    if isinstance(stream_cfg, TextDataStreamConfig):
        return TextDataStream(stream_cfg, dp_rank=dp_rank, dp_world_size=dp_world_size)
    return VisionDataStream(
        stream_cfg,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        s3_client=s3_client,
    )


def _io_main(
    config: KiwiDataLoader.Config,
    dp_rank: int,
    dp_world_size: int,
    mixed_queue: Any,
    stream_starts: dict[str, dict[str, int]],
    stream_skips: dict[str, dict[str, set[int]]],
    stop_event: Any,
) -> None:
    seed = _configure_worker(
        base_seed=config.seed,
        dp_rank=dp_rank,
        worker_id=-1,
        log_level=config.worker_log_level,
    )
    s3_client = (
        init_s3_client(config.s3_client) if config.s3_client is not None else None
    )
    weights = queue_weights(config.datasets)
    streams: dict[tuple[str, str], Iterator[Any]] = {}
    try:
        for dataset_config in config.datasets:
            ds_starts = stream_starts.get(dataset_config.name, {})
            ds_skips = stream_skips.get(dataset_config.name, {})
            for stream_cfg in dataset_config.streams:
                stream = _build_stream(
                    stream_cfg,
                    dp_rank=dp_rank,
                    dp_world_size=dp_world_size,
                    s3_client=s3_client,
                )
                streams[(dataset_config.name, stream_cfg.name)] = stream.iter_samples(
                    start=ds_starts.get(stream_cfg.name, 0),
                    skip_idx=ds_skips.get(stream_cfg.name),
                )
        mixer = StreamMixer(streams, weights, rng=random.Random(seed))
        for item in mixer:
            if not _put_sample(mixed_queue, item, stop_event):
                return
    except Exception:
        logger.exception("kiwi IO mixer failed")
        stop_event.set()
        raise
    finally:
        for iterator in streams.values():
            with contextlib.suppress(Exception):
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
        with contextlib.suppress(Exception):
            mixed_queue.cancel_join_thread()


def _put_batch(
    batch_queue: Any,
    batch: tuple[list[SampleRef], ModelBatchData],
    stop_event: Any,
) -> bool:
    while not stop_event.is_set():
        try:
            batch_queue.put(batch, timeout=IPC_TIMEOUT)
            return True
        except Full:
            continue
    return False


def _preprocess_raw(
    preprocessor: Preprocessor,
    raw: dict[str, Any],
    *,
    type_of_content: str,
) -> SampleClass | None:
    try:
        sample = preprocessor(**raw)
    except Exception as e:
        logger.error(f"preprocessor failed: {e}")
        return None
    sample.type_of_content = type_of_content
    return sample


def _packing_main(
    config: KiwiDataLoader.Config,
    dp_rank: int,
    worker_id: int,
    mixed_queue: Any,
    batch_queue: Any,
    stop_event: Any,
) -> None:
    _configure_worker(
        base_seed=config.seed,
        dp_rank=dp_rank,
        worker_id=worker_id,
        log_level=config.worker_log_level,
    )
    preprocessor = Preprocessor(config.preprocessor)
    content = {dataset.name: dataset.type_of_content for dataset in config.datasets}
    reserve = _count_paddings(config.supported_modalities)
    router = BankRouter(
        token_budget=config.token_budget - reserve,
        config=config.packing,
    )
    try:
        stalled = 0
        for (idx, payload), (dataset_name, stream_name) in iter_with_warnings_on_slow(
            iter_queue(
                mixed_queue,
                stop_event=stop_event,
                timeout=config.startup_timeout,
                name="mixed",
            ),
            name="mixed",
        ):
            ref = SampleRef(
                idx=idx,
                source_dataset=dataset_name,
                source_stream=stream_name,
            )
            sample = (
                None
                if payload is None
                else _preprocess_raw(
                    preprocessor,
                    payload,
                    type_of_content=content[dataset_name],
                )
            )
            if sample is None:
                router.remember(ref)
                stalled += 1
            elif router.add(sample, dataset_name, ref):
                stalled = 0
            else:
                stalled += 1
            if stalled >= 128:
                raise RuntimeError(
                    "dropped 128 samples in a row without filling a bank "
                    f"(token_budget={config.token_budget}); "
                    "samples are too long or keep failing"
                )
            if not router.ready():
                continue
            refs, batch = router.pop()
            _pad_batch(
                preprocessor,
                batch,
                config.token_budget,
                config.supported_modalities,
            )
            batch.postprocessing()
            if not _put_batch(batch_queue, (refs, batch.to_model_input()), stop_event):
                return
    except Exception:
        stop_event.set()
        raise
    finally:
        with contextlib.suppress(Exception):
            batch_queue.cancel_join_thread()


_MODALITY_FIELDS = ("tokens", "position_ids", "cu_seqlens", "target_mask", "span_ids")
_IGNORED_TRAINER_ARGS = (
    "tokenizer",
    "seq_len",
    "local_batch_size",
    "snapshot_every_n_steps",
)


def _warn_ignored_trainer_args(**kwargs: Any) -> None:
    """Warn for trainer kwargs this loader accepts and does not read."""
    ignored = [name for name in _IGNORED_TRAINER_ARGS if kwargs.get(name) is not None]
    if not ignored:
        return
    logger.warning(
        "KiwiDataLoader ignores %s; sequence length is config.token_budget "
        "and the tokenizer is preprocessor.tokenizer_path",
        ", ".join(ignored),
    )


def _encode_span_properties(
    spans: list[set[str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``list[set[str]]`` into a uint8 blob and per-span byte offsets."""
    chunks: list[bytes] = []
    offsets = [0]
    for props in spans:
        if not isinstance(props, set):
            raise ValueError(
                f"span_properties entries must be sets, got {type(props).__name__}"
            )
        blob = bytearray()
        for prop in sorted(props):
            if not isinstance(prop, str):
                raise ValueError(
                    f"span property must be a str, got {type(prop).__name__}"
                )
            encoded = prop.encode("utf-8")
            blob += len(encoded).to_bytes(4, "little")
            blob += encoded
        chunks.append(bytes(blob))
        offsets.append(offsets[-1] + len(blob))
    raw = b"".join(chunks)
    if raw:
        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    else:
        data = torch.empty(0, dtype=torch.uint8)
    return data, torch.tensor(offsets, dtype=torch.int64)


def _decode_span_properties(
    data: torch.Tensor, offsets: torch.Tensor
) -> list[set[str]]:
    if offsets.ndim != 1 or offsets.numel() < 1:
        raise ValueError(
            "span_properties_offsets must be a 1D tensor with at least the "
            f"end marker, got shape {tuple(offsets.shape)}"
        )
    raw = bytes(data.detach().cpu().tolist())
    bounds = [int(value) for value in offsets.detach().cpu().tolist()]
    spans: list[set[str]] = []
    for start, end in zip(bounds, bounds[1:]):
        region = raw[start:end]
        props: set[str] = set()
        cursor = 0
        while cursor < len(region):
            if cursor + 4 > len(region):
                raise ValueError("truncated span property length prefix")
            length = int.from_bytes(region[cursor : cursor + 4], "little")
            cursor += 4
            if cursor + length > len(region):
                raise ValueError("truncated span property bytes")
            props.add(region[cursor : cursor + length].decode("utf-8"))
            cursor += length
        spans.append(props)
    return spans


def model_batch_to_wire(
    batch: ModelBatchData,
) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
    """Flatten a model batch into the replicated-dataloader pair.

    Every modality key is always present, so the broadcast schema stays fixed
    for the iteration. Absent modalities are ``None``. ``labels`` is an empty
    placeholder: supervision lives in each modality's ``target_mask``.
    """
    if batch.magi_ranges.dist_attn_key is not None:
        raise ValueError(
            "dist_attn_key is built during batch prep and cannot be replicated"
        )
    span_bytes, span_offsets = _encode_span_properties(batch.span_properties)
    inputs: dict[str, torch.Tensor | None] = {
        "position_ids": batch.position_ids,
        "magi_seq_len": torch.tensor(batch.magi_ranges.seq_len, dtype=torch.int64),
        "magi_q_ranges": batch.magi_ranges.q_ranges,
        "magi_k_ranges": batch.magi_ranges.k_ranges,
        "magi_attn_type_map": batch.magi_ranges.attn_type_map,
        "span_properties_bytes": span_bytes,
        "span_properties_offsets": span_offsets,
    }
    for name in ENCODER_ORDER:
        modality = getattr(batch, name)
        for field_name in _MODALITY_FIELDS:
            key = f"{name}.{field_name}"
            inputs[key] = None if modality is None else getattr(modality, field_name)
    return inputs, torch.empty(0, dtype=torch.int64)


def model_batch_from_wire(
    inputs: dict[str, torch.Tensor | None],
    labels: torch.Tensor,
) -> ModelBatchData:
    """Inverse of :func:`model_batch_to_wire`."""
    if not isinstance(labels, torch.Tensor) or labels.numel() != 0:
        raise ValueError("kiwi replicated batches carry an empty labels placeholder")

    def require(key: str) -> torch.Tensor:
        value = inputs[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"replicated kiwi batch is missing tensor {key!r}")
        return value

    modalities: dict[str, ModalityData | None] = {}
    for name in ENCODER_ORDER:
        values = [inputs[f"{name}.{field_name}"] for field_name in _MODALITY_FIELDS]
        if all(value is None for value in values):
            modalities[name] = None
            continue
        if any(value is None for value in values):
            raise ValueError(f"modality {name!r} is only partially present")
        modalities[name] = ModalityData(
            *(value for value in values if isinstance(value, torch.Tensor))
        )
    seq_len = require("magi_seq_len")
    return ModelBatchData(
        position_ids=require("position_ids"),
        magi_ranges=MagiRanges(
            seq_len=int(seq_len.item()),
            q_ranges=require("magi_q_ranges"),
            k_ranges=require("magi_k_ranges"),
            attn_type_map=require("magi_attn_type_map"),
        ),
        span_properties=_decode_span_properties(
            require("span_properties_bytes"),
            require("span_properties_offsets"),
        ),
        **modalities,
    )


class KiwiDataLoader(BaseDataLoader):
    """Packs multimodal samples into fixed token-budget batches.

    One instance per ``batch`` coordinate. ``replicate_batch_over`` tells
    the trainer to broadcast its batches across ``pp``, ``cp``, and ``tp``.
    """

    Config: TypeAlias = KiwiDataLoaderConfig

    iter_queue = staticmethod(iter_queue)

    def __init__(
        self,
        config: Config,
        *,
        dp_rank: int,
        dp_world_size: int,
        tokenizer: Any = None,
        seq_len: int | None = None,
        local_batch_size: int | None = None,
        snapshot_every_n_steps: int | None = None,
    ) -> None:
        _warn_ignored_trainer_args(
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )
        self.config = config
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size

        self._ctx = torch.multiprocessing.get_context("spawn")
        self.batch_queue = self._ctx.Queue(maxsize=self.config.batch_queue_size)
        mixed_size = max(
            (
                stream.prefetch_size or 1
                for dataset_config in self.config.datasets
                for stream in dataset_config.streams
            ),
            default=1,
        )
        self.mixed_queue = self._ctx.Queue(maxsize=mixed_size)
        self.stop_event = self._ctx.Event()

        self._cursors: dict[str, dict[str, StreamCursor]] = {
            dataset_config.name: {
                stream.name: StreamCursor() for stream in dataset_config.streams
            }
            for dataset_config in self.config.datasets
        }
        self._processes: list[Any] = []
        self._io_process: Any = None
        self._started = False
        self._terminated = False

    def _all_queues(self) -> list[Any]:
        return [self.batch_queue, self.mixed_queue]

    def start(self) -> None:
        """Spawn IO and packing processes.

        Idempotent, and called from ``__iter__`` so constructing the loader
        stays side-effect free and checkpoint state can be restored first.
        Workers prefetch while the parent process does other setup; the first
        ``next`` waits up to ``startup_timeout`` for a batch.
        """
        if self._started:
            return
        if self._terminated:
            raise RuntimeError("kiwi dataloader cannot be restarted after terminate()")

        config = self.config
        self._processes = [
            self._ctx.Process(
                target=_packing_main,
                args=(
                    config,
                    self.dp_rank,
                    worker_id,
                    self.mixed_queue,
                    self.batch_queue,
                    self.stop_event,
                ),
                name=f"kiwi-pack-{worker_id}",
                daemon=True,
            )
            for worker_id in range(config.num_packing_workers)
        ]
        stream_starts = {
            dataset_name: {
                stream_name: cursor.watermark for stream_name, cursor in streams.items()
            }
            for dataset_name, streams in self._cursors.items()
        }
        stream_skips = {
            dataset_name: {
                stream_name: set(cursor.skip) for stream_name, cursor in streams.items()
            }
            for dataset_name, streams in self._cursors.items()
        }
        self._io_process = self._ctx.Process(
            target=_io_main,
            args=(
                config,
                self.dp_rank,
                self.dp_world_size,
                self.mixed_queue,
                stream_starts,
                stream_skips,
                self.stop_event,
            ),
            name="kiwi-io",
            daemon=True,
        )

        for process in self._processes:
            process.start()
        self._io_process.start()
        self._started = True

    def _workers_dead(self) -> str | None:
        def crashed(process: Any, label: str) -> str | None:
            if process is None or process.is_alive():
                return None
            clean = self.stop_event.is_set() and process.exitcode == 0
            if clean:
                return None
            return f"{label}; see the worker traceback above"

        for process in self._processes:
            msg = crashed(
                process, f"kiwi dataloader packing worker {process.name} exited"
            )
            if msg is not None:
                return msg
        return crashed(self._io_process, "the kiwi dataloader IO process exited")

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        self.stop_event.set()

        processes = [*self._processes]
        if self._io_process is not None:
            processes.append(self._io_process)
        for process in processes:
            with contextlib.suppress(Exception):
                process.join(timeout=IPC_TIMEOUT + 10.0)
            if process.is_alive():
                logger.warning(
                    f"kiwi dataloader worker still alive after stop_event: {process.name}"
                )

        for queue in self._all_queues():
            with contextlib.suppress(Exception):
                queue.cancel_join_thread()
                queue.close()

        logger.info("Kiwi dataloader stopped.")

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.terminate()

    def iter_model_batches(self) -> Iterator[ModelBatchData]:
        """Yield packed batches before they are flattened for replication."""
        self.start()
        for packed in iter_with_warnings_on_slow(
            iter_queue(
                self.batch_queue,
                stop_event=self.stop_event,
                timeout=self.config.startup_timeout,
                name="batch",
            ),
            name="batch",
        ):
            dead = self._workers_dead()
            if dead is not None:
                raise RuntimeError(dead)
            refs, batch = packed
            self._ack_batch(refs)
            yield batch
        dead = self._workers_dead()
        if dead is not None:
            raise RuntimeError(dead)

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        for batch in self.iter_model_batches():
            inputs, labels = model_batch_to_wire(batch)
            # None marks an absent modality. The trainer pair type does not
            # mention it; ReplicatedBatchDataLoader already transports None.
            yield cast(dict[str, torch.Tensor], inputs), labels

    def _ack_batch(self, refs: list[SampleRef]) -> None:
        for ref in refs:
            streams = self._cursors.get(ref.source_dataset)
            if streams is None:
                continue
            cursor = streams.get(ref.source_stream)
            if cursor is not None:
                cursor.ack(ref.idx)

    def state_dict(self) -> dict[str, Any]:
        datasets = {
            dataset_name: {
                stream_name: cursor.to_dict() for stream_name, cursor in streams.items()
            }
            for dataset_name, streams in self._cursors.items()
        }
        return {"datasets": datasets, "dp_world_size": self.dp_world_size}

    def load_state_dict(
        self, state_dict: dict[str, Any], *, strict: bool = False
    ) -> None:
        """Restore per-stream watermarks. Must run before ``start()``."""
        if not state_dict:
            return
        if self._started:
            raise RuntimeError(
                "kiwi dataloader state must be loaded before iteration starts"
            )

        saved_world_size = state_dict.get("dp_world_size")
        if saved_world_size != self.dp_world_size:
            logger.warning(
                f"Kiwi dataloader checkpoint was written with dp_world_size "
                f"{saved_world_size}, resuming with {self.dp_world_size}: shard "
                "assignment differs, so data will be revisited or skipped."
            )

        saved_datasets = state_dict.get("datasets", {})
        for dataset_name, streams in self._cursors.items():
            if dataset_name not in saved_datasets:
                if strict:
                    raise KeyError(
                        f"dataset {dataset_name!r} missing from dataloader checkpoint"
                    )
                logger.warning(
                    f"Dataset {dataset_name!r} missing from dataloader checkpoint; "
                    "it restarts from sample 0."
                )
                continue
            saved_streams = saved_datasets[dataset_name]
            for stream_name, cursor in streams.items():
                if stream_name in saved_streams:
                    restored = StreamCursor.from_dict(saved_streams[stream_name])
                    cursor.watermark = restored.watermark
                    cursor.skip = restored.skip
                elif strict:
                    raise KeyError(
                        f"stream {stream_name!r} missing from dataset "
                        f"{dataset_name!r} checkpoint"
                    )
                else:
                    logger.warning(
                        f"Stream {stream_name!r} of dataset {dataset_name!r} "
                        "missing from the checkpoint; it restarts from sample 0."
                    )

    def get_metrics(
        self,
        device: torch.device | str,
        process_group: dist.ProcessGroup | None = None,
    ) -> dict[str, int]:
        """Queue occupancy, reduced with MIN so the slowest rank is reported."""
        metrics = {
            "dataloader/qsize": self.batch_queue.qsize(),
            "dataloader/mixed_qsize": self.mixed_queue.qsize(),
        }

        keys = sorted(metrics)
        values = torch.tensor(
            [metrics[key] for key in keys], dtype=torch.int64, device=device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.MIN, group=process_group)
        return dict(zip(keys, values.cpu().tolist(), strict=True))
