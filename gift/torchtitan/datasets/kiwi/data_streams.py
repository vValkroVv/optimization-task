from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
import random
import tarfile
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from boto3.s3.transfer import TransferConfig

from torchtitan.tools.logging import logger

from .configs import (
    BaseDataStreamConfig,
    TextDataStreamConfig,
    VisionDataStreamConfig,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class DataStream(ABC):
    """Iterable shard stream."""

    def __init__(
        self,
        cfg: BaseDataStreamConfig,
        *,
        dp_rank: int,
        dp_world_size: int,
        strict: bool = False,
    ) -> None:
        self.cfg = cfg
        self.strict = strict

        with open(self.cfg.data_file) as f:
            files = f.read().strip("\n").split("\n")
        if len(files) < dp_world_size:
            raise ValueError(
                f"Stream {self.cfg.name!r} lists {len(files)} shards in "
                f"{self.cfg.data_file}, fewer than the {dp_world_size} data-parallel "
                "ranks reading it: some ranks would get no data."
            )
        self.dp_rank = dp_rank
        self.chunk_files = files[(dp_rank % len(files)) :: dp_world_size]
        self._counts = self._load_or_build_counts()
        self._epoch_len = int(self._counts.sum())
        if self._epoch_len == 0:
            raise ValueError(
                f"Stream {self.cfg.name!r} has no rows after filter={self.cfg.filter!r}"
            )

    @abstractmethod
    def _load_data(self, markup: pd.DataFrame) -> list[dict[str, Any]]:
        """One dict per markup row; ``{}`` if the row could not be loaded."""

    def _load_parquet_data(
        self, parquet_idx: int, rows: list[int], idxs: list[int]
    ) -> list[tuple[int, dict[str, Any] | None]]:
        epoch = idxs[0] // self._epoch_len
        markup = self._markup(self.chunk_files[parquet_idx], parquet_idx, epoch).iloc[
            rows
        ]
        loaded = self._load_data(markup)
        out: list[tuple[int, dict[str, Any] | None]] = []
        for sample, idx in zip(loaded, idxs, strict=True):
            out.append((idx, sample or None))
        return out

    def _filter_df(self, data_df: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.filter:
            return data_df
        expr = compile(self.cfg.filter, "<filter>", "eval")
        keep = [
            i
            for i, row in enumerate(data_df.to_dict("records"))
            if eval(expr, {"__builtins__": {}}, row)
        ]
        return data_df.iloc[keep]

    def _load_or_build_counts(self) -> np.ndarray:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "dp_rank": self.dp_rank,
                    "filter": self.cfg.filter,
                    "data_file": os.path.abspath(self.cfg.data_file),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        path = os.path.join(os.path.dirname(self.cfg.data_file) or ".", f"{digest}.npy")
        if os.path.isfile(path):
            return np.load(path, allow_pickle=False).astype(np.int64, copy=False)

        def _count(chunk: str) -> int:
            return len(self._filter_df(pd.read_parquet(chunk)))

        with ThreadPoolExecutor(max_workers=min(32, len(self.chunk_files))) as pool:
            counts = np.fromiter(
                pool.map(_count, self.chunk_files),
                dtype=np.int64,
                count=len(self.chunk_files),
            )
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as f:
            np.save(f, counts)
        os.replace(tmp_path, path)
        return counts

    def _markup(self, chunk_file: str, parquet_idx: int, epoch: int) -> pd.DataFrame:
        return self._filter_df(
            pd.read_parquet(chunk_file).sample(
                frac=1.0,
                random_state=np.random.default_rng((int(epoch), int(parquet_idx))),
            )
        )

    def __iter__(self) -> Iterator[tuple[int, dict[str, Any] | None]]:
        yield from self.iter_samples()

    def _shard_jobs(
        self,
        start: int,
        skip_idx: frozenset[int],
    ) -> Iterator[tuple[int, list[int], list[int]]]:
        """Yield ``(parquet_idx, rows, idxs)`` from ``start`` across epochs."""
        start_epoch, skip = divmod(start, self._epoch_len)
        idx = start
        for epoch in itertools.count(start_epoch):
            order = np.random.default_rng(epoch).permutation(len(self.chunk_files))
            if skip:
                prefix = np.cumsum(self._counts[order])
                i = int(np.searchsorted(prefix, skip, side="right"))
                skip -= int(prefix[i - 1] if i else 0)
                order = order[i:]
            for parquet_idx in map(int, order):
                row0, skip = skip, 0
                n = int(self._counts[parquet_idx]) - row0
                if n <= 0:
                    continue
                rows: list[int] = []
                idxs: list[int] = []
                for i in range(n):
                    if idx + i not in skip_idx:
                        rows.append(row0 + i)
                        idxs.append(idx + i)
                idx += n
                if rows:
                    yield parquet_idx, rows, idxs

    def iter_samples(
        self,
        start: int = 0,
        skip_idx: set[int] | frozenset[int] | None = None,
    ) -> Iterator[tuple[int, dict[str, Any] | None]]:
        workers = max(1, self.cfg.num_reader_threads or 1)
        budget = max(1, self.cfg.prefetch_size or 1)
        pending: deque[tuple[Future[list[tuple[int, dict[str, Any] | None]]], int]] = (
            deque()
        )
        queued = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                for parquet_idx, rows, idxs in self._shard_jobs(
                    start, frozenset(skip_idx or ())
                ):
                    n = len(idxs)
                    while pending and queued + n > budget:
                        fut, k = pending.popleft()
                        queued -= k
                        yield from fut.result()
                    pending.append(
                        (
                            pool.submit(
                                self._load_parquet_data, parquet_idx, rows, idxs
                            ),
                            n,
                        )
                    )
                    queued += n
            finally:
                for fut, _ in pending:
                    fut.cancel()


def _iter_pt(
    tar: tarfile.TarFile, ids: list[str], strict: bool, tar_name: str
) -> Iterator[tuple[str, torch.Tensor]]:
    for id in ids:
        try:
            fileobj = tar.extractfile(f"{id}.pt")
        except Exception as e:
            if strict:
                raise
            logger.info(f"Failed to open {id} in {tar_name}:\n {e}")
            continue
        if fileobj is None:
            if strict:
                raise FileNotFoundError(f"{id}.pt is not a file in {tar_name}")
            continue
        yield id, torch.load(fileobj, map_location="cpu", weights_only=True)


class FSTarLoader:
    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def load_tar(
        self, tar_name: str, ids: list[str]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        try:
            with tarfile.open(tar_name) as tar:
                yield from _iter_pt(tar, ids, self.strict, tar_name)
        except Exception as e:
            if self.strict:
                raise
            logger.warning(
                f"Failed to get {tar_name} from NFS with the following error:\n{e}"
            )


class S3TarLoader:
    def __init__(self, s3_client: S3Client, strict: bool = False) -> None:
        self.s3_client = s3_client
        self.strict = strict

    def load_tar(
        self, tar_name: str, ids: list[str]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        try:
            bucket, key = tar_name.split("/", maxsplit=1)
            with io.BytesIO() as tar_buffer:
                self.s3_client.download_fileobj(
                    Bucket=bucket,
                    Key=key,
                    Fileobj=tar_buffer,
                    Config=TransferConfig(use_threads=False, num_download_attempts=2),
                )
                tar_buffer.seek(0)
                with tarfile.open(fileobj=tar_buffer) as tar:
                    yield from _iter_pt(tar, ids, self.strict, tar_name)
        except Exception as e:
            if self.strict:
                raise
            logger.warning(
                f"Failed to get {tar_name} from S3 with the following error:\n{e}"
            )


class TarLoader:
    def __init__(
        self,
        s3_client: S3Client | None = None,
        path_replace: str | None = None,
        strict: bool = False,
    ) -> None:
        self.fs_loader = FSTarLoader(strict)
        self.s3_loader = (
            S3TarLoader(s3_client, strict) if s3_client is not None else None
        )
        self.old, self.new = path_replace.split("->", 1) if path_replace else ("", "")

    def _replace_path(self, name: str) -> str:
        return name.replace(self.old, self.new) if self.old else name

    def load_tar(
        self, tar_name: str, ids: list[str]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        tar_name = self._replace_path(tar_name)
        if tar_name.startswith("/"):
            yield from self.fs_loader.load_tar(tar_name, ids)
        elif self.s3_loader is not None:
            yield from self.s3_loader.load_tar(tar_name, ids)
        else:
            logger.warning(
                f"{tar_name!r} looks like an S3 path but no s3_client was provided"
            )


class VisionDataStream(DataStream):
    cfg: VisionDataStreamConfig

    def __init__(
        self,
        cfg: VisionDataStreamConfig,
        *,
        dp_rank: int,
        dp_world_size: int,
        s3_client: S3Client | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__(
            cfg, dp_rank=dp_rank, dp_world_size=dp_world_size, strict=strict
        )
        self.tar_loader = TarLoader(
            s3_client, path_replace=cfg.tar_path_replace, strict=strict
        )

    def _domain_adaptation(self, sample: dict[str, Any]) -> None:
        if "gen_image" in sample:
            layout = "<text><gen_image>"
        elif "gen_video" in sample:
            layout = "<text><gen_video>"
        elif "semantic_image" in sample:
            layout = "<semantic_image><text>"
        else:
            raise ValueError(f"Not supported: {sample.keys()}")
        sample["layout"] = layout

    def _load_data(self, markup: pd.DataFrame) -> list[dict[str, Any]]:
        loaded: dict[str, torch.Tensor] = {}
        for tar_name, group_df in markup.groupby(
            self.cfg.visual_field_name, sort=False
        ):
            ids = [row["id"] for row in group_df.to_dict(orient="records")]
            for _id, tensor in self.tar_loader.load_tar(str(tar_name), ids):
                loaded[_id] = tensor
        samples: list[dict[str, Any]] = []
        for row in markup.to_dict(orient="records"):
            data = loaded.get(row["id"])
            if data is None:
                samples.append({})
                continue
            text_key, text = self._choose_text(row)
            sample = {
                self.cfg.visual_type: data,
                "text": text,
                "text_loss": self.cfg.text_loss,
            }
            _sample_properties = {f"stream:{self.cfg.name}", f"text_key:{text_key}"}
            properties = {
                self.cfg.visual_type: [
                    {f"type:{self.cfg.visual_type}"} | _sample_properties
                ],
                "text": [{"type:text"} | _sample_properties],
            }
            sample["properties"] = properties
            self._domain_adaptation(sample)
            samples.append(sample)
        return samples

    def _choose_text(self, data: dict[str, str]) -> tuple[str, str]:
        uncond_prob = self.cfg.uncond_prob or 0.0
        if random.random() < uncond_prob:
            return "uncond", ""
        probs = {
            key: value
            for key, value in self.cfg.text_probs.items()
            if isinstance(data.get(key), str)
        }
        total = sum(probs.values())
        if not probs or total <= 0:
            return "uncond", ""
        keys, weights = zip(*probs.items(), strict=True)
        key = np.random.choice(keys, p=np.array(weights) / total)
        return key, data[key]


class TextDataStream(DataStream):
    cfg: TextDataStreamConfig

    ENCODING2DTYPE_MAPPING_DICT = {
        0: np.uint16,
        1: np.uint32,
        2: np.int64,
        3: np.int32,
    }

    def _load_data(self, markup: pd.DataFrame) -> list[dict[str, Any]]:
        field = self.cfg.text_field_name
        samples: list[dict[str, Any]] = []
        for row in markup.to_dict(orient="records"):
            try:
                samples.append(
                    dict(
                        layout="<text>",
                        text_loss=True,
                        text=self._read_binary_tokenized_sample(row, field=field),
                        fim_rates=self.cfg.fim_rates,
                    )
                )
            except Exception:
                samples.append({})
        return samples

    def _read_binary_tokenized_sample(
        self, sample: dict[str, Any], field: str = "tokens"
    ) -> torch.Tensor:
        sample_dtype_code = sample.get("dtype", 2)
        assert sample_dtype_code in self.ENCODING2DTYPE_MAPPING_DICT, (
            f"Use unsupported dtype code{sample_dtype_code}"
        )
        tokens_dtype = self.ENCODING2DTYPE_MAPPING_DICT[sample_dtype_code]
        return torch.from_numpy(
            np.frombuffer(sample[field], dtype=tokens_dtype).copy().astype(np.int64)
        )
