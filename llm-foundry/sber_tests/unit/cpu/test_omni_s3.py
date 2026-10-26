"""CPU regressions for S3 payloads and mixed local/remote shard caches.

Run with: python -m unittest sber_tests.unit.cpu.test_omni_s3 -v
The Omni definitions are compiled from production source to avoid importing the
GPU model stack. Streaming, shard IO and S3 download code run normally; only the
S3 client's transport is replaced with copies from a temporary fixture bucket.
"""

import ast
import io
import json
import logging
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "contrib" / "streaming"))
from streaming import MDSWriter, Stream, StreamingDataset  # noqa: E402
from streaming.base.dataset import NEVER, _ShardState  # noqa: E402


def _load_omni():
    path = ROOT / "llmfoundry" / "data" / "omni_data.py"
    names = {"StreamingOmniDataset", "build_omni_dataloader"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    data_path = ROOT / "llmfoundry" / "data" / "data.py"
    data_names = {"DTYPE2ENCODING_MAPPING_DICT", "ENCODING2DTYPE_MAPPING_DICT"}
    data_nodes = [node for node in ast.parse(data_path.read_text()).body
                  if (isinstance(node, ast.FunctionDef) and node.name == "read_binary_tokenized_sample")
                  or (isinstance(node, ast.Assign) and any(
                      isinstance(target, ast.Name) and target.id in data_names
                      for target in node.targets))]
    logger = logging.getLogger("omni_s3_test")
    logger.disabled = True
    namespace = dict(io=io, json=json, tarfile=tarfile, torch=torch, np=np,
                     Path=Path, List=List, Dict=Dict, Any=Any, Tuple=Tuple,
                     BaseProcessor=object, StreamingTextDataset=StreamingDataset,
                     logger=logger)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *data_nodes, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


OMNI = _load_omni()
OmniDataset = OMNI["StreamingOmniDataset"]


def _tar(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class TestOmniS3Payload(unittest.TestCase):
    def setUp(self):
        self.dataset = OmniDataset.__new__(OmniDataset)
        self.dataset.num_samples = 12000
        self.stream = SimpleNamespace(name="s3", files_folder_path=None)
        self.dataset._get_stream = lambda idx: self.stream

    def _read(self, files, entries):
        sample = {"files": json.dumps(files).encode(), "payloads": _tar(entries)}
        return self.dataset._read_files_from_sample(0, sample)

    def test_duplicate_names_preserve_image_and_audio_contents(self):
        files = [{"name": "same", "type": kind} for kind in ["image", "audio"]]
        result = self._read(files, [("same", b"image"), ("same", b"audio")])
        self.assertEqual([file["content"] for file in result], [b"image", b"audio"])

    def test_tar_member_bytes_are_preserved(self):
        entries = [("image.png", b"\x89PNG\r\n\x1a\nimage bytes"),
                   ("audio.wav", b"RIFF audio bytes")]
        files = [{"name": name, "type": kind}
                 for (name, _), kind in zip(entries, ["image", "audio"])]
        result = self._read(files, entries)
        self.assertEqual([file["content"] for file in result],
                         [content for _, content in entries])

    def test_mismatched_metadata_is_rejected(self):
        files = [{"name": "expected", "type": "image"}]
        for entries, message in [([], "metadata contains"),
                                 ([("other", b"data")], "file order mismatch")]:
            with self.subTest(entries=entries):
                with self.assertRaisesRegex(ValueError, message):
                    self._read(files, entries)

    def test_local_files_keep_the_files_folder_path(self):
        self.stream.files_folder_path = "/dataset/files"
        sample = {"files": json.dumps([{"path": "audio.wav", "type": "audio"}]).encode()}
        self.assertEqual(self.dataset._read_files_from_sample(0, sample),
                         [{"path": "/dataset/files/audio.wav", "type": "audio"}])

    def test_broken_sample_falls_back_to_the_next_global_index(self):
        expected = {"text": {"input_ids": torch.tensor([1, 2])}}
        self.dataset.get_sample = Mock(side_effect=[ValueError("bad tar"), expected])
        self.assertIs(self.dataset[9], expected)
        self.assertEqual([call.args for call in self.dataset.get_sample.call_args_list],
                         [(9,), (10,)])

    def test_retries_are_bounded(self):
        self.dataset.get_sample = Mock(side_effect=ValueError("bad tar"))
        with self.assertRaisesRegex(ValueError, "after 10000 attempts"):
            self.dataset[0]
        self.assertEqual(self.dataset.get_sample.call_count, 10000)

    def test_last_broken_sample_wraps_to_a_valid_index(self):
        self.dataset.num_samples = 3
        expected = {"text": {"input_ids": torch.tensor([1, 2])}}
        self.dataset.get_sample = Mock(side_effect=[ValueError("bad sample"), expected])
        self.assertIs(self.dataset[2], expected)
        self.assertEqual([call.args for call in self.dataset.get_sample.call_args_list],
                         [(2,), (0,)])

    def test_all_broken_samples_are_tried_only_once(self):
        self.dataset.num_samples = 3
        failure = ValueError("bad sample")
        self.dataset.get_sample = Mock(side_effect=failure)
        with self.assertRaisesRegex(ValueError, "after 3 attempts") as error:
            self.dataset[2]
        self.assertIs(error.exception.__cause__, failure)
        self.assertEqual([call.args for call in self.dataset.get_sample.call_args_list],
                         [(2,), (0,), (1,)])

    def test_invalid_indices_do_not_trigger_sample_replacement(self):
        self.dataset.num_samples = 3
        self.dataset.get_sample = Mock()
        for idx in (-1, 3):
            with self.assertRaisesRegex(ValueError, "Invalid sample index"):
                self.dataset[idx]
        self.dataset.get_sample.assert_not_called()

    def test_allow_short_inputs_is_forwarded_from_config(self):
        for allow_short_inputs in (False, True):
            with self.subTest(allow_short_inputs=allow_short_inputs):
                cfg = OmegaConf.create({
                    "dataset": {"streams": {}, "allow_short_inputs": allow_short_inputs},
                    "num_workers": 0, "drop_last": False,
                })
                dataset_factory = Mock()
                text_factory = Mock(return_value=SimpleNamespace(data_modality="text"))
                with patch.dict(OMNI, processors=SimpleNamespace(build_processors=lambda *args: []),
                                collators=SimpleNamespace(), om=OmegaConf,
                                DataModality=SimpleNamespace(TEXT="text"),
                                TextCollator=text_factory, StreamingOmniDataset=dataset_factory,
                                OmniCollator=Mock(), DataLoader=Mock(), wrap_dataloader=Mock()):
                    OMNI["build_omni_dataloader"](cfg, tokenizer=Mock(), device_batch_size=1)
                self.assertIs(dataset_factory.call_args.kwargs["allow_short_inputs"],
                              allow_short_inputs)

    def test_cache_limit_can_be_configured(self):
        for configured, expected in [(None, "8gb"), ("64gb", "64gb")]:
            with self.subTest(configured=configured):
                cfg = OmegaConf.create({"dataset": {"streams": {}}, "num_workers": 0,
                                        "drop_last": False})
                if configured is not None:
                    cfg.dataset.cache_limit = configured
                dataset_factory = Mock()
                text_factory = Mock(return_value=SimpleNamespace(data_modality="text"))
                with patch.dict(OMNI, processors=SimpleNamespace(build_processors=lambda *args: []),
                                collators=SimpleNamespace(), om=OmegaConf,
                                DataModality=SimpleNamespace(TEXT="text"),
                                TextCollator=text_factory, StreamingOmniDataset=dataset_factory,
                                OmniCollator=Mock(), DataLoader=Mock(), wrap_dataloader=Mock()):
                    OMNI["build_omni_dataloader"](cfg, tokenizer=Mock(), device_batch_size=1)
                self.assertEqual(dataset_factory.call_args.kwargs["cache_limit"], expected)


class TestMixedS3Cache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.bucket = self.root / "bucket"
        self.cache = self.root / "cache"
        self.datasets = []
        self.addCleanup(self.datasets.clear)

    def _write(self, path, count, compression=None):
        with MDSWriter(out=str(path), columns={"data": "bytes"},
                       size_limit=400, compression=compression) as writer:
            for i in range(count):
                writer.write({"data": bytes([i]) * 140})

    def _snapshot(self):
        return {path.name: path.read_bytes() for path in self.source.iterdir()}

    def _dataset(self, compression=None, remote_is_local=False, raw_only=False):
        self._write(self.source, 30, compression)
        self._write(self.bucket, 8, compression)
        if raw_only:
            from streaming.base.compression import decompress
            for path in self.source.glob("*.zstd"):
                path.with_suffix("").write_bytes(decompress("zstd", path.read_bytes()))
                path.unlink()
        index = json.loads((self.bucket / "index.json").read_text())
        shard_size = max(shard["raw_data"]["bytes"] +
                         (shard["zip_data"]["bytes"] if shard["zip_data"] else 0)
                         for shard in index["shards"])
        self.limit = (self.bucket / "index.json").stat().st_size + shard_size
        local = str(self.source)
        streams = [Stream(local=local, remote=local if remote_is_local else None),
                   Stream(remote="s3://fixture/train", local=str(self.cache))]
        dataset = StreamingDataset(streams=streams, cache_limit=self.limit,
                                   batch_size=1, predownload=1)
        self.datasets.append(dataset)
        return dataset

    def _download(self, bucket, key, filename, **kwargs):
        self.assertEqual(bucket, "fixture")
        self.assertTrue(key.startswith("train/"))
        shutil.copyfile(self.bucket / key.removeprefix("train/"), filename)

    def _mock_s3(self):
        session = Mock()
        session.client.return_value.download_file.side_effect = self._download
        return patch("boto3.session.Session", return_value=session)

    def test_last_broken_local_mds_sample_falls_back_to_valid_data(self):
        with MDSWriter(out=str(self.source),
                       columns={"tokens": "bytes", "labels": "bytes", "files": "bytes"}) as writer:
            for idx in range(3):
                tokens = np.array([idx], dtype=np.int64).tobytes()
                writer.write({"tokens": tokens, "labels": tokens,
                              "files": b"{" if idx == 2 else b"[]"})
        stream = Stream(local=str(self.source))
        stream.name = "local"
        stream.files_folder_path = str(self.root / "files")
        dataset = OmniDataset(processors=[], streams=[stream], batch_size=1,
                              allow_short_inputs=True)
        dataset.max_seq_len = 16
        self.datasets.append(dataset)
        before = self._snapshot()
        self.assertEqual(dataset[2]["text"]["input_ids"].tolist(), [0])
        self.assertEqual(self._snapshot(), before)

    def test_minimum_sequence_length_can_be_disabled(self):
        lengths = [1, 64, 65]
        with MDSWriter(out=str(self.source),
                       columns={"tokens": "bytes", "labels": "bytes"}) as writer:
            for length in lengths:
                tokens = np.arange(length, dtype=np.int64).tobytes()
                writer.write({"tokens": tokens, "labels": tokens})
        stream = Stream(local=str(self.source))
        stream.name = "local"
        stream.files_folder_path = str(self.root / "files")
        dataset = OmniDataset(processors=[], streams=[stream], batch_size=1)
        dataset.max_seq_len = 128
        self.datasets.append(dataset)
        self.assertFalse(dataset.allow_short_inputs)

        for allow_short_inputs in (False, True):
            dataset.allow_short_inputs = allow_short_inputs
            for idx, length in enumerate(lengths):
                with self.subTest(allow_short_inputs=allow_short_inputs, length=length):
                    if not allow_short_inputs and length <= 64:
                        with self.assertRaisesRegex(ValueError, "too short sequence length"):
                            dataset.get_sample(idx)
                    else:
                        sample = dataset.get_sample(idx)
                        self.assertEqual(sample["text"]["input_ids"].tolist(), list(range(length)))
                        self.assertEqual(sample["text"]["labels"].tolist(), list(range(length)))

        def replace_text(files, dataset_sample):
            return {"text": {"input_ids": torch.arange(129), "labels": torch.arange(129)}}

        dataset.processors = [replace_text]
        for allow_short_inputs in (False, True):
            dataset.allow_short_inputs = allow_short_inputs
            with self.subTest(allow_short_inputs=allow_short_inputs):
                with self.assertRaisesRegex(ValueError, "exceeding max_seq_len=128"):
                    dataset.get_sample(0)

    def test_eviction_preserves_local_sources_in_mixed_streams(self):
        for compression, remote_is_local, raw_only in [(None, False, False),
                                                       (None, True, False),
                                                       ("zstd", False, False),
                                                       ("zstd", False, True)]:
            with self.subTest(compression=compression, remote_is_local=remote_is_local,
                              raw_only=raw_only), self._mock_s3():
                dataset = self._dataset(compression, remote_is_local, raw_only)
                before = self._snapshot()
                self.assertEqual(dataset.cache_usage,
                                 (self.bucket / "index.json").stat().st_size)
                remote_ids = [i for i, stream_id in enumerate(dataset.stream_per_shard)
                              if stream_id == 1]
                self.assertTrue(all(dataset._shard_access_times[i] == NEVER for i in remote_ids))
                for _ in range(2):
                    for sample_id in range(dataset.num_samples):
                        sample = dataset[sample_id]
                        expected = sample_id if sample_id < 30 else sample_id - 30
                        self.assertEqual(sample["data"], bytes([expected]) * 140)
                    self.assertLessEqual(dataset.cache_usage, self.limit)
                    self.assertEqual({name: (self.source / name).read_bytes() for name in before},
                                     before)
                with self.assertRaisesRegex(ValueError, "Cannot evict shard"):
                    dataset.evict_shard(0)
                self.assertEqual({name: (self.source / name).read_bytes() for name in before},
                                 before)
                self.assertTrue(any(dataset._shard_states[i] == _ShardState.REMOTE
                                    for i in remote_ids))
                self.datasets.clear()
                del dataset
                for directory in [self.source, self.bucket, self.cache]:
                    shutil.rmtree(directory)

    def test_impossible_cache_limit_is_rejected_before_training(self):
        with self._mock_s3():
            self._dataset()
            with self.assertRaisesRegex(ValueError, "largest remote-backed shard"):
                StreamingDataset(remote="s3://fixture/train",
                                 local=str(self.root / "other_cache"),
                                 cache_limit=self.limit - 1, batch_size=1)

    def test_failed_eviction_releases_lock_and_resets_shard_state(self):
        with self._mock_s3():
            dataset = self._dataset()
            shard_id = next(i for i, stream_id in enumerate(dataset.stream_per_shard)
                            if stream_id == 1)
            dataset.cache_limit = dataset.cache_usage
            with self.assertRaisesRegex(ValueError, "no remote-backed shards"):
                dataset.prepare_shard(shard_id)
            self.assertEqual(dataset._shard_states[shard_id], _ShardState.REMOTE)
            self.assertFalse(dataset._cache_filelock.is_locked)
            dataset.cache_limit = self.limit
            self.assertEqual(dataset[30]["data"], bytes([0]) * 140)


if __name__ == "__main__":
    unittest.main()
