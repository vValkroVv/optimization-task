# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Regressions for source preservation, shard validation, and failed preparation."""

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from threading import Event
from unittest.mock import patch

from filelock import FileLock

from streaming import JSONWriter, MDSWriter, Stream, StreamingDataset
from streaming.base.compression import decompress
from streaming.base.dataset import NEVER, _ShardState


class TestLocalSources(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.datasets = []
        self.addCleanup(self.datasets.clear)

    def _write(self, path, compression=None, writer=MDSWriter):
        with writer(out=str(path),
                    columns={'data': 'int'},
                    size_limit=300,
                    compression=compression) as out:
            for i in range(16):
                out.write({'data': i})

    def _dataset(self, **kwargs):
        dataset = StreamingDataset(batch_size=1, predownload=1, **kwargs)
        self.datasets.append(dataset)
        return dataset

    def _assert_source_unchanged(self, source, before):
        self.assertEqual({name: (source / name).read_bytes() for name in before}, before)

    def test_source_aliases_are_preserved_during_mixed_cache_eviction(self):
        for alias_kind, compression in product(('trailing_slash', 'symlink', 'relative'),
                                               (None, 'zstd')):
            with self.subTest(alias=alias_kind, compression=compression):
                directory = self.root / f'{alias_kind}-{compression}'
                source, remote, cache = (
                    directory / name for name in ('source', 'remote', 'cache'))
                self._write(source, compression)
                self._write(remote, compression)
                if alias_kind == 'trailing_slash':
                    alias = str(source) + '/'
                elif alias_kind == 'symlink':
                    link = directory / 'alias'
                    link.symlink_to(source, target_is_directory=True)
                    alias = str(link)
                else:
                    alias = os.path.relpath(source)
                before = {path.name: path.read_bytes() for path in source.iterdir()}
                index = json.loads((remote / 'index.json').read_text())
                limit = (remote / 'index.json').stat().st_size + max(
                    shard['raw_data']['bytes'] +
                    (shard['zip_data']['bytes'] if shard['zip_data'] else 0)
                    for shard in index['shards'])
                local_stream = Stream(local=str(source), remote=alias, keep_zip=False)
                self.assertTrue(local_stream.local_is_source)
                self.assertTrue(local_stream.safe_keep_zip)
                dataset = self._dataset(
                    streams=[local_stream,
                             Stream(remote=str(remote), local=str(cache))],
                    cache_limit=limit)
                for _ in range(2):
                    self.assertEqual([dataset[i]['data'] for i in range(32)], list(range(16)) * 2)
                self._assert_source_unchanged(source, before)
                self.assertLessEqual(dataset.cache_usage, limit)
                self.assertTrue(
                    any(dataset._shard_states[i] == _ShardState.REMOTE
                        for i in range(dataset.num_shards)
                        if dataset._cacheable_shards[i]))

    def test_cloud_uris_are_not_normalized_as_filesystem_paths(self):
        for remote in ('s3://bucket/data', 'gs://bucket/data', 'sftp://host/data'):
            stream = Stream(remote=remote, local=str(self.root / 'cache'))
            with patch('streaming.base.stream.os.path.realpath', side_effect=AssertionError):
                self.assertFalse(stream.local_is_source)

    def test_reader_eviction_cannot_delete_local_sources(self):
        source = self.root / 'source'
        self._write(source, 'zstd')
        before = {path.name: path.read_bytes() for path in source.iterdir()}
        dataset = self._dataset(local=str(source))
        shard = dataset.shards[0]
        for evict in (shard.evict, shard._evict_raw, shard._evict_zip):
            with self.assertRaisesRegex(ValueError, 'local dataset source'):
                evict()
            self._assert_source_unchanged(source, before)

    def test_incomplete_local_shards_fail_before_shared_state_is_allocated(self):
        for writer in (MDSWriter, JSONWriter):
            for compression in (None, 'zstd'):
                with self.subTest(writer=writer.__name__, compression=compression):
                    source = self.root / f'{writer.__name__}-{compression}'
                    self._write(source, compression, writer)
                    info = json.loads((source / 'index.json').read_text())['shards'][0]
                    missing = source / info['zip_data' if compression else 'raw_data']['basename']
                    missing.unlink()
                    before = {path.name: path.read_bytes() for path in source.iterdir()}
                    with patch('streaming.base.dataset.get_shm_prefix') as shared_state:
                        with self.assertRaises(ValueError) as error:
                            self._dataset(local=str(source))
                        self.assertIn(str(missing), str(error.exception))
                        shared_state.assert_not_called()
                    self._assert_source_unchanged(source, before)

    def test_local_zip_raw_and_both_forms_remain_readable(self):
        for writer in (MDSWriter, JSONWriter):
            for form in ('zip', 'raw', 'both'):
                with self.subTest(writer=writer.__name__, form=form):
                    source = self.root / f'{writer.__name__}-{form}'
                    self._write(source, 'zstd', writer)
                    if form != 'zip':
                        for path in source.glob('*.zstd'):
                            path.with_suffix('').write_bytes(decompress('zstd', path.read_bytes()))
                            if form == 'raw':
                                path.unlink()
                    before = {path.name: path.read_bytes() for path in source.iterdir()}
                    dataset = self._dataset(local=str(source), cache_limit=1)
                    self.assertEqual([dataset[i]['data'] for i in range(16)], list(range(16)))
                    self.assertEqual(dataset.cache_usage, 0)
                    self._assert_source_unchanged(source, before)

    def test_partial_local_raw_form_can_be_recovered_from_complete_zip(self):
        source = self.root / 'source'
        self._write(source, 'zstd', JSONWriter)
        info = json.loads((source / 'index.json').read_text())['shards'][0]
        zip_meta = source / info['zip_meta']['basename']
        raw_meta = source / info['raw_meta']['basename']
        raw_meta.write_bytes(decompress('zstd', zip_meta.read_bytes()))
        before = {path.name: path.read_bytes() for path in source.iterdir()}
        dataset = self._dataset(local=str(source), cache_limit=1)
        self.assertEqual(dataset[0]['data'], 0)
        self._assert_source_unchanged(source, before)

    def test_missing_local_source_error_releases_lock(self):
        source = self.root / 'source'
        self._write(source)
        dataset = self._dataset(local=str(source))
        dataset._shard_states[0] = _ShardState.REMOTE
        with self.assertRaisesRegex(ValueError, 'local dataset source'):
            dataset.prepare_shard(0)
        self.assertEqual(dataset._shard_states[0], _ShardState.REMOTE)
        self.assertFalse(dataset._cache_filelock.is_locked)
        with FileLock(dataset._cache_filelock_path).acquire(timeout=0.5):
            pass

    def test_failed_remote_download_resets_state_and_can_be_retried(self):
        source, cache = self.root / 'remote', self.root / 'cache'
        self._write(source)
        dataset = self._dataset(remote=str(source), local=str(cache))
        with patch.object(dataset.streams[0],
                          'prepare_shard',
                          side_effect=OSError('download failed')):
            with self.assertRaisesRegex(OSError, 'download failed'):
                dataset.prepare_shard(0)
        self.assertEqual(dataset._shard_states[0], _ShardState.REMOTE)
        self.assertFalse(dataset._cache_filelock.is_locked)
        dataset.prepare_shard(0)
        self.assertEqual(dataset[0]['data'], 0)

    def test_failed_decompression_restores_local_state_and_can_be_retried(self):
        source = self.root / 'source'
        self._write(source, 'zstd')
        dataset = self._dataset(local=str(source))
        before = {path.name: path.read_bytes() for path in source.iterdir()}
        with patch.object(dataset.streams[0], 'prepare_shard', side_effect=OSError('bad zip')):
            with self.assertRaisesRegex(OSError, 'bad zip'):
                dataset.prepare_shard(0)
        self.assertEqual(dataset._shard_states[0], _ShardState.LOCAL)
        self.assertFalse(dataset._cache_filelock.is_locked)
        self._assert_source_unchanged(source, before)
        self.assertEqual(dataset[0]['data'], 0)

    def test_waiting_worker_can_retry_after_another_download_fails(self):
        source, cache = self.root / 'remote', self.root / 'cache'
        self._write(source)
        dataset = self._dataset(remote=str(source), local=str(cache))
        stream = dataset.streams[0]
        prepare = stream.prepare_shard
        entered, release = Event(), Event()

        def fail_once(shard):
            if not entered.is_set():
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('Test failed to release the download')
                raise OSError('download failed')
            return prepare(shard)

        with patch.object(stream, 'prepare_shard', side_effect=fail_once):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(dataset.prepare_shard, 0)
                self.assertTrue(entered.wait(3))
                second = executor.submit(dataset.prepare_shard, 0)
                release.set()
                with self.assertRaisesRegex(OSError, 'download failed'):
                    first.result(timeout=3)
                second.result(timeout=3)
        self.assertEqual(dataset[0]['data'], 0)

    def test_eviction_mask_does_not_modify_unavailable_shard_timestamps(self):
        source, cache = self.root / 'remote', self.root / 'cache'
        self._write(source)
        dataset = self._dataset(remote=str(source), local=str(cache))
        dataset.prepare_shard(0)
        dataset._shard_access_times[1] = 0
        dataset.evict_coldest_shard()
        self.assertEqual(dataset._shard_access_times[1], 0)
        self.assertEqual(dataset._shard_states[0], _ShardState.REMOTE)
        self.assertEqual(dataset._shard_access_times[0], NEVER)


if __name__ == '__main__':
    unittest.main()
