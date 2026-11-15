#!/usr/bin/env python3
"""Exact CPU/Gloo census of the production stream through the active loader."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import importlib.metadata
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import uuid
import traceback
from datetime import timedelta

import numpy as np

import data_contract as c

RAW = c.DATA_TASK_DIR / 'artifacts/raw/loader-verification'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def row_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype='<u4').tobytes()).hexdigest()


def census(tokens):
    tokens = np.asarray(tokens)
    if tokens.size == 0 or int(tokens.min()) < 0 or int(tokens.max()) >= c.VOCAB_SIZE:
        raise AssertionError('input token outside production vocabulary')
    unique = np.unique(tokens)
    raw = np.bincount((tokens.ravel() // c.EMBEDDING_OWNER_WIDTH), minlength=8)
    owned_unique = np.bincount(unique // c.EMBEDDING_OWNER_WIDTH, minlength=8)
    return {'unique_token_ids': int(unique.size), 'token_min': int(tokens.min()),
            'token_max': int(tokens.max()), 'owner_raw_requests': raw.tolist(),
            'owner_unique_requests': owned_unique.tolist()}, unique


def checked_manifest():
    rows = [json.loads(line) for line in c.ROW_MANIFEST.read_text().splitlines()]
    if len(rows) != c.NUM_ROWS:
        raise AssertionError('manifest row count changed')
    lookup = {}
    for index, row in enumerate(rows):
        if row['index'] != index or row['domain'] != c.DOMAIN_ORDER[index % 4]:
            raise AssertionError('manifest order/domain changed')
        if row['sha256'] in lookup:
            raise AssertionError('duplicate prepared row hash')
        lookup[row['sha256']] = row
    return lookup


def assert_unique_rows(indices, expected):
    if len(indices) != expected or len(set(indices)) != expected:
        raise AssertionError('consumed rows repeat or count differs')


class WorkerCollator:
    """Attach diagnostics outside the model input without changing native collation."""
    def __init__(self, native):
        self.native = native

    def __call__(self, samples):
        import torch
        inputs, labels = self.native(samples)
        worker = torch.utils.data.get_worker_info()
        return inputs, labels, worker.id, os.getpid()


def make_loader(depth, namespace):
    import torch.distributed as dist
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.config.schema.datasets.streaming.text.configs import (
        StreamingTextDatasetConfig, TextStreamConfig)
    from torchtitan.config.schema.datasets.streaming.text.dataloader import StreamingTextDataLoaderConfig
    from torchtitan.datasets.streaming.text.dataloader import StreamingTextDataLoader
    from torchtitan.distributed.data_world import DataWorld

    rank = dist.get_rank()
    world = DataWorld(rank=rank, world_size=8, local_rank=rank, local_world_size=8,
                      process_group=dist.group.WORLD, shared_memory_namespace=namespace)
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(c.TOKENIZER_DIR))
    dataset = StreamingTextDatasetConfig(
        streams=[TextStreamConfig(name=c.DATASET_ID, local=str(c.DATASET_ROOT), split='train')],
        fim_rates={'vanilla_fim': 0.0, 'inline_fim': 0.0, 'multiline_fim': 0.0, 'syntax_aware_fim': 0.0},
        shuffle=False, shuffle_seed=2048, num_canonical_nodes=1,
        epoch_size=c.NUM_ROWS, partition_algo='orig', legacy=True)
    config = StreamingTextDataLoaderConfig(dataset=dataset, num_workers=2,
        persistent_workers=True, pin_memory=False, drop_last=True,
        num_prefetch_batches=depth, infinite=True)
    loader = StreamingTextDataLoader(config, data_world=world, tokenizer=tokenizer,
        seq_len=c.SEQ_LEN, local_batch_size=c.LOCAL_BATCH_SIZE)
    loader._loader.collate_fn = WorkerCollator(loader._loader.collate_fn)
    assert loader.num_prefetch_batches == depth
    assert loader._loader.num_workers == 2 and loader._loader.persistent_workers
    assert loader.data_world.process_group is dist.group.WORLD
    assert tokenizer.get_vocab_size() == c.VOCAB_SIZE
    return loader, tokenizer, dataclasses.asdict(config)


def check_batch(batch, lookup, eos_id):
    import torch
    inputs, labels, worker_id, worker_pid = batch
    assert set(inputs) == {'input', 'positions'}
    tokens, positions = inputs['input'], inputs['positions']
    assert tokens.shape == labels.shape == positions.shape == (2, 4096)
    assert tokens.dtype == labels.dtype == torch.int64 and positions.dtype == torch.int32
    assert torch.equal(labels[:, :-1], tokens[:, 1:])
    assert bool(torch.all(labels[:, -1] == c.IGNORE_INDEX))
    assert int((labels != c.IGNORE_INDEX).sum()) == 8190
    array = tokens.numpy()
    rows = [lookup[row_hash(row)] for row in array]
    # Independent document-start scan: EOS itself belongs to the previous document.
    expected = np.empty((2, 4096), dtype=np.int32)
    for row in range(2):
        last_start = 0
        for col in range(4096):
            if col == 0 or array[row, col - 1] == eos_id:
                last_start = col
            expected[row, col] = col - last_start
    assert np.array_equal(expected, positions.numpy())
    stats, unique = census(array)
    stats.update(row_indices=[row['index'] for row in rows],
                 row_hashes=[row['sha256'] for row in rows],
                 domains=[row['domain'] for row in rows], worker_id=worker_id,
                 worker_pid=worker_pid, physical_tokens=8192, main_valid_targets=8190,
                 eos_tokens=int(np.count_nonzero(array == eos_id)),
                 positions_sha256=hashlib.sha256(positions.numpy().tobytes()).hexdigest(),
                 labels_sha256=hashlib.sha256(labels.numpy().tobytes()).hexdigest())
    return stats, unique


def child(depth, batches, output, namespace):
    import torch
    import torch.distributed as dist
    from torchtitan.datasets.streaming.text.dataloader import StreamingTextDataLoader

    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.manual_seed(2048)
    dist.init_process_group('gloo', timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    assert dist.get_world_size() == 8
    loader = None
    try:
        lookup = checked_manifest()
        loader, tokenizer, config = make_loader(depth, namespace)
        records, step_unique, local_unique = [], {}, []
        iterator = iter(loader)
        for batch_index in range(batches):
            record, unique = check_batch(next(iterator), lookup, tokenizer.eos_id)
            record['microbatch'] = batch_index
            records.append(record)
            local_unique.append(unique)
            if (batch_index + 1) % c.ACCUMULATION_STEPS == 0 or batch_index + 1 == batches:
                step_unique[str(batch_index // c.ACCUMULATION_STEPS)] = np.unique(np.concatenate(local_unique))
                local_unique = []
        indices = [index for record in records for index in record['row_indices']]
        assert_unique_rows(indices, batches * 2)
        assert {record['worker_id'] for record in records} == {0, 1}
        workers = {record['worker_id']: record['worker_pid'] for record in records}
        assert all(record['worker_pid'] == workers[record['worker_id']] for record in records)
        loader.close()
        loader = None
        assert not any(Path(f'/proc/{pid}').exists() for pid in workers.values())
        assert not torch.cuda.is_initialized()
        source = Path(inspect.getsourcefile(StreamingTextDataLoader)).resolve()
        assert source.is_relative_to((c.TASK_ROOT / 'gift').resolve())
        import torchtitan.datasets.streaming.core.dataloader as core
        import torchtitan.datasets.streaming.text.collators as collators
        import torchtitan.datasets.streaming.text.datasets as datasets
        import torchtitan.components.tokenizer as tokenizer_module
        source_files = {module.__name__: {'path': module.__file__, 'sha256': sha256(module.__file__)}
                        for module in (core, collators, datasets, tokenizer_module)}
        write_json(output / f'rank{rank}.json', {
            'status': 'PASS', 'rank': rank, 'backend': dist.get_backend(),
            'cuda_initialized': torch.cuda.is_initialized(), 'workers_stopped': True,
            'depth': depth, 'config': config, 'records': records,
            'runtime': {'python': sys.version, 'executable': sys.executable,
                        'torch': torch.__version__, 'torch_git': torch.version.git_version,
                        'mosaicml_streaming': importlib.metadata.version('mosaicml-streaming')},
            'active_source_files': source_files,
            'tokenizer_path': str(c.TOKENIZER_DIR.resolve()),
            'loader_source': {'path': str(source), 'sha256': sha256(source)}})
        np.savez_compressed(output / f'rank{rank}-step-tokens.npz', **step_unique)
        dist.barrier()
    finally:
        if loader is not None:
            loader.close()
        dist.destroy_process_group()


def summarize(directory, batches):
    ranks = [json.loads((directory / f'rank{rank}.json').read_text()) for rank in range(8)]
    assert all(rank['status'] == 'PASS' and rank['rank'] == i and not rank['cuda_initialized']
               and rank['workers_stopped'] for i, rank in enumerate(ranks))
    indices = [index for rank in ranks for record in rank['records'] for index in record['row_indices']]
    assert_unique_rows(indices, 8 * batches * 2)
    files = [np.load(directory / f'rank{rank}-step-tokens.npz') for rank in range(8)]
    updates = []
    try:
        for step in range((batches + 31) // 32):
            union = np.unique(np.concatenate([file[str(step)] for file in files]))
            raw = np.zeros(8, dtype=np.int64)
            for rank in ranks:
                for record in rank['records'][step * 32:(step + 1) * 32]:
                    raw += record['owner_raw_requests']
            unique = np.bincount(union // c.EMBEDDING_OWNER_WIDTH, minlength=8)
            updates.append({'update': step + 1, 'global_unique_token_ids': len(union),
                'owner_raw_requests': raw.tolist(), 'owner_unique_requests': unique.tolist(),
                'owner_raw_max_share': float(raw.max() / raw.sum()),
                'owner_raw_max_over_mean': float(raw.max() / raw.mean()),
                'owner_unique_max_share': float(unique.max() / unique.sum())})
    finally:
        for file in files:
            file.close()
    return {'status': 'PASS', 'ranks_complete': 8, 'rows_per_rank': 2 * batches,
            'global_unique_rows': len(indices), 'physical_tokens': len(indices) * 4096,
            'main_valid_targets': len(indices) * 4095, 'updates': updates,
            'rank_receipts': [str(directory / f'rank{rank}.json') for rank in range(8)]}


def order_signature(directory):
    result = []
    for rank in range(8):
        records = json.loads((directory / f'rank{rank}.json').read_text())['records']
        result.append([{key: record[key] for key in ('row_indices', 'row_hashes', 'labels_sha256',
                       'positions_sha256')} for record in records])
    return result


def run(batches, tag):
    if batches < 2 or batches > 672:
        raise ValueError('bounded run requires 2..672 microbatches')
    preparation = json.loads(c.PREPARATION_STATUS.read_text())
    assert preparation['status'] == 'prepared', 'producer has not completed'
    checked_manifest()
    identity = json.loads(c.DATA_RECEIPT.read_text())
    # Hash every shard and tokenizer once before opening either comparison arm.
    assert identity['status'] == 'prepared'
    mds = identity['dataset']
    assert sha256(c.ROW_MANIFEST) == mds['row_manifest']['sha256']
    assert mds['rows'] == mds['row_count'] == c.NUM_ROWS
    for item in mds['files']:
        assert sha256(c.DATASET_ROOT / item['path']) == item['sha256']
    for name, digest in c.TOKENIZER_FILE_SHA256.items():
        assert sha256(c.TOKENIZER_DIR / name) == digest
    root = RAW / tag
    root.mkdir(parents=True, exist_ok=False)
    summaries = []
    for depth in (0, 32):
        output = root / f'depth{depth}'
        output.mkdir()
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
            '--nproc-per-node=8', str(Path(__file__).resolve()), 'child', '--depth', str(depth),
            '--batches', str(batches), '--output', str(output), '--namespace', f'prod_{tag}_{depth}_{uuid.uuid4().hex}']
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
               'PYTHONPATH': str(c.TASK_ROOT / 'gift')}
        with (root / f'depth{depth}.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                returncode = process.wait(timeout=1800)
            except BaseException:
                # The launcher owns this fresh session, including its ranks/workers.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if returncode:
            raise RuntimeError(f'depth {depth} failed; see {root / f"depth{depth}.log"}')
        summary = summarize(output, batches)
        summary['depth'] = depth
        summaries.append(summary)
        write_json(c.DATA_TASK_DIR / f'{c.DATASET_ROOT.name}-loader-{tag}-depth{depth}.json', summary)
    assert order_signature(root / 'depth0') == order_signature(root / 'depth32'), 'fresh opening order differs'
    summary = {'status': 'PASS', 'scope': 'CPU/Gloo actual loader only; no GPU/training/performance claim',
        'batches_per_rank': batches, 'fresh_opening_exact_order_equal': True,
        'seed': 2048, 'data_receipt_sha256': sha256(c.DATA_RECEIPT),
        'row_manifest_sha256': sha256(c.ROW_MANIFEST), 'dataset': mds,
        'tokenizer_path': str(c.TOKENIZER_DIR), 'tokenizer_hashes': c.TOKENIZER_FILE_SHA256,
        'arms': summaries, 'logs': str(root), 'command': sys.argv}
    destination = c.LOADER_SUMMARY if batches == 672 else c.DATA_TASK_DIR / f'{c.DATASET_ROOT.name}-loader-{tag}.json'
    write_json(destination, summary)
    print(json.dumps({'status': 'PASS', 'receipt': str(destination)}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'child'))
    parser.add_argument('--batches', type=int, default=672)
    parser.add_argument('--tag', default='full')
    parser.add_argument('--depth', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--namespace')
    args = parser.parse_args(argv)
    if args.mode == 'child':
        child(args.depth, args.batches, args.output, args.namespace)
    else:
        try:
            run(args.batches, args.tag)
        except BaseException:
            write_json(RAW / args.tag / 'failure.json',
                       {'status': 'FAIL', 'command': sys.argv, 'traceback': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
