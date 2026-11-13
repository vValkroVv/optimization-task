"""Bounded collection contracts; synthetic parser fixtures are not GPU evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import measure
import parse_results
import run as runner
from test_measure import _adapter, _profile_parser_fixture


def _bounded_fixture(root: Path):
    _profile_parser_fixture(root)
    metadata = json.loads((root / 'metadata.json').read_text())
    metadata['profile_capture']['profile_microbatches'] = 3
    (root / 'metadata.json').write_text(json.dumps(metadata))
    for rank in range(8):
        path = root / f'measurement/rank-{rank:02d}.json'
        receipt = json.loads(path.read_text())
        selected = rank < 2
        receipt['profile_capture'].update(profile_microbatches=3, collected_microbatches=3 if selected else 0,
            collection_stopped=selected, early_stop_cuda_drains=int(selected), export_update=2 if selected else None,
            token_reduction_calls=int(selected))
        path.write_text(json.dumps(receipt))
    for rank in (0, 1):
        events = [dict(ph='X', cat='user_annotation', name=f'task.profile_update.2.train_step.rank{rank}', ts=0, dur=50),
                  dict(ph='X', cat='user_annotation', name='task.token_count.global_reduction', ts=1, dur=1),
                  dict(ph='X', cat='kernel', name='ncclDevKernel_AllReduce_Sum', ts=2, dur=1)]
        for index in range(3):
            start = 10 + index * 10
            events.extend([dict(ph='X', cat='user_annotation', name=f'task.profile_update.2.microbatch.rank{rank}', ts=start, dur=8),
                           dict(ph='X', cat='gpu_user_annotation', name=f'task.profile_update.2.microbatch.rank{rank}', ts=start, dur=8),
                           dict(ph='X', cat='cpu_op', name='autograd::engine::evaluate_function: TestBackward', ts=start+1, dur=2)])
        path = root / f'native/profiling/traces/iteration_2/rank{rank}_trace.json'
        path.write_text(json.dumps(dict(traceEvents=events)))


@pytest.mark.parametrize('damage', ['none', 'fourth', 'optimizer', 'no_prefix', 'no_token', 'no_gpu', 'no_backward', 'no_collective', 'no_drain', 'trained_only_three'])
def test_bounded_parser_requires_complete_prefix_and_only_three_microbatches(tmp_path: Path, damage: str):
    _bounded_fixture(tmp_path)
    path = tmp_path / 'native/profiling/traces/iteration_2/rank0_trace.json'
    trace = json.loads(path.read_text())
    if damage == 'fourth':
        trace['traceEvents'].append(dict(ph='X', cat='user_annotation', name='task.profile_update.2.microbatch.rank0', ts=40, dur=8))
    elif damage == 'optimizer':
        trace['traceEvents'].append(dict(ph='X', cat='user_annotation', name='task.profile_update.2.optimizer_step.rank0', ts=40, dur=8))
    elif damage in ('no_prefix', 'no_token', 'no_gpu', 'no_backward', 'no_collective'):
        word = {'no_prefix':'train_step', 'no_token':'global_reduction', 'no_backward':'TestBackward', 'no_collective':'nccl'}
        trace['traceEvents'] = [event for event in trace['traceEvents']
            if not (event['cat'] == 'gpu_user_annotation' if damage == 'no_gpu' else word[damage] in event['name'])]
    elif damage in ('no_drain', 'trained_only_three'):
        receipt_path = tmp_path / 'measurement/rank-00.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['profile_capture']['early_stop_cuda_drains' if damage == 'no_drain' else 'microbatch_calls'] = 0 if damage == 'no_drain' else 3
        receipt_path.write_text(json.dumps(receipt))
    path.write_text(json.dumps(trace))
    summary = parse_results.summarize_run(tmp_path)
    assert summary['status'] == ('complete' if damage == 'none' else 'incomplete'), summary.get('errors')


def test_prefetch_cli_default_candidate_and_invalid_values(tmp_path: Path):
    assert runner._parse_args(['--run-id', 'default']).num_prefetch_batches == 0
    args = runner._parse_args(['--run-id', 'candidate', '--num-prefetch-batches', '32'])
    overrides = runner._training_overrides(args, {'native': tmp_path})
    index = overrides.index('--dataloader.num_prefetch_batches')
    assert overrides[index + 1] == '32'
    for flags in (['--num-prefetch-batches', '-1'], ['--profile-microbatches', '3'],
                  ['--profile-microbatches', '4']):
        with pytest.raises(SystemExit):
            runner._parse_args(['--run-id', 'invalid', *flags])


def test_host_memory_reads_are_outside_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    order = []
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(measure, 'host_memory_snapshot', lambda: order.append('host') or {'self': {'VmRSS_bytes': 100}})
    trainer_type, adapter, _, _ = _adapter(tmp_path, steps=2, warmup=1, window=1, monitor_interval=0, capture_microbatches=False)
    adapter.window_options['clock_ns'] = lambda: order.append('clock') or len(order) * 1000
    adapter.window_options['wall_clock_ns'] = lambda: order.append('wall') or len(order) * 1000
    with adapter:
        trainer_type().train()
    assert order == ['host', 'wall', 'clock', 'clock', 'wall', 'host']
    record = json.loads((tmp_path / 'measurement/rank-00.json').read_text())
    assert record['measurement']['memory']['host_before_start_timestamp']['self']['VmRSS_bytes'] == 100


def test_new_output_root_routes_rank_cache_and_shared_cache(monkeypatch: pytest.MonkeyPatch):
    import os
    import shutil
    import uuid

    output_root = runner.TASK_ROOT / 'runs/v2-prefetch-baseline/artifacts/raw'
    paths = runner._make_run_dirs('runner-layout-test-' + uuid.uuid4().hex, output_root)
    try:
        assert paths['shared_hf_cache'] == output_root / 'shared-cache/huggingface'
        monkeypatch.setenv('RANK', '7')
        monkeypatch.setenv('TRITON_CACHE_DIR', str(paths['cache_triton']))
        rank, selected = measure._route_rank_triton_cache(paths['measurement'])
        assert rank == 7
        assert selected == paths['cache_triton'] / 'rank_7'
        assert os.environ['TRITON_CACHE_DIR'] == str(selected)
        assert all((paths['cache_triton'] / f'rank_{rank}').is_dir() for rank in range(8))
    finally:
        shutil.rmtree(paths['attempt'])
