"""Tiny GPU lifecycle diagnostic; never a throughput measurement."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'gift'))
sys.path.insert(0, str(ROOT / 'runs/overnight-streaming-perf'))
import measure
import run as runner


def main():
    import torch
    import torch.distributed as dist
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.config.schema.components.optimizer import OptimizersContainerConfig
    from torchtitan.config.schema.tools.profiler import ProfilerConfig
    from torchtitan.datasets.streaming.core.prefetch import PrefetchIterator
    from torchtitan.tools.profiler import Profiler

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fail-microbatch', type=int, default=0)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.manual_seed(17)
    model = torch.nn.Linear(16, 16, device='cuda')
    optimizers = OptimizersContainer(OptimizersContainerConfig(name='AdamW', implementation='for-loop'), model_parts=[model])
    config = ProfilerConfig(enable_profiling=True, profile_ranks=[0, 1], record_shapes=False,
                            profile_freq=2, profiler_warmup=1, profiler_active=1,
                            profiler_repeat=1, profiler_skip_first=0, profiler_skip_first_wait=0)

    def token_sum(value):
        dist.all_reduce(value)
        return value

    reductions = SimpleNamespace(dist_sum=token_sum)

    class Trainer:
        def __init__(self):
            self.step = 0
            self.model_parts = [model]
            self.optimizers = optimizers
            self.metrics_processor = SimpleNamespace(ntokens_since_last_log=0)
            self.ntokens_seen = 0
            self.completed = 0
            self.optimized = 0

        def forward_backward_step(self, value):
            loss = model(value).square().mean() / 32
            loss.backward()
            self.completed += 1
            if self.step == 2 and args.fail_microbatch and self.completed == 32 + args.fail_microbatch:
                raise RuntimeError('intentional early-stop cleanup probe')
            return loss.detach()

        def train_step(self):
            queue = PrefetchIterator(lambda: iter(range(32)), capacity=32)
            try:
                inputs = [next(queue) for _ in range(32)]
            finally:
                queue.close()
                queue.join()
            reductions.dist_sum(torch.tensor([len(inputs)], device='cuda'))
            self.optimizers.zero_grad()
            for _ in inputs:
                self.forward_backward_step(torch.ones((2, 16), device='cuda'))
            self.optimizers.step()
            self.optimized += 1

        def train(self):
            with Profiler(config, base_folder=str(args.output / 'native')) as profiler:
                for step in (1, 2):
                    self.step = step
                    self.train_step()
                    profiler.step()

    trainer = Trainer()
    adapter = measure.NativeAdapter(trainer_class=Trainer, profiler_class=Profiler,
        optimizer_class=OptimizersContainer, metrics_class=type(trainer.metrics_processor),
        metrics_module=SimpleNamespace(), token_reduction_module=reductions, prefetch_class=PrefetchIterator,
        options=dict(run_id=args.output.name, output_dir=args.output / 'measurement', expected_steps=2,
                     warmup_updates=1, window_updates=0, monitor_interval=0, capture_microbatches=False,
                     profile_update=2, profile_microbatches=3),
        cuda_synchronize=torch.cuda.synchronize, memory_stats=torch.cuda.memory_stats)
    failure = None
    try:
        with adapter:
            trainer.train()
    except RuntimeError as error:
        if not args.fail_microbatch or 'intentional early-stop' not in str(error):
            raise
        failure = str(error)
    torch.cuda.synchronize()
    receipt = dict(rank=rank, forward_backward_calls=trainer.completed, optimizer_calls=trainer.optimized,
                   expected_injected_failure=failure, torch=torch.__version__, cuda=torch.version.cuda)
    (args.output / f'probe-rank-{rank}.json').write_text(json.dumps(receipt, indent=2) + '\n')
    dist.barrier()
    if rank == 0:
        cli = runner._parse_args(['--run-id', args.output.name, '--steps', '2', '--warmup', '1', '--window', '0',
                                 '--profile-update', '2', '--profile-microbatches', '3'])
        metadata = dict(run_id=args.output.name, steps=2, warmup_updates=1, window_updates=0, monitor_interval=0,
                        profile_capture=runner._profile_capture_spec(cli))
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps({'output': str(args.output), 'diagnostic_only': True, 'failure_mode': args.fail_microbatch}))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
