import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import nullcontext
import functools
import multiprocessing
import torch.multiprocessing as mp

import torch
import torch.distributed as dist

from giga_fsdp import GigaFSDP
from giga_fsdp.logger import logger

from tests.utils.models import get_model, prep_data, Models
from tests.utils.freezing import freeze_model, FreezeStrategy
from tests.utils.wrappers import (
    wrap_model, 
    WrapStrategy,
)


@dataclass
class TestSetup:
    batch_size: int = 4
    seq_len: int = 32
    gradient_accumulation_steps: int = 2
    zero_stage: int = 3
    lr: float = 10
    n_iter: int = 10
    buffer_count: int = 2
    hpz_first_layers_num: int = 0
    hpz_group_size: Optional[int] = None
    hybrid_group_size: Optional[int] = None
    param_dtype: torch.dtype = torch.bfloat16
    wrap_strategy: WrapStrategy = WrapStrategy.CLASSIC
    seed: int = 0
    freeze_strategy: FreezeStrategy = FreezeStrategy.FULL_TRAINABLE
    model_type: Models = Models.TEXT
    profile: bool = False
    collector: Dict[str, Any] = field(default_factory=dict)
    num_minimum_gpu: int = 0

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "1"
    os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "30"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def error_handler(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            result = func(self, *args, **kwargs)
            self.add_to_collector(f'{func.__name__}_done', True)
            return result
        except Exception as e:
            self.add_to_collector(f'{func.__name__}_done', e)
    return wrapper

def spawn_with_manager(args):
    manager = multiprocessing.Manager()
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus

    shared_dict = manager.dict()
    mp.spawn(exec, args=(args, world_size, shared_dict), nprocs=world_size, join=True)
    args.collector = dict(shared_dict)[0]

class Executor:
    def __init__(self, rank, args, world_size):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.collector = {}
    
    def get_context_manager(self):
        if self.args.profile:
            return torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch.profiler.schedule(
                            wait=1,
                            warmup=10,
                            active=2,
                            repeat=1),
                        on_trace_ready=self.trace_handler
                        )
        else:
            return nullcontext()
    
    def trace_handler(self, p):
        if self.rank == 0:
            p.export_chrome_trace(
                    os.path.join(str(Path(__file__).parent.parent), 
                    "traces", 
                    f"{self.args.model_type}_{self.args.wrap_strategy}_{self.args.freeze_strategy}_trace.json")
                )

    def exec(self):
        self._setup()
        self.init_model()
        if self.collector['init_model_done'] == True:
            self.train_loop()
        self._finish()

    def _setup(self):
        setup(self.rank, self.world_size)
        torch.cuda.set_device(self.rank)
        torch.manual_seed(self.args.seed + self.rank)

    def _finish(self):
        dist.barrier()
        cleanup()  

    def add_to_collector(self, key, value):
        self.collector[key] = value

    @error_handler
    def init_model(self):
        self.data = prep_data(self.args, self.rank)
        self.model = get_model(self.args).to(self.rank)
        self.wrapper_dict = wrap_model(self.model, self.args.wrap_strategy)
        self.freeze_dict = freeze_model(self.model, self.args.freeze_strategy)
        self.add_to_collector("freeze_dict", self.freeze_dict)

        self.model = GigaFSDP(self.model, 
                zero_stage=self.args.zero_stage,
                fsdp_process_group=None,
                model_parallel_process_group=None,
                param_dtype=self.args.param_dtype,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                buffer_count=self.args.buffer_count,
                modules_to_wrap_with_names=self.wrapper_dict['giga_fsdp_modules_to_wrap_with_names'],
                rogue_layer_norm_modules_with_names=self.wrapper_dict['giga_fsdp_rogue_layer_norm_modules_with_names'],
                layer_norm_module_cls=self.wrapper_dict['giga_fsdp_layer_norm_module_cls'],
                device_id=torch.cuda.current_device(),
                hpz_first_layers_num=self.args.hpz_first_layers_num,
                hpz_group_size=self.args.hpz_group_size,
                hybrid_group_size=self.args.hybrid_group_size,
                )
        
        logger.debug(f'The model has been successfully wrapped in GigaFSDP!')

        named_param = set(name for name, _ in self.model.named_parameters())
        named_buffer = set(name for name, _ in self.model.named_buffers(recurse=False))
        for name in ("_layernormfreezed_parameter", "_layernorm_parameter"):
            if name in named_param:
                named_buffer.add(name)

        self.add_to_collector('named_parameters', named_param)
        self.add_to_collector('named_buffers', named_buffer)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
    
    def forward_backward(self):
        self.optimizer.zero_grad(set_to_none=False)
        for i in range(self.args.gradient_accumulation_steps):
            loss, *_ = self.model(**self.data)
            if loss.grad_fn is not None: # for inference mode case
                loss.backward()
        self.model.clip_grad_norm_(1.0)
        self.optimizer.step()
    
    @error_handler
    def train_loop(self): 
        self.add_to_collector("state_dict_before_training", self.model.state_dict())
        
        with self.get_context_manager() as context:
            for i in range(self.args.n_iter if not self.args.profile else 13):
                logger.debug(f"iter: {i}")
                self.forward_backward()
                if self.args.profile:
                    context.step()

        self.add_to_collector("state_dict_after_training", self.model.state_dict())
        local_state_dict = self.model.local_state_dict()
        self.add_to_collector("local_state_dict", {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in local_state_dict.items() })

        logger.debug(f'Train_loop completed successfully!')

def exec(rank, args, world_size, shared_dict):
    executor = Executor(rank, args, world_size)
    executor.exec()
    shared_dict[rank] = executor.collector


if __name__ == "__main__":
    pass