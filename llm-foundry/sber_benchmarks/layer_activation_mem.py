import contextlib
import dataclasses
from typing import Optional
import gc

import torch

from llmfoundry.models.gigar.configuration_gigar import GigarConfig
from llmfoundry.models.gigar.modelling_gigar import BLOCK_CLASS_REGISTRY

gc.disable()

@dataclasses.dataclass
class GigarLayerConfig:
    name: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None
    fused_mlp: bool = True
    fused_mlp_checkpoint_lvl: int = 1
    tp_size: int = 1

    dict = dataclasses.asdict


ctx_lens = [2048, 4096, 8192, 16384, 32768]
batch_sizes = [1, 2]

layer_cls = BLOCK_CLASS_REGISTRY["LlamaDecoderLayer"]

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

@contextlib.contextmanager
def torch_memory(message, minus_memory_bytes: int=0):

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mem_begin = torch.cuda.memory_stats()['active_bytes.all.current'] / 2 ** 20
    yield

    gc.collect()
    torch.cuda.synchronize()
    # delete all intermediate tensors
    torch.cuda.empty_cache()
    memory = (torch.cuda.memory_stats()['active_bytes.all.current'] - minus_memory_bytes) / 2 ** 20 - mem_begin
    print(f"{message}: {memory} MB")

for layer_cfg in [
    GigarLayerConfig(name="gigar-7b", hidden_size=4096, intermediate_size=11008, num_attention_heads=32),
    GigarLayerConfig(name="gigar-29b", hidden_size=6656, intermediate_size=17920, num_attention_heads=52, num_key_value_heads=13),
    GigarLayerConfig(name="llama-70b", hidden_size=8192, intermediate_size=28672, num_attention_heads=64, num_key_value_heads=8),
]:
    hf_config = GigarConfig(**layer_cfg.dict())

    for ctx_len in ctx_lens:
        for batch_size in batch_sizes:
            print(10*'===')
            print(f"{layer_cfg.name} with {ctx_len=} and {batch_size=}")
            print(10 * '---')
            with torch_memory('init memory per 1 layer'):
                layer = layer_cls(hf_config).cuda(torch.cuda.current_device())
                layer.train()

            hidden_states = torch.randn(
                batch_size, ctx_len, layer_cfg.hidden_size,
                dtype=torch.bfloat16
            ).cuda()

            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # warm up
                    for _ in range(3):
                        layer(torch.randn(
                            batch_size, ctx_len, layer_cfg.hidden_size,
                            dtype=torch.bfloat16
                        ).cuda())
                    ####
                    with torch_memory(
                            'activation memory per 1 layer',
                            minus_memory_bytes=hidden_states.storage().size() * hidden_states.element_size()
                    ):
                        output = layer(hidden_states)

            except RuntimeError as e:
                if "CUDA out of memory" not in str(e):
                    raise
                print(f"activation memmory OOM")
            finally:
                del output, layer, hidden_states

            gc.collect()
            torch.cuda.empty_cache()
            print(10 * '===' + '\n')
    print('\n'+10*'!!!!')


