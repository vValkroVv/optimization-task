"""Native dense Qwen hybrid backbone with last-marker pooling and FP32 Mean-Var head."""

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.entrypoints.openai.mean_var import SIGMA_FLOOR
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)


class Qwen3_5ForMeanVar(nn.Module):
    supports_lora = False
    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
    hf_to_sglang_mapper = None
    start_layer = Qwen3_5ForConditionalGeneration.start_layer
    end_layer = Qwen3_5ForConditionalGeneration.end_layer

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.pp_group = get_pp_group()
        if self.pp_group.world_size != 1 or get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Qwen Mean-Var supports TP=PP=1; use independent replicas")
        if quant_config is not None or getattr(config, "model_type", None) != "qwen3_5":
            raise ValueError(
                "Qwen Mean-Var requires an unquantized dense Qwen3.5/3.8 export"
            )
        if getattr(config, "mean_var_schema", None) != "gemma-mean-var-v1":
            raise ValueError(
                "use the Mean-Var exporter, not an ordinary Qwen checkpoint"
            )
        self.config = config
        self.quant_config = None
        self.model = Qwen3_5ForCausalLM(
            config=config.text_config, quant_config=None, prefix="model"
        )
        self.mean_var_head = nn.Linear(
            config.text_config.hidden_size, 2, bias=True, dtype=torch.float32
        )
        self.pooler = Pooler(PoolingType.LAST, normalize=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=True,
        **kwargs,
    ):
        if not get_embedding or input_embeds is not None:
            raise ValueError("Qwen Mean-Var requires --is-embedding and text token IDs")
        hidden = self.model(input_ids, positions, forward_batch)
        pooled = self.pooler(hidden, forward_batch).embeddings
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            raw = self.mean_var_head(pooled.float())
        return EmbeddingPoolerOutput(
            embeddings=torch.stack((raw[:, 0], raw[:, 1].abs() + SIGMA_FLOOR), dim=-1)
        )

    def load_weights(self, weights):
        loaded = Qwen3_5ForConditionalGeneration.load_weights(self, weights)
        missing = set(dict(self.named_parameters())) - set(loaded)
        if missing:
            raise ValueError(f"incomplete Qwen Mean-Var checkpoint: {sorted(missing)}")
        if self.mean_var_head.weight.dtype != torch.float32:
            raise ValueError("Mean-Var head must remain FP32")
        return loaded


EntryClass = Qwen3_5ForMeanVar
