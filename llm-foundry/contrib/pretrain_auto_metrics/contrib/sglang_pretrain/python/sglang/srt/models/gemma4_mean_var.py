"""Native SGLang text-only Gemma4 prefill, last-marker pooling and FP32 Mean-Var head."""

import torch
from sglang.srt.distributed import get_pp_group
from sglang.srt.entrypoints.openai.mean_var import SIGMA_FLOOR
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM as NativeGemma4ForCausalLM
from sglang.srt.models.gemma4_causal import Gemma4TextModel
from torch import nn


class Gemma4ForMeanVar(nn.Module):
    supports_lora = False
    packed_modules_mapping = NativeGemma4ForCausalLM.packed_modules_mapping
    _get_k_eq_v_layers = NativeGemma4ForCausalLM._get_k_eq_v_layers
    get_attention_sliding_window_size = NativeGemma4ForCausalLM.get_attention_sliding_window_size

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        config = getattr(config, "text_config", config)
        self.pp_group = get_pp_group()
        if self.pp_group.world_size != 1:
            raise ValueError("Mean-Var pooling/head require PP=1")
        if quant_config is not None or getattr(config, "enable_moe_block", False):
            raise ValueError("Mean-Var v1 supports unquantized dense Gemma4 only")
        if getattr(config, "mean_var_schema", None) != "gemma-mean-var-v1":
            raise ValueError("use the Mean-Var exporter, not an ordinary Gemma checkpoint")
        self.config = config
        self.quant_config = None
        self.model = Gemma4TextModel(config=config, quant_config=None, prefix="model")
        self.mean_var_head = nn.Linear(config.hidden_size, 2, bias=True, dtype=torch.float32)
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
            raise ValueError("Mean-Var requires --is-embedding and text token IDs")
        # Same zero-based positions as the frozen HF evaluator. No +1 mutation.
        hidden = self.model(input_ids, positions, forward_batch)
        pooled = self.pooler(hidden, forward_batch).embeddings
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            raw = self.mean_var_head(pooled.float())
        return EmbeddingPoolerOutput(embeddings=torch.stack((raw[:, 0], raw[:, 1].abs() + SIGMA_FLOOR), dim=-1))

    def load_weights(self, weights):
        # Delegate all fused qkv/gate-up and tied K/V handling to native Gemma4.
        loaded = NativeGemma4ForCausalLM.load_weights(self, weights)
        missing = set(dict(self.named_parameters())) - set(loaded)
        if missing:
            raise ValueError(f"incomplete Mean-Var checkpoint: {sorted(missing)}")
        if self.mean_var_head.weight.dtype != torch.float32:
            raise ValueError("Mean-Var head must remain FP32")
        return loaded


EntryClass = Gemma4ForMeanVar
