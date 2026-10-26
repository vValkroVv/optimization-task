from dataclasses import dataclass


@dataclass(frozen=True)
class MLAModules:
    LlamaRotaryEmbedding: type
    apply_rotary_emb: callable
    apply_rotary_emb_mla: callable
