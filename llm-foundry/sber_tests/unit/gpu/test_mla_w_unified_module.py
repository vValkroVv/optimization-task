from dataclasses import dataclass
import pytest

import torch


@dataclass
class RoPEConfig:
    max_position_embeddings: int
    head_dim: int
    init_device: int | str | torch.device
    rope_theta: float = 100000.0
    dtype: torch.dtype = torch.float32


class TestMLA:
    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize(
        "batch,tokens,nheads,head_size",
        [
            (1, 4096, 4, 32),
            (2, 4096, 8, 64),
            (4, 4096, 16, 128),
            (1, 8192, 64, 256),
            (2, 8192, 32, 128),
        ],
    )
    def test_interleaved_rope_for_query(
        self, batch, tokens, nheads, head_size, unified_module
    ):

        device = torch.cuda.current_device()

        head_offset = head_size // 4 * 3
        q_rope_head_size = head_size - head_offset

        config = RoPEConfig(
            head_dim=q_rope_head_size,
            max_position_embeddings=tokens,
            init_device=device,
        )
        rotary_emb = unified_module(
            "llmfoundry.models.layers.custom_embedding", "LlamaRotaryEmbedding"
        )(config=config, device=device)

        query = torch.randn(
            (batch, tokens, nheads, head_size),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        query_clone = query.detach().clone().requires_grad_(True)

        cos, sin = rotary_emb(
            query,
            seq_len=tokens,
            position_ids=None,
            cu_seqlens=None,
        )

        rot_query_new = unified_module(
            "llmfoundry.models.layers.triton_rotary_embeddings", "apply_rotary_emb_mla"
        )(
            query,
            cos,
            sin,
            inplace=False,
            interleaved=True,
            head_offset=head_offset,
        )
        rot_query_new.sum().backward()

        q_nope = query_clone[..., :head_offset].contiguous()
        q_rope = query_clone[..., head_offset:].contiguous()

        q_rope = unified_module(
            "llmfoundry.models.layers.custom_embedding", "apply_rotary_emb"
        )(
            q_rope,
            cos,
            sin,
            inplace=False,
            interleaved=True,
            cu_seqlens=None,
            max_seqlen=None,
        ).view(*q_rope.shape)

        rot_query_old = torch.cat([q_nope, q_rope], dim=-1)
        rot_query_old.sum().backward()

        torch.testing.assert_close(rot_query_new, rot_query_old)
        torch.testing.assert_close(query.grad, query_clone.grad)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
