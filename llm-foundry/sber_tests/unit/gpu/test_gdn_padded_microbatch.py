"""Qwen3NextGatedDeltaNet on a padded microbatch of more than one row.

Training with `varlen_input: false` hands the layer an `attention_mask` and no
`cu_seqlens`, so its unpad path runs and sets `cu_seqlens` itself; the varlen
flatten one line later must not fire on top of it, or the epilogue undoes the
unpad and then asserts a leading axis of 1 against `batch_size`.
"""

import pytest
import torch

from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM
from sber_tests.unit.gpu.test_mtp_sp_loss_equivalence import _sp_config

SEQ = 128  # > 64, or the layer picks fused_recurrent, which training rejects
ROWS = 2


def _hybrid_config():
    config = _sp_config(num_hidden_layers=2)
    config.use_mtp = False
    config.mtp_predictor_num = 0
    config.varlen_input = False
    config.linear_attention_type = "Qwen3NextGatedDeltaNet"
    config.full_attention_layers = [1]
    config.linear_key_head_dim = 32
    config.linear_value_head_dim = 32
    config.linear_num_key_heads = 2
    config.linear_num_value_heads = 2
    config.linear_conv_kernel_dim = 4
    config.linear_use_short_conv = True
    return config


@pytest.mark.unit
@pytest.mark.requires_gpu
class TestPaddedMicrobatch:
    def test_forward_and_backward_run(self):
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        torch.manual_seed(0)
        config = _hybrid_config()
        model = GigaMixForCausalLM(config).to("cuda").to(torch.bfloat16).train()

        ids = torch.randint(0, config.vocab_size, (ROWS, SEQ), device="cuda")
        mask = torch.ones(ROWS, SEQ, dtype=torch.long, device="cuda")
        mask[1, SEQ // 2 :] = 0  # second row is padded: the unpad path fires
        labels = ids.masked_fill(mask == 0, config.ignore_index)
        position_ids = torch.arange(SEQ, device="cuda").repeat(ROWS, 1)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=position_ids,
            labels=labels,
            return_dict=False,
        )
        loss, logits = outputs[0], outputs[1]
        assert torch.isfinite(loss)
        assert logits.shape[:2] == (ROWS, SEQ)
        loss.backward()
