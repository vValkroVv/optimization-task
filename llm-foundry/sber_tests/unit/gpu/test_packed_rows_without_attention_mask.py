"""Packed, end-padded rows need no ``attention_mask``.

``TextFlashAttnCollator`` sends no mask for end padding by default. MLA then
takes its ``cu_seqlens``-only branch (document boundaries from the
collator's ``position_ids``) instead of the masked unpad branch. Every real
token must get the same logits either way, and a document packed into a row
must get the logits it gets alone.
"""

import pytest
import torch

from llmfoundry.data.collators.text_flash_attn_collator import TextFlashAttnCollator
from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM
from sber_tests.unit.gpu.test_mtp_sp_loss_equivalence import _sp_config

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


def _document(generator, length):
    body = torch.randint(3, 256, (length - 2,), generator=generator)
    return torch.cat([torch.tensor([BOS_ID]), body, torch.tensor([EOS_ID])])


def _logits(model, batch):
    inputs = {key: value.cuda() for key, value in batch.items() if key != "labels"}

    with torch.no_grad():
        return model(**inputs, return_dict=True).logits.float()


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_packed_row_logits_do_not_depend_on_the_mask():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    config = _sp_config(num_hidden_layers=2)
    config.use_mtp = False
    config.mtp_predictor_num = 0
    config.pad_token_id = PAD_ID
    model = GigaMixForCausalLM(config).to("cuda").to(torch.bfloat16).train()
    generator = torch.Generator().manual_seed(0)
    doc_a, doc_b = _document(generator, 24), _document(generator, 30)
    packed = torch.cat([doc_a, doc_b])
    examples = [
        {"input_ids": row, "labels": row.clone()} for row in (packed, doc_a, doc_b)
    ]

    def collate(**kwargs):
        collator = TextFlashAttnCollator(
            scale=16.0,
            pad_token_id=PAD_ID,
            bos_token_id=BOS_ID,
            eos_token_id=EOS_ID,
            **kwargs,
        )
        return collator(examples)

    unmasked = collate()
    masked = collate(return_attention_mask=True)
    assert "attention_mask" not in unmasked
    assert unmasked["input_ids"].shape[1] > packed.numel()

    no_mask_logits = _logits(model, unmasked)
    mask_logits = _logits(model, masked)

    tolerance = {"rtol": 1e-2, "atol": 1e-2}
    for row, document in enumerate((packed, doc_a, doc_b)):
        torch.testing.assert_close(
            no_mask_logits[row, : len(document)],
            mask_logits[row, : len(document)],
            **tolerance,
        )

    torch.testing.assert_close(
        no_mask_logits[0, : len(doc_a)], no_mask_logits[1, : len(doc_a)], **tolerance
    )
    torch.testing.assert_close(
        no_mask_logits[0, len(doc_a) : len(packed)],
        no_mask_logits[2, : len(doc_b)],
        **tolerance,
    )
