"""The MoE router skips pads when giga_mix gets ``inputs_embeds``.

GigaOmni hands the language model ``inputs_embeds``, no ``attention_mask`` for
end-padded rows and ``padding_mask`` from its final ``input_ids``. The router
must then count and balance exactly the tokens the text model does, which
derives the mask from ``input_ids`` itself; without ``padding_mask`` every pad
lands in the expert counts, the seq-aux loss and the router z-loss.
"""

import pytest
import torch

from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM
from sber_tests.unit.gpu.test_mtp_sp_loss_equivalence import _sp_config

PAD_ID = 0
SEQ = 64
REAL_LENGTHS = (SEQ, 40)
TOP_K = 2


def _moe_config():
    config = _sp_config(num_hidden_layers=2)
    config.use_mtp = False
    config.mtp_predictor_num = 0
    config.varlen_input = False
    config.first_k_dense_replace = 1
    config.num_routed_experts = 8
    config.num_experts_per_tok = TOP_K
    config.n_shared_experts = 1
    config.aux_loss_free = True
    config.router_z_loss_eps = 1e-2
    config.pad_token_id = PAD_ID
    return config


def _batch(vocab_size):
    torch.manual_seed(0)
    input_ids = torch.randint(1, vocab_size, (len(REAL_LENGTHS), SEQ), device="cuda")
    for row, real in enumerate(REAL_LENGTHS):
        input_ids[row, real:] = PAD_ID
    labels = input_ids.masked_fill(input_ids == PAD_ID, -100)
    return input_ids, labels


def _run(model, routers, labels, **inputs):
    for router in routers:
        router.local_tokens_per_expert.zero_()
    model.zero_grad(set_to_none=True)

    loss = model(labels=labels, return_dict=False, **inputs)[0]
    loss.backward()

    counts = torch.stack([router.local_tokens_per_expert.clone() for router in routers])
    gate_grads = torch.cat([router.gate.weight.grad.flatten() for router in routers])
    return loss.detach(), counts, gate_grads


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_inputs_embeds_with_padding_mask_route_like_input_ids():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    config = _moe_config()
    model = GigaMixForCausalLM(config).to("cuda").to(torch.bfloat16).train()
    routers = [
        module
        for module in model.modules()
        if getattr(module, "local_tokens_per_expert", None) is not None
    ]
    assert routers
    input_ids, labels = _batch(config.vocab_size)

    def embeds():
        return model.get_input_embeddings()(input_ids)

    text = _run(model, routers, labels, input_ids=input_ids)
    omni = _run(
        model,
        routers,
        labels,
        inputs_embeds=embeds(),
        padding_mask=input_ids != PAD_ID,
    )
    unmasked = _run(model, routers, labels, inputs_embeds=embeds())

    real_slots = sum(REAL_LENGTHS) * TOP_K
    assert text[1].sum(dim=1).tolist() == [real_slots] * len(routers)
    assert torch.equal(omni[1], text[1])
    torch.testing.assert_close(omni[0], text[0])
    torch.testing.assert_close(omni[2], text[2], rtol=1e-2, atol=1e-4)
    all_slots = len(REAL_LENGTHS) * SEQ * TOP_K
    assert unmasked[1].sum(dim=1).tolist() == [all_slots] * len(routers)
