import types

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from llmfoundry.models.layers import attention

DEV = "cuda"
DOCS = [2048, 1536, 2560]
PADS = 1024
S_R = sum(DOCS)
S_P = S_R + PADS

CFG = types.SimpleNamespace(
    hidden_size=1024,  # slim hidden; head structure is the 700B one
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=32,
    linear_num_value_heads=64,
    linear_conv_kernel_dim=4,
    rms_norm_eps=1e-6,
    linear_attn_o_norm_eps=1e-6,
    linear_gating_type="gated_rmsnorm_sigmoid",
    linear_sigmoid_gate_scale=2.0,
    linear_conv_init_std=0.1,
)


@pytest.fixture(scope="module")
def layer():
    torch.manual_seed(7)
    layer = attention.Qwen3NextGatedDeltaNet(CFG, layer_idx=0)
    layer.reset_parameters()
    for p in layer.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)
    return layer.to(DEV, torch.bfloat16).train()


@pytest.fixture(scope="module")
def x_full():
    gen = torch.Generator(device=DEV).manual_seed(99)
    return (torch.randn(1, S_P, CFG.hidden_size, device=DEV, dtype=torch.float32,
                        generator=gen) * 0.5).to(torch.bfloat16)


def _cu(doc_lens, total=None):
    b = [0]
    for d in doc_lens:
        b.append(b[-1] + d)
    if total is not None:
        b[-1] = total
    return torch.tensor(b, device=DEV, dtype=torch.int32)


def _fwd(layer, x, cu, need_grad=False):
    x = x.clone()
    if need_grad:
        x.requires_grad_(True)
    o, _, _ = layer(x, cu_seqlens=cu)
    return o, x


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_pad_tail_merged_into_last_doc_is_bitwise_safe(layer, x_full):
    o_r, x_r = _fwd(layer, x_full[:, :S_R], _cu(DOCS), need_grad=True)
    o_p, x_p = _fwd(layer, x_full, _cu(DOCS, total=S_P), need_grad=True)
    assert torch.equal(o_p[:, :S_R], o_r), "padded forward must match on real rows"

    gen = torch.Generator(device=DEV).manual_seed(5)
    dout = (torch.randn(1, S_R, CFG.hidden_size, device=DEV, dtype=torch.float32,
                        generator=gen) * 0.1).to(torch.bfloat16)
    o_r.backward(dout)
    dout_p = torch.zeros_like(o_p)
    dout_p[:, :S_R] = dout
    o_p.backward(dout_p)
    assert torch.equal(x_p.grad[:, :S_R], x_r.grad), "padded input-grad must match on real rows"
    assert x_p.grad[:, S_R:].abs().max() == 0, "pad rows must get no gradient"


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_recurrent_state_resets_exactly_at_doc_boundaries(layer, x_full):
    cu = _cu(DOCS)
    o_r, _ = _fwd(layer, x_full[:, :S_R], cu)
    x_mut = x_full[:, :S_R].clone()
    gen = torch.Generator(device=DEV).manual_seed(11)
    x_mut[:, :DOCS[0]] = (torch.randn(1, DOCS[0], CFG.hidden_size, device=DEV,
                                      dtype=torch.float32, generator=gen) * 0.5).to(torch.bfloat16)
    o_m, _ = _fwd(layer, x_mut, cu)
    assert (o_m[:, :DOCS[0]].float() - o_r[:, :DOCS[0]].float()).abs().max() > 1e-3, \
        "doc1 must actually change"
    assert torch.equal(o_m[:, DOCS[0]:], o_r[:, DOCS[0]:]), \
        "docs 2..n must be untouched by doc1 (state must reset at boundaries)"


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_pads_as_own_segment_also_safe(layer, x_full):
    o_r, _ = _fwd(layer, x_full[:, :S_R], _cu(DOCS))
    o_s, _ = _fwd(layer, x_full, _cu(DOCS + [PADS]))
    assert torch.equal(o_s[:, :S_R], o_r)
