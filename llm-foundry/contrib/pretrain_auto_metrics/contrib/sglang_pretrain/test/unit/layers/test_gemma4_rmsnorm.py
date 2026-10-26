import pytest
import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.layernorm import Gemma4RMSNorm

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_scale,shift", [(True, 0.0), (True, 1.0), (False, 0.0)])
def test_native_norm_scale_and_dtype(dtype, with_scale, shift):
    norm = Gemma4RMSNorm(4, eps=1e-4, with_scale=with_scale, scale_shift=shift)
    if with_scale:
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    values = torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.0, 0.0, 0.0, 0.0]], dtype=dtype)
    expected = values.float() * torch.rsqrt(
        values.float().square().mean(-1, keepdim=True) + 1e-4
    )
    if with_scale:
        expected *= norm.weight + shift
    result = norm.forward_native(values)
    assert result.dtype == dtype
    torch.testing.assert_close(result, expected.to(dtype))
    assert ("weight" in norm.state_dict()) == with_scale
