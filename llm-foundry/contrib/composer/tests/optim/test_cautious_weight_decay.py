import pytest
import torch

from composer.functional.triton_kernels import adamw_cautious_parameter_update


@pytest.mark.gpu
@pytest.mark.parametrize('numel', [1, 257, 4099])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_adamw_cautious_parameter_update(numel: int, dtype: torch.dtype):
    generator = torch.Generator(device='cuda').manual_seed(17)
    param = torch.randn(numel, device='cuda', dtype=dtype, generator=generator)
    exp_avg = torch.randn(numel, device='cuda', dtype=dtype, generator=generator)
    variance = torch.rand(numel, device='cuda', dtype=dtype, generator=generator) + 0.1

    step_size = 0.03
    sqrt_bias_correction2 = 0.2
    eps = 1e-8
    weight_decay = 0.01

    param_fp32 = param.float()
    exp_avg_fp32 = exp_avg.float()
    variance_fp32 = variance.float()
    update = exp_avg_fp32 / (variance_fp32.sqrt() / sqrt_bias_correction2 + eps)
    decay = torch.where(param_fp32 * exp_avg_fp32 >= 0, weight_decay * param_fp32, 0.0)
    expected = (param_fp32 - decay - step_size * update).to(dtype)

    adamw_cautious_parameter_update(
        param,
        exp_avg,
        variance,
        step_size,
        sqrt_bias_correction2,
        eps,
        weight_decay,
    )

    torch.testing.assert_close(param, expected, rtol=2e-3, atol=2e-3)


def test_adamw_cautious_parameter_update_requires_cuda():
    tensor = torch.ones(1)
    with pytest.raises(RuntimeError, match='requires CUDA'):
        adamw_cautious_parameter_update(tensor, tensor, tensor, 0.1, 0.2, 1e-8, 0.01)
