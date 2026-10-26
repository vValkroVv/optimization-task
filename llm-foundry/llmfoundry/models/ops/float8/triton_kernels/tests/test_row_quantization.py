import pytest
import torch
from utils import _setup_standalone_path

_setup_standalone_path()

from llmfoundry.models.ops.float8._cudacherry import \
    quantize_vec128
from llmfoundry.models.ops.float8.triton_kernels.row_quantization import \
    row_quantization_1x128_fn


@pytest.mark.parametrize("hid_dim", [  # pyright: ignore[reportUntypedFunctionDecorator]
    1280, 128, 256, 640,
])
def test_row_quantization(hid_dim: int) -> None:
    device = torch.cuda.current_device()

    token_group_sizes = [128*17, 128*5, 128*9, 128]

    tensor_groups = torch.randn(
        (sum(token_group_sizes), hid_dim),
        device=device,
        dtype=torch.bfloat16)

    tq_triton = row_quantization_1x128_fn(tensor_groups)
    tq_triton_bfp16 = tq_triton[0].to(torch.bfloat16)

    tq_cuda = quantize_vec128(tensor_groups)
    tq_cuda = tq_cuda._data.view(torch.float8_e4m3fn)
    tq_cuda_bfp16 = tq_cuda.to(torch.bfloat16)

    assert torch.allclose(tq_cuda_bfp16, tq_triton_bfp16)
