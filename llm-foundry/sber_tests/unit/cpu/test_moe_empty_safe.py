import importlib
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load_grouped_gemm_module():
    path = Path(__file__).parents[3] / "llmfoundry" / "models" / "ops" / "grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("_llmfoundry_grouped_gemm_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.fast
def test_grouped_gemm_proxy_empty_input_keeps_autograd_edges():
    grouped_gemm_module = _load_grouped_gemm_module()

    class FailingOps:
        def gmm(self, *args, **kwargs):
            raise AssertionError("Kernel path must not be called for empty input.")

    proxy = grouped_gemm_module._GroupedGemmOpsProxy(FailingOps())
    x = torch.empty((0, 16), dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((2, 16, 8), dtype=torch.bfloat16, requires_grad=True)

    out = proxy.gmm(x, w, torch.tensor([0, 0], dtype=torch.int32), trans_b=False)
    assert out.shape == (0, 8)
    assert out.dtype == x.dtype

    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert w.grad is not None and w.grad.shape == w.shape
    assert torch.count_nonzero(w.grad).item() == 0


@pytest.mark.unit
@pytest.mark.fast
def test_grouped_gemm_proxy_non_empty_uses_underlying_kernel():
    grouped_gemm_module = _load_grouped_gemm_module()

    class RecordingOps:
        def __init__(self):
            self.called = False

        def gmm(self, x, w, batch_sizes, trans_b=False):
            self.called = True
            return x @ w[0]

    base_ops = RecordingOps()
    proxy = grouped_gemm_module._GroupedGemmOpsProxy(base_ops)

    x = torch.randn((3, 4), dtype=torch.bfloat16)
    w = torch.randn((1, 4, 5), dtype=torch.bfloat16)
    out = proxy.gmm(x, w, torch.tensor([3], dtype=torch.int32), trans_b=False)

    assert base_ops.called is True
    assert out.shape == (3, 5)


@pytest.mark.unit
@pytest.mark.fast
def test_build_m_indices_zero_tokens_returns_empty_without_kernel_launch():
    try:
        triton_utils = importlib.import_module("llmfoundry.models.ops.float8.triton_kernels.utils")
    except ImportError as e:
        pytest.skip(f"FP8/Triton stack is not importable in this environment: {e}")

    group_sizes_tensor = torch.tensor([0, 0, 0], dtype=torch.int32)
    m_indices = triton_utils.build_m_indices([0, 0, 0], group_sizes_tensor)

    assert m_indices.shape == (0,)
    assert m_indices.dtype == torch.int32
    assert m_indices.device == group_sizes_tensor.device


@pytest.mark.unit
@pytest.mark.fast
def test_grouped_gemm_fp8_wrapper_empty_input_keeps_autograd_edges(monkeypatch):
    try:
        grouped_gemm_fp8 = importlib.import_module("llmfoundry.models.ops.float8.grouped_gemm")
    except ImportError as e:
        pytest.skip(f"FP8 grouped GEMM stack is not importable in this environment: {e}")

    wrapper = object.__new__(grouped_gemm_fp8.GroupedGemmFp8Wrapper)

    def _fail_apply(*args, **kwargs):
        raise AssertionError("FP8 kernel path must not be called for empty input.")

    monkeypatch.setattr(grouped_gemm_fp8._GroupedGemm, "apply", _fail_apply)

    x = torch.empty((0, 16), dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((2, 32, 16), dtype=torch.bfloat16, requires_grad=True)

    out = grouped_gemm_fp8.GroupedGemmFp8Wrapper.group_gemm(
        wrapper,
        tensor_groups=x,
        tensor_groups_sizes=[0, 0],
        weights=w,
        weights_quantized=None,
        m_indices=None,
        tensor_groups_sizes_padded=torch.tensor([0, 0], dtype=torch.int32),
        triton_row2col=False,
    )
    assert out.shape == (0, 32)
    assert out.dtype == torch.bfloat16

    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert w.grad is not None and w.grad.shape == w.shape
    assert torch.count_nonzero(w.grad).item() == 0
