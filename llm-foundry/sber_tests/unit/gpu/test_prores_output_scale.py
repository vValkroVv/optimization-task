"""ProRes `output_scale` on the real norm forwards, on CUDA.

`output_scale` threads through all five norm classes, and three of the five paths
cannot be exercised on CPU at all: `GatedNorm`/`ZeroCenteredGatedNorm` call
`mark_dynamic` (which rejects a fixed-shape CPU tensor) and `LlamaRMSNorm`
dispatches to the CUDA-only `dropout_add_rms_norm`. A norm that silently dropped
the kwarg would leave the residual unscaled with no error anywhere, so each class
is checked numerically here.

Two invariants per class:
  * `output_scale=None` is bit-identical to not passing it -- the default path,
    which every non-ProRes run takes, must not move;
  * `output_scale=a` scales the output by exactly `a`.

Run: `bash run_unit_gpu_tests.sh sber_tests/unit/gpu/test_prores_output_scale.py`.
"""

import pytest
import torch

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY, resolve_norm_class

pytestmark = [pytest.mark.unit, pytest.mark.requires_gpu]

HIDDEN = 256


@pytest.fixture(autouse=True)
def _dynamo_eager_fallback():
    """Let dynamo fall back to eager instead of raising.

    `GatedNorm.compute_norm` calls `mark_dynamic(x, 2)`, but the hidden dim is
    pinned by the norm weight, so inductor specializes it and dynamo raises
    `ConstraintViolationError` for a norm called on its own. This reproduces on
    `origin/main` verbatim -- it has nothing to do with `output_scale` -- and the
    fallback keeps the math under test the same math. The compiled kernels
    themselves are covered by the end-to-end training runs.
    """
    before = torch._dynamo.config.suppress_errors
    torch._dynamo.config.suppress_errors = True
    yield
    torch._dynamo.config.suppress_errors = before


def _norm(norm_type: str):
    config = GigaMixConfig(hidden_size=HIDDEN, tp_size=1, init_device="cuda")
    norm = resolve_norm_class(norm_type).from_config(config).cuda()
    with torch.no_grad():
        torch.nn.init.normal_(norm.weight, std=0.5)
    return norm


@pytest.mark.parametrize("norm_type", sorted(NORM_CLASS_REGISTRY))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_none_is_bit_identical_to_omitting_it(norm_type, dtype):
    norm = _norm(norm_type)
    x = torch.randn(4, 8, HIDDEN, device="cuda", dtype=dtype)
    with torch.no_grad():
        assert torch.equal(norm(x), norm(x, output_scale=None))


@pytest.mark.parametrize("norm_type", sorted(NORM_CLASS_REGISTRY))
@pytest.mark.parametrize("scale", [0.0, 0.25, 0.5])
def test_output_is_scaled_by_alpha(norm_type, scale):
    norm = _norm(norm_type)
    x = torch.randn(4, 8, HIDDEN, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        ref = norm(x)
        got = norm(x, output_scale=scale)
    assert ref.abs().max().item() > 0
    torch.testing.assert_close(got, ref * scale, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("norm_type", sorted(NORM_CLASS_REGISTRY))
def test_tensor_scale_matches_float_scale(norm_type):
    """The callback hands over a 0-dim buffer, not a Python float."""
    norm = _norm(norm_type)
    x = torch.randn(4, 8, HIDDEN, device="cuda", dtype=torch.float32)
    buf = torch.tensor(0.375, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(
            norm(x, output_scale=buf), norm(x, output_scale=0.375)
        )


@pytest.mark.parametrize("norm_type", sorted(NORM_CLASS_REGISTRY))
def test_gradient_flows_through_a_scaled_norm(norm_type):
    norm = _norm(norm_type)
    x = torch.randn(4, 8, HIDDEN, device="cuda", requires_grad=True)
    norm(x, output_scale=0.5).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
