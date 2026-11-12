"""Softcap arithmetic, saved storage, compilation and CUDA memory checks."""

import pytest
import torch

import torchtitan.ops.softcap as implementation


def reference(x, cap):
    return (cap * torch.tanh(x.float() / cap)).to(x.dtype)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("cap", [0.5, 30.0])
def test_cpu_fallback_is_exact(dtype, cap):
    x = torch.linspace(-100, 100, 257, dtype=dtype).requires_grad_()
    expected = reference(x, cap)
    actual = implementation.softcap(x, cap)
    grad = torch.linspace(-2, 2, x.numel(), dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, grad)[0],
        torch.autograd.grad(expected, x, grad)[0],
        rtol=0,
        atol=0,
    )


def test_zero_cap_is_identity():
    x = torch.randn(9, requires_grad=True)
    assert implementation.softcap(x, 0.0) is x


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_custom_backward_matches_eager_and_retains_only_input(monkeypatch, dtype):
    # Exercise the autograd boundary on CPU without invoking a CUDA compiler.
    monkeypatch.setattr(implementation, "_fused_forward", implementation._forward)
    monkeypatch.setattr(implementation, "_fused_backward", implementation._backward)
    x = torch.linspace(-100, 100, 257, dtype=dtype).requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append(t) or t, lambda t: t
    ):
        actual = implementation._Softcap.apply(x, 30.0)
    assert len(saved) == 1
    assert saved[0] is x
    grad = torch.linspace(-2, 2, x.numel(), dtype=dtype)
    expected = reference(x, 30.0)
    actual_grad = torch.autograd.grad(actual, x, grad, create_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, x, grad, create_graph=True)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual_grad.sum(), x)[0],
        torch.autograd.grad(expected_grad.sum(), x)[0],
        rtol=0,
        atol=0,
    )


def test_pointwise_helpers_compile_fullgraph():
    x = torch.randn(13, 17, requires_grad=True)
    grad = torch.randn_like(x)
    forward = torch.compile(
        implementation._forward, backend="aot_eager", fullgraph=True
    )
    backward = torch.compile(
        implementation._backward, backend="aot_eager", fullgraph=True
    )
    torch.testing.assert_close(forward(x, 30.0), reference(x, 30.0), rtol=0, atol=0)
    torch.testing.assert_close(
        backward(x, grad, 30.0), implementation._backward(x, grad, 30.0), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Inductor")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cuda_numeric_and_peak_memory(dtype):
    x = torch.linspace(-100, 100, 2**22, device="cuda", dtype=dtype).requires_grad_()
    grad = torch.ones_like(x)
    # Warm compiler and allocator before measuring each path's incremental peak.
    implementation.softcap(x, 30.0).backward(grad)
    x.grad = None
    torch.cuda.synchronize()

    def run(fn):
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        out = fn(x, 30.0)
        (dx,) = torch.autograd.grad(out, x, grad)
        torch.cuda.synchronize()
        return out, dx, torch.cuda.max_memory_allocated() - baseline

    expected, expected_grad, eager_peak = run(reference)
    actual, actual_grad, fused_peak = run(implementation.softcap)
    # Inductor's fused arithmetic/transcendentals may differ in final rounding.
    tolerance = 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        actual_grad, expected_grad, rtol=tolerance, atol=tolerance
    )
    assert fused_peak < eager_peak


@pytest.mark.parametrize("training", [False, True])
def test_previous_loss_chunk_is_released_before_next_head_call(training):
    import weakref

    from torchtitan.components.loss import ChunkedCELoss

    class Head(torch.nn.Linear):
        previous = None
        calls = 0

        def forward(self, x):
            if self.previous is not None:
                assert self.previous() is None, "previous chunk logits are still live"
            output = reference(super().forward(x), 30.0)
            self.previous = weakref.ref(output)
            self.calls += 1
            return output

    head = Head(7, 19)
    loss = ChunkedCELoss(ChunkedCELoss.Config(num_chunks=4))
    loss.set_lm_head(head)
    loss.reset_components()
    hidden = torch.randn(2, 8, 7, requires_grad=training)
    labels = torch.randint(0, 19, (2, 8))
    with torch.set_grad_enabled(training):
        result = loss(hidden, labels, torch.tensor(16.0))
        if training:
            result.backward()
    assert head.calls == 4
    assert head.previous() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Inductor")
def test_cuda_noncontiguous_and_compiled_model():
    x = torch.randn(13, 17, device="cuda", dtype=torch.bfloat16).T.requires_grad_()
    fn = torch.compile(implementation.softcap, fullgraph=True)
    expected = reference(x, 30.0)
    actual = fn(x, 30.0)
    torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.016)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0],
        torch.autograd.grad(expected.sum(), x)[0],
        rtol=0.016,
        atol=0.016,
    )
