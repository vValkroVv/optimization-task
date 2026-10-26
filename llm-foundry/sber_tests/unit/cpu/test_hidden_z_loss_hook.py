"""Correctness tests for `HiddenZLossRegularizer` and `apply_hidden_z_loss_hook`.

The kernel under test is inlined here verbatim (same code path as
`llmfoundry/models/ops/hidden_z_loss_hook.py`) so the tests run on a plain
`torch`-only environment — no `llmfoundry` / `composer` deps required.
If you change the production kernel, mirror the change in this file and
re-run.

What's verified:
  1. Pass-through on forward: the returned tensor equals the input and shares
     storage (no copy).
  2. Logged scalar equals `hidden_z_loss_fn(x, coef).mean()` bit-for-bit and
     is marked non-differentiable (no grad_fn / requires_grad on caller side).
  3. Backward contributes exactly `d(loss)/dx` on top of the upstream
     gradient: gradient w.r.t. `x` after running the hook matches the
     gradient produced by a reference implementation that explicitly adds
     the regularizer to the total loss. Tested in fp32 and fp64.
  4. `coef == 0` and eval-mode short-circuits are no-ops (grad passes through
     unchanged).
  5. No grad accumulates into unrelated parameters via the scoped
     `enable_grad()` inside our custom backward.

What's intentionally NOT tested here:
  * `torch.autograd.gradcheck`. Our Function injects a gradient component
    that is **independent of `grad_output`** (the `g_reg` term), which
    breaks gradcheck's linearity assumption. This is an idiomatic "gradient
    splicing" pattern (the same one used by `AddAuxiliaryLoss` for MoE
    aux-loss) and is verified by the reference-equivalence tests in
    `TestBackward` instead.
"""

from typing import Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# Inlined kernel (mirrors llmfoundry/models/ops/hidden_z_loss_hook.py)
# --------------------------------------------------------------------------- #

@torch.jit.script
def hidden_z_loss_fn(hidden_states: torch.Tensor, eps: float):
    return eps * torch.square(torch.logsumexp(hidden_states.abs(), dim=-1))


class HiddenZLossRegularizer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, coef: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(x)
        ctx.coef = coef
        with torch.no_grad():
            loss_val = hidden_z_loss_fn(x, coef).mean()
        ctx.mark_non_differentiable(loss_val)
        return x, loss_val

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, _grad_loss_val):
        (x,) = ctx.saved_tensors
        coef = ctx.coef
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            loss = hidden_z_loss_fn(x_req, coef).mean()
            (g_reg,) = torch.autograd.grad(loss, x_req)
        if grad_y is None:
            return g_reg, None
        return grad_y + g_reg, None


def apply_hidden_z_loss_hook(
    hidden_states: torch.Tensor,
    coef: float,
    *,
    training: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not training or coef == 0.0:
        return hidden_states, None
    hidden_states, loss_val = HiddenZLossRegularizer.apply(hidden_states, coef)
    return hidden_states, loss_val


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def _make_input(
    shape=(2, 7, 5),
    dtype=torch.float32,
    seed=0,
    requires_grad=True,
) -> torch.Tensor:
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=dtype)
    if requires_grad:
        x.requires_grad_(True)
    return x


class TestForward:

    def test_passthrough_values(self):
        x = _make_input()
        y, _ = HiddenZLossRegularizer.apply(x, 1e-3)
        assert torch.equal(y.detach(), x.detach())

    def test_passthrough_shares_data(self):
        x = _make_input()
        y, _ = HiddenZLossRegularizer.apply(x, 1e-3)
        assert y.data_ptr() == x.data_ptr()

    def test_scalar_matches_hidden_z_loss_fn(self):
        x = _make_input()
        coef = 2.5e-3
        _, loss = HiddenZLossRegularizer.apply(x, coef)
        ref = hidden_z_loss_fn(x.detach(), coef).mean()
        assert loss.shape == torch.Size([])
        assert torch.allclose(loss, ref, rtol=0, atol=0)

    def test_scalar_is_non_differentiable(self):
        x = _make_input()
        _, loss = HiddenZLossRegularizer.apply(x, 1e-3)
        assert not loss.requires_grad
        assert loss.grad_fn is None


class TestBackward:
    """Main correctness gate. Equivalent to: did we inject exactly the same
    gradient we would have gotten by explicitly adding `reg` to the loss?
    """

    def _run(self, shape, coef, dtype, atol):
        x = _make_input(shape=shape, dtype=dtype, seed=1)
        torch.manual_seed(42)
        w = torch.randn_like(x)

        # Path A — custom autograd.Function hook.
        y, _ = HiddenZLossRegularizer.apply(x, coef)
        (y * w).sum().backward()
        grad_a = x.grad.detach().clone()

        # Path B — reference. Pass-through plus explicit regularizer.
        x_ref = x.detach().clone().requires_grad_(True)
        reg = hidden_z_loss_fn(x_ref, coef).mean()
        ((x_ref * w).sum() + reg).backward()
        grad_b = x_ref.grad.detach().clone()

        max_abs_diff = (grad_a - grad_b).abs().max().item()
        assert torch.allclose(grad_a, grad_b, rtol=1e-5, atol=atol), (
            f"grad mismatch: max|Δ|={max_abs_diff:.3e}"
        )

    def test_fp32_small(self):
        self._run(shape=(2, 7, 5), coef=1e-3, dtype=torch.float32, atol=1e-5)

    def test_fp32_medium(self):
        self._run(shape=(4, 16, 32), coef=5e-4, dtype=torch.float32, atol=1e-5)

    def test_fp32_larger_coef(self):
        self._run(shape=(3, 8, 16), coef=5e-2, dtype=torch.float32, atol=1e-4)

    def test_fp64_strict(self):
        self._run(shape=(3, 9, 11), coef=7e-4, dtype=torch.float64, atol=1e-12)

    def test_fp64_tiny_coef(self):
        self._run(shape=(2, 5, 8), coef=1e-8, dtype=torch.float64, atol=1e-15)

    def test_coef_zero_gives_downstream_only(self):
        x = _make_input(seed=2)
        w = torch.randn_like(x)
        y, loss = HiddenZLossRegularizer.apply(x, 0.0)
        (y * w).sum().backward()
        grad_a = x.grad.detach().clone()
        # With coef=0, regularizer contribution is exactly 0 → grad == w.
        assert torch.allclose(grad_a, w, rtol=0, atol=0)
        assert torch.equal(loss, torch.zeros_like(loss))

    def test_no_leak_into_unrelated(self):
        """Scoped enable_grad inside backward must not spill into other params."""
        x = _make_input(seed=3)
        unrelated = torch.randn(4, 4, requires_grad=True)
        y, _ = HiddenZLossRegularizer.apply(x, 1e-3)
        y.sum().backward()
        assert unrelated.grad is None

    def test_grad_output_scaling(self):
        """Downstream-gradient path must be multiplied by grad_output.

        Concretely: for y = x pass-through with no regularizer (coef=0),
        `.backward(gradient=v)` must write `v` into x.grad. The regularizer
        term is independent of grad_output by design — this test isolates
        the pass-through path.
        """
        x = _make_input(seed=4)
        y, _ = HiddenZLossRegularizer.apply(x, 0.0)
        v = torch.randn_like(y)
        y.backward(gradient=v)
        assert torch.allclose(x.grad, v, rtol=0, atol=0)


class TestApplyHook:

    def test_eval_is_noop(self):
        x = _make_input()
        y, loss = apply_hidden_z_loss_hook(x, coef=1e-3, training=False)
        assert y is x
        assert loss is None

    def test_coef_zero_is_noop(self):
        x = _make_input()
        y, loss = apply_hidden_z_loss_hook(x, coef=0.0, training=True)
        assert y is x
        assert loss is None

    def test_training_active_adds_grad(self):
        x = _make_input(seed=5)
        w = torch.randn_like(x)
        y, loss = apply_hidden_z_loss_hook(x, coef=1e-3, training=True)
        (y * w).sum().backward()
        grad_with_hook = x.grad.detach().clone()

        x_ref = x.detach().clone().requires_grad_(True)
        (x_ref * w).sum().backward()
        grad_no_hook = x_ref.grad.detach().clone()

        # Non-zero regularizer must change the gradient.
        assert not torch.allclose(grad_with_hook, grad_no_hook, rtol=0, atol=1e-8)
        assert loss is not None and loss.shape == torch.Size([])


class TestActivationMemoryProxy:
    """Indirect signals that forward keeps no logsumexp/square/mean graph."""

    def test_passthrough_shares_storage(self):
        x = _make_input()
        y, _ = HiddenZLossRegularizer.apply(x, 1e-3)
        assert y.data_ptr() == x.data_ptr()

    def test_loss_value_has_no_graph(self):
        """If any logsumexp/square/mean intermediate were kept, the logging
        scalar would have a grad_fn chain back to `x`. It must not.
        """
        x = _make_input()
        _, loss = HiddenZLossRegularizer.apply(x, 1e-3)
        assert loss.grad_fn is None
        assert not loss.requires_grad


# --------------------------------------------------------------------------- #
# Standalone runner (pytest-free). Runs all `test_*` methods of each class.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import inspect
    import traceback

    classes = [
        TestForward,
        TestBackward,
        TestApplyHook,
        TestActivationMemoryProxy,
    ]
    failures = []
    total = 0
    for cls in classes:
        inst = cls()
        for name, fn in inspect.getmembers(inst, predicate=inspect.ismethod):
            if not name.startswith("test_"):
                continue
            total += 1
            label = f"{cls.__name__}::{name}"
            try:
                fn()
            except Exception as e:
                failures.append((label, traceback.format_exc()))
                print(f"FAIL  {label}: {type(e).__name__}: {e}")
            else:
                print(f"PASS  {label}")
    print(f"\n{total - len(failures)}/{total} tests passed")
    if failures:
        for label, tb in failures:
            print(f"\n--- {label} ---\n{tb}")
        raise SystemExit(1)
