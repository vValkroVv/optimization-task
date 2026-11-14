"""CPU regression through the pinned FLA dispatch signature and real hook."""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import torch


def test_capture_flattens_actual_fla_keyword_arguments(tmp_path, monkeypatch):
    # Cache ownership must be established before importing backend/FLA modules.
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "tilelang-cache"))
    monkeypatch.setenv("TILELANG_TMP_DIR", str(tmp_path / "tilelang-tmp"))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("FLA_DISABLE_BACKEND_DISPATCH", raising=False)
    from fla.ops.backends import BackendRegistry
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from torchtitan.experiments.flash_qla import backend

    # The public pinned dispatch is real; only the native GPU implementation is
    # replaced. Borrowed registry state is restored by pytest's monkeypatch.
    monkeypatch.setattr(BackendRegistry, "_registries", {})
    monkeypatch.setattr(BackendRegistry, "_initialized", {"gated_delta_rule"})
    path = Path(__file__).resolve().parents[1] / "capture.py"
    spec = importlib.util.spec_from_file_location("capture_regression", path)
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    inputs = [torch.randn(1, 3, 2, 8)[..., ::2].detach().requires_grad_() for _ in range(3)]
    inputs += [torch.randn(1, 3, 2, requires_grad=True) for _ in range(2)]
    inputs += [torch.randn(2, requires_grad=True) for _ in range(2)]
    q, k, v, raw_g, beta, A_log, dt_bias = inputs
    kwargs = dict(q=q, k=k, v=v, g=raw_g, beta=beta, A_log=A_log,
                  dt_bias=dt_bias, use_gate_in_kernel=True,
                  use_qk_l2norm_in_kernel=False, output_final_state=False,
                  cu_seqlens=None)
    signature = inspect.signature(chunk_gated_delta_rule)
    bound = signature.bind(**kwargs)
    bound.apply_defaults()
    keyword_name = next(name for name, p in signature.parameters.items()
                        if p.kind == inspect.Parameter.VAR_KEYWORD)
    # This is the original broken binding: semantic flags and parameters are
    # nested, so its raw-gate guard would reject this valid GIFT invocation.
    assert not bound.arguments.get("use_gate_in_kernel", False)
    assert "A_log" not in bound.arguments
    assert bound.arguments[keyword_name]["use_gate_in_kernel"] is True
    calls = []

    def native_double(variant, *actual, cu_seqlens=None, scale=None):
        assert variant == "fla"
        assert all(a is b for a, b in zip(actual, inputs))
        assert cu_seqlens is None and scale is None
        calls.append(actual)
        q_, k_, v_, g_, beta_, A_, bias_ = actual
        return q_ + k_ + v_ + (g_ + beta_ + A_ + bias_).unsqueeze(-1), None

    monkeypatch.setattr(backend, "run_gdn", native_double)
    identity = capture.install_capture(tmp_path / "fixtures", max_fixtures=1)
    output, state = chunk_gated_delta_rule(**kwargs)
    assert state is None and len(calls) == 1
    upstream = torch.randn_like(output)
    gradients = torch.autograd.grad(output, inputs, upstream)
    fixture_path = tmp_path / "fixtures/fixture-01.pt"
    fixture = torch.load(fixture_path, weights_only=True)
    manifest = json.loads(fixture_path.with_suffix(".json").read_text())
    assert identity["counters"] == {"calls": 1, "vjp_calls": 1, "saved": 1}
    assert manifest["fixture_sha256"] == capture.sha256(fixture_path)
    for name, original in zip(capture.NAMES, inputs):
        torch.testing.assert_close(fixture[name], original.detach(), rtol=0, atol=0)
        assert fixture[name].stride() == original.stride()
    torch.testing.assert_close(fixture["grad_out"], upstream, rtol=0, atol=0)
    torch.testing.assert_close(fixture["output"], output.detach(), rtol=0, atol=0)
    for gradient in gradients[:3]:
        torch.testing.assert_close(gradient, upstream, rtol=0, atol=0)
    for gradient in gradients[3:5]:
        torch.testing.assert_close(gradient, upstream.sum(-1), rtol=0, atol=0)
    for gradient in gradients[5:]:
        torch.testing.assert_close(gradient, upstream.sum(-1).sum((0, 1)))
