"""CPU checks for strict backend selection; no kernel or numerical GPU claims."""
import importlib
from pathlib import Path

import pytest
import torch

from torchtitan.experiments.flash_qla import backend


@pytest.fixture
def registry():
    backend._stage_sources()
    from fla.ops.gated_delta_rule.backends import gdr_registry
    from torchtitan.models.common.attention.linear.fla_custom_ops import _DIRECT_VJP

    saved = dict(gdr_registry._backends)
    saved_vjp = dict(_DIRECT_VJP)
    try:
        yield gdr_registry
    finally:
        gdr_registry._backends = saved
        gdr_registry._update_active_backend()
        _DIRECT_VJP.clear()
        _DIRECT_VJP.update(saved_vjp)


def inputs():
    q = torch.empty(1, 2, 16, 128, dtype=torch.bfloat16)
    v = torch.empty(1, 2, 32, 128, dtype=torch.bfloat16)
    g = torch.empty(1, 2, 32, dtype=torch.bfloat16)
    p = torch.empty(32, dtype=torch.float32)
    return q, q.clone(), v, g, g.clone(), p, p.clone()


def test_import_does_not_register(registry):
    saved = dict(registry._backends)
    importlib.reload(backend)
    assert registry._backends == saved


def test_install_fla_is_explicit_priority_zero(registry, monkeypatch):
    monkeypatch.delenv("FLA_DISABLE_BACKEND_DISPATCH", raising=False)
    identity = backend.install_backend("fla")
    selected = registry.get_active()
    assert selected.backend_type == "experimental_flash_qla_gate"
    assert selected.priority == 0
    assert identity["variant"] == "fla"
    assert len(identity["task_head"]) == 40
    expected = Path(backend.__file__).resolve().parents[3] / "contrib/flash-linear-attention"
    assert Path(identity["fla"]["path"]).is_relative_to(expected)
    assert len(identity["backend"]["sha256"]) == 64


def test_disabled_dispatch_fails_before_registration(registry, monkeypatch):
    saved = dict(registry._backends)
    monkeypatch.setenv("FLA_DISABLE_BACKEND_DISPATCH", "1")
    with pytest.raises(RuntimeError, match="FLA_DISABLE_BACKEND_DISPATCH"):
        backend.install_backend("fla")
    assert saved == registry._backends


def test_import_time_disabled_dispatch_fails(registry, monkeypatch):
    import fla.ops.backends as dispatch

    monkeypatch.delenv("FLA_DISABLE_BACKEND_DISPATCH", raising=False)
    monkeypatch.setattr(dispatch, "_DISPATCH_DISABLED", True)
    with pytest.raises(RuntimeError, match="disabled when imported"):
        backend.install_backend("fla")


def test_direct_backward_requires_external_gate(registry, monkeypatch):
    monkeypatch.setenv("GIFT_FLASH_QLA_DIRECT_BACKWARD", "true")
    with pytest.raises(ValueError, match="GIFT_FLASH_QLA_DIRECT_BACKWARD must be 0 or 1"):
        backend.install_backend("external_gate")
    monkeypatch.setenv("GIFT_FLASH_QLA_DIRECT_BACKWARD", "1")
    with pytest.raises(ValueError, match="requires external_gate"):
        backend.install_backend("fused_gate")


def test_direct_external_gate_vjp_orchestration(registry, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from fla.ops.gated_delta_rule import gate
    from torchtitan.models.common.attention.linear.fla_custom_ops import fla_vjp

    q, k, v, raw, beta, decay, bias = (
        torch.empty(*tensor.shape, 2, dtype=tensor.dtype)[..., 0] for tensor in inputs()
    )
    grad_o = torch.empty(*v.shape, 2, dtype=v.dtype)[..., 0]
    cu = torch.tensor([0, -1, 2, -1], dtype=torch.int32)[::2]
    gate_values = torch.zeros_like(raw, dtype=torch.float32)
    cumulative = torch.ones_like(gate_values)
    kkt = torch.empty(0)
    natural_dg = torch.full_like(gate_values, 2.0)
    calls = []

    def gate_forward(**kw):
        assert all(kw[name].is_contiguous() for name in ("g", "A_log", "dt_bias"))
        assert kw["output_dtype"] == torch.float32
        calls.append("gate")
        return gate_values

    def cumsum(**kw):
        assert kw["g"] is gate_values and kw["chunk_size"] == 64
        assert kw["cu_seqlens"].is_contiguous() and torch.equal(kw["cu_seqlens"], cu)
        calls.append("cumsum")
        return cumulative

    def solve(**kw):
        assert kw["chunk_size"] == 64 and kw["k"].is_contiguous() and kw["b"].is_contiguous()
        calls.append("kkt")
        return kkt

    def scan_backward(**kw):
        assert kw["g"] is cumulative and kw["A"] is kkt and not kw["auto_cp"]
        assert kw["scale"] == 128 ** -0.5
        assert all(kw[name].is_contiguous() for name in ("q", "k", "v", "beta", "do"))
        calls.append("scan_backward")
        return tuple(torch.zeros_like(t, dtype=torch.float32) for t in (q, k, v, beta)) + (natural_dg, None)

    def gate_backward(**kw):
        assert kw["dyg"] is natural_dg
        calls.append("gate_backward")
        return torch.ones_like(raw), torch.ones_like(decay), torch.ones_like(bias)

    root = Path(backend.__file__).resolve().parents[3]
    qla = SimpleNamespace(
        __file__=str(root / "contrib/flash-qla/flash_qla/ops/gated_delta_rule/chunk/__init__.py"),
        __name__="flash_qla.ops.gated_delta_rule.chunk",
        CHUNK_SIZE=64, chunk_local_cumsum=cumsum, kkt_solve=solve,
        chunk_gated_delta_rule_bwd=scan_backward,
    )
    monkeypatch.setattr(backend, "_qla", lambda: qla)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(gate, "gdn_gate_fwd", gate_forward)
    monkeypatch.setattr(gate, "gdn_gate_bwd", gate_backward)
    monkeypatch.setenv("GIFT_FLASH_QLA_DIRECT_BACKWARD", "1")
    assert backend.install_backend("external_gate")["direct_backward"]
    leaves = [q, k, v, raw, beta, decay, bias]
    output = fla_vjp("gdn_chunk", leaves, [grad_o], [cu], [], "")
    assert calls == ["gate", "cumsum", "kkt", "scan_backward", "gate_backward"]
    for actual, leaf, value in zip(output, leaves, (0, 0, 0, 1, 0, 1, 1), strict=True):
        assert actual.dtype == leaf.dtype and torch.equal(actual, torch.full_like(leaf, value))


@pytest.mark.parametrize("variant", ["fla", "external_gate", "fused_gate"])
def test_cpu_inputs_fail_without_loading_qla(variant, monkeypatch):
    def forbidden():
        raise AssertionError("Kernel import reached for unsupported CPU inputs")

    monkeypatch.setattr(backend, "_qla", forbidden)
    with pytest.raises(ValueError, match="same CUDA device"):
        backend.run_gdn(variant, *inputs())


def test_shape_and_dtype_guards():
    q, k, v, g, b, a, dt = inputs()
    with pytest.raises(ValueError, match="16,128"):
        backend.run_gdn("fla", q[:, :, :8], k, v, g, b, a, dt)
    with pytest.raises(ValueError, match="BF16 q/k/v"):
        backend.run_gdn("fla", q.float(), k, v, g, b, a, dt)
    with pytest.raises(ValueError, match="BF16 or FP32"):
        backend.run_gdn("fla", q, k, v, g, b, a.double(), dt)
    with pytest.raises(ValueError, match="Unknown GDN variant"):
        backend.run_gdn("unknown", q, k, v, g, b, a, dt)


@pytest.mark.parametrize("extra", [
    {"output_final_state": True}, {"initial_state": object()},
    {"use_qk_l2norm_in_kernel": True}, {"cp_context": object()},
    {"use_beta_sigmoid_in_kernel": True}, {"allow_neg_eigval": True},
    {"state_v_first": True}, {"unrecognized_flag": True},
    {"use_gate_in_kernel": False},
])
def test_selected_dispatch_never_falls_back(registry, monkeypatch, extra):
    monkeypatch.delenv("FLA_DISABLE_BACKEND_DISPATCH", raising=False)
    backend.install_backend("fla")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    q, k, v, g, b, a, dt = inputs()
    options = {"use_gate_in_kernel": True, "A_log": a, "dt_bias": dt}
    options.update(extra)
    # Exercise the real FLA dispatcher: a swallowed verifier error would reach fallback.
    with pytest.raises(ValueError, match="Explicit GDN backend requires"):
        chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=b, **options)


def test_stale_source_is_rejected():
    with pytest.raises(RuntimeError, match="Wrong staged source"):
        backend._source(backend, Path("/unrelated-checkout"))


@pytest.mark.parametrize("parameter_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_fused_autograd_orchestration_cpu(monkeypatch, parameter_dtype, strided):
    """Run the real custom Function with CPU doubles only at GPU kernel boundaries."""
    from types import SimpleNamespace
    import fla.ops.gated_delta_rule.gate as gate

    q, k, v, raw, beta, a, dt = inputs()
    a, dt = a.to(parameter_dtype), dt.to(parameter_dtype)
    if strided:
        q, k, v, raw, beta, a, dt = (
            torch.empty(*t.shape, 2, dtype=t.dtype)[..., 0]
            for t in (q, k, v, raw, beta, a, dt)
        )
    values = (q, k, v, raw, beta, a, dt)
    for value in values:
        value.requires_grad_(True)
    calls = []
    cp = SimpleNamespace(cu_seqlens=None, seq_map_c2r=None)
    cumulative = torch.zeros_like(raw, dtype=torch.float32)
    natural_dg = torch.ones_like(raw, dtype=torch.float32)

    def cumulative_gate(**kw):
        assert all(kw[name].is_contiguous() for name in ("g", "A_log", "dt_bias"))
        assert kw["g"].shape == raw.shape and kw["A_log"].dtype == parameter_dtype
        assert kw["chunk_size"] == 64 and kw["scale"] is None
        assert kw["output_dtype"] == torch.float32
        calls.append("gate_cumsum")
        return cumulative

    def preprocess(context, **kw):
        assert context is cp and kw["g"] is cumulative
        assert not kw["enable_fwd_cp_cache"]
        calls.append("cp_noop")
        return None, None

    def forward(**kw):
        assert kw["g"] is cumulative and kw["initial_state"] is None
        assert all(kw[name].is_contiguous() for name in ("q", "k", "v", "b"))
        assert not kw["output_final_state"] and not kw["output_h"]
        calls.append("qla_forward")
        return v.clone(), None, None

    def backward(**kw):
        assert kw["cp_context"] is cp and not kw["auto_cp"]
        assert kw["g"] is cumulative and kw["do"].is_contiguous()
        calls.append("qla_backward")
        return q.float().zero_(), k.float().zero_(), v.float().zero_(), beta.float().zero_(), natural_dg, None

    def gate_backward(**kw):
        assert kw["dyg"] is natural_dg  # No second reverse cumsum or log2 scaling.
        calls.append("gate_backward")
        return torch.ones_like(raw), torch.ones_like(a), torch.ones_like(dt)

    qla = SimpleNamespace(
        CHUNK_SIZE=64, _auto_intra_cp_context=lambda *args: cp,
        kkt_solve=lambda **kw: torch.empty(0), cp_preprocess_fwd=preprocess,
        fused_gdr_fwd=forward, chunk_gated_delta_rule_bwd=backward,
    )
    monkeypatch.setattr(backend, "_qla", lambda: qla)
    monkeypatch.setattr(gate, "gdn_gate_chunk_cumsum", cumulative_gate)
    monkeypatch.setattr(gate, "gdn_gate_bwd", gate_backward)
    output = backend._FusedGate.apply(*values, None, 128 ** -0.5)
    output.float().sum().backward()
    assert calls == ["gate_cumsum", "cp_noop", "qla_forward", "qla_backward", "gate_backward"]
    for value in values:
        assert value.grad is not None and value.grad.dtype == value.dtype


def test_install_rejects_stale_imported_fla(registry, monkeypatch):
    from types import SimpleNamespace

    original_import = importlib.import_module
    stale = SimpleNamespace(__name__="fla.ops.gated_delta_rule.chunk",
                            __file__="/tmp/stale-checkout/fla/ops/gated_delta_rule/chunk.py")

    def load(name):
        return stale if name == stale.__name__ else original_import(name)

    monkeypatch.setattr(backend.importlib, "import_module", load)
    with pytest.raises(RuntimeError, match="Wrong staged source"):
        backend.install_backend("fla")


def test_qla_module_provenance_is_cached(monkeypatch):
    from types import SimpleNamespace

    backend._qla.cache_clear()
    module = SimpleNamespace()
    checked = []
    monkeypatch.setattr(backend.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(backend, "_source", lambda *args: checked.append(args))
    try:
        assert backend._qla() is module
        assert backend._qla() is module
        assert len(checked) == 1
    finally:
        backend._qla.cache_clear()
