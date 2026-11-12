"""CUDA-free integration coverage for per-block Inductor graph exclusions."""

import pytest
import torch

_CUDA_API_NAMES = ("is_available", "device_count", "init", "_lazy_init")
_CUDA_API_ORIGINALS = {name: getattr(torch.cuda, name) for name in _CUDA_API_NAMES}


def _forbid_cuda(*args, **kwargs):
    raise AssertionError("CUDA discovery or initialization is forbidden in this test")


# Guard before importing the compiler/model path or discovering the model tree.
try:
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.init = _forbid_cuda
    torch.cuda._lazy_init = _forbid_cuda

    from torchtitan.config import ActivationCheckpointConfig, CompileConfig  # noqa: E402
    from torchtitan.distributed.activation_checkpoint import apply_ac  # noqa: E402
    from torchtitan.distributed.compile import apply_compile  # noqa: E402
    from torchtitan.protocols.blocks import Block, peel, walk_segments  # noqa: E402
    from torchtitan.protocols.module import Module, ModuleDict  # noqa: E402
    from tests.unit_tests.torch_free import REPO_ROOT, run_blocked  # noqa: E402
finally:
    for _name, _original in _CUDA_API_ORIGINALS.items():
        setattr(torch.cuda, _name, _original)


@pytest.fixture(autouse=True)
def _guard_cuda_during_each_test(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "init", _forbid_cuda)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _forbid_cuda)


PRESET = REPO_ROOT / "configs" / "experiments" / "gift_moe_3b_hybridep_fp8_compile.yaml"
EXCLUDED_FQNS = [f"layers.{index}" for index in range(1, 18)]


class _TinyBlock(Block):
    def forward(self, x):
        return x


class _TinyModel(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleDict({str(index): _TinyBlock() for index in range(2)})


def _load_meta_campaign_model(monkeypatch):
    from torchtitan.config.build import load_model_spec
    from torchtitan.models.common.moe.dispatch.hybridep import (
        HybridEPTokenDispatcher,
    )

    # The test checks the production module type and nesting without importing
    # HybridEP's CUDA runtime or creating its dispatcher kernels.
    monkeypatch.setattr(
        HybridEPTokenDispatcher, "__init__", lambda self, config: None
    )
    model_spec = load_model_spec(str(PRESET))
    with torch.device("meta"):
        return model_spec.model.build()


def test_compile_config_torch_free_default_and_resolved_round_trip():
    selected = ["layers.1", "layers.17"]
    code = f"""
from torchtitan.config import CompileConfig, load_document
from torchtitan.config.build import prepare_document
from torchtitan.config.configurable import to_plain
from torchtitan.config.save import check_round_trip, get_typed_document

first = CompileConfig()
second = CompileConfig()
assert first.cudagraph_exclude_blocks == []
assert second.cudagraph_exclude_blocks == []
first.cudagraph_exclude_blocks.append("layers.1")
assert second.cudagraph_exclude_blocks == []

document, _ = load_document(
    {str(PRESET)!r},
    [("compile.cudagraph_exclude_blocks", "{','.join(selected)}")],
)
prepared = prepare_document(document)
assert prepared.trainer_config.compile.cudagraph_exclude_blocks == {selected!r}
resolved = get_typed_document(prepared.trainer_config, prepared.model_section)
assert resolved["compile"]["cudagraph_exclude_blocks"] == {selected!r}
check_round_trip(
    prepared.trainer_config,
    document.data,
    resolved,
    "cudagraph-exclusion-config-test",
    model_section=prepared.model_section,
)
assert to_plain(prepared.trainer_config.compile)["cudagraph_exclude_blocks"] == {selected!r}
print("torch-free defaults and resolved config round-trip passed")
"""
    result = run_blocked(code)
    assert result.stdout.strip() == "torch-free defaults and resolved config round-trip passed"


def test_empty_exclusions_keep_the_canonical_compile_kwargs(monkeypatch):
    calls = []

    def fake_compile(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = _TinyModel()

    apply_compile(model, CompileConfig(enable=True))

    assert len(calls) == 2
    assert all(args == () for _, args, _ in calls)
    assert all(
        kwargs == {"backend": "inductor", "fullgraph": True}
        for _, _, kwargs in calls
    )


def test_campaign_selects_17_moe_blocks_through_selective_ac(monkeypatch, caplog):
    from torchtitan.models.common.moe.dispatch.hybridep import (
        HybridEPTokenDispatcher,
    )

    model = _load_meta_campaign_model(monkeypatch)
    before_ac = [segment for segment in walk_segments(model) if segment.is_block]
    assert [segment.fqn for segment in before_ac] == [
        f"layers.{index}" for index in range(18)
    ]
    assert not peel(before_ac[0].module).moe_enabled
    for segment in before_ac[1:]:
        block = peel(segment.module)
        assert block.moe_enabled
        assert isinstance(
            block.moe.experts.token_dispatcher, HybridEPTokenDispatcher
        )

    apply_ac(
        model,
        ActivationCheckpointConfig(mode="selective"),
        model_compile_enabled=True,
    )
    segments = [segment for segment in walk_segments(model) if segment.is_block]
    assert [segment.fqn for segment in segments] == [
        f"layers.{index}" for index in range(18)
    ]
    assert all(
        getattr(segment.module, "_checkpoint_wrapped_module", None) is not None
        for segment in segments
    )
    assert all(isinstance(peel(segment.module), Block) for segment in segments)
    assert all(
        isinstance(
            peel(segment.module).moe.experts.token_dispatcher,
            HybridEPTokenDispatcher,
        )
        for segment in segments[1:]
    )

    calls = []

    def fake_compile(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    config = CompileConfig(
        enable=True,
        components=["model", "loss"],
        cudagraph_exclude_blocks=EXCLUDED_FQNS.copy(),
    )
    with caplog.at_level("INFO"):
        apply_compile(model, config)

    fqn_by_module_id = {id(segment.module): segment.fqn for segment in segments}
    kwargs_by_fqn = {}
    for fn, args, kwargs in calls:
        assert args == ()
        assert fn.__self__ is not None
        kwargs_by_fqn[fqn_by_module_id[id(fn.__self__)]] = kwargs

    assert len(kwargs_by_fqn) == 18
    assert {
        fqn for fqn, kwargs in kwargs_by_fqn.items() if "options" in kwargs
    } == set(EXCLUDED_FQNS)
    for fqn, kwargs in kwargs_by_fqn.items():
        expected = {"backend": "inductor", "fullgraph": True}
        if fqn in EXCLUDED_FQNS:
            expected["options"] = {"triton.cudagraphs": False}
        assert kwargs == expected, f"{fqn}: {kwargs}"

    graph_logs = [
        record.getMessage()
        for record in caplog.records
        if "Disabled Inductor CUDA Graphs" in record.getMessage()
    ]
    assert len(graph_logs) == 1
    assert "17 block(s)" in graph_logs[0]
    assert "triton.cudagraphs" in graph_logs[0]
    assert all(fqn in graph_logs[0] for fqn in EXCLUDED_FQNS)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"enable": False}, "compile.enable=true"),
        ({"components": ["loss"]}, "'model' in compile.components"),
        ({"backend": "aot_eager"}, "compile.backend='inductor'"),
    ],
)
def test_config_rejects_exclusions_when_model_compile_would_ignore_them(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CompileConfig(
            enable=kwargs.get("enable", True),
            components=kwargs.get("components", ["model"]),
            backend=kwargs.get("backend", "inductor"),
            cudagraph_exclude_blocks=["layers.1"],
        )


def test_config_rejects_duplicate_exclusion_names():
    with pytest.raises(ValueError, match="duplicate block FQNs"):
        CompileConfig(
            enable=True,
            cudagraph_exclude_blocks=["layers.1", "layers.1"],
        )


@pytest.mark.parametrize(
    ("names", "backend", "enable", "components", "message"),
    [
        (["layers.1", "layers.1"], "inductor", True, ["model"], "duplicate"),
        (["layers.99"], "inductor", True, ["model"], "unmatched"),
        (["layers.1"], "aot_eager", True, ["model"], "compile.backend"),
        (["layers.1"], "inductor", False, ["model"], "compile.enable"),
        (["layers.1"], "inductor", True, ["loss"], "compile.components"),
    ],
)
def test_apply_compile_revalidates_mutated_exclusion_config(
    names, backend, enable, components, message
):
    model = _TinyModel()
    config = CompileConfig(enable=True, cudagraph_exclude_blocks=["layers.0"])
    config.cudagraph_exclude_blocks = names
    config.backend = backend
    config.enable = enable
    config.components = components

    with pytest.raises(ValueError, match=message):
        apply_compile(model, config)


def test_loss_compile_path_does_not_receive_graph_exclusion_options(monkeypatch):
    from torchtitan.components.loss import BaseLoss

    calls = []

    def fake_compile(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    class LossStub:
        fn = object()

    config = CompileConfig(
        enable=True,
        components=["model", "loss"],
        cudagraph_exclude_blocks=EXCLUDED_FQNS.copy(),
    )
    BaseLoss._maybe_compile(LossStub(), config)

    assert len(calls) == 1
    assert calls[0][1] == ()
    assert calls[0][2] == {"backend": "inductor"}
