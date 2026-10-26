"""Invariants of the structural traversal that AC, compile, FSDP, CP and PP share.

A container added to a model cannot be picked up by one consumer and missed by
another, and the config-side and runtime-side walks cannot drift apart.
"""

import ast
import dataclasses
import functools
import logging
import pathlib

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from tests.unit_tests.config_helpers import load_model_spec
from torchtitan.distributed.fsdp import _build_shard_units
from torchtitan.models.common.moe import MoE
from torchtitan.protocols.blocks import Block, walk_segments


def _spec(preset, **model_overrides):
    return load_model_spec(preset, model_overrides)


# gift_moe has no dedicated MTP block: its heads take the MoE block.
_GIFT_MOE_MTP = {"mtp": {"layers": [{"block": "mla_moe", "repeat": 2}]}}

# One representative per structural shape, not per preset.
CASES = {
    "llama3": lambda: _spec("llama3/debugmodel.yaml"),
    "qwen3_tied": lambda: _spec("qwen3/debugmodel.yaml"),
    "deepseek_v3": lambda: _spec("deepseek_v3/debugmodel.yaml"),
    "gift_moe_mtp": lambda: _spec("gift_moe/debugmodel.yaml", config=_GIFT_MOE_MTP),
    "gift_3_5_mtp": lambda: _spec(
        "gift_moe/gift_3_5/debugmodel.yaml",
        config={"mtp": {"layers": [{"block": "mtp", "repeat": 2}]}},
    ),
    "qwen3_vl": lambda: _spec("qwen3_vl/debugmodel.yaml"),
    "gpt_oss": lambda: _spec("gpt_oss/debugmodel.yaml"),
    "flux": lambda: _spec("flux/debug.yaml"),
}


@pytest.fixture(scope="module")
def built():
    models = {}
    for name, make in CASES.items():
        spec = make()
        with torch.device("meta"):
            models[name] = (spec.model, spec.model.build())
    return models


@pytest.mark.parametrize("case", list(CASES))
def test_segment_fqns_resolve(built, case):
    """Every FQN is a real path in the module tree, not a literal built by hand."""
    _, model = built[case]
    for segment in model.named_segments():
        assert model.get_submodule(segment.fqn) is segment.module, segment.fqn


@pytest.mark.parametrize("case", list(CASES))
def test_blocks_are_exactly_the_outermost_blocks(built, case):
    """Nothing missed, nothing counted twice, never a block inside a block.

    ``MTPBlock`` contains a ``ResidualTransformerBlock``, but the wrap unit is
    the head.
    """
    _, model = built[case]
    walked = {s.fqn for s in model.named_segments() if s.is_block}
    all_blocks = {f for f, m in model.named_modules() if isinstance(m, Block)}
    outermost = {
        f
        for f in all_blocks
        if not any(f != other and f.startswith(f"{other}.") for other in all_blocks)
    }
    assert walked == outermost
    assert all_blocks, "the case has no Block at all; it tests nothing"


@pytest.mark.parametrize("case", list(CASES))
def test_segment_order_matches_forward(built, case):
    """Segment order is the forward order, checked against real forward hooks.

    Registration order is only the default, so this is what makes the ordering a
    contract rather than a coincidence (``qwen3_vl`` registers its tower last and
    runs it first).
    """
    if case in NO_FORWARD_ORDER_COVERAGE:
        # Passing pixel_values=None would run the text path only and pass without
        # ever entering the tower -- worse than skipping, because it would read as
        # covered. Real modality inputs are the fix.
        pytest.skip(f"{case}: forward order is NOT covered (needs modality inputs)")
    if case in REAL_INPUT_CASES:
        model, run_forward = REAL_INPUT_CASES[case]()
    else:
        _, model = built[case]
        run_forward = functools.partial(_meta_text_forward, model)
    segments = list(model.named_segments())

    fired: list[str] = []
    by_module = {id(s.module): s.fqn for s in segments}
    handles = [
        m.register_forward_pre_hook(
            lambda mod, _args, _fqn=by_module[id(m)]: fired.append(_fqn)
        )
        for m in (s.module for s in segments)
    ]
    # No except around the forward: a model that cannot forward is a defect to fix,
    # or an entry in NO_FORWARD_ORDER_COVERAGE with a reason. A blanket
    # skip-on-exception hid an unsupported-autocast bug in the MoE router for three
    # of these cases, and a skip nobody reads looks exactly like a pass.
    try:
        run_forward()
    finally:
        for handle in handles:
            handle.remove()

    order = [s.fqn for s in segments if s.fqn in set(fired)]
    first_seen = []
    for fqn in fired:
        if fqn not in first_seen:
            first_seen.append(fqn)
    assert order == first_seen


# Parameters owned directly by a module the walk descends *through* belong to no
# segment, so the root ``fully_shard(model)`` takes them. Listed per case so a new
# one fails this test instead of quietly landing in the root group.
BARE_PARAMS = {"qwen3_vl": {"vision_encoder.pos_embed"}}


# Cases whose forward order is NOT checked against real hooks: their forward needs
# conditioning a token-only call cannot supply. Listed, not inlined, so the gap is
# visible and a model added here needs a reason.
# TODO: build flux's modality inputs and empty this. gpt_oss needs
# ``get_attention_masks`` to take a device: half its blocks index the mask dict, and
# ``create_block_mask`` defaults to cuda, so no CPU call reaches them.
NO_FORWARD_ORDER_COVERAGE = {"flux", "gpt_oss"}


def _meta_text_forward(model):
    """Tokens alone fire every segment of a text model."""
    with torch.device("meta"):
        tokens = torch.zeros(1, 8, dtype=torch.long)
        model(tokens, positions=torch.zeros(1, 8, dtype=torch.int32))


def _qwen3_vl_with_one_image():
    """A CPU qwen3_vl and a call carrying an image through its tower.

    Meta tensors cannot serve: placing vision tokens reads tensor *values*
    (``torch.where(...).tolist()``). Only the vocabulary is narrowed, to keep the
    embedding and head out of a unit test's memory; the structure ships as-is.
    """
    config = _spec("qwen3_vl/debugmodel.yaml", config={"vocab_size": 512}).model
    model = config.build()
    model.init_weights(buffer_device=torch.device("cpu"))

    # One 4x4-patch image: 2x2 merging leaves the 4 image tokens in the text.
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    patch_dim = config.vision_encoder.patch_embed.proj.in_features
    image_id, video_id = 100, 101
    tokens = torch.tensor([[7, 8, image_id, image_id, image_id, image_id, 9, 10]])

    def run_forward():
        # FlexAttention has no CPU backward; the order needs no gradients.
        with torch.no_grad():
            model(
                tokens,
                pixel_values=torch.randn(1, int(grid_thw.prod()), patch_dim),
                grid_thw=grid_thw,
                special_tokens={"image_id": image_id, "video_id": video_id},
            )

    return model, run_forward


# Cases built for real rather than on meta, with their own forward call.
REAL_INPUT_CASES = {"qwen3_vl": _qwen3_vl_with_one_image}


@pytest.mark.parametrize("case", list(CASES))
def test_shard_units_cover_every_parameter_once(built, case):
    """The FSDP grouping policy partitions the parameters: no gap, no overlap."""
    _, model = built[case]
    units = _build_shard_units(
        list(model.named_segments()),
        reshard_after_forward=True,
        output_reshard_after_forward=False,
    )
    covered = {
        id(param)
        for unit in units
        for module in unit.modules
        for param in module.parameters()
    }
    by_id = {id(param): fqn for fqn, param in model.named_parameters()}
    leftover = {by_id[pid] for pid in set(by_id) - covered}
    assert leftover == BARE_PARAMS.get(case, set())


def test_unknown_coalesce_group_warns(built, caplog):
    """A stale group name falls back to one group per block, and that fallback is
    what the option exists to avoid -- so it must not be silent."""
    _, model = built["qwen3_vl"]
    segments = list(model.named_segments())
    units = dict(
        reshard_after_forward=True,
        output_reshard_after_forward=False,
    )
    with caplog.at_level(logging.WARNING):
        good = _build_shard_units(
            segments, **units, coalesce_groups=frozenset({"vision_encoder.layers"})
        )
        assert not caplog.records, "a real group name must not warn"
        stale = _build_shard_units(
            segments,
            **units,
            coalesce_groups=frozenset({"model.vision_encoder.layers"}),
        )
    assert len(stale) > len(good), "the stale name should not have coalesced anything"
    assert any(
        "model.vision_encoder.layers" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("case", list(CASES))
def test_tied_parameters_land_in_one_group(built, case):
    """Weight tying is detected by parameter identity, not by a model flag."""
    _, model = built[case]
    units = _build_shard_units(
        list(model.named_segments()),
        reshard_after_forward=True,
        output_reshard_after_forward=False,
    )
    owner: dict[int, int] = {}
    for index, unit in enumerate(units):
        for module in unit.modules:
            for param in module.parameters():
                assert owner.setdefault(id(param), index) == index, (
                    "a shared parameter is claimed by two FSDP groups"
                )


@pytest.mark.parametrize("case", list(CASES))
def test_grouped_param_axes_are_the_expert_weights(built, case):
    """The declared group axes are exactly the routed-expert weights, dim 0.

    An MTP head's MoE is reached with no special case, which is what makes dense
    and MoE symmetric here.
    """
    _, model = built[case]
    declared = {id(p): axis for p, axis in model.grouped_param_axes()}
    # Keyed on the MoE, not on the expert class: an oracle asking
    # ``isinstance(m, GroupedExperts)`` would share the blind spot it exists to
    # catch -- a family with its own expert module compares an empty set to an
    # empty one and passes. ``shared_experts`` is dense, so it stays out.
    expected = {
        id(p)
        for module in model.modules()
        if isinstance(module, MoE)
        for p in module.experts.parameters()
    }
    assert set(declared) == expected
    for param, axis in model.grouped_param_axes():
        assert param.shape[0] == axis


@pytest.mark.parametrize("case", list(CASES))
def test_config_and_runtime_walks_agree(built, case):
    """``traverse(Block.Config)`` and ``named_segments()`` name the same blocks.

    A container in one walk but not the other would let FLOPs, sharding-config
    fill or PP metadata disagree with what actually gets wrapped.
    """
    config, model = built[case]
    from_config = {fqn for fqn, *_ in config.traverse(Block.Config)}
    from_runtime = {s.fqn for s in model.named_segments() if s.is_block}
    assert from_config == from_runtime


def test_pipeline_split_handles_nested_containers():
    """A stage may name a block inside a nested container, not just ``layers.N``.

    Naming heads individually must prune to exactly those heads; this is what lets
    a wrapper model (``decoder.layers.3``) or a tower be split at all.
    """
    from torchtitan.distributed.pipeline_parallel import _split_module

    spec = _spec("gift_moe/debugmodel.yaml", config=_GIFT_MOE_MTP)
    with torch.device("meta"):
        model = spec.model.build()

    # Naming both heads is equivalent to naming the container they live in.
    whole = sorted(_split_module(model, ["norm", "lm_head", "mtp"]).state_dict())
    per_head = sorted(
        _split_module(
            model, ["norm", "lm_head", "mtp.layers.0", "mtp.layers.1"]
        ).state_dict()
    )
    assert whole == per_head

    one_head = _split_module(model, ["norm", "lm_head", "mtp.layers.1"])
    kept = {k.split(".")[2] for k in one_head.state_dict() if k.startswith("mtp.")}
    assert kept == {"1"}

    # A stage that names no head keeps no MTP parameters at all.
    none = _split_module(model, ["layers.0"])
    assert not [k for k in none.state_dict() if k.startswith("mtp.")]


def test_what_a_towers_forward_calls_is_cut_onto_the_towers_stage():
    """A cut must not separate a module from the forward that calls it.

    Every merger is called by ``Qwen3VLVisionEncoder.forward``, so glued to the
    following block it would be pruned off the stage calling it: an empty
    ``ModuleList``, the wrong merger once renumbered, or ``self.merger`` None.
    """
    from torchtitan.distributed.pipeline_parallel import _partition_segments

    with torch.device("meta"):
        model = _spec("qwen3_vl/debugmodel.yaml").model.build()
    tower = model.vision_encoder
    _, body, tail, _ = _partition_segments(model)
    stage_of = {fqn: index for index, item in enumerate(body) for fqn in item}
    called = {
        f"vision_encoder.deepstack_merger_list.{merger}": layer
        for merger, layer in enumerate(tower.deepstack_visual_indices)
    }
    called["vision_encoder.merger"] = len(tower.layers) - 1
    for fqn, layer in called.items():
        assert fqn not in tail, fqn
        assert stage_of[fqn] == stage_of[f"vision_encoder.layers.{layer}"], fqn


def test_an_unpaired_deepstack_merger_is_refused():
    """The walk pairs merger ``i`` with layer ``deepstack_visual_indices[i]``.

    An unpaired one falls out of the reported structure -- and so out of AC, FSDP
    and PP -- with nothing to notice it.
    """
    tower = _spec("qwen3_vl/debugmodel.yaml").model.vision_encoder
    with pytest.raises(ValueError, match="deepstack_mergers"):
        dataclasses.replace(tower, deepstack_mergers=tower.deepstack_mergers[:-1])
    with pytest.raises(ValueError, match="name no layer"):
        dataclasses.replace(
            tower,
            deepstack_visual_indices=[
                *tower.deepstack_visual_indices[:-1],
                len(tower.layers),
            ],
        )


@pytest.mark.parametrize("case", list(CASES))
def test_output_run_segments_stay_separate_and_unresharded(built, case):
    """Output-run segments (norm, lm_head) each get their own no-reshard group.

    Separate, because under a chunked loss the head and the norm are reduced by
    different backward passes and a shared group breaks HSDP's deferred
    all-reduce; no-reshard, because backward re-gathers them immediately.
    ``is_output`` marks the head only: backward starts there.
    """
    from torchtitan.distributed.fsdp import _output_run_bounds

    _, model = built[case]
    segments = list(model.named_segments())
    units = _build_shard_units(
        segments,
        reshard_after_forward=True,
        output_reshard_after_forward=False,
    )
    bounds = _output_run_bounds(segments)
    run_fqns = [] if bounds is None else [s.fqn for s in segments[slice(*bounds)]]
    outputs = [u for u in units if u.is_output]
    assert len(outputs) == (1 if run_fqns else 0)
    for unit in units:
        # Merged groups join fqns with "+" (shared-param merges only).
        in_run = any(fqn in unit.fqn.split("+") for fqn in run_fqns)
        assert unit.reshard_after_forward is not in_run, unit.fqn
    if run_fqns:
        assert run_fqns[-1] in outputs[0].fqn.split("+")


def test_leading_stem_is_not_mistaken_for_an_output():
    """``[embedding, blocks...]`` — a first pipeline stage — keeps resharding."""
    from torchtitan.distributed.pipeline_parallel import _split_module

    with torch.device("meta"):
        model = _spec("llama3/debugmodel.yaml").model.build()
    stage = _split_module(model, ["tok_embeddings", "layers.0", "layers.1"])
    units = _build_shard_units(
        list(stage.named_segments()),
        reshard_after_forward=True,
        output_reshard_after_forward=False,
    )
    assert units and all(u.reshard_after_forward for u in units)
    assert not any(u.is_output for u in units)


def test_zero_blocks_warns_and_skips(caplog):
    """AC and compile skip a model with no repeated unit, and say so.

    Both are handed pipeline stages, and a looped schedule can cut one holding only
    the stem or the tail, so neither can refuse. The success log goes with the work,
    so an undeclared block cannot read as "applied".
    """
    from torchtitan.config import ActivationCheckpointConfig, CompileConfig
    from torchtitan.distributed.activation_checkpoint import apply_ac
    from torchtitan.distributed.compile import apply_compile
    from torchtitan.distributed.pipeline_parallel import _partition_segments
    from torchtitan.protocols.module import Module, ModuleDict

    class Foreign(nn.Module):
        """Stands in for a third-party block: has its own forward, not a Block."""

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x)

    class NoBlocks(Module):
        def __init__(self):
            super().__init__()
            self.layers = ModuleDict({str(i): Foreign() for i in range(2)})

    model = NoBlocks()
    with caplog.at_level(logging.WARNING):
        apply_ac(model, ActivationCheckpointConfig(mode="full"))
        apply_compile(model, CompileConfig(enable=True))
    assert sum("no repeated block" in r.getMessage() for r in caplog.records) == 2
    assert not any(
        isinstance(m, CheckpointWrapper) for _, m in model.layers.named_children()
    )
    # The splitter still refuses: it cannot cut what it cannot find.
    with pytest.raises(ValueError, match="no repeated block"):
        _partition_segments(model)

    # A model may declare a foreign class as its unit by overriding the walk.
    class Declaring(NoBlocks):
        def named_segments(self):
            for segment in walk_segments(self):
                yield dataclasses.replace(segment, is_block=segment.group == "layers")

    declaring = Declaring()
    apply_ac(declaring, ActivationCheckpointConfig(mode="full"))
    assert all(
        isinstance(m, CheckpointWrapper) for _, m in declaring.layers.named_children()
    )


def test_ac_and_compile_tolerate_a_block_free_stage(caplog):
    """A looped schedule cuts stages that hold only the stem or the tail.

    Not exotic: ``pipeline_parallel_layers_per_stage 1`` yields these two at any
    depth, since the stem and tail weights consume a one-block stage budget.
    """
    from torchtitan.config import ActivationCheckpointConfig, CompileConfig
    from torchtitan.distributed.activation_checkpoint import apply_ac
    from torchtitan.distributed.compile import apply_compile
    from torchtitan.distributed.pipeline_parallel import _split_module

    spec = _spec("llama3/debugmodel.yaml")
    for names in (["tok_embeddings"], ["norm", "lm_head"]):
        with torch.device("meta"):
            stage = _split_module(spec.model.build(), names)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            apply_ac(stage, ActivationCheckpointConfig(mode="full"))
            apply_compile(stage, CompileConfig(enable=True))
        warned = sum("no repeated block" in r.getMessage() for r in caplog.records)
        assert warned == 2, f"{names}: expected AC and compile to warn, got {warned}"


def test_prune_keeps_modulelist_entries_past_index_zero():
    """Pruning a ``ModuleList`` renumbers it, so recursion must run first.

    Recursing after the rebuild looked up the pre-renumbering index, found it
    absent from the stage's names, and nulled the block that was meant to stay.
    """
    from torchtitan.distributed.pipeline_parallel import _split_module

    with torch.device("meta"):
        model = _spec("flux/debug.yaml").model.build()
    for names in (["double_blocks.0"], ["double_blocks.1"], ["single_blocks.1"]):
        stage = _split_module(model, names)
        assert sum(1 for _ in stage.parameters()) > 0, names


def test_three_way_tying_keeps_every_module_in_a_group():
    """Chained parameter sharing must not drop a group's modules.

    A single pass of "point at the first owner" can send group C at group B while
    B is itself merged into A; C's modules then belong to no surviving group and
    are silently never sharded. Union-find is what prevents that.
    """
    from torchtitan.distributed.fsdp import _ShardUnit, _merge_units_sharing_params

    shared_a, shared_b = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2))

    def holder(*params):
        module = nn.Module()
        for index, param in enumerate(params):
            module.register_parameter(f"p{index}", param)
        return module

    # A shares `shared_a` with B; C shares only `shared_b` with B.
    units = [
        _ShardUnit([holder(shared_a)], "a", False, True),
        _ShardUnit([holder(shared_a, shared_b)], "b", False, True),
        _ShardUnit([holder(shared_b)], "c", False, True),
    ]
    merged = _merge_units_sharing_params(units)
    assert len(merged) == 1
    assert len(merged[0].modules) == 3


def test_peel_looks_through_the_ac_wrapper_and_nothing_else():
    """A container is not a wrapper around its entry, whatever its arity.

    Peeling one would report the container as the block, so AC would wrap a
    module whose forward never runs — and only for a one-entry container.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    from torchtitan.protocols.blocks import peel
    from torchtitan.protocols.module import ModuleList

    with torch.device("meta"):
        block = _spec("llama3/debugmodel.yaml").model.layers[0].build()
    assert peel(checkpoint_wrapper(block)) is block
    one = ModuleList([block])
    for container in (one, ModuleList([block, block])):
        assert peel(container) is container
    assert [s.fqn for s in walk_segments(one)] == ["0"]


def test_coalesce_block_groups_rejects_an_unknown_name():
    """A misspelled group name must not silently leave the blocks ungrouped.

    No process group needed: ``apply_fsdp`` validates the names before it touches a
    mesh. This is the public-entry-point half of the contract, whose internal half
    (warn and fall back, because under PP another stage owns those blocks) is
    ``test_unknown_coalesce_group_warns``.
    """
    from torchtitan.distributed.fsdp import apply_fsdp

    with torch.device("meta"):
        model = _spec("llama3/debugmodel.yaml").model.build()
    with pytest.raises(ValueError, match="coalesce_block_groups"):
        apply_fsdp(
            model,
            None,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            coalesce_block_groups=("layerz",),
        )


def test_cp_accounts_for_every_attention_kernel():
    """CP must raise on a kernel it cannot wrap, never skip it.

    The kernel classes are derived from ``InnerAttentionConfig`` rather than
    listed, so a newly added kernel cannot fall outside the collection: on a mixed
    stack that would mean CP silently applied to only some layers.
    """
    from torchtitan.distributed.context_parallel import (
        _inner_attention_classes,
        apply_cp_to_forward,
        CP_SUPPORTED_ATTENTION,
    )
    from torchtitan.models.common.attention import FlashAttention3
    from torchtitan.models.common.attention.softmax.kernels import InnerAttentionConfig
    from torchtitan.protocols.module import Module, ModuleList

    derived = _inner_attention_classes()
    assert Module not in derived, "the config base must not be mistaken for a kernel"
    for supported in CP_SUPPORTED_ATTENTION:
        assert supported in derived
    # On the class: every other kernel needs a flash-attn build (the varlen one
    # too, through torch's own ``_fa3``), which a CPU test runner has none of.
    assert FlashAttention3 in derived, "flash kernels must be accounted for"

    class NewKernel(Module):
        """A kernel CP has never heard of -- the case this test is about."""

        @dataclasses.dataclass(kw_only=True, slots=True)
        class Config(InnerAttentionConfig):
            pass

        def __init__(self, config):
            del config
            super().__init__()

    assert NewKernel in _inner_attention_classes(), "declaring a kernel is enough"

    class Holder(Module):
        def __init__(self, kernels):
            super().__init__()
            self.kernels = ModuleList(kernels)

    new = NewKernel.Config().build()
    sdpa = CP_SUPPORTED_ATTENTION[1].Config().build()
    for kernels in ([new], [new, sdpa]):
        with pytest.raises(NotImplementedError, match="NewKernel"):
            apply_cp_to_forward(Holder(kernels), None)


def _code_tokens(path: pathlib.Path) -> tuple[set[str], set[str]]:
    """String literals and attribute names in real code, excluding docstrings.

    A docstring may show ``["tok_embeddings", "layers.0"]`` as an example; live
    code may not.
    """
    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }
    names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return literals, names


def test_core_does_not_name_model_internals():
    """Distributed code and the trainer must not hardcode a model's shape.

    A mechanism-absence test: asserting no FQN literals and no duck-typed
    attribute probes keeps the coupling from creeping back. Model structure reaches
    core only through ``named_segments`` and ``grouped_param_axes``.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "torchtitan"
    forbidden_literals = {"layers", "tok_embeddings", "lm_head", "mtp"}
    forbidden_names = {"moe_enabled", "iter_blocks", "enable_weight_tying"}

    offenders = []
    for path in [root / "trainer.py", *sorted((root / "distributed").rglob("*.py"))]:
        literals, names = _code_tokens(path)
        rel = path.relative_to(root)
        offenders += [
            f"{rel}: literal {t!r}" for t in sorted(literals & forbidden_literals)
        ]
        offenders += [f"{rel}: attribute {t}" for t in sorted(names & forbidden_names)]
    assert not offenders, "model-specific names leaked into core: " + ", ".join(
        offenders
    )
