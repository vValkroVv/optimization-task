"""The conformer component's flagless encoder-SP gate and per-stack feasibility.

Single process; ``composer.utils.dist`` inside the modeling module is replaced
by a stand-in pinned to a fake SP group and the SP gather by a local stand-in,
so the tests check the DECISION, not the sharded numerics — those are the
torchrun cases in test_conformer_sp_parity. Runs a 2-layer, d16 conformer on
CPU with ``attention_backend="sdpa"``, bf16 like the bucketing tests.
"""

import ast
import contextlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sber_tests.test_utils.conformer_cpu import import_conformer_modules

modeling, configuration = import_conformer_modules()

pytestmark = [pytest.mark.unit, pytest.mark.fast]

FEAT_IN = 8
ATTN_CHUNK = 8
N_LAYERS = 2
SUBSAMPLING = 4
HALO = ATTN_CHUNK * N_LAYERS
# shortest mel length whose post-stem stack gives every rank a core >= halo
SHARD_FROM = {sp: HALO * SUBSAMPLING * sp for sp in (2, 4)}
CFG = dict(
    feat_in=FEAT_IN,
    n_layers=N_LAYERS,
    d_model=16,
    subsampling="striding_conv1d",
    subsampling_factor=SUBSAMPLING,
    subsampling_conv_channels=16,
    ff_expansion_factor=2,
    n_heads=2,
    pos_emb_max_len=64,
    conv_kernel_size=5,
    dropout=0.0,
    dropout_emb=0.0,
    dropout_att=0.0,
    attn_chunk_size=ATTN_CHUNK,
    max_position_seq_len=256,
    attention_backend="sdpa",
)


class _Dist:
    """Stand-in for ``composer.utils.dist`` pinned to an SP group of ``size``
    (``None`` = no group initialised). Records every all_reduce."""

    def __init__(self, size):
        self.size, self.reduces = size, []

    def get_sp_group_size(self):
        return self.size

    def get_sp_group_rank(self):
        return 0

    def get_sp_group(self):
        return None

    def all_reduce(self, tensor, reduce_operation="SUM", group=None, async_op=True):
        self.reduces.append(reduce_operation)


@contextlib.contextmanager
def pinned_sp(size):
    """Pin the SP group size the gate sees and replace the gather by a local
    stand-in with the right shape (``size`` identical slices), counting calls."""
    saved = (modeling.composer_dist, modeling.gather_from_sequence_parallel_region)
    dist = _Dist(size) if size != "absent" else None
    gathers = []

    def local_gather(x):
        gathers.append(tuple(x.shape))
        return x.repeat(1, size, 1)

    modeling.composer_dist = dist
    modeling.gather_from_sequence_parallel_region = local_gather
    try:
        yield dist, gathers
    finally:
        modeling.composer_dist, modeling.gather_from_sequence_parallel_region = saved


def build(top_level_training=True, train=True, **overrides):
    torch.manual_seed(0)
    config = configuration.ConformerConfig(**{**CFG, **overrides})
    model = modeling.ConformerForAudioLanguageModeling(config)
    for param in model.parameters():
        param.data.normal_(0, 0.2)
    model.train(train)
    model.set_top_level_training(top_level_training)
    return model.to(torch.bfloat16)


def bundle(*lengths, dummy=False):
    torch.manual_seed(1)
    item = {
        "spectrograms": torch.randn(sum(lengths), FEAT_IN).to(torch.bfloat16),
        "spectrogram_lengths": torch.tensor(lengths),
    }
    if dummy:
        item["dummy_value"] = True
    return [item]


def test_off_without_a_stamp_without_composer_without_a_group_or_at_sp1():
    model = build()
    model._top_level_training = (
        modeling.ConformerForAudioLanguageModeling._top_level_training
    )
    assert model._top_level_training is False
    with pinned_sp(2) as (dist, gathers):
        model(bundle(SHARD_FROM[2]))
    assert dist.reduces == [] and gathers == []
    model.set_top_level_training(True)
    for size in ("absent", None, 1):
        with pinned_sp(size) as (dist, gathers):
            model(bundle(SHARD_FROM[2]))
        assert gathers == [] and (dist is None or dist.reduces == []), size


def test_top_level_eval_is_off_component_eval_is_not():
    """Eval/predict batches differ per SP rank: the TOP-LEVEL phase gates. A
    component put in eval mode by a frozen-encoder wrap keeps sharding."""
    model = build(top_level_training=False)
    with pinned_sp(2) as (dist, gathers):
        model(bundle(SHARD_FROM[2]))
    assert gathers == []
    frozen = build(train=False)
    assert frozen.training is False
    with pinned_sp(2) as (dist, gathers):
        out, out_len = frozen(bundle(SHARD_FROM[2]))
    assert len(gathers) == 1 and dist.reduces == []
    assert out.shape[0] == 1 and out_len.tolist() == [SHARD_FROM[2] // SUBSAMPLING]


@pytest.mark.parametrize("sp_size", [2, 4])
def test_stack_shards_only_when_every_core_is_at_least_a_halo(sp_size):
    model = build()
    with pinned_sp(sp_size) as (_, gathers):
        model(bundle(SHARD_FROM[sp_size]))
    assert len(gathers) == 1
    with pinned_sp(sp_size) as (_, gathers):
        model(bundle(SHARD_FROM[sp_size] // 2))
    assert gathers == []


def test_sp_above_twice_the_layer_count_runs_replicated():
    """With sp_size > 2 * n_layers a rank's slice could start past the end of
    the stack; such groups run replicated whatever the clip length."""
    model = build()
    with pinned_sp(8) as (_, gathers):
        model(bundle(HALO * SUBSAMPLING * 8 * 4))
    assert gathers == []


def test_dummy_bundle_never_shards():
    """The all-ranks dummy (20 frames): no gather, no assert, at any sp."""
    model = build()
    for sp_size in (2, 4):
        with pinned_sp(sp_size) as (dist, gathers):
            out, out_len = model(bundle(20, dummy=True))
        assert gathers == [] and dist.reduces == []
        assert out_len.tolist() == [5]


def test_bucketed_groups_decide_per_stack():
    """Two bucket groups: the long group shards, the short group's core is
    below the halo and runs replicated (one gather total)."""
    lengths = [300, 300, 20, 20]
    groups = modeling._length_bucketed_groups(torch.tensor(lengths))
    assert len(groups) == 2, groups
    model = build()
    with pinned_sp(2) as (_, gathers):
        model(bundle(*lengths))
    assert len(gathers) == 1


def test_batch_norm_asserts_when_a_stack_would_shard():
    model = build(length_bucketing=False, conv_norm_type="batch_norm")
    with pinned_sp(2), pytest.raises(AssertionError, match="layer_norm"):
        model(bundle(SHARD_FROM[2]))
    with pinned_sp(1):
        model(bundle(SHARD_FROM[2]))


def test_top_level_model_stamps_every_component_before_its_forward():
    """The stamp lives at the call site the gate depends on: in
    ``_forward_components`` the stamp call precedes ``component(inputs)``."""
    src = (
        Path(__file__).resolve().parents[3]
        / "llmfoundry"
        / "models"
        / "gigaomni"
        / "modelling_gigaomni.py"
    ).read_text()
    fn = next(
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_components"
    )
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "component.set_top_level_training" in calls
    assert calls.index("component.set_top_level_training") < calls.index(
        "component.unwrap_tuple_output"
    )


def test_stamp_reaches_the_component_through_a_wrapper():
    """gigafsdp wrappers replace the component in the ModuleDict and forward
    unknown attribute reads to it; the stamp call lands on the wrapped module."""

    class _Wrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self._module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self._module, name)

    model = build(top_level_training=False)
    wrapped = _Wrapper(model)
    wrapped.set_top_level_training(True)
    assert model._top_level_training is True
    assert "_top_level_training" not in wrapped.__dict__
