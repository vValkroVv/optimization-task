# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What fp8 selection accepts, and what it refuses.

``enable_fp8_grouped_experts`` is the gate in front of the whole sparse fp8
path: every constraint the DeepGEMM grouped GEMMs and the two EP backends
impose on the layout is stated there, and a config that slips through it fails
much later, on a GPU, inside a kernel. The rules are pure config, so they are
checked here rather than in an integration run.

The last test walks the gift_moe registry: a preset is fp8's only entry point
in production, and several of them are never built anywhere else.
"""

import pytest

from torchtitan.components.quantization.utils import (
    apply_float8,
    enable_fp8_grouped_experts,
    has_quantization,
)
from torchtitan.config.configs import FP8Config
from torchtitan.models.common.ffn import FeedForward
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.moe.dispatch import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
    LocalTokenDispatcher,
)

_EXPERT_INIT = {
    "w13_E2FD": lambda p: None,
    "w2_EDF": lambda p: None,
}


def _experts(
    *,
    dispatcher=None,
    dim: int = 512,
    score_before_experts: bool = False,
    param_init=_EXPERT_INIT,
) -> GroupedExperts.Config:
    """A GroupedExperts config the fp8 selection would accept, minus overrides."""
    if dispatcher is None:
        dispatcher = HybridEPTokenDispatcher.Config(pad_multiple=128)
    return GroupedExperts.Config(
        dim=dim,
        hidden_dim=1024,
        num_experts=8,
        token_dispatcher=dispatcher,
        score_before_experts=score_before_experts,
        param_init=dict(param_init) if param_init is not None else None,
    )


_DENSE_FP8 = r".*\.(w1|w2|w3)$"


def _moe(*, dispatcher=None, dim: int = 512, hidden_dim: int = 1024) -> MoE.Config:
    """The smallest tree ``apply_float8`` walks: routed experts, a router, an FFN."""
    experts = _experts(dispatcher=dispatcher, dim=dim)
    return MoE.Config(
        experts=experts,
        router=TokenChoiceTopKRouter.Config(
            gate=Linear.Config(
                in_features=dim, out_features=experts.num_experts, bias=False
            ),
            top_k=2,
        ),
        # Projections spelled out: ``resolve_child`` refuses to size one
        # without a ``param_init``, and nothing here is ever built.
        shared_experts=FeedForward.Config(
            dim=dim,
            hidden_dim=hidden_dim,
            w1=Linear.Config(in_features=dim, out_features=hidden_dim),
            w2=Linear.Config(in_features=hidden_dim, out_features=dim),
            w3=Linear.Config(in_features=dim, out_features=hidden_dim),
        ),
    )


# --- the backend the layout requires ---


@pytest.mark.parametrize(
    "dispatcher",
    [LocalTokenDispatcher.Config(), AllToAllTokenDispatcher.Config()],
    ids=["local", "all_to_all"],
)
def test_grouped_gemm_requires_an_ep_backend(dispatcher):
    """Only DeepEP/HybridEP produce the padded, expert-major token groups."""
    with pytest.raises(ValueError, match="deepep"):
        enable_fp8_grouped_experts(_experts(dispatcher=dispatcher))


def test_the_experts_config_refuses_the_same_pairing_on_its_own():
    """Derived state is checked before any runtime tensor/kernel allocation."""
    config = GroupedExperts.Config(
        dim=512,
        hidden_dim=1024,
        num_experts=8,
        token_dispatcher=LocalTokenDispatcher.Config(),
    )
    config.use_float8_grouped_gemm = True
    with pytest.raises(ValueError, match="DeepEP or HybridEP"):
        config.build()


def test_scores_must_be_applied_after_the_experts():
    """The fp8 experts apply probs between their GEMMs, so not before dispatch."""
    with pytest.raises(ValueError, match="score_before_experts=False"):
        enable_fp8_grouped_experts(_experts(score_before_experts=True))


# --- the HybridEP-only layout constraints ---


@pytest.mark.parametrize("dim", [256, 384, 1024 + 128])
def test_hybridep_fp8_requires_512_aligned_dim(dim):
    with pytest.raises(ValueError, match="dim % 512"):
        enable_fp8_grouped_experts(_experts(dim=dim))


@pytest.mark.parametrize("pad_multiple", [None, 1, 32, 64, 192])
def test_hybridep_fp8_requires_128_aligned_pad_multiple(pad_multiple):
    """DeepGEMM starts every expert group on a 128-row boundary."""
    with pytest.raises(ValueError, match="pad_multiple"):
        enable_fp8_grouped_experts(
            _experts(
                dispatcher=HybridEPTokenDispatcher.Config(pad_multiple=pad_multiple)
            )
        )


@pytest.mark.parametrize("capacity_factor", [1.0, 0.5])
def test_hybridep_fp8_refuses_non_blocking_dispatch(capacity_factor):
    """The fixed-capacity tail is unmasked on the way to the router; see
    ``fused_swiglu_quantization``'s ``HAS_MASK`` branch for the kernel side."""
    with pytest.raises(ValueError, match="non_blocking_capacity_factor"):
        enable_fp8_grouped_experts(
            _experts(
                dispatcher=HybridEPTokenDispatcher.Config(
                    pad_multiple=128, non_blocking_capacity_factor=capacity_factor
                )
            )
        )


@pytest.mark.parametrize("dim", [256, 512])
def test_deepep_fp8_only_needs_128_aligned_dim(dim):
    """DeepEP pads with our own kernel, so the 512 TMA constraint is HybridEP's."""
    experts = _experts(dispatcher=DeepEPTokenDispatcher.Config(), dim=dim)
    enable_fp8_grouped_experts(experts)
    assert experts.use_float8_grouped_gemm


# --- the fused w13 parameter ---


def test_w13_initializer_is_left_alone():
    sentinel = lambda p: None  # noqa: E731
    experts = _experts(param_init={**_EXPERT_INIT, "w13_E2FD": sentinel})
    enable_fp8_grouped_experts(experts)
    assert experts.param_init["w13_E2FD"] is sentinel


def test_separate_w1_w3_initializers_are_a_config_error():
    """The parameter is the fused w13_E2FD; nothing derives its initializer."""
    experts = _experts(
        param_init={
            "w1_EFD": lambda p: None,
            "w2_EDF": lambda p: None,
            "w3_EFD": lambda p: None,
        }
    )
    with pytest.raises(ValueError, match="w13_E2FD"):
        enable_fp8_grouped_experts(experts)


def test_missing_param_init_is_a_config_error():
    with pytest.raises(ValueError, match="param_init"):
        enable_fp8_grouped_experts(_experts(param_init=None))


# --- apply_float8: the whole-tree selection ---


def test_dense_fp8_and_tensor_parallelism_are_exclusive():
    with pytest.raises(ValueError, match="tensor parallelism"):
        apply_float8(
            _moe(),
            FP8Config(dense_include_linear_layers=_DENSE_FP8),
            tensor_parallel_degree=2,
        )


def test_sparse_fp8_survives_tensor_parallelism():
    """Only the dense linears are refused under TP; the experts are not."""
    model = _moe(dispatcher=DeepEPTokenDispatcher.Config())

    apply_float8(
        model, FP8Config(sparse_use_grouped_gemm=True), tensor_parallel_degree=2
    )

    assert has_quantization(model)


def test_shared_quantizer_is_enabled_on_every_feed_forward():
    model = _moe()

    apply_float8(
        model,
        FP8Config(
            dense_include_linear_layers=_DENSE_FP8, dense_use_shared_quantizer=True
        ),
        tensor_parallel_degree=1,
    )

    ffns = [ffn for _fqn, ffn, _p, _a in model.traverse(FeedForward.Config)]
    assert ffns and all(ffn.use_shared_quantizer for ffn in ffns)


def test_the_dense_regex_decides_which_projections_are_swapped():
    """The router's gate is a Linear too, and it must be left alone."""
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    model = _moe()

    apply_float8(
        model,
        FP8Config(dense_include_linear_layers=_DENSE_FP8),
        tensor_parallel_degree=1,
    )

    swapped = {
        fqn
        for fqn, lin, _p, _a in model.traverse(Linear.Config)
        if isinstance(lin, FP8Linear.Config)
    }
    assert swapped == {
        "shared_experts.w1",
        "shared_experts.w2",
        "shared_experts.w3",
    }, swapped


# --- the fused SwiGLU+quant switch ---


def test_unfused_swiglu_without_its_fp8_path_is_a_config_error():
    """Turning off a fusion that is not there would silently do nothing."""
    with pytest.raises(ValueError, match="dense_include_linear_layers"):
        FP8Config(dense_fused_swiglu_quant=False)
    with pytest.raises(ValueError, match="sparse_use_grouped_gemm"):
        FP8Config(sparse_fused_swiglu_quant=False)


@pytest.mark.parametrize("fused", [True, False])
def test_fused_swiglu_switch_reaches_ffns_and_experts(fused):
    model = _moe(dispatcher=DeepEPTokenDispatcher.Config())

    apply_float8(
        model,
        FP8Config(
            dense_include_linear_layers=_DENSE_FP8,
            dense_fused_swiglu_quant=fused,
            sparse_use_grouped_gemm=True,
            sparse_fused_swiglu_quant=fused,
        ),
        tensor_parallel_degree=1,
    )

    ffns = [ffn for _fqn, ffn, _p, _a in model.traverse(FeedForward.Config)]
    experts = [e for _fqn, e, _p, _a in model.traverse(GroupedExperts.Config)]
    assert ffns and all(ffn.dense_fused_swiglu_quant is fused for ffn in ffns)
    assert experts and all(e.sparse_fused_swiglu_quant is fused for e in experts)


def test_expert_z_loss_refuses_the_fused_fp8_swiglu():
    """The fused backward hands w13 an fp8 gradient: nothing to add the z-loss to."""
    experts = _experts(dispatcher=DeepEPTokenDispatcher.Config())
    experts.w1_hidden_z_loss_coef = 1e-3
    enable_fp8_grouped_experts(experts)
    with pytest.raises(ValueError, match="sparse_fused_swiglu_quant=False"):
        experts.build()


def test_sparse_selection_reaches_every_moe_layer():
    model = _moe(dispatcher=DeepEPTokenDispatcher.Config())

    apply_float8(
        model, FP8Config(sparse_use_grouped_gemm=True), tensor_parallel_degree=1
    )

    experts = [e for _fqn, e, _p, _a in model.traverse(GroupedExperts.Config)]
    assert experts and all(e.use_float8_grouped_gemm for e in experts)
    # Sparse-only: no Linear was swapped, so ``has_quantization`` must not be
    # answering from the dense side alone.
    assert has_quantization(model)


def test_current_fp8_policy_overwrites_derived_flags_and_is_idempotent(caplog):
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    model = _moe(dispatcher=DeepEPTokenDispatcher.Config())
    initial = FP8Config(
        dense_include_linear_layers=_DENSE_FP8,
        dense_use_shared_quantizer=True,
        dense_fused_swiglu_quant=False,
        sparse_use_grouped_gemm=True,
        sparse_deep_gemm_num_sms=112,
        sparse_fused_swiglu_quant=False,
    )
    apply_float8(model, initial, tensor_parallel_degree=1)
    converted = [config for _, config, _, _ in model.traverse(FP8Linear.Config)]
    assert converted
    apply_float8(model, initial, tensor_parallel_degree=1)
    repeated = [config for _, config, _, _ in model.traverse(FP8Linear.Config)]
    assert len(repeated) == len(converted)
    assert all(actual is original for actual, original in zip(repeated, converted))
    assert "matched no" not in caplog.text
    current = FP8Config(
        dense_include_linear_layers=_DENSE_FP8, sparse_deep_gemm_num_sms=108
    )
    apply_float8(model, current, tensor_parallel_degree=1)
    ffns = [config for _, config, _, _ in model.traverse(FeedForward.Config)]
    experts = [config for _, config, _, _ in model.traverse(GroupedExperts.Config)]
    assert ffns and all(
        not config.use_shared_quantizer and config.dense_fused_swiglu_quant
        for config in ffns
    )
    assert experts and all(
        not config.use_float8_grouped_gemm
        and config.sparse_fused_swiglu_quant
        and config.deep_gemm_num_sms == 108
        for config in experts
    )


def test_disabled_sparse_fp8_does_not_validate_or_preserve_stale_expert_state():
    model = _moe(dispatcher=LocalTokenDispatcher.Config())
    model.experts.use_float8_grouped_gemm = True
    model.experts.deep_gemm_num_sms = 99
    model.experts.sparse_fused_swiglu_quant = False
    apply_float8(model, FP8Config(), tensor_parallel_degree=1)
    assert model.experts.use_float8_grouped_gemm is False
    assert model.experts.deep_gemm_num_sms == 116
    assert model.experts.sparse_fused_swiglu_quant is True


@pytest.mark.parametrize("copy_first", [False, True])
def test_derived_fp8_score_invariant_is_checked_before_runtime_allocation(
    monkeypatch, copy_first
):
    import torch

    config = _experts(
        dispatcher=DeepEPTokenDispatcher.Config(), score_before_experts=True
    )
    config.use_float8_grouped_gemm = True
    if copy_first:
        config = config._copy()
    allocated = []

    def forbidden_allocation(*args, **kwargs):
        allocated.append(True)
        raise AssertionError("invalid FP8 configuration reached tensor allocation")

    monkeypatch.setattr(torch, "empty", forbidden_allocation)
    with pytest.raises(ValueError, match="score_before_experts=False"):
        config.build()
    assert not allocated


def test_config_build_preserves_derived_fp8_flags_at_the_owner_boundary(monkeypatch):
    import torch.nn as nn

    class Captured(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.received_config = config

    monkeypatch.setattr(FeedForward.Config, "owner", classmethod(lambda cls: Captured))
    monkeypatch.setattr(
        GroupedExperts.Config, "owner", classmethod(lambda cls: Captured)
    )
    ffn = FeedForward.Config(dim=512, hidden_dim=1024)
    ffn.use_shared_quantizer = True
    ffn.dense_fused_swiglu_quant = False
    received = ffn.build().received_config
    assert received is not ffn
    assert received.use_shared_quantizer is True
    assert received.dense_fused_swiglu_quant is False
    experts = GroupedExperts.Config(dim=512, hidden_dim=1024, num_experts=4)
    experts.use_float8_grouped_gemm = True
    experts.deep_gemm_num_sms = 112
    experts.sparse_fused_swiglu_quant = False
    received = experts.build().received_config
    assert received is not experts
    assert received.use_float8_grouped_gemm is True
    assert received.deep_gemm_num_sms == 112
    assert received.sparse_fused_swiglu_quant is False
