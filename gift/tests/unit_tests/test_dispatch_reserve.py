"""reserve_hybridep_tokens: a full-length microbatch per rank, reserved up front.

HybridEP compiles its kernels for its buffer's token capacity and grows it by
reallocating and recompiling; the reservation keeps sequence-length warmup
from triggering that at each new maximum. No other backend is touched.
"""

from types import SimpleNamespace

import pytest
import torch

from torchtitan.models.common.moe import reserve_hybridep_tokens
from torchtitan.models.common.moe.dispatch import (
    BaseTokenDispatcher,
    HybridEPTokenDispatcher,
    LocalTokenDispatcher,
)


def _hybridep_dispatcher(seen: list[int]) -> HybridEPTokenDispatcher:
    # Not built: building imports deep_ep. Only the type and the method matter.
    dispatcher = HybridEPTokenDispatcher.__new__(HybridEPTokenDispatcher)
    dispatcher.reserve_tokens_per_rank = seen.append
    return dispatcher


def _reserve(model: torch.nn.Module, *, pp: bool, cp: int = 1, tp: int = 1) -> None:
    reserve_hybridep_tokens(
        model,
        training=SimpleNamespace(local_batch_size=8, seq_len=4096),
        parallelism=SimpleNamespace(pipeline_parallel_microbatch_size=2),
        parallel_dims=SimpleNamespace(pp_enabled=pp, cp=cp, tp=tp),
    )


def _model(dispatcher: BaseTokenDispatcher) -> torch.nn.Module:
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts.token_dispatcher = dispatcher
    return model


@pytest.mark.parametrize(
    "pp, cp, tp, tokens",
    [
        (True, 1, 1, 2 * 4096),  # the pp microbatch, not the local batch
        (False, 1, 1, 8 * 4096),
        (False, 2, 1, 8 * 4096 // 2),  # CP splits the sequence
        (True, 1, 2, 2 * 4096 // 2),  # EP shards the MoE input over TP
    ],
)
def test_reserves_a_full_length_microbatch_per_rank(pp, cp, tp, tokens):
    seen: list[int] = []
    _reserve(_model(_hybridep_dispatcher(seen)), pp=pp, cp=cp, tp=tp)
    assert seen == [tokens]


def test_other_backends_are_not_called():
    dispatcher = LocalTokenDispatcher.Config().build()
    called: list[int] = []
    dispatcher.reserve_tokens_per_rank = called.append
    _reserve(_model(dispatcher), pp=False)
    assert called == []
