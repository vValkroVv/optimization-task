# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The two overlaps in the MoE path: the side stream, and the deferred wait.

1. ``_launch_on_comm_stream`` must issue the work it was given to overlap
   (the expert weight quantization) *before* launching the all-to-all. Swap the
   two and there is no overlap, while every numeric test stays green.
2. ``MoE.forward`` must run ``shared_experts`` between the combine and
   ``sync_combine``. DeepEP's combine is asynchronous, so moving the wait up by
   one line is what turns the shared experts from free into serial.
3. Everything crossing the stream boundary must be tied to the stream that will
   read it -- inputs to the communication stream, its results back to the
   compute stream. A missed ``record_stream`` lets the allocator hand that
   memory to someone else while the other stream is still reading it, which
   surfaces as garbage numbers under memory pressure and nowhere else.

The first two are about call order and need neither ranks nor a GPU. The third
needs one CUDA device -- ``record_stream`` is meaningless without one -- but
still no ranks and no HybridEP buffer, because what is under test is the
bookkeeping, not the communication.
"""

import pytest
import torch

from torchtitan.models.common.ffn import FeedForward
from torchtitan.models.common.moe import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
    GroupedExperts,
    HybridEPTokenDispatcher,
    LocalTokenDispatcher,
    MoE,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.moe.dispatch import BaseTokenDispatcher
from torchtitan.models.common.nn_modules import Linear

DIM, HIDDEN, EXPERTS = 64, 128, 4


# --- the side stream ---


@pytest.mark.parametrize("comm_stream", [False, True])
def test_the_overlapped_work_is_issued_before_the_communication(comm_stream):
    """On either path: with one stream the two serialize, but the order still
    has to be the one that overlaps when there are two."""
    pytest.importorskip("deep_ep")
    from torchtitan.distributed.deepep import hybridep

    calls: list[str] = []

    def communicate():
        calls.append("comm")
        return "comm-out"

    def quantize_weights():
        calls.append("overlap")
        return "overlap-out"

    hybridep.set_kernel_options(comm_stream=comm_stream, fuse_permute=True)
    try:
        out, overlapped, join = hybridep._launch_on_comm_stream(
            communicate, torch.ones(4), overlap_on_compute=quantize_weights
        )
        join()
    finally:
        hybridep.set_kernel_options(comm_stream=False, fuse_permute=True)

    assert calls == ["overlap", "comm"]
    assert (out, overlapped) == ("comm-out", "overlap-out")


def test_nothing_to_overlap_is_not_an_error():
    """The bf16 path launches the same helper with no weights to quantize."""
    pytest.importorskip("deep_ep")
    from torchtitan.distributed.deepep import hybridep

    assert hybridep._run_on_comm_stream(lambda: "out", torch.ones(4)) == "out"


# --- the deferred wait ---


class _Recorder(torch.nn.Module):
    """Stands in for a submodule and writes down when it was called."""

    def __init__(self, calls: list[str], name: str) -> None:
        super().__init__()
        self.calls = calls
        self.name = name

    def forward(self, x, *args, **kwargs):
        self.calls.append(self.name)
        return torch.zeros_like(x)

    def sync_combine(self) -> None:
        self.calls.append("sync_combine")


def _moe() -> MoE:
    config = MoE.Config(
        experts=GroupedExperts.Config(dim=DIM, hidden_dim=HIDDEN, num_experts=EXPERTS),
        router=TokenChoiceTopKRouter.Config(
            gate=Linear.Config(in_features=DIM, out_features=EXPERTS, bias=False),
            top_k=2,
        ),
        # Present so there is something to overlap the combine with.
        shared_experts=FeedForward.Config(
            dim=DIM,
            hidden_dim=HIDDEN,
            w1=Linear.Config(in_features=DIM, out_features=HIDDEN),
            w2=Linear.Config(in_features=HIDDEN, out_features=DIM),
            w3=Linear.Config(in_features=DIM, out_features=HIDDEN),
        ),
    )
    return config.build()


def test_the_shared_experts_run_between_the_combine_and_its_wait():
    moe = _moe()
    calls: list[str] = []
    moe.experts = _Recorder(calls, "experts")
    moe.shared_experts = _Recorder(calls, "shared_experts")

    moe(torch.randn(1, 8, DIM))

    assert calls == ["experts", "shared_experts", "sync_combine"]


def test_only_deepep_waits_on_its_combine():
    """HybridEP's combine is synchronous, so it must not override the no-op --
    and the others must not grow a wait that costs a sync for nothing."""
    assert DeepEPTokenDispatcher.sync_combine is not BaseTokenDispatcher.sync_combine
    for cls in (
        HybridEPTokenDispatcher,
        LocalTokenDispatcher,
        AllToAllTokenDispatcher,
    ):
        assert cls.sync_combine is BaseTokenDispatcher.sync_combine, cls.__name__


def test_the_experts_pass_the_wait_down_to_their_dispatcher():
    """``MoE.forward`` calls ``experts.sync_combine()``; the experts own no
    communication, so the only thing it can mean is the dispatcher's."""
    calls: list[str] = []
    experts = GroupedExperts.Config(
        dim=DIM,
        hidden_dim=HIDDEN,
        num_experts=EXPERTS,
        token_dispatcher=LocalTokenDispatcher.Config(),
    ).build()
    experts.token_dispatcher.sync_combine = lambda: calls.append("dispatcher")

    experts.sync_combine()

    assert calls == ["dispatcher"]


# --- what crosses the stream boundary ---


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="record_stream needs a CUDA device"
)
def test_everything_crossing_the_stream_boundary_is_recorded(monkeypatch):
    """Tied in both directions, and only what lives on the device.

    The arguments are the shape the dispatch op passes: an fp8 payload (one
    argument, two allocations), a nested list, a host tensor and a plain int.
    """
    pytest.importorskip("deep_ep")
    from torchtitan.distributed.deepep import hybridep
    from torchtitan.ops.float8.fp8_tensor import (
        Float8BlockwiseQTensor,
        InputLayoutType,
    )

    recorded: list[tuple[int, object]] = []
    record_stream = torch.Tensor.record_stream

    def spy(self, stream):
        recorded.append((self.data_ptr(), stream))
        return record_stream(self, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", spy)

    payload = Float8BlockwiseQTensor(
        torch.zeros(128, 256, dtype=torch.float8_e4m3fn, device="cuda"),
        torch.zeros(128, 2, dtype=torch.float32, device="cuda"),
        torch.bfloat16,
        InputLayoutType.ROWWISE,
        1,
        scale_block_first=False,
    )
    scaling_factor = torch.zeros(128, 2, device="cuda")
    on_the_host = torch.zeros(4)
    produced_on_comm = (torch.ones(8, device="cuda"), [torch.ones(4, device="cuda")])

    hybridep.set_kernel_options(comm_stream=True, fuse_permute=True)
    try:
        out, _overlapped, join = hybridep._launch_on_comm_stream(
            lambda: produced_on_comm,
            payload,
            [scaling_factor, None],
            7,
            on_the_host,
            overlap_on_compute=lambda: None,
        )
        comm_stream = hybridep._comm_stream
        compute_stream = torch.cuda.current_stream()
        join()
    finally:
        hybridep.set_kernel_options(comm_stream=False, fuse_permute=True)

    assert comm_stream is not None and comm_stream != compute_stream
    on_comm = {ptr for ptr, stream in recorded if stream == comm_stream}
    on_compute = {ptr for ptr, stream in recorded if stream == compute_stream}

    # In: everything the compute stream produced, both planes of the wrapper
    # and through the nested list.
    for tensor in (payload._data, payload._scale, scaling_factor):
        assert tensor.data_ptr() in on_comm

    # Out: everything the communication stream produced, back to its consumer.
    assert out is produced_on_comm
    assert produced_on_comm[0].data_ptr() in on_compute
    assert produced_on_comm[1][0].data_ptr() in on_compute

    # A host tensor has no stream to be tied to, and an int is not a tensor.
    assert on_the_host.data_ptr() not in on_comm | on_compute
