from dataclasses import replace

import pytest
import torch

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.stats.token_loss import TokenLossStats
from metric_sampler.contract.metrics import ValueType
from stand import bench_aggregator
from stand import check_token_loss_kernel as loss_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("detailed", [False, True])
def test_token_loss_oracle_preserves_distinct_partial_microbatches(detailed):
    shape = loss_check._Shape("small", layers=2, experts=8, top_k=2, mtp_heads=2, batch_size=3, sequence_length=17)
    layout = Layout(loss_check._layout_requests(shape))
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(17)
    inputs = (
        loss_check._make_inputs(shape, device, generator),
        loss_check._make_inputs(replace(shape, batch_size=2), device, generator),
    )
    reference = loss_check._make_rig(layout, shape, device)
    actual = loss_check._make_rig(layout, shape, device)
    try:
        loss_check._run(
            reference, inputs, detailed, loss_check._reference_observe_routing, loss_check._reference_observe_loss
        )
        loss_check._run(actual, inputs, detailed, TokenLossStats.observe_routing, TokenLossStats.observe_loss)
        assert loss_check._compare(shape, detailed, reference.recorder, actual.recorder) == []
        for microbatch, values in enumerate(inputs):
            for layer, indices in values.topk_indices.items():
                expected = torch.bincount(indices.flatten(), minlength=shape.experts).to(torch.int32)
                assert torch.equal(
                    loss_check._read(actual.recorder, "router/tokens_per_expert", layer, microbatch), expected
                )
            counts = loss_check._read(actual.recorder, "loss/stream_count", -1, microbatch)
            valid = values.labels_valid[0] & torch.isfinite(values.loss[0])
            expected_counts = torch.stack(
                [valid[values.stream_ids == stream].sum() for stream in range(loss_check._STREAMS)]
            )
            assert torch.equal(counts.long(), expected_counts)

        loss_check._read(actual.recorder, "router/tokens_per_expert", 0, 1)[0] += 1
        problems = loss_check._compare(shape, detailed, reference.recorder, actual.recorder)
        assert len(problems) == 1
        assert "router/tokens_per_expert layer=0 mb=1: mismatch" in problems[0]
    finally:
        reference.close()
        actual.close()


def test_token_loss_oracle_rejects_one_entropy_bin_shift():
    shape = loss_check._Shape("entropy", layers=1, experts=2, top_k=1, mtp_heads=0, batch_size=1, sequence_length=4096)
    layout = Layout(loss_check._layout_requests(shape))
    reference = loss_check._make_rig(layout, shape, torch.device("cpu"))
    actual = loss_check._make_rig(layout, shape, torch.device("cpu"))
    try:
        reference.recorder.begin_step(False)
        actual.recorder.begin_step(False)
        expected = loss_check._read(reference.recorder, "router/entropy_hist", 0, 0)
        observed = loss_check._read(actual.recorder, "router/entropy_hist", 0, 0)
        expected[18] = 4096
        observed[18] = 4095
        observed[19] = 1
        assert loss_check._compare(shape, False, reference.recorder, actual.recorder) == [
            "entropy mtp=0 detailed=False router/entropy_hist layer=0 mb=0: mismatch at [18, 19] "
            "(ref [4096, 0] vs got [4095, 1])"
        ]
    finally:
        reference.close()
        actual.close()


def test_aggregator_benchmark_keeps_router_dispatcher_and_activation_microbatches():
    config = bench_aggregator.BenchConfig(
        "small", layers=2, experts=8, microbatches=3, activation_points=("residual",)
    )
    layout = bench_aggregator._build_layout(config)
    expected = {
        (name, layer, microbatch)
        for name in (
            "router/tokens_per_expert", "dispatcher/assignments_before_capacity",
            "dispatcher/valid_assignments_before_capacity", "dispatcher/assignments_after_capacity",
            "dispatcher/valid_assignments_after_capacity", "act/residual/moments",
            "act/residual/counts", "act/residual/amax",
        )
        for layer in range(config.layers)
        for microbatch in range(config.microbatches)
    }
    observed = {
        (slot.metric.name, slot.layer, slot.microbatch)
        for slot in layout.slots if slot.metric.required
    }
    assert observed == expected
    assert all(not slot.metric.detailed for slot in layout.slots)
    counts = [slot for slot in layout.slots if slot.metric.name == "act/residual/counts"]
    assert len(counts) == 6
    assert all(slot.width == 130 and slot.metric.value_type is ValueType.INT32 for slot in counts)
