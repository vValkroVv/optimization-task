from __future__ import annotations

import math
import uuid

import pytest
import torch

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder
from metric_sampler.capture.ring import ShmRing
from metric_sampler.capture.stats.expert_balance import ExpertBalanceStats
from metric_sampler.contract.catalog import metric_by_name

pytestmark = pytest.mark.unit


@pytest.fixture
def capture():
    names = [('router/scores_mean', 4), ('monitor/router_balance_local', 14),
             ('monitor/router_balance_counts_local', 3), ('monitor/router_bias_local', 4),
             ('monitor/expert_activation_sumsq_local', 8), ('monitor/expert_activation_count_local', 4)]
    layout = Layout([(metric_by_name(name), layer, microbatch, width)
                     for layer, microbatch in [(0, 0), (1, 0), (0, 1)] for name, width in names])
    ring = ShmRing.create('expert-cpu-' + uuid.uuid4().hex, layout, 1)
    recorder = RankRecorder(layout, ring, torch.device('cpu'))
    recorder.begin_step(True)
    stats = ExpertBalanceStats(recorder, 4, 2, 2)

    delivered = []

    def read(name, layer=0, microbatch=0):
        # What the step delivers: the device buffer with host-staged words overlaid.
        if not delivered:
            recorder.step_end(step=1)
            delivered.append(torch.from_numpy(ring.payload(ring.ready_slots()[0]).copy()))
        slot = layout.slot(name, layer, microbatch)
        result = delivered[0][slot.offset:slot.offset + slot.width].clone()
        if slot.metric.value_type.value == 'float32':
            return result.view(torch.float32)
        if name in ('monitor/router_balance_local', 'monitor/expert_activation_sumsq_local'):
            return result.view(torch.float64)
        return result

    yield stats, read
    recorder.close()
    ring.unlink()


def router(stats, scores, valid=None, bias=None, layer=0, microbatch=0):
    raw = scores[:, :2]
    normalized = raw / raw.sum(-1, keepdim=True)
    biased = scores if bias is None else scores + bias
    stats.observe_router(layer, microbatch, scores, biased, normalized, raw, biased[:, :2], valid, bias,
                         torch.tensor(0.25), torch.tensor(0.125), 0.01)


def test_all_batch_rows_repeated_calls_and_masked_denominators_preserved(capture):
    stats, read = capture
    first = torch.tensor([[.25, .5, .125, .25], [.5, .25, .25, .125]])
    second = torch.tensor([[.125, .25, .5, .25], [.25, .5, .25, .125]])
    bias = torch.tensor([.125, 0., .25, 0.])
    router(stats, first, torch.tensor([[True, False]]), bias)
    router(stats, second, torch.tensor([[True, True]]), bias)
    source = torch.cat((first, second)).double()
    valid = [True, False, True, True]
    expected = [0., 0., 0., 0., .5, .25, .02]
    for use, row in zip(valid, source.tolist()):
        expected[3] += sum(-(value + float(shift)) * math.log(value + float(shift) + 1e-12)
                           for value, shift in zip(row, bias))
        if use:
            expected[0] += 1.
            expected[1] += row[0] + row[1]
            expected[2] += row[0] + row[1] + float(bias[0] + bias[1])
    first.fill_(99)
    bias.fill_(99)
    stats.flush()
    torch.testing.assert_close(
        read('monitor/router_balance_local'), torch.tensor(expected, dtype=torch.float64), rtol=1e-7, atol=1e-14
    )
    assert read('monitor/router_balance_counts_local').tolist() == [4, 3, 2]
    assert read('monitor/router_bias_local').tolist() == [.125, 0., .25, 0.]
    assert torch.equal(read('router/scores_mean'), source.mean(0).float())
    stats.flush()


def test_router_bias_disabled_and_all_invalid_tokens_are_explicit(capture):
    stats, read = capture
    values = torch.tensor([[.5, .25, .125, .0625]])
    router(stats, values, torch.zeros(1, dtype=torch.bool))
    stats.flush()
    assert read('monitor/router_balance_counts_local').tolist() == [1, 0, 1]
    assert read('monitor/router_balance_local')[:3].tolist() == [0., 0., 0.]
    assert read('monitor/router_bias_local').tolist() == [0., 0., 0., 0.]
    assert read('monitor/router_balance_local')[3] > 0


def test_router_empty_population_retains_undefined_mean_and_call_count(capture):
    stats, read = capture
    router(stats, torch.empty(0, 4))
    stats.flush()
    assert read('monitor/router_balance_counts_local').tolist() == [0, 0, 1]
    assert torch.isnan(read('router/scores_mean')).all()


def test_layer_and_microbatch_are_distinct_and_flush_releases_population(capture):
    stats, read = capture
    router(stats, torch.ones(1, 4), layer=1)
    stats.flush()
    router(stats, torch.full((2, 4), 2.), microbatch=1)
    stats.flush()
    assert read('router/scores_mean', 1).tolist() == [1.] * 4
    assert read('router/scores_mean', microbatch=1).tolist() == [2.] * 4
    assert read('monitor/router_balance_counts_local', microbatch=1).tolist() == [2, 2, 1]


def test_router_nonfinite_entropy_is_preserved_instead_of_excluding_rows(capture):
    stats, read = capture
    router(stats, torch.tensor([[1., 1., float('nan'), float('inf')]]))
    stats.flush()
    assert torch.isnan(read('monitor/router_balance_local')[3])
    assert read('monitor/router_balance_counts_local').tolist() == [1, 1, 1]


def test_router_changed_bias_rejects_second_snapshot_without_overwriting(capture):
    stats, read = capture
    router(stats, torch.ones(1, 4), bias=torch.zeros(4))
    with pytest.raises(ValueError, match='bias changed'):
        router(stats, torch.ones(1, 4), bias=torch.ones(4))
    stats.flush()
    assert read('monitor/router_balance_counts_local').tolist() == [1, 1, 1]
    assert read('monitor/router_bias_local').tolist() == [0.] * 4


def test_router_invalid_mask_fails(capture):
    stats, _ = capture
    with pytest.raises(ValueError, match='every source token'):
        router(stats, torch.ones(2, 4), torch.ones(3, dtype=torch.bool))


def test_router_capacity_checked_before_allocating_large_population(capture):
    stats, _ = capture
    values = torch.empty((2**31, 4), device='meta')
    with pytest.raises(CaptureCapacityError, match='Int32'):
        router(stats, values)


def test_empty_router_calls_have_an_independent_exact_count_limit(capture):
    stats, read = capture
    router(stats, torch.empty(0, 4))
    stats._routers[0, 0].calls = 2**31 - 1
    with pytest.raises(CaptureCapacityError, match='call counts'):
        router(stats, torch.empty(0, 4))
    stats.flush()
    assert read('monitor/router_balance_counts_local').tolist() == [0, 0, 2**31 - 1]


@pytest.mark.parametrize('field,value', [
    ('scores', torch.ones(4)), ('scores', torch.ones(2, 3)), ('scores', torch.ones(2, 4, dtype=torch.int32)),
    ('biased_scores', torch.ones(1, 4)), ('biased_scores', torch.ones(2, 4, device='meta')),
    ('biased_scores', torch.ones(2, 4, dtype=torch.int32)),
    ('valid', torch.ones(2)), ('valid', torch.ones(2, dtype=torch.bool, device='meta')),
    ('normalized_topk', torch.ones(2)), ('normalized_topk', torch.ones(2, 0)),
    ('normalized_topk', torch.ones(3, 2)), ('raw_topk', torch.ones(2, 2, device='meta')),
    ('biased_topk', torch.ones(2, 2, dtype=torch.int32)),
    ('bias', torch.ones(1, 4)), ('bias', torch.ones(4, device='meta')),
    ('bias', torch.ones(4, dtype=torch.int32)),
])
def test_router_shape_dtype_and_device_guards_preserve_population(capture, field, value):
    stats, read = capture
    arguments = dict(scores=torch.ones(2, 4), biased_scores=torch.ones(2, 4),
                     normalized_topk=torch.ones(2, 2), raw_topk=torch.ones(2, 2), biased_topk=torch.ones(2, 2),
                     valid=None, bias=None, z_loss=torch.tensor(.25), seq_aux_loss=torch.tensor(.125), alpha=.01)
    arguments[field] = value
    with pytest.raises(ValueError):
        stats.observe_router(0, 0, **arguments)
    router(stats, torch.ones(1, 4))
    stats.flush()
    assert read('monitor/router_balance_counts_local').tolist() == [1, 1, 1]


def test_disabled_sequence_auxiliary_loss_is_explicit_nan(capture):
    stats, read = capture
    scores = torch.ones(2, 4)
    stats.observe_router(0, 0, scores, scores, scores[:, :2], scores[:, :2], scores[:, :2],
                         None, None, torch.tensor(.25), None, 0.)
    stats.flush()
    values = read('monitor/router_balance_local')
    assert values[:3].tolist() == [4., 4., 4.]
    assert values[4].item() == .25 and torch.isnan(values[5]) and values[6].item() == 0.
    assert read('monitor/router_balance_counts_local').tolist() == [2, 2, 1]


def test_disabled_router_z_loss_is_an_integer_zero(capture):
    stats, read = capture
    scores = torch.ones(2, 4)
    stats.observe_router(0, 0, scores, scores, scores[:, :2], scores[:, :2], scores[:, :2],
                         None, None, torch.tensor(0), torch.tensor(.125), 0.)
    stats.flush()
    assert read('monitor/router_balance_local')[4:6].tolist() == [0., .125]


def test_operands_that_require_grad_are_recorded_like_detached_ones_and_keep_no_graph(capture):
    stats, read = capture
    scores = torch.tensor([[.25, .5, .125, .25], [.5, .25, .25, .125]])
    bias = torch.tensor([.125, 0., .25, 0.])
    grad_operands = [
        operand.clone().requires_grad_() for operand in (scores, scores, scores[:, :2], scores[:, :2], scores[:, :2])
    ]
    z_loss, aux_loss = torch.tensor(.25, requires_grad=True), torch.tensor(.125, requires_grad=True)

    stats.observe_router(0, 0, scores, scores, scores[:, :2], scores[:, :2], scores[:, :2], None, bias,
                         torch.tensor(.25), torch.tensor(.125), 0.01)
    stats.observe_router(1, 0, *grad_operands, None, bias.clone().requires_grad_(), z_loss, aux_loss, 0.01)

    population = stats._routers[1, 0]
    assert not any(tensor.requires_grad for tensor in (population.values, population.valid_tokens,
                                                      population.score_sums,
                                                      population.bias))
    stats.flush()
    for name in ('router/scores_mean', 'monitor/router_balance_local', 'monitor/router_balance_counts_local',
                 'monitor/router_bias_local'):
        assert read(name, layer=1).tolist() == read(name, layer=0).tolist()


@pytest.mark.parametrize('dtype', [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_complete_integer_padding_mask_counts_its_valid_tokens_for_every_integer_dtype(capture, dtype):
    stats, read = capture
    scores = torch.tensor([[.5, .25, .125, .125]]).repeat(256, 1)
    mask = torch.ones(2, 128, dtype=dtype)
    mask[1, 126:] = 0
    router(stats, scores, mask)
    stats.flush()
    assert read('monitor/router_balance_counts_local').tolist() == [256, 254, 1]
    # Float32 2/3 + 1/3 is not exactly one; the sum itself runs in float64.
    normalized_row = (torch.tensor([.5, .25]) / .75).double().sum().item()
    assert read('monitor/router_balance_local')[:3].tolist() == pytest.approx(
        [254 * normalized_row, 190.5, 190.5], rel=1e-15, abs=0
    )


def test_one_padding_mask_given_to_every_layer_records_as_separate_masks_would(capture):
    stats, read = capture
    scores = torch.tensor([[.5, .25, .125, .125], [.25, .25, .25, .25], [.125, .5, .25, .125]])
    shared = torch.tensor([1, 0, 1], dtype=torch.int32)
    router(stats, scores, shared, layer=0)
    router(stats, scores * 2, shared, layer=1)
    router(stats, scores * 3, torch.tensor([1, 0, 1], dtype=torch.int32), microbatch=1)
    stats.flush()
    assert read('monitor/router_balance_counts_local', 0, 0).tolist() == [3, 2, 1]
    assert read('monitor/router_balance_counts_local', 1, 0).tolist() == [3, 2, 1]
    assert read('router/scores_mean', 1, 0).tolist() == (scores * 2).double().mean(0).float().tolist()
    assert read('monitor/router_balance_counts_local', 0, 1).tolist() == [3, 2, 1]


def test_a_padding_mask_changed_in_place_between_layers_is_read_again(capture):
    stats, read = capture
    scores = torch.ones(3, 4) / 4
    mask = torch.tensor([1, 1, 1], dtype=torch.int32)
    router(stats, scores, mask, layer=0)
    mask[1] = 0
    router(stats, scores, mask, layer=1)
    stats.flush()
    assert read('monitor/router_balance_counts_local', 0, 0).tolist() == [3, 3, 1]
    assert read('monitor/router_balance_counts_local', 1, 0).tolist() == [3, 2, 1]


@pytest.mark.parametrize('mask', [torch.tensor([0, 2], dtype=torch.int32), torch.tensor([-1, 1], dtype=torch.int64)])
def test_nonbinary_integer_padding_masks_are_reported_once_at_the_next_step(capture, mask):
    stats, _ = capture
    router(stats, torch.ones(2, 4), mask)
    router(stats, torch.ones(1, 4), torch.ones(1, dtype=torch.int32), layer=1)
    stats.require_binary_masks()
    with pytest.raises(ValueError, match='zero or one'):
        stats.require_binary_masks()
    stats.require_binary_masks()
    stats.finish()


def test_nonbinary_mask_of_the_last_step_is_reported_by_finish(capture):
    stats, _ = capture
    router(stats, torch.ones(2, 4), torch.tensor([0, 2], dtype=torch.int32))
    stats.require_binary_masks()
    with pytest.raises(ValueError, match='zero or one'):
        stats.finish()
    stats.finish()


def test_binary_integer_and_boolean_masks_pass_the_step_check(capture):
    stats, _ = capture
    stats.require_binary_masks()
    router(stats, torch.ones(2, 4), torch.tensor([0, 1], dtype=torch.int32))
    router(stats, torch.ones(2, 4), torch.tensor([True, False]), layer=1)
    stats.require_binary_masks()
    stats.require_binary_masks()
    stats.finish()


@pytest.mark.parametrize('experts,local,offset', [(0, 1, 0), (4, 0, 0), (4, 3, 2), (4, 1, -1)])
def test_invalid_ownership_rejected(experts, local, offset):
    with pytest.raises(ValueError, match='ownership'):
        ExpertBalanceStats(None, experts, local, offset)


def test_exact_expert_integer_counts_above_bfloat16_precision_and_repeated_calls(capture):
    stats, read = capture
    first = torch.ones(260, 2, dtype=torch.bfloat16)
    first[-3:] *= 2
    stats.observe_experts(0, 0, first, [257, 3])
    stats.observe_experts(0, 0, torch.tensor([[3., 4.], [1., 2.]]), [1, 1])
    first.zero_()
    stats.flush()
    assert read('monitor/expert_activation_count_local').tolist() == [0, 0, 258, 4]
    assert read('monitor/expert_activation_sumsq_local').tolist() == [0., 0., 539., 29.]


def test_expert_zero_tokens_and_empty_local_expert_have_exact_zero(capture):
    stats, read = capture
    stats.observe_experts(1, 0, torch.empty(0, 8), [0, 0])
    stats.observe_experts(0, 0, torch.tensor([[1., 2.]]), [0, 1])
    stats.flush()
    assert read('monitor/expert_activation_count_local', 1).tolist() == [0, 0, 0, 0]
    assert read('monitor/expert_activation_sumsq_local', 1).tolist() == [0.] * 4
    assert read('monitor/expert_activation_sumsq_local').tolist() == [0., 0., 0., 5.]


@pytest.mark.parametrize('counts', [[1], [-1, 2], [True, 0], [0., 1.]])
def test_expert_noninteger_negative_or_incomplete_counts_rejected(capture, counts):
    stats, _ = capture
    with pytest.raises(ValueError, match='exact nonnegative'):
        stats.observe_experts(0, 0, torch.ones(1, 2), counts)


@pytest.mark.parametrize('value', [None, torch.ones(1), torch.ones(3, 2)])
def test_non_tensor_or_padded_expert_population_rejected(capture, value):
    stats, _ = capture
    with pytest.raises(ValueError, match='complete unpadded'):
        stats.observe_experts(0, 0, value, [1, 1])


def test_expert_capacity_is_checked_without_allocating_population(capture):
    stats, _ = capture
    with pytest.raises(CaptureCapacityError, match='Int32'):
        stats.observe_experts(0, 0, None, [2**31, 0])


def test_expert_nonfinite_values_are_not_silently_dropped(capture):
    stats, read = capture
    stats.observe_experts(0, 0, torch.tensor([[float('nan')], [float('inf')]]), [1, 1])
    stats.flush()
    values = read('monitor/expert_activation_sumsq_local')
    assert torch.isnan(values[2]) and torch.isposinf(values[3])
    assert read('monitor/expert_activation_count_local').tolist() == [0, 0, 1, 1]

