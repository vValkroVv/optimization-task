import pytest
import torch

from metric_sampler.capture.stats.activations import ActivationStats, point_slots
from stand.activation_kernel_checks.fake_recorder import (
    AMAX_OFFSET,
    BUFFER_WORDS,
    COUNTS_OFFSET,
    FLOAT_SUM_WORDS,
    MOMENTS_OFFSET,
    POWER_SUMS_OFFSET,
    QUANTILE_COUNTS_OFFSET,
    QUANTILES_OFFSET,
    WORDS_WITH_QUANTILES,
    make_recorder,
)

pytestmark = pytest.mark.unit


def test_activation_oracle_recorder_uses_disjoint_microbatch_slots():
    recorder = make_recorder(torch.device("cpu"))
    stats = ActivationStats(recorder)
    stats.begin_step(False)
    for microbatch, values in enumerate((torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))):
        stats.begin_microbatch(microbatch)
        stats.observe("point", 0, values)
    expected_moments = (torch.tensor([3.0, 5.0, 0.0, 0.0]), torch.tensor([18.0, 86.0, 0.0, 0.0]))
    for microbatch, expected in enumerate(expected_moments):
        slot = recorder.resolve("activation/point/moments", 0, microbatch)
        assert torch.equal(recorder.float_words[slot.offset:slot.offset + slot.width], expected)
        counts = recorder.resolve("activation/point/counts", 0, microbatch)
        assert recorder.buffer[counts.offset:counts.offset + 2].tolist() == [0, (2, 4)[microbatch]]
        assert recorder.buffer[counts.offset + 2:counts.offset + 66].sum().item() == (2, 4)[microbatch]


def test_word_layout_follows_the_slots_of_an_activation_point_and_the_quantile_words_follow_them():
    widths = dict(point_slots("point"))
    moments = 0
    counts = moments + widths["act/point/moments"]
    amax = counts + widths["act/point/counts"]
    power_sums = amax + widths["act/point/amax"]

    assert (moments, counts, amax, power_sums) == (MOMENTS_OFFSET, COUNTS_OFFSET, AMAX_OFFSET, POWER_SUMS_OFFSET)
    assert power_sums + widths["act/point/power_sums"] == BUFFER_WORDS
    assert (QUANTILES_OFFSET, QUANTILE_COUNTS_OFFSET) == (BUFFER_WORDS, BUFFER_WORDS + 9)
    assert QUANTILE_COUNTS_OFFSET + 5 == WORDS_WITH_QUANTILES


def test_float_sum_words_are_the_sums_of_the_moments_and_of_the_power_sums():
    assert FLOAT_SUM_WORDS == (MOMENTS_OFFSET, MOMENTS_OFFSET + 1, POWER_SUMS_OFFSET, POWER_SUMS_OFFSET + 1)


def test_the_recorder_views_the_count_slot_as_int32_words_of_its_float_words():
    recorder = make_recorder(torch.device("cpu"))

    recorder.tensor("counts")[1] = 7

    assert recorder.buffer[COUNTS_OFFSET + 1].item() == 7
    assert recorder.float_buffer() is recorder.float_words
    assert recorder.tensor("moments").dtype == torch.float32
    assert recorder.tensor("exp_hist").dtype == torch.int32
