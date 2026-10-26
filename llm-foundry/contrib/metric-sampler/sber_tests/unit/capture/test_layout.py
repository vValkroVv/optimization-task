from __future__ import annotations

import json

import pytest

from metric_sampler.capture.layout import Layout
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.contract.metrics import SCHEMA_VERSION

pytestmark = pytest.mark.unit

_LOSS = metric_by_name("loss/total")
_COUNTS = metric_by_name("router/tokens_per_expert")
_BIAS = metric_by_name("router/expert_bias_before_update")
_HIST = metric_by_name("expert_loss/hist")
_STREAM_COUNT = metric_by_name("expert_loss/stream_count")


def test_offsets_are_contiguous_in_request_order() -> None:
    layout = Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4), (_BIAS, 0, -1, 2)])

    assert layout.words == 7
    assert [slot.offset for slot in layout.slots] == [0, 1, 5]
    assert [slot.width for slot in layout.slots] == [1, 4, 2]


def test_empty_layout_has_zero_words() -> None:
    layout = Layout([])

    assert layout.words == 0
    assert layout.slots == ()


def test_slot_looks_up_by_metric_name_layer_and_microbatch() -> None:
    layout = Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4), (_COUNTS, 1, -1, 4)])

    slot = layout.slot("router/tokens_per_expert", 1, -1)

    assert slot.offset == 5
    assert slot.width == 4
    assert slot.metric is _COUNTS


def test_slot_index_returns_the_request_order_position() -> None:
    layout = Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4), (_COUNTS, 1, -1, 4)])

    assert layout.slot_index("loss/total", -1, -1) == 0
    assert layout.slot_index("router/tokens_per_expert", 0, -1) == 1
    assert layout.slot_index("router/tokens_per_expert", 1, -1) == 2


def test_slot_index_raises_key_error_when_absent() -> None:
    layout = Layout([(_LOSS, -1, -1, 1)])

    with pytest.raises(KeyError):
        layout.slot_index("router/tokens_per_expert", 0, -1)


def test_slot_raises_key_error_when_absent() -> None:
    layout = Layout([(_LOSS, -1, -1, 1)])

    with pytest.raises(KeyError):
        layout.slot("router/tokens_per_expert", 0, -1)


def test_duplicate_request_raises_value_error() -> None:
    with pytest.raises(ValueError):
        Layout([(_LOSS, -1, -1, 1), (_LOSS, -1, -1, 1)])


def test_to_json_round_trips_through_from_json() -> None:
    layout = Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4), (_BIAS, 0, -1, 2)])

    restored = Layout.from_json(layout.to_json())

    assert restored.words == layout.words
    assert [(slot.metric.name, slot.layer, slot.microbatch, slot.width, slot.offset) for slot in restored.slots] == [
        (slot.metric.name, slot.layer, slot.microbatch, slot.width, slot.offset) for slot in layout.slots
    ]


def test_to_json_contains_the_schema_version_and_metric_ids() -> None:
    layout = Layout([(_LOSS, -1, -1, 1)])

    payload = json.loads(layout.to_json())

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["slots"] == [{"metric_id": _LOSS.metric_id, "layer": -1, "microbatch": -1, "width": 1, "offset": 0}]


def test_from_json_raises_value_error_on_schema_mismatch() -> None:
    payload = json.dumps({"schema_version": SCHEMA_VERSION + 1, "slots": []})

    with pytest.raises(ValueError):
        Layout.from_json(payload)


def test_non_detailed_words_equal_words_when_no_slot_is_detailed() -> None:
    layout = Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4)])

    assert layout.non_detailed_words == layout.words == 5
    assert layout.non_detailed_slots == len(layout.slots) == 2


def test_non_detailed_words_stop_at_the_first_detailed_slot() -> None:
    layout = Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4), (_HIST, 0, -1, 8), (_STREAM_COUNT, 0, -1, 2)])

    assert layout.non_detailed_words == 5
    assert layout.words == 15
    assert layout.non_detailed_slots == 2


def test_empty_layout_has_no_non_detailed_words() -> None:
    assert Layout([]).non_detailed_words == 0
    assert Layout([]).non_detailed_slots == 0


def test_non_detailed_slot_after_a_detailed_slot_raises_value_error() -> None:
    with pytest.raises(ValueError):
        Layout([(_HIST, 0, -1, 8), (_LOSS, -1, -1, 1)])


def test_default_stream_names_is_all() -> None:
    layout = Layout([(_LOSS, -1, -1, 1)])

    assert layout.stream_names == ("all",)


def test_stream_names_round_trip_through_json() -> None:
    layout = Layout([(_LOSS, -1, -1, 1)], stream_names=("all", "code", "web"))

    restored = Layout.from_json(layout.to_json())

    assert restored.stream_names == ("all", "code", "web")
