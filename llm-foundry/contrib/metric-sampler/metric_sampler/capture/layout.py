from __future__ import annotations

import json
from dataclasses import dataclass

from metric_sampler.contract.catalog import metric_by_id
from metric_sampler.contract.metrics import SCHEMA_VERSION, MetricSpec

_DEFAULT_STREAM_NAMES = ("all",)


@dataclass(frozen=True)
class Slot:
    metric: MetricSpec
    layer: int
    microbatch: int
    width: int
    offset: int


class Layout:
    """Assigns each (metric, layer, microbatch) request a contiguous offset in the flat buffer.

    Detailed-mode slots (`metric.detailed`) must all come after every non-detailed slot, so
    that a non-detailed step can copy just the leading `non_detailed_words` words to the host."""

    def __init__(
        self,
        requests: list[tuple[MetricSpec, int, int, int]],
        stream_names: tuple[str, ...] = _DEFAULT_STREAM_NAMES,
    ) -> None:
        slots: list[Slot] = []
        slot_by_key: dict[tuple[str, int, int], Slot] = {}
        index_by_key: dict[tuple[str, int, int], int] = {}
        offset = 0
        non_detailed_words = 0
        non_detailed_slots = 0
        seen_detailed = False
        for index, (metric, layer, microbatch, width) in enumerate(requests):
            key = (metric.name, layer, microbatch)
            if key in slot_by_key:
                raise ValueError(f"duplicate slot request for {key}")
            if metric.detailed:
                seen_detailed = True
            elif seen_detailed:
                raise ValueError(f"non-detailed slot request for {key} follows a detailed slot")
            else:
                non_detailed_words = offset + width
                non_detailed_slots = index + 1
            slot = Slot(metric=metric, layer=layer, microbatch=microbatch, width=width, offset=offset)
            slots.append(slot)
            slot_by_key[key] = slot
            index_by_key[key] = index
            offset += width
        self._slots = tuple(slots)
        self._slot_by_key = slot_by_key
        self._index_by_key = index_by_key
        self._words = offset
        self._non_detailed_words = non_detailed_words
        self._non_detailed_slots = non_detailed_slots
        self._stream_names = tuple(stream_names)

    @property
    def slots(self) -> tuple[Slot, ...]:
        return self._slots

    @property
    def words(self) -> int:
        return self._words

    @property
    def non_detailed_words(self) -> int:
        """Offset of the first detailed slot, or `words` when there is none."""
        return self._non_detailed_words

    @property
    def non_detailed_slots(self) -> int:
        """Index of the first detailed slot, or the slot count when there is none."""
        return self._non_detailed_slots

    @property
    def stream_names(self) -> tuple[str, ...]:
        return self._stream_names

    def slot(self, metric_name: str, layer: int, microbatch: int) -> Slot:
        return self._slot_by_key[(metric_name, layer, microbatch)]

    def slot_index(self, metric_name: str, layer: int, microbatch: int) -> int:
        """This slot's position in `slots`, i.e. its bit position in the recorded-slots mask."""
        return self._index_by_key[(metric_name, layer, microbatch)]

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "stream_names": list(self._stream_names),
                "slots": [
                    {
                        "metric_id": slot.metric.metric_id,
                        "layer": slot.layer,
                        "microbatch": slot.microbatch,
                        "width": slot.width,
                        "offset": slot.offset,
                    }
                    for slot in self._slots
                ],
            }
        )

    @classmethod
    def from_json(cls, text: str) -> Layout:
        payload = json.loads(text)
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"layout schema version {payload['schema_version']} does not match "
                f"the running schema version {SCHEMA_VERSION}"
            )
        requests = [
            (metric_by_id(entry["metric_id"]), entry["layer"], entry["microbatch"], entry["width"])
            for entry in payload["slots"]
        ]
        return cls(requests, stream_names=tuple(payload["stream_names"]))
