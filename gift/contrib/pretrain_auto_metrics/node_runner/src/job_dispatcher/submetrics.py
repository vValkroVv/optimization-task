import math
from collections.abc import Callable
from typing import Literal, TypedDict, get_args

from pydantic import BaseModel


MetricStatus = Literal["failed"]
MetricValue = float | MetricStatus


class SubmetricEntry(TypedDict):
    value: MetricValue
    submetrics: dict[str, MetricValue]


JDMetricEntry = MetricValue | SubmetricEntry
Aggregate = Callable[[dict[str, float]], float]


def mean(members: dict[str, float]) -> float:
    return sum(members.values()) / len(members)


def is_number(value: MetricValue | None) -> bool:
    return isinstance(value, (int, float))


def is_reportable(value: MetricValue | None) -> bool:
    return value in get_args(MetricStatus) or (
        is_number(value) and math.isfinite(value)
    )


def _headline(entry: JDMetricEntry) -> MetricValue:
    if isinstance(entry, dict):
        return entry["value"]
    return entry


class SubmetricGroup(BaseModel):
    key: str
    members: list[str]
    value: str | Aggregate
    diagnostics: list[str] = []


class JDSubmetrics:
    @classmethod
    def nest_submetrics(
        cls, metrics: dict[str, JDMetricEntry], groups: list[SubmetricGroup]
    ) -> dict[str, JDMetricEntry]:
        nested: dict[str, JDMetricEntry] = dict(metrics)
        for group in groups:
            cls._apply_group(nested, group)
        return nested

    @classmethod
    def _apply_group(
        cls, nested: dict[str, JDMetricEntry], group: SubmetricGroup
    ) -> None:
        diagnostics = {
            key: _headline(nested[key]) for key in group.diagnostics if key in nested
        }
        members = {
            key: _headline(nested[key])
            for key in group.members
            if key in nested and is_reportable(_headline(nested[key]))
        }
        if callable(group.value):
            if not members:
                return
            statuses = [v for v in members.values() if not is_number(v)]
            value = statuses[0] if statuses else group.value(members)
        else:
            if group.value not in nested or not (members or diagnostics):
                return
            value = _headline(nested.pop(group.value))
        for key in [*members, *diagnostics]:
            if not isinstance(nested[key], dict):
                del nested[key]
        nested[group.key] = {"value": value, "submetrics": {**members, **diagnostics}}
