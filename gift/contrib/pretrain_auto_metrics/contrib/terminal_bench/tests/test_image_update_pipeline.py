from __future__ import annotations

import re

import pytest

from terminal_bench.rli_image_update.scripts.pipeline import (
    MAX_PARALLEL_BUILDS,
    PILOT_TASKS,
    render_child_pipeline,
    select_tasks,
)


def _records(task_ids: list[str]) -> list[dict[str, object]]:
    return [{"task_id": task_id} for task_id in task_ids]


def test_all_mode_excludes_pilot_tasks_and_uses_five_resource_groups() -> None:
    records = _records([*PILOT_TASKS, *[f"task-{index}" for index in range(83)]])

    pipeline = render_child_pipeline(records, "all")

    assert pipeline.count("extends: .terminal-image-build") == 83
    for task_id in PILOT_TASKS:
        assert f'"build-{task_id}"' not in pipeline
    assert set(re.findall(r"resource_group: terminal-image-build-(\d+)", pipeline)) == {
        str(index) for index in range(MAX_PARALLEL_BUILDS)
    }
    assert "--mode batch" not in pipeline


def test_pilot_mode_selects_fixed_tasks() -> None:
    records = _records(["another-task", *PILOT_TASKS])

    selected = select_tasks(records, "pilot")

    assert [record["task_id"] for record in selected] == list(PILOT_TASKS)


def test_pilot_mode_rejects_missing_task() -> None:
    with pytest.raises(ValueError, match="pilot tasks missing from manifest"):
        select_tasks(_records(list(PILOT_TASKS[:-1])), "pilot")
