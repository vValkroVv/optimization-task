"""Validate the selected eight-GPU NV18 topology from nvidia-smi output."""

from __future__ import annotations

import re


GPU_LABELS = tuple(f"GPU{index}" for index in range(8))
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_topology_matrix(text: str) -> dict[str, list[str]]:
    lines = [ANSI_ESCAPE.sub("", line).split() for line in text.splitlines()]
    if not any(line[:8] == list(GPU_LABELS) for line in lines):
        return {}
    matrix = {}
    for row in lines:
        if (
            row
            and row[0] in GPU_LABELS
            and len(row) >= 9
            and all(value in {"X", "NV18"} for value in row[1:9])
        ):
            matrix[row[0]] = row[1:9]
    return matrix


def is_exact_nv18(matrix: dict[str, list[str]]) -> bool:
    if set(matrix) != set(GPU_LABELS):
        return False
    return all(
        len(matrix[label]) == 8
        and matrix[label][column] == ("X" if label == GPU_LABELS[column] else "NV18")
        for label in GPU_LABELS
        for column in range(8)
    )
