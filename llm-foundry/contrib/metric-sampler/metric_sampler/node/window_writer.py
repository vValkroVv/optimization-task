from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa

from metric_sampler.contract.catalog import metric_by_id
from metric_sampler.contract.metrics import SCHEMA_VERSION, Family
from metric_sampler.contract.raw_layout import (
    MANIFEST_NAME,
    AttemptDir,
    RawRoot,
    Window,
    family_file_name,
    window_name,
    write_atomically,
)
from metric_sampler.node.parquet_files import VALUES_LEAF_COLUMN, write_parquet_atomically
from metric_sampler.node.spool import Spool
from metric_sampler.node.step_merger import Row
from metric_sampler.node.window_counters import WindowCounters

_PARQUET_COMPRESSION_LEVEL = 3

_BASE_FIELDS = [
    # Plain string, not an Arrow dictionary: Parquet already dictionary-encodes repeated
    # strings, so an explicit dictionary type only adds schema overhead on top.
    ("node", pa.string()),
    ("step", pa.uint32()),
    ("rank", pa.int16()),
    ("microbatch", pa.int16()),
    ("layer", pa.int16()),
    ("metric_id", pa.uint16()),
    # 0 for an ordinary row; for a summary/* row (`StepMerger._summary_rows`) the source EXPERT_COUNTS
    # metric id, so summaries of different sources don't collide under deduplication.
    ("source_metric_id", pa.uint16()),
    ("contributors", pa.uint8()),
    ("expected_contributors", pa.uint8()),
    ("identical", pa.bool_()),
    ("verbose", pa.bool_()),
]
_EXPERT_COUNTS_SCHEMA = pa.schema(
    [
        *_BASE_FIELDS,
        # Non-zero when the row repeats another metric's row of the same rank (MetricSpec.alias_of) and
        # keeps no values of its own.
        ("alias_metric_id", pa.uint16()),
        ("values", pa.list_(pa.uint32())),
    ]
)
_FLOAT_SCHEMA = pa.schema([*_BASE_FIELDS, ("values", pa.list_(pa.float32()))])

_FAMILY_FILE_NAMES = {family: family_file_name(family.value) for family in Family}


class WindowWriter:
    """Buffers the settled steps and writes them as windows: one Parquet file per metric family plus a manifest.
    The manifest is first made durable as the spool's publish intent, so a stopped node process can finish the
    publication."""

    def __init__(
        self,
        node: str,
        node_directory: Path,
        attempt_dir: AttemptDir,
        raw_root: RawRoot,
        nodes_in_job: int,
        ranks: list[int],
        absent_ranks: list[int],
        capture_interval: int,
        spool: Spool,
        counters: WindowCounters,
    ) -> None:
        self._node = node
        self._node_dir = node_directory
        self._attempt_dir = attempt_dir
        self._raw_root = raw_root
        self._nodes_in_job = nodes_in_job
        self._ranks = ranks
        self._absent_ranks = absent_ranks
        self._capture_interval = capture_interval
        self._spool = spool
        self._counters = counters
        self._buffered_steps: list[int] = []
        self._rows_by_step: dict[int, list[Row]] = {}
        self._missing_by_step: dict[int, list[int]] = {}

    @property
    def buffered_step_count(self) -> int:
        return len(self._buffered_steps)

    def add_step(self, step: int, rows: list[Row], missing_ranks: list[int]) -> None:
        self._missing_by_step[step] = missing_ranks
        self._rows_by_step[step] = rows
        self._buffered_steps.append(step)

    def write_window(self, count: int) -> list[Path]:
        if not self._spool.has_intent():
            self._write_window_files(count)
        steps, written = self.finish_publication()
        self._forget_published_steps(steps)
        self._counters.reset()
        return written

    def finish_publication(self) -> tuple[list[int], list[Path]]:
        """Publishes the window the spool holds an intent for, unless it is published already, and clears the
        intent."""
        content = self._spool.read_intent()
        manifest = json.loads(content)
        steps = manifest["steps"]
        written = []
        if not set(steps) <= self._spool.committed_steps:
            file_names = [_FAMILY_FILE_NAMES[Family(family)] for family in manifest["rows_per_family"]]
            published_dir = self._publish_manifest(steps[0], steps[-1], content, file_names)
            written = [published_dir / name for name in [*file_names, MANIFEST_NAME]]
            self._spool.commit(steps)
        self._spool.clear_intent()
        return steps, written

    def _write_window_files(self, count: int) -> None:
        """Writes the family files of the oldest `count` buffered steps, then the publication intent, which is the
        manifest of the window."""
        steps = sorted(self._buffered_steps)[:count]
        rows: list[Row] = []
        missing_ranks: dict[int, list[int]] = {}
        for step in steps:
            rows.extend(self._rows_by_step[step])
            missing = self._missing_by_step[step]
            if missing:
                missing_ranks[step] = missing
        rows.sort(key=lambda row: (row.step, row.metric_id, row.source_metric_id, row.layer, row.microbatch, row.rank))

        rows_by_family: dict[Family, list[Row]] = {}
        # Looked up once per metric: hashing a Family is a Python call.
        family_rows_by_metric: dict[int, list[Row]] = {}
        for row in rows:
            family_rows = family_rows_by_metric.get(row.metric_id)
            if family_rows is None:
                family = metric_by_id(row.metric_id).family
                family_rows = family_rows_by_metric[row.metric_id] = rows_by_family.setdefault(family, [])
            family_rows.append(row)

        window_dir = self._window_dir(steps[0], steps[-1])
        window_dir.mkdir(parents=True, exist_ok=True)
        rows_per_family = {}
        for family, family_rows in rows_by_family.items():
            self._write_parquet(window_dir / _FAMILY_FILE_NAMES[family], family, family_rows)
            rows_per_family[family.value] = len(family_rows)
        # Freeze publication before the collector can see or move the window.
        self._write_manifest(steps, rows_per_family, missing_ranks)

    def _forget_published_steps(self, steps: list[int]) -> None:
        step_set = set(steps)
        self._buffered_steps = [step for step in self._buffered_steps if step not in step_set]
        for step in steps:
            del self._rows_by_step[step]
            del self._missing_by_step[step]
        self._spool.discard_slots(steps)

    def _publish_manifest(self, first_step: int, last_step: int, content: bytes, file_names: list[str]) -> Path:
        """The directory holding the window's published manifest: where the collector or an earlier attempt already
        published it, else the window directory, where it is published now."""
        window_dir = self._window_dir(first_step, last_step)
        collected_dir = self._attempt_dir.collected_path(
            self._raw_root,
            Window(path=window_dir, node=self._node, first_step=first_step, last_step=last_step),
        )
        for directory in (window_dir, collected_dir):
            try:
                published = (directory / MANIFEST_NAME).read_bytes()
            except FileNotFoundError:
                continue
            if published != content:
                raise ValueError(f"published window differs from its durable intent: {directory}")
            return directory
        for name in file_names:
            # The manifest makes the window visible: every family file it names must exist first.
            (window_dir / name).stat()
        write_atomically(window_dir / MANIFEST_NAME, content)
        return window_dir

    def _write_parquet(self, path: Path, family: Family, rows: list[Row]) -> None:
        columns = {
            "node": [self._node] * len(rows),
            "step": [row.step for row in rows],
            "rank": [row.rank for row in rows],
            "microbatch": [row.microbatch for row in rows],
            "layer": [row.layer for row in rows],
            "metric_id": [row.metric_id for row in rows],
            "source_metric_id": [row.source_metric_id for row in rows],
            "contributors": [row.contributors for row in rows],
            "expected_contributors": [row.expected_contributors for row in rows],
            "identical": [row.identical for row in rows],
            "verbose": [row.verbose for row in rows],
        }
        schema = _FLOAT_SCHEMA
        if family is Family.EXPERT_COUNTS:
            columns["alias_metric_id"] = [row.alias_metric_id for row in rows]
            schema = _EXPERT_COUNTS_SCHEMA
        columns["values"] = self._values_array(family, rows)
        table = pa.table(columns, schema=schema)
        # Dictionary encoding is off, since many distinct values defeat it; float values are split into byte streams,
        # which compress better.
        write_options: dict[str, object] = {"compression_level": _PARQUET_COMPRESSION_LEVEL, "use_dictionary": False}
        if family is not Family.EXPERT_COUNTS:
            write_options["use_byte_stream_split"] = [VALUES_LEAF_COLUMN]
        write_parquet_atomically(path, table, **write_options)

    @staticmethod
    def _values_array(family: Family, rows: list[Row]) -> pa.Array:
        # Built from each row's numpy array directly: tolist() would convert every element to a Python object, which
        # dominates _write_parquet for wide vectors.
        value_dtype = np.uint32 if family is Family.EXPERT_COUNTS else np.float32
        offsets = np.zeros(len(rows) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum([row.values.size for row in rows])
        flat_values = np.concatenate([row.values for row in rows]) if rows else np.empty(0, dtype=value_dtype)
        return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), pa.array(flat_values))

    def _write_manifest(
        self, steps: list[int], rows_per_family: dict[str, int], missing_ranks: dict[int, list[int]]
    ) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "attempt": self._attempt_dir.attempt,
            "node": self._node,
            "nodes_in_job": self._nodes_in_job,
            "ranks": self._ranks,
            "absent_ranks": self._absent_ranks,
            "first_step": steps[0],
            "last_step": steps[-1],
            "steps": steps,
            "capture_interval": self._capture_interval,
            "rows_per_family": rows_per_family,
            "missing_ranks": {str(step): ranks for step, ranks in missing_ranks.items()},
            "dropped_steps_per_rank": dict(self._counters.dropped_steps_per_rank),
            "torn_slots_per_rank": dict(self._counters.torn_slots_per_rank),
            "late_slots_per_rank": dict(self._counters.late_slots_per_rank),
            "identical_violations": self._counters.identical_violations,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        self._spool.write_intent(json.dumps(manifest, indent=2).encode("utf-8"))

    def _window_dir(self, first_step: int, last_step: int) -> Path:
        return self._node_dir / window_name(first_step, last_step)
