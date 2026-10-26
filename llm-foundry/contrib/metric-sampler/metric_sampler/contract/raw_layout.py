from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

COLLECTED_DIR_NAME = "_collected"
MANIFEST_NAME = "manifest.json"
LOGGED_DIR_NAME = "logged"
LOGGED_DATA_NAME = "logged.parquet"
DUMPS_DIR_NAME = "dumps"
# The node aggregator's working directory, and in it the window it is publishing, which the collector checks
# before deleting a collected copy of that window.
SPOOL_DIR_NAME = "_spool"
PUBLISH_INTENT_NAME = "publish.json"

# Readable by the owner's group: the collector need not run as the writer.
_GROUP_READABLE_FILE_MODE = 0o640

_ATTEMPT_DIR_PATTERN = re.compile(r"^attempt=(.+)$")
_NODE_DIR_PATTERN = re.compile(r"^node=(.+)$")
_WINDOW_DIR_PATTERN = re.compile(r"^w(\d{8})-(\d{8})$")
_DUMP_STEP_DIR_PATTERN = re.compile(r"^step=(\d{8})$")
_DUMP_FILE_PATTERN = re.compile(r"^rank=(\d{5})\.parquet$")


def attempt_dir_name(attempt: str) -> str:
    return f"attempt={attempt}"


def node_dir_name(node: str) -> str:
    return f"node={node}"


def node_dir(output_dir: Path, attempt: str, node: str) -> Path:
    """A node's directory under a run's output directory: `<output>/attempt=<A>/node=<node>`."""
    return output_dir / attempt_dir_name(attempt) / node_dir_name(node)


def window_name(first_step: int, last_step: int) -> str:
    return f"w{first_step:08d}-{last_step:08d}"


def family_file_name(family_name: str) -> str:
    return f"{family_name}.parquet"


def dump_relative_path(step: int, rank: int) -> str:
    return f"{DUMPS_DIR_NAME}/step={step:08d}/rank={rank:05d}.parquet"


def write_atomically(path: Path, data: bytes) -> None:
    """Write a sibling, fsync, rename, so readers only ever see a complete file. The sibling's
    name is unique per call: concurrent node aggregators can write the same attempt-level file
    (metrics.json, streams.json) at once, and a shared temporary name would race."""
    file_descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    directory_descriptor = None
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_name, _GROUP_READABLE_FILE_MODE)
        if os.name == "posix":  # pragma: no cover -- POSIX directory durability, unavailable on Windows.
            # The collector can move the whole window as soon as its manifest is visible.
            directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.replace(temporary_name, path)
        if directory_descriptor is not None:  # pragma: no cover -- POSIX only.
            os.fsync(directory_descriptor)
    finally:
        if directory_descriptor is not None:  # pragma: no cover -- POSIX only.
            os.close(directory_descriptor)
        Path(temporary_name).unlink(missing_ok=True)


def _complete_window_dirs(parent: Path) -> list[tuple[Path, re.Match[str]]]:
    """The window directories of `parent` that hold a manifest, each with its parsed name."""
    window_dirs = []
    for window_dir in sorted(parent.iterdir()):
        window_match = _WINDOW_DIR_PATTERN.match(window_dir.name)
        if window_match is None or not window_dir.is_dir():
            continue
        if not (window_dir / MANIFEST_NAME).exists():
            continue
        window_dirs.append((window_dir, window_match))
    return window_dirs


@dataclass(frozen=True)
class Window:
    """One complete node aggregator window: a manifest plus the family files that had rows."""

    path: Path
    node: str
    first_step: int
    last_step: int

    @property
    def name(self) -> str:
        return window_name(self.first_step, self.last_step)

    @property
    def relative_dir(self) -> str:
        return f"{node_dir_name(self.node)}/{self.name}"

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_NAME

    def family_path(self, family_name: str) -> Path:
        return self.path / family_file_name(family_name)


@dataclass(frozen=True)
class Dump:
    """One activation dump file: the sampled activations of one rank at one step."""

    path: Path
    step: int
    rank: int

    @property
    def relative_path(self) -> str:
        return dump_relative_path(self.step, self.rank)


@dataclass(frozen=True)
class LoggedWindow:
    """One complete `LoggedWriter` window: a manifest plus `logged.parquet`, with no owning node."""

    path: Path
    first_step: int
    last_step: int

    @property
    def name(self) -> str:
        return window_name(self.first_step, self.last_step)

    @property
    def relative_dir(self) -> str:
        return f"{LOGGED_DIR_NAME}/{self.name}"

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_NAME

    @property
    def data_path(self) -> Path:
        return self.path / LOGGED_DATA_NAME


@dataclass(frozen=True)
class AttemptDir:
    """`<root>/<exp>/<run>/attempt=<A>`, holding one `node=<node>` directory per node."""

    path: Path
    exp: str
    run: str
    attempt: str

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.json"

    @property
    def streams_path(self) -> Path:
        return self.path / "streams.json"

    @property
    def monitor_parameters_path(self) -> Path:
        return self.path / "monitor_parameters.json"

    @property
    def monitor_expert_ownership_path(self) -> Path:
        return self.path / "monitor_expert_ownership.json"

    def complete_windows(self) -> list[Window]:
        windows = []
        for node_dir in sorted(self.path.iterdir()):
            node_match = _NODE_DIR_PATTERN.match(node_dir.name)
            if node_match is None or not node_dir.is_dir():
                continue
            for window_dir, window_match in _complete_window_dirs(node_dir):
                windows.append(
                    Window(
                        path=window_dir,
                        node=node_match.group(1),
                        first_step=int(window_match.group(1)),
                        last_step=int(window_match.group(2)),
                    )
                )
        return windows

    def complete_logged_windows(self) -> list[LoggedWindow]:
        logged_dir = self.path / LOGGED_DIR_NAME
        if not logged_dir.is_dir():
            return []
        return [
            LoggedWindow(
                path=window_dir,
                first_step=int(window_match.group(1)),
                last_step=int(window_match.group(2)),
            )
            for window_dir, window_match in _complete_window_dirs(logged_dir)
        ]

    def dump_path(self, step: int, rank: int) -> Path:
        return self.path / dump_relative_path(step, rank)

    def complete_dumps(self) -> list[Dump]:
        dumps_dir = self.path / DUMPS_DIR_NAME
        if not dumps_dir.is_dir():
            return []
        dumps = []
        for step_dir in sorted(dumps_dir.iterdir()):
            step_match = _DUMP_STEP_DIR_PATTERN.match(step_dir.name)
            if step_match is None or not step_dir.is_dir():
                continue
            for dump_file in sorted(step_dir.iterdir()):
                file_match = _DUMP_FILE_PATTERN.match(dump_file.name)
                if file_match is None or not dump_file.is_file():
                    continue
                dumps.append(Dump(path=dump_file, step=int(step_match.group(1)), rank=int(file_match.group(1))))
        return dumps

    def collected_path(self, raw_root: RawRoot, window: Window | LoggedWindow) -> Path:
        return self._collected_attempt_path(raw_root) / window.relative_dir

    def collected_dump_path(self, raw_root: RawRoot, dump: Dump) -> Path:
        return self._collected_attempt_path(raw_root) / dump.relative_path

    def _collected_attempt_path(self, raw_root: RawRoot) -> Path:
        return raw_root.path / COLLECTED_DIR_NAME / self.exp / self.run / attempt_dir_name(self.attempt)


@dataclass(frozen=True)
class RawRoot:
    """The root directory holding one subdirectory per experiment, plus `_collected`."""

    path: Path

    def attempt_dirs(self) -> list[AttemptDir]:
        attempt_dirs = []
        for exp_dir in sorted(self.path.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name == COLLECTED_DIR_NAME:
                continue
            for run_dir in sorted(exp_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                for attempt_dir in sorted(run_dir.iterdir()):
                    attempt_match = _ATTEMPT_DIR_PATTERN.match(attempt_dir.name)
                    if attempt_match is None or not attempt_dir.is_dir():
                        continue
                    attempt_dirs.append(
                        AttemptDir(
                            path=attempt_dir,
                            exp=exp_dir.name,
                            run=run_dir.name,
                            attempt=attempt_match.group(1),
                        )
                    )
        return attempt_dirs

    def has_uncollected(self, dumps: bool) -> bool:
        """Whether any attempt holds a complete window or logged window, or with `dumps` a complete dump, that no
        collector has moved to `_collected` yet."""
        return any(
            attempt.complete_windows() or attempt.complete_logged_windows() or (dumps and attempt.complete_dumps())
            for attempt in self.attempt_dirs()
        )
