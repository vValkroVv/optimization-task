import base64
import gzip
import io
import json
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

import zstandard as zstd

from ...logger import Logger

OBM_ENCODED_PREFIX = "obm_enc_"


@dataclass(frozen=True)
class CacheScoresSpec:
    raw_fields: Sequence[str] = ()
    output_filename: str = "cache_scores.jsonl.zst"
    merge_keys: Sequence[str] = ()


def project_row(row: dict, raw_fields: Sequence[str], derived: dict) -> dict:
    record = dict(derived)
    for field_name in raw_fields:
        if field_name in row:
            record[field_name] = row[field_name]
    return record


def merge_key(fields: Sequence[str], record: dict) -> Optional[tuple]:
    values = tuple(record.get(name) for name in fields)
    return values if all(value is not None for value in values) else None


def _iter_json_lines(lines: Iterable[str], source: object) -> Iterator[dict]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            Logger.warning(f"Skipping unparsable line in {source}")


def decoded_name(path: Path) -> str:
    if not path.name.startswith(OBM_ENCODED_PREFIX):
        return path.name
    return base64.urlsafe_b64decode(path.name[len(OBM_ENCODED_PREFIX) :]).decode()


def encoded_path(path: Path) -> Path:
    name = base64.urlsafe_b64encode(path.name.encode()).decode()
    return path.with_name(f"{OBM_ENCODED_PREFIX}{name}")


def iter_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.name.startswith(OBM_ENCODED_PREFIX) else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:
            yield from _iter_json_lines(f, path)
    except (EOFError, gzip.BadGzipFile, zlib.error) as e:
        Logger.warning(f"Truncated gzip tail in {path.name}: {e}")


def iter_compressed_jsonl_bytes(data: bytes) -> Iterator[dict]:
    reader = zstd.ZstdDecompressor().stream_reader(io.BytesIO(data))
    text = io.TextIOWrapper(reader, encoding="utf-8")
    yield from _iter_json_lines(text, "compressed scores stream")


class CacheScoresBuilder(ABC):
    cache_subdir: Optional[str] = None

    def __init__(
        self,
        metrics_dir: Path,
        spec: CacheScoresSpec,
        run_files: Optional[set[Path]] = None,
    ):
        self.metrics_dir = Path(metrics_dir)
        self.spec = spec
        self.run_files = run_files
        self.existing_rows_loader: Optional[Callable[[], Iterable[dict]]] = None
        # Merge keys this build replaces; rows of the remote file under them are stale.
        self.rebuilt_keys: set[tuple] = set()

    def _glob(self, pattern: str) -> list[Path]:
        paths = sorted(self.metrics_dir.glob(pattern))
        if self.run_files is None:
            return paths
        return [path for path in paths if path in self.run_files]

    @abstractmethod
    def iter_source_files(self) -> Iterable[Path]: ...

    @abstractmethod
    def derived_fields(self, source_file: Path) -> dict: ...

    def existing_rows(self) -> Iterable[dict]:
        if self.existing_rows_loader is None:
            return ()
        return self.existing_rows_loader()

    def iter_records(self, row: dict, derived: dict) -> Iterator[dict]:
        yield project_row(row, self.spec.raw_fields, derived)

    def covered_keys(self) -> Iterable[tuple]:
        return ()

    def build(self, out_dir: Path) -> Optional[Path]:
        out_dir = Path(out_dir)
        out_path = out_dir / self.spec.output_filename
        sources = list(self.iter_source_files())
        self.rebuilt_keys = set(self.covered_keys())
        if not sources and not self.rebuilt_keys:
            Logger.warning(f"No cache source files for {type(self).__name__}; skipping")
            return None

        part_path = out_path.with_name(f"{out_path.name}.part")
        writer = zstd.ZstdCompressor(level=10).stream_writer(open(part_path, "wb"))
        Logger.info(f"Building {out_path.name} from {len(sources)} cache file(s)")

        n_rows = 0
        merge_keys = self.spec.merge_keys
        try:
            for source_file in sources:
                derived = self.derived_fields(source_file)
                if merge_keys:
                    key = merge_key(merge_keys, derived)
                    if key is not None:
                        self.rebuilt_keys.add(key)
                for row in iter_jsonl(source_file):
                    for record in self.iter_records(row, derived):
                        writer.write(
                            (json.dumps(record, ensure_ascii=False) + "\n").encode(
                                "utf-8"
                            )
                        )
                        n_rows += 1
            n_rows += self._merge_existing(writer, self.rebuilt_keys)
        except Exception as e:
            Logger.error(f"Failed to build {out_path.name}: {e}")
            writer.close()
            part_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
            return None
        writer.close()

        part_path.replace(out_path)
        Logger.info(f"Wrote {n_rows} scorable row(s) to {out_path}")
        return out_path

    def _merge_existing(self, writer, local_keys: set) -> int:
        merge_keys = self.spec.merge_keys
        if not merge_keys:
            return 0
        n_merged = 0
        for row in self.existing_rows():
            if merge_key(merge_keys, row) in local_keys:
                continue
            writer.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
            n_merged += 1
        if n_merged:
            Logger.info(
                f"Merged {n_merged} row(s) from existing scores "
                f"({len(local_keys)} group(s) rebuilt locally)"
            )
        return n_merged
