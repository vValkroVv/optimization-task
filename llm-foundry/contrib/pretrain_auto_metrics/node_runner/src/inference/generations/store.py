from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple, Optional


def request_key(*, profile: str, path: str, body: dict[str, Any]) -> str:
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    payload = f"{profile}\0{path}\0{canonical}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


SHARD_BUFFER_BYTES = 1 << 20


class ShardWriter:
    def __init__(self, dump_dir: Path, profile: str) -> None:
        self._profile = profile
        dump_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(
            dump_dir / f"{profile}.jsonl",
            "a",
            encoding="utf-8",
            buffering=SHARD_BUFFER_BYTES,
        )

    def write(
        self,
        *,
        path: str,
        body: dict[str, Any],
        status_code: int,
        response: str,
    ) -> None:
        record = {
            "key": request_key(profile=self._profile, path=path, body=body),
            "profile": self._profile,
            "path": path,
            "request": body,
            "status": status_code,
            "response": response,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._file.close()


class RecordLocation(NamedTuple):
    shard: Path
    offset: int
    length: int


class DumpIndex:
    def __init__(self, dump_dir: Path, profile: str) -> None:
        self.profile = profile
        self._locations: dict[str, RecordLocation] = {}
        shards = sorted(dump_dir.glob(f"{profile}*.jsonl"))
        if not shards:
            raise ValueError(
                f"No {profile}*.jsonl shards under {dump_dir}: replay-generations-dir must "
                "point at the dump directory itself, not at a single shard and not "
                "at a directory of dumps."
            )
        for shard in shards:
            self._index_shard(shard)

    def __len__(self) -> int:
        return len(self._locations)

    def _index_shard(self, shard: Path) -> None:
        offset = 0
        with open(shard, "rb") as handle:
            for line in handle:
                record = self._parse(line, shard, offset)
                if record is not None:
                    self._locations.setdefault(
                        record["key"], RecordLocation(shard, offset, len(line))
                    )
                offset += len(line)

    @staticmethod
    def _parse(line: bytes, shard: Path, offset: int) -> Optional[dict[str, Any]]:
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            print(
                f"Skipping unreadable record in {shard} at offset {offset}: "
                f"{len(line)} bytes that are not JSON ({exc}). A recording job "
                "killed mid-write leaves its last line truncated.",
                flush=True,
            )
            return None

    def lookup(self, *, path: str, body: dict[str, Any]) -> Optional[dict[str, Any]]:
        key = request_key(profile=self.profile, path=path, body=body)
        location = self._locations.get(key)
        if location is None:
            return None
        with open(location.shard, "rb") as handle:
            handle.seek(location.offset)
            return json.loads(handle.read(location.length))
