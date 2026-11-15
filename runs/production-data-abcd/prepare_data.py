#!/usr/bin/env python3
"""Prepare the bounded four-domain GigaSets MDS stream for CPU validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import streaming
from streaming import MDSWriter
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from data_contract import (
    DATA_RECEIPT,
    DATASET_ID,
    DATASET_ROOT,
    DOMAIN_ORDER,
    MDS_SHARD_SIZE_LIMIT,
    MDS_SPLIT,
    NUM_ROWS,
    PREPARATION_STATUS,
    ROW_MANIFEST,
    ROWS_PER_DOMAIN,
    SEQ_LEN,
    SOURCE_RECIPES,
    SOURCE_ROOT,
    TASK_ROOT,
    TOKENIZER_DIR,
    TOKENIZER_FILE_SHA256,
    TOKEN_DTYPE_CODE,
    TOKEN_DTYPE_NAME,
    VOCAB_SIZE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_candidates() -> dict[str, list[Path]]:
    candidates: dict[str, list[Path]] = {}
    for domain, recipe in SOURCE_RECIPES.items():
        if "files" in recipe:
            files = [SOURCE_ROOT / relative for relative in recipe["files"]]
        else:
            relative_directories = recipe.get("directories")
            if relative_directories is None:
                relative_directories = (recipe["directory"],)
            files = []
            for relative_directory in relative_directories:
                directory = SOURCE_ROOT / relative_directory
                if not directory.is_dir():
                    raise FileNotFoundError(
                        f"Source directory does not exist: {directory}"
                    )
                files.extend(
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.name.endswith(recipe["suffix"])
                )
            files.sort()
        if not files:
            raise FileNotFoundError(f"No source shards found for domain {domain}")
        max_files = recipe["max_files"]
        if max_files is not None:
            files = files[:max_files]
        for path in files:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(SOURCE_ROOT.resolve()):
                raise RuntimeError(f"Source shard escapes the pinned corpus root: {path}")
            if not resolved.is_file():
                raise FileNotFoundError(f"Source shard is not a file: {path}")
        candidates[domain] = files
    if tuple(candidates) != DOMAIN_ORDER:
        raise AssertionError("Source recipe order differs from the data contract")
    return candidates


def tokenizer_contract() -> tuple[HuggingFaceTokenizer, dict[str, Any]]:
    actual_hashes = {
        name: sha256_file(TOKENIZER_DIR / name) for name in TOKENIZER_FILE_SHA256
    }
    if actual_hashes != TOKENIZER_FILE_SHA256:
        raise RuntimeError(
            f"Canonical tokenizer hashes changed: {actual_hashes!r}"
        )

    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(TOKENIZER_DIR))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise RuntimeError(
            f"Tokenizer vocabulary is {tokenizer.get_vocab_size()}, expected {VOCAB_SIZE}"
        )
    special_ids = {
        "bos": tokenizer.bos_id,
        "eos": tokenizer.eos_id,
        "unk": tokenizer.token_to_id("<unk>"),
        "fim_prefix": tokenizer.token_to_id("<|fim_prefix|>"),
        "fim_middle": tokenizer.token_to_id("<|fim_middle|>"),
        "fim_suffix": tokenizer.token_to_id("<|fim_suffix|>"),
    }
    expected_special_ids = {
        "bos": 1,
        "eos": 2,
        "unk": 0,
        "fim_prefix": 128253,
        "fim_middle": 128254,
        "fim_suffix": 128255,
    }
    if special_ids != expected_special_ids:
        raise RuntimeError(f"Canonical tokenizer special IDs changed: {special_ids!r}")
    empty_encoded = tokenizer.encode("", add_bos=True, add_eos=True)
    if empty_encoded != [tokenizer.bos_id, tokenizer.eos_id]:
        raise RuntimeError(f"Unexpected wrapper BOS/EOS behavior: {empty_encoded!r}")

    tokenizer_source = Path(inspect.getsourcefile(HuggingFaceTokenizer) or "").resolve()
    metadata = {
        "path": str(TOKENIZER_DIR.resolve()),
        "files": actual_hashes,
        "vocab_size": tokenizer.get_vocab_size(),
        "special_token_ids": special_ids,
        "wrapper": {
            "source_path": str(tokenizer_source),
            "source_sha256": sha256_file(tokenizer_source),
            "hf_adds_bos": bool(tokenizer.hf_adds_bos),
            "hf_adds_eos": bool(tokenizer.hf_adds_eos),
            "empty_text_with_requested_bos_eos": empty_encoded,
            "empty_text_underlying_ids": tokenizer.tokenizer.encode("").ids,
        },
    }
    return tokenizer, metadata


class TokenPacker:
    """Join document token sequences and emit nonoverlapping full rows."""

    def __init__(self, seq_len: int = SEQ_LEN) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        self.seq_len = seq_len
        self._tokens: list[int] = []
        self._offset = 0

    def append(self, token_ids: list[int]) -> Iterator[np.ndarray]:
        self._tokens.extend(token_ids)
        while len(self._tokens) - self._offset >= self.seq_len:
            end = self._offset + self.seq_len
            row = np.asarray(self._tokens[self._offset : end], dtype="<u4")
            self._offset = end
            if self._offset == len(self._tokens):
                self._tokens.clear()
                self._offset = 0
            elif self._offset >= 16 * self.seq_len:
                del self._tokens[: self._offset]
                self._offset = 0
            yield row

    @property
    def pending_tokens(self) -> int:
        return len(self._tokens) - self._offset


def make_mds_sample(row: np.ndarray) -> dict[str, Any]:
    tokens = np.ascontiguousarray(row, dtype="<u4")
    if tokens.shape != (SEQ_LEN,):
        raise ValueError(f"MDS rows must have shape ({SEQ_LEN},), got {tokens.shape}")
    if int(tokens.min()) < 0 or int(tokens.max()) >= VOCAB_SIZE:
        raise ValueError("MDS row contains a token outside the canonical vocabulary")
    return {"tokens": tokens.tobytes(order="C"), "dtype": TOKEN_DTYPE_CODE}


class DomainRowStream:
    def __init__(
        self,
        *,
        domain: str,
        files: list[Path],
        tokenizer: HuggingFaceTokenizer,
        seen_text_hashes: set[str],
    ) -> None:
        self.domain = domain
        self.files = files
        self.tokenizer = tokenizer
        self.seen_text_hashes = seen_text_hashes
        self.packer = TokenPacker()
        self.text_hash_chain = hashlib.sha256()
        self.duplicate_text_hash_chain = hashlib.sha256()
        self.stats: dict[str, Any] = {
            "candidate_file_count": len(files),
            "source_files": [],
            "documents_consumed": 0,
            "documents_encoded": 0,
            "duplicate_documents_skipped": 0,
            "encoded_token_count": 0,
            "token_min": None,
            "token_max": None,
            "unique_rows_written": 0,
            "duplicate_rows_skipped": 0,
            "duplicate_row_hash_chain_sha256": hashlib.sha256().hexdigest(),
            "pending_tokens_not_written": 0,
            "source_exhausted": False,
        }
        self.duplicate_row_hash_chain = hashlib.sha256()

    def iter_rows(self) -> Iterator[np.ndarray]:
        try:
            for path in self.files:
                before = path.stat()
                source_hash = sha256_file(path)
                after_hash = path.stat()
                if (before.st_size, before.st_mtime_ns) != (
                    after_hash.st_size,
                    after_hash.st_mtime_ns,
                ):
                    raise RuntimeError(f"Source shard changed while hashing: {path}")
                record = {
                    "path": str(path.resolve()),
                    "path_relative_to_gigasets_root": str(
                        path.resolve().relative_to(SOURCE_ROOT.resolve())
                    ),
                    "bytes": before.st_size,
                    "mtime_ns": before.st_mtime_ns,
                    "sha256": source_hash,
                    "lines_consumed": 0,
                    "documents_encoded": 0,
                    "duplicate_documents_skipped": 0,
                }
                self.stats["source_files"].append(record)
                raw_stream = pa.input_stream(str(path), compression="zstd")
                with io.TextIOWrapper(raw_stream, encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        record["lines_consumed"] = line_number
                        self.stats["documents_consumed"] += 1
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Invalid JSON at {record['path_relative_to_gigasets_root']}:{line_number}"
                            ) from exc
                        text = item.get("text") if isinstance(item, dict) else None
                        if not isinstance(text, str):
                            raise TypeError(
                                f"Expected text:str at "
                                f"{record['path_relative_to_gigasets_root']}:{line_number}"
                            )
                        text_digest = hashlib.sha256(text.encode("utf-8")).digest()
                        text_hash = text_digest.hex()
                        self.text_hash_chain.update(text_digest)
                        if text_hash in self.seen_text_hashes:
                            self.stats["duplicate_documents_skipped"] += 1
                            record["duplicate_documents_skipped"] += 1
                            self.duplicate_text_hash_chain.update(text_digest)
                            continue
                        self.seen_text_hashes.add(text_hash)
                        token_ids = self.tokenizer.encode(
                            text, add_bos=True, add_eos=True
                        )
                        if not token_ids:
                            raise RuntimeError(
                                "The tokenizer returned no tokens for a document"
                            )
                        token_min = min(token_ids)
                        token_max = max(token_ids)
                        if token_min < 0 or token_max >= VOCAB_SIZE:
                            raise ValueError(
                                f"Token IDs [{token_min}, {token_max}] exceed "
                                f"the vocabulary bounds [0, {VOCAB_SIZE})"
                            )
                        self.stats["token_min"] = (
                            token_min
                            if self.stats["token_min"] is None
                            else min(self.stats["token_min"], token_min)
                        )
                        self.stats["token_max"] = (
                            token_max
                            if self.stats["token_max"] is None
                            else max(self.stats["token_max"], token_max)
                        )
                        self.stats["documents_encoded"] += 1
                        self.stats["encoded_token_count"] += len(token_ids)
                        record["documents_encoded"] += 1
                        for row in self.packer.append(token_ids):
                            yield row
                final_stat = path.stat()
                if (before.st_size, before.st_mtime_ns) != (
                    final_stat.st_size,
                    final_stat.st_mtime_ns,
                ):
                    raise RuntimeError(f"Source shard changed while reading: {path}")
            self.stats["source_exhausted"] = True
        finally:
            self.stats["pending_tokens_not_written"] = self.packer.pending_tokens
            self.stats["text_hash_sequence_sha256"] = self.text_hash_chain.hexdigest()
            self.stats[
                "duplicate_text_hash_sequence_sha256"
            ] = self.duplicate_text_hash_chain.hexdigest()
            self.stats[
                "duplicate_row_hash_chain_sha256"
            ] = self.duplicate_row_hash_chain.hexdigest()


def _runtime_receipt() -> dict[str, Any]:
    from streaming import MDSWriter as InstalledMDSWriter

    writer_source = Path(inspect.getsourcefile(InstalledMDSWriter) or "").resolve()
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "mosaicml_streaming": importlib.metadata.version("mosaicml-streaming"),
        "streaming_package_path": str(Path(streaming.__file__).resolve()),
        "mds_writer_source_path": str(Path(inspect.getfile(InstalledMDSWriter)).resolve()),
        "mds_writer_source_sha256": sha256_file(writer_source),
        "pyarrow_module_path": str(Path(pa.__file__).resolve()),
        "little_endian_host": sys.byteorder == "little",
        "task_git_head": subprocess.run(
            ["git", "-C", str(TASK_ROOT), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
    }


def _check_import_identity() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required, got {platform.python_version()}")
    expected_prefix = Path("/opt/torchtitan-venv").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError(f"Unexpected Python environment: {sys.prefix}")
    if importlib.metadata.version("mosaicml-streaming") != "0.13.0":
        raise RuntimeError("Preparation requires mosaicml-streaming 0.13.0")
    streaming_path = Path(streaming.__file__).resolve()
    site_packages = expected_prefix / "lib/python3.12/site-packages"
    if not streaming_path.is_relative_to(site_packages):
        raise RuntimeError(f"Streaming must use the installed package, got {streaming_path}")
    if streaming_path.is_relative_to((TASK_ROOT / "gift/contrib/streaming").resolve()):
        raise RuntimeError(f"Bundled Streaming import selected: {streaming_path}")


def _index_and_dataset_receipt() -> dict[str, Any]:
    index_path = MDS_SPLIT / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if sum(int(shard["samples"]) for shard in index["shards"]) != NUM_ROWS:
        raise RuntimeError("MDS index sample counts do not sum to the requested rows")

    files: list[dict[str, Any]] = []
    for path in sorted(file for file in MDS_SPLIT.iterdir() if file.is_file()):
        relative = path.relative_to(DATASET_ROOT)
        actual_hash = sha256_file(path)
        size = path.stat().st_size
        expected_hash: str | None = None
        if path.name != "index.json":
            shard = next(
                (
                    item
                    for item in index["shards"]
                    if item["raw_data"]["basename"] == path.name
                ),
                None,
            )
            if shard is None:
                raise RuntimeError(f"Unexpected MDS output file: {path.name}")
            expected_hash = shard["raw_data"].get("hashes", {}).get("sha256")
            if expected_hash != actual_hash:
                raise RuntimeError(
                    f"MDS writer hash mismatch for {path.name}: "
                    f"{expected_hash!r} != {actual_hash}"
                )
        files.append(
            {
                "path": str(relative),
                "bytes": size,
                "sha256": actual_hash,
                "writer_sha256": expected_hash,
            }
        )

    index_hash = next(item["sha256"] for item in files if item["path"] == "train/index.json")
    dataset_digest = hashlib.sha256()
    for item in files:
        dataset_digest.update(item["path"].encode("utf-8") + b"\0")
        dataset_digest.update(str(item["bytes"]).encode("ascii") + b"\0")
        dataset_digest.update(bytes.fromhex(item["sha256"]))
    return {
        "index_sha256": index_hash,
        "dataset_sha256": dataset_digest.hexdigest(),
        "files": files,
        "shard_count": len(index["shards"]),
        "row_count": sum(int(shard["samples"]) for shard in index["shards"]),
        "output_bytes": sum(item["bytes"] for item in files),
        "compression": None,
        "size_limit_bytes": MDS_SHARD_SIZE_LIMIT,
    }


def prepare() -> dict[str, Any]:
    output_paths = (DATASET_ROOT, DATA_RECEIPT, ROW_MANIFEST)
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite task artifacts: " + ", ".join(existing)
        )
    if sys.byteorder != "little":
        raise RuntimeError("This MDS contract requires a little-endian host")
    started = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    status: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "status": "preparing",
        "started_utc": start_utc,
        "output_dataset_root": str(DATASET_ROOT),
        "rows_target": NUM_ROWS,
        "rows_written": 0,
        "error": None,
    }
    write_json_atomic(PREPARATION_STATUS, status)

    streams: dict[str, DomainRowStream] = {}
    iterators: dict[str, Iterator[np.ndarray]] = {}
    seen_text_hashes: set[str] = set()
    seen_row_hashes: set[str] = set()
    row_hash_chain = hashlib.sha256()
    domain_token_min: int | None = None
    domain_token_max: int | None = None
    try:
        _check_import_identity()
        tokenizer, tokenizer_metadata = tokenizer_contract()
        candidates = source_candidates()
        selection = {
            domain: {
                "candidate_file_count": len(files),
                "files": [
                    str(path.resolve().relative_to(SOURCE_ROOT.resolve()))
                    for path in files
                ],
            }
            for domain, files in candidates.items()
        }
        streams = {
            domain: DomainRowStream(
                domain=domain,
                files=candidates[domain],
                tokenizer=tokenizer,
                seen_text_hashes=seen_text_hashes,
            )
            for domain in DOMAIN_ORDER
        }
        iterators = {domain: stream.iter_rows() for domain, stream in streams.items()}

        DATASET_ROOT.mkdir(parents=True)
        ROW_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        row_manifest_hash = hashlib.sha256()
        row_manifest_count = 0
        with ROW_MANIFEST.open("x", encoding="utf-8") as row_handle:
            with MDSWriter(
                out=str(MDS_SPLIT),
                columns={"tokens": "bytes", "dtype": "int8"},
                compression=None,
                hashes=["sha256"],
                size_limit=MDS_SHARD_SIZE_LIMIT,
            ) as writer:
                for cycle in range(ROWS_PER_DOMAIN):
                    for domain in DOMAIN_ORDER:
                        stream = streams[domain]
                        while True:
                            try:
                                row = next(iterators[domain])
                            except StopIteration as exc:
                                raise RuntimeError(
                                    f"Domain {domain} exhausted after "
                                    f"{stream.stats['unique_rows_written']} unique rows; "
                                    f"needed {ROWS_PER_DOMAIN}"
                                ) from exc
                            row = np.ascontiguousarray(row, dtype="<u4")
                            if row.shape != (SEQ_LEN,):
                                raise RuntimeError(
                                    f"Domain {domain} emitted row shape {row.shape}"
                                )
                            digest = hashlib.sha256(row.tobytes()).digest()
                            row_hash = digest.hex()
                            if row_hash in seen_row_hashes:
                                stream.stats["duplicate_rows_skipped"] += 1
                                stream.duplicate_row_hash_chain.update(digest)
                                stream.stats["duplicate_row_hash_chain_sha256"] = (
                                    stream.duplicate_row_hash_chain.hexdigest()
                                )
                                continue
                            seen_row_hashes.add(row_hash)
                            break

                        row_min = int(row.min())
                        row_max = int(row.max())
                        if row_min < 0 or row_max >= VOCAB_SIZE:
                            raise RuntimeError(
                                f"Domain {domain} row has token range "
                                f"[{row_min}, {row_max}] outside [0, {VOCAB_SIZE})"
                            )
                        domain_token_min = (
                            row_min
                            if domain_token_min is None
                            else min(domain_token_min, row_min)
                        )
                        domain_token_max = (
                            row_max
                            if domain_token_max is None
                            else max(domain_token_max, row_max)
                        )
                        sample_index = row_manifest_count
                        entry = {
                            "index": sample_index,
                            "domain": domain,
                            "sha256": row_hash,
                        }
                        encoded_entry = (
                            json.dumps(entry, separators=(",", ":"), sort_keys=True)
                            + "\n"
                        ).encode("utf-8")
                        writer.write(make_mds_sample(row))
                        row_handle.write(encoded_entry.decode("utf-8"))
                        row_manifest_hash.update(encoded_entry)
                        row_hash_chain.update(digest)
                        row_manifest_count += 1
                        stream.stats["unique_rows_written"] += 1
                        status["rows_written"] = row_manifest_count

                    if (cycle + 1) % 1024 == 0:
                        status["domains"] = {
                            domain: stream.stats for domain, stream in streams.items()
                        }
                        status["elapsed_seconds"] = round(time.monotonic() - started, 2)
                        write_json_atomic(PREPARATION_STATUS, status)
                        print(
                            json.dumps(
                                {
                                    "event": "packing_progress",
                                    "rows_written": row_manifest_count,
                                    "rows_target": NUM_ROWS,
                                    "elapsed_seconds": status["elapsed_seconds"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            row_handle.flush()
            os.fsync(row_handle.fileno())

        for iterator in iterators.values():
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

        if row_manifest_count != NUM_ROWS:
            raise RuntimeError(
                f"Wrote {row_manifest_count} rows, expected {NUM_ROWS}"
            )
        if len(seen_row_hashes) != NUM_ROWS:
            raise RuntimeError("Output row hash set contains a duplicate")
        if sum(stream.stats["unique_rows_written"] for stream in streams.values()) != NUM_ROWS:
            raise RuntimeError("Domain row counts do not sum to the requested corpus size")
        if any(
            stream.stats["unique_rows_written"] != ROWS_PER_DOMAIN
            for stream in streams.values()
        ):
            raise RuntimeError("Per-domain row counts differ from the requested equal mix")

        index_receipt = _index_and_dataset_receipt()
        if index_receipt["row_count"] != NUM_ROWS:
            raise RuntimeError("MDS writer did not persist every requested row")
        if sha256_file(ROW_MANIFEST) != row_manifest_hash.hexdigest():
            raise RuntimeError("Row manifest hash changed after writing")

        source_manifest_hash = hashlib.sha256()
        for domain in DOMAIN_ORDER:
            for source_file in streams[domain].stats["source_files"]:
                source_manifest_hash.update(
                    source_file["path_relative_to_gigasets_root"].encode("utf-8")
                    + b"\0"
                )
                source_manifest_hash.update(str(source_file["bytes"]).encode("ascii") + b"\0")
                source_manifest_hash.update(bytes.fromhex(source_file["sha256"]))

        receipt = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "status": "prepared",
            "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "task_root": str(TASK_ROOT),
            "source": {
                "corpus": "GigaSets",
                "version": "5.3.1",
                "root": str(SOURCE_ROOT),
                "source_manifest_sha256": source_manifest_hash.hexdigest(),
                "selection": selection,
                "domains": {
                    domain: streams[domain].stats for domain in DOMAIN_ORDER
                },
                "duplicate_documents_skipped_total": sum(
                    stream.stats["duplicate_documents_skipped"]
                    for stream in streams.values()
                ),
                "duplicate_rows_skipped_total": sum(
                    stream.stats["duplicate_rows_skipped"]
                    for stream in streams.values()
                ),
            },
            "tokenizer": tokenizer_metadata,
            "packing": {
                "sequence_length": SEQ_LEN,
                "rows": NUM_ROWS,
                "rows_per_domain": ROWS_PER_DOMAIN,
                "domain_row_order": list(DOMAIN_ORDER),
                "interleaving": "one full row per domain in domain_row_order, repeated",
                "document_encoding": "GIFT HuggingFaceTokenizer.encode(text, add_bos=True, add_eos=True)",
                "document_boundary": "concatenate complete BOS/text/EOS token sequences before packing",
                "row_boundaries": "nonoverlapping contiguous 4096-token slices; no padding or truncation",
                "row_dtype": TOKEN_DTYPE_NAME,
                "row_byte_order": "little-endian",
                "mds_dtype_code": TOKEN_DTYPE_CODE,
                "token_id_bounds": [0, VOCAB_SIZE - 1],
                "observed_token_id_bounds": [domain_token_min, domain_token_max],
                "row_hashes_unique": True,
                "row_hash_sequence_sha256": row_hash_chain.hexdigest(),
            },
            "dataset": {
                "root": str(DATASET_ROOT),
                "split": "train",
                "rows": NUM_ROWS,
                "row_manifest": {
                    "path": str(ROW_MANIFEST),
                    "sha256": row_manifest_hash.hexdigest(),
                    "rows": row_manifest_count,
                },
                **index_receipt,
            },
            "runtime": _runtime_receipt(),
            "command": (
                "PYTHONPATH=gift "
                "/opt/torchtitan-venv/bin/python3.12 "
                "runs/overnight-streaming-perf/prepare_mds.py prepare"
            ),
            "performance_review": {
                "training_runtime_change": "none; only dataset bytes and input identity change",
                "row_payload_bytes_per_sample": SEQ_LEN * 4,
                "global_batch_input_bytes_per_update": 512 * SEQ_LEN * 4,
                "loader_workers": 2,
                "prefetch_depths_to_verify": [0, 32],
                "unmeasured_risk": "real-data NFS read/cache behavior may differ from the old tiny fixture",
            },
        }
        write_json_atomic(DATA_RECEIPT, receipt)
        status.update(
            {
                "status": "prepared",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "receipt": str(DATA_RECEIPT),
                "dataset_sha256": index_receipt["dataset_sha256"],
                "rows_written": row_manifest_count,
            }
        )
        write_json_atomic(PREPARATION_STATUS, status)
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "receipt": str(DATA_RECEIPT),
                    "dataset_root": str(DATASET_ROOT),
                    "rows": row_manifest_count,
                    "dataset_sha256": index_receipt["dataset_sha256"],
                    "elapsed_seconds": status["elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return receipt
    except BaseException as exc:
        for iterator in iterators.values():
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        status.update(
            {
                "status": "failed",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "rows_written": status.get("rows_written", 0),
                "domains": {
                    domain: stream.stats for domain, stream in streams.items()
                },
            }
        )
        write_json_atomic(PREPARATION_STATUS, status)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare()


if __name__ == "__main__":
    main()
