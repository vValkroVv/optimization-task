from __future__ import annotations

import sqlite3

import pytest

from sqlite_s3_query.sqlite_file import (
    SQLITE_MAGIC_HEADER,
    decode_chunked_payload,
    normalize_sqlite_file,
)


def _make_sqlite_payload(tmp_path):
    db_path = tmp_path / "seed.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO sample(value) VALUES ('ok')")
    conn.commit()
    conn.close()
    return db_path.read_bytes()


def test_decode_chunked_payload_roundtrip(tmp_path):
    payload = _make_sqlite_payload(tmp_path)
    chunked = f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n0\r\n\r\n"
    decoded = decode_chunked_payload(chunked)
    assert decoded == payload


def test_decode_chunked_payload_accepts_decimal_sizes(tmp_path):
    payload = _make_sqlite_payload(tmp_path)
    chunked = f"{len(payload)}\r\n".encode("ascii") + payload + b"\r\n0\r\n\r\n"
    decoded = decode_chunked_payload(chunked)
    assert decoded == payload


def test_decode_chunked_payload_accepts_missing_terminal_chunk(tmp_path):
    payload = _make_sqlite_payload(tmp_path)
    # Damaged payload: no trailing '0\\r\\n\\r\\n' terminator.
    chunked = f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n"
    decoded = decode_chunked_payload(chunked)
    assert decoded == payload


def test_normalize_sqlite_file_repairs_chunked_payload(tmp_path):
    payload = _make_sqlite_payload(tmp_path)
    chunked = f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n0\r\n\r\n"

    db_path = tmp_path / "broken.sqlite"
    db_path.write_bytes(chunked)

    normalized, repaired = normalize_sqlite_file(db_path)
    assert repaired is True
    assert normalized.startswith(SQLITE_MAGIC_HEADER)
    assert db_path.read_bytes().startswith(SQLITE_MAGIC_HEADER)


def test_normalize_sqlite_file_repairs_single_size_prefix_payload(tmp_path):
    payload = _make_sqlite_payload(tmp_path)
    # Real-world corruption seen in S3: payload starts with "100000\\r\\n" then
    # raw SQLite bytes.
    broken = b"100000\r\n" + payload

    db_path = tmp_path / "prefixed.sqlite"
    db_path.write_bytes(broken)

    normalized, repaired = normalize_sqlite_file(db_path)
    assert repaired is True
    assert normalized.startswith(SQLITE_MAGIC_HEADER)
    assert db_path.read_bytes().startswith(SQLITE_MAGIC_HEADER)


def test_normalize_sqlite_file_rejects_invalid_payload(tmp_path):
    db_path = tmp_path / "invalid.sqlite"
    db_path.write_bytes(b"not a sqlite file")

    with pytest.raises(ValueError, match="not a SQLite database"):
        normalize_sqlite_file(db_path)
