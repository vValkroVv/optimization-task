from __future__ import annotations

from pathlib import Path

SQLITE_MAGIC_HEADER = b"SQLite format 3\x00"


def _decode_chunked_payload(
    payload: bytes,
    *,
    base: int,
    allow_incomplete_final_chunk: bool,
) -> bytes | None:
    pos = 0
    decoded = bytearray()

    while True:
        line_end = payload.find(b"\r\n", pos)
        if line_end < 0:
            return None

        size_token = payload[pos:line_end]
        size_token = size_token.split(b";", 1)[0].strip()
        if not size_token:
            return None

        try:
            chunk_size = int(size_token, base)
        except ValueError:
            return None

        if chunk_size < 0:
            return None

        pos = line_end + 2
        data_end = pos + chunk_size
        if data_end > len(payload):
            return None

        decoded.extend(payload[pos:data_end])
        pos = data_end

        if chunk_size == 0:
            # Optional trailer headers may follow. For corruption recovery we allow
            # either no trailer bytes or a tail ending with CRLF.
            if pos == len(payload):
                return bytes(decoded)
            if payload[pos : pos + 2] == b"\r\n":
                pos += 2
            tail = payload[pos:]
            if not tail or tail.endswith(b"\r\n"):
                return bytes(decoded)
            return None

        if pos == len(payload):
            if allow_incomplete_final_chunk:
                return bytes(decoded)
            return None

        if payload[pos : pos + 2] != b"\r\n":
            return None
        pos += 2

        if pos == len(payload):
            if allow_incomplete_final_chunk:
                return bytes(decoded)
            return None


def decode_chunked_payload(payload: bytes) -> bytes | None:
    """Decode HTTP chunked transfer payload.

    Returns decoded bytes or None if payload is not valid chunked data.
    """
    # The nominal format uses hexadecimal chunk sizes and a terminating "0" chunk.
    # In practice, we occasionally see damaged payloads that are:
    # - decimal chunk sizes instead of hex;
    # - missing the final "0\\r\\n\\r\\n".
    # Try strict variants first, then recovery variants.
    for base, allow_incomplete_final_chunk in (
        (16, False),
        (10, False),
        (16, True),
        (10, True),
    ):
        decoded = _decode_chunked_payload(
            payload,
            base=base,
            allow_incomplete_final_chunk=allow_incomplete_final_chunk,
        )
        if decoded is not None:
            return decoded
    return None


def strip_single_chunk_prefix(payload: bytes) -> bytes | None:
    """Best-effort recovery for payloads like '<size>\\r\\n<sqlite-bytes>'."""
    line_end = payload.find(b"\r\n")
    if line_end <= 0:
        return None

    size_token = payload[:line_end].split(b";", 1)[0].strip()
    if not size_token:
        return None

    # Accept both hex and decimal tokens.
    parsed = False
    for base in (16, 10):
        try:
            if int(size_token, base) >= 0:
                parsed = True
                break
        except ValueError:
            continue
    if not parsed:
        return None

    body = payload[line_end + 2 :]
    if body.startswith(SQLITE_MAGIC_HEADER):
        return body
    return None


def normalize_sqlite_file(path: Path) -> tuple[bytes, bool]:
    """Return valid SQLite payload and repair known chunked corruption in-place.

    Returns:
        (payload, repaired)
    Raises:
        ValueError if file does not contain a valid SQLite database.
    """
    payload = path.read_bytes()
    if payload.startswith(SQLITE_MAGIC_HEADER):
        return payload, False

    decoded = decode_chunked_payload(payload)
    if decoded is not None and decoded.startswith(SQLITE_MAGIC_HEADER):
        path.write_bytes(decoded)
        return decoded, True

    stripped = strip_single_chunk_prefix(payload)
    if stripped is not None:
        path.write_bytes(stripped)
        return stripped, True

    raise ValueError(f"file is not a SQLite database: {path} (header={payload[:16]!r})")
