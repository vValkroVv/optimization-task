import re
from base64 import b64decode
from pathlib import Path
from typing import Optional, TextIO


class Anonymizer:
    _B64_WORDS = (
        b"c2Jlcg==",
        b"Z2lnYQ==",
        b"0YHQsdC10YA=",
        b"0LPQuNCz0LA=",
    )
    _WORDS_TO_REPLACE = tuple(b64decode(w).decode("utf-8") for w in _B64_WORDS)
    _REPLACE_PATTERN = re.compile(
        "|".join(
            re.escape(w) for w in sorted(_WORDS_TO_REPLACE, key=len, reverse=True)
        ),
        re.IGNORECASE,
    )
    _CHUNK_SIZE = 1024 * 1024
    _TAIL_SIZE = max(len(word) for word in _WORDS_TO_REPLACE)

    @classmethod
    def anonymize(cls, message: str) -> str:
        return cls._REPLACE_PATTERN.sub("papaya", message)

    @classmethod
    def anonymize_file(
        cls,
        file_path: str | Path,
        output_file_path: Optional[str | Path] = None,
        *,
        append: bool = False,
    ) -> None:
        source_path = Path(file_path)
        output_path = Path(output_file_path) if output_file_path is not None else None
        if output_path is None or source_path.resolve() == output_path.resolve():
            if append:
                raise ValueError("append requires a separate output file")
            input_path = cls._move_source_file(source_path)
            output_path = source_path
        else:
            input_path = source_path

        output_mode = "a" if append else "w"
        with (
            input_path.open("r") as input_file,
            output_path.open(output_mode) as output_file,
        ):
            cls._write_anonymized_file(input_file, output_file)

    @classmethod
    def _move_source_file(cls, source_path: Path) -> Path:
        moved_path = cls._get_moved_file_path(source_path)
        source_path.rename(moved_path)
        return moved_path

    @classmethod
    def _get_moved_file_path(cls, source_path: Path) -> Path:
        moved_path = source_path.with_name(f"{source_path.name}.raw")
        if not moved_path.exists():
            return moved_path

        index = 1
        while True:
            moved_path = source_path.with_name(f"{source_path.name}.raw.{index}")
            if not moved_path.exists():
                return moved_path
            index += 1

    @classmethod
    def _write_anonymized_file(cls, input_file: TextIO, output_file: TextIO) -> None:
        tail = ""
        chunk = input_file.read(cls._CHUNK_SIZE)
        while True:
            if not chunk:
                break

            next_chunk = input_file.read(cls._CHUNK_SIZE)
            text = tail + chunk
            if not next_chunk:
                output_file.write(cls.anonymize(text))
                break

            if len(text) <= cls._TAIL_SIZE:
                tail = text
                chunk = next_chunk
                continue

            output_file.write(cls.anonymize(text[: -cls._TAIL_SIZE]))
            tail = text[-cls._TAIL_SIZE :]
            chunk = next_chunk
