"""Pack ``tests/assets/c4_test`` into the tokenized MDS dataset the ``_mds`` presets read.

Rows are not the HF loader's windows: it cuts every ``seq_len + 1`` tokens because it
shifts labels inside the window, while the streaming collator shifts within the row.
Same corpus and tokenizer either way, and eos still marks the document boundaries the
collator turns into positions.

Regenerate from the repo root::

    python tests/assets/c4_test_mds/pack_test_dataset.py
"""

import json
from pathlib import Path

import numpy as np
from streaming.base import MDSWriter

from torchtitan.components.tokenizer import HuggingFaceTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPO_ROOT / "tests/assets/c4_test/data.json"
TOKENIZER = REPO_ROOT / "tests/assets/tokenizer"
OUT = Path(__file__).parent / "train"

SEQ_LEN = 2048
# One epoch of the widest case reading it (HSDP draws 8 * 4 ranks per step over 10
# steps) plus headroom: shuffle is off, so wrapping replays the opening batches.
NUM_ROWS = 384
# Keeps every committed shard under the 500KiB pre-commit cap.
SHARD_ROWS = 32
# llm-foundry dtype code for uint32, see core/mds_decode.py
UINT32_CODE = 1


def pack_mds_dataset(
    source: Path,
    out: Path,
    *,
    tokenizer_path: Path,
    seq_len: int,
    num_rows: int,
    shard_rows: int,
) -> None:
    """Tokenize ``source`` (one JSON object per line) into ``num_rows`` MDS rows."""
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tokenizer_path))
    rows: list[list[int]] = []
    buffer: list[int] = []
    with open(source) as f:
        for line in f:
            text = json.loads(line)["text"]
            buffer.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
            while len(buffer) >= seq_len and len(rows) < num_rows:
                rows.append(buffer[:seq_len])
                buffer = buffer[seq_len:]
            if len(rows) == num_rows:
                break

    if len(rows) < num_rows:
        raise ValueError(
            f"{source} yields only {len(rows)} rows of {seq_len} tokens, "
            f"{num_rows} requested"
        )

    # Uncompressed, because the reader decompresses next to the shards and would
    # have every test run write into the repo tree. ``exist_ok`` wipes ``out``,
    # hence the split subdir rather than writing next to this script.
    with MDSWriter(
        out=str(out),
        columns={"tokens": "bytes", "dtype": "int8"},
        compression=None,
        size_limit=shard_rows * seq_len * 4,
        exist_ok=True,
    ) as writer:
        for row in rows:
            tokens = np.asarray(row, dtype=np.uint32).tobytes()
            writer.write({"tokens": tokens, "dtype": UINT32_CODE})

    # The writer omits the trailing newline that end-of-file-fixer then adds; match
    # it so regeneration stays a no-op.
    index = out / "index.json"
    index.write_text(index.read_text().rstrip("\n") + "\n")


if __name__ == "__main__":
    pack_mds_dataset(
        SOURCE,
        OUT,
        tokenizer_path=TOKENIZER,
        seq_len=SEQ_LEN,
        num_rows=NUM_ROWS,
        shard_rows=SHARD_ROWS,
    )
