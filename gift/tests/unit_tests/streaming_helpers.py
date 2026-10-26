"""
Shared helpers for the streaming dataloader unit tests

Importing this module requires the ``streaming`` package, so test modules must
call ``pytest.importorskip("streaming")`` before importing it. The dataloader
factory additionally needs ``tree_sitter`` (imported lazily), so its users
must importorskip that too
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from streaming.base import MDSWriter

from torchtitan.distributed.data_world import DataWorld
from torchtitan.config.schema.datasets.streaming.text.configs import (
    StreamingTextDatasetConfig,
    TextStreamConfig,
)

if TYPE_CHECKING:
    from torchtitan.datasets.streaming.text.dataloader import StreamingTextDataLoader


class StubTokenizer:
    """Bare tokenizer stand-in: non-FIM streaming text paths only need eos_id"""

    __slots__ = ("eos_id",)

    def __init__(self, eos_id: int = 999) -> None:
        self.eos_id = eos_id


class CharTokenizer:
    """Char-level tokenizer for FIM tests: id = ord(char) + 100, lossless decode"""

    _FIM_TOKEN_IDS = {"<|fim_prefix|>": 1, "<|fim_middle|>": 2, "<|fim_suffix|>": 3}

    def __init__(self, *, with_fim_tokens: bool = True) -> None:
        self.eos_id = 0
        self._special_token_ids = self._FIM_TOKEN_IDS if with_fim_tokens else {}

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        add_special_tokens: bool = False,
    ) -> list[int]:
        return [ord(char) + 100 for char in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return "".join(chr(token_id - 100) for token_id in token_ids if token_id >= 100)

    def token_to_id(self, token: str) -> int | None:
        return self._special_token_ids.get(token)


def install_single_process_shm_gather(monkeypatch) -> None:
    """Allow constructing dp_world_size > 1 datasets in one test process.

    streaming's shm-prefix rendezvous all_gathers node-leader prefixes, which
    needs a real process group. With local_world_size=1 every simulated rank
    reads back only its own slot, so returning this rank's value for all slots
    is faithful; production uses the data-axis group from DataWorld.from_mesh.
    """
    from streaming.base import distributed as streaming_dist
    from streaming.base.shared import prefix as shm_prefix

    monkeypatch.setattr(
        shm_prefix,
        "all_gather_object",
        lambda obj: [obj] * streaming_dist.get_world_size(),
    )


def make_sequential_rows(
    num_rows: int, row_len: int, *, base: int = 1000
) -> list[list[int]]:
    """Token rows where the first token (base + i) uniquely identifies the sample"""
    return [[base + i] * row_len for i in range(num_rows)]


def write_tokenized_mds(
    out_dir: Path, rows: list[list[int]], *, token_field: str = "tokens"
) -> None:
    """Write rows as a tokenized MDS dataset (llm-foundry bytes + dtype convention)"""
    columns = {token_field: "bytes", "dtype": "int8"}
    with MDSWriter(columns=columns, out=str(out_dir)) as writer:
        for row in rows:
            writer.write(
                {token_field: np.asarray(row, dtype=np.uint32).tobytes(), "dtype": 1}
            )


def make_text_dataloader(
    local_dir: Path,
    *,
    seq_len: int,
    local_batch_size: int,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    shuffle: bool = False,
    drop_last: bool = True,
    infinite: bool = False,
    num_workers: int = 0,
    tokenizer: object | None = None,
) -> "StreamingTextDataLoader":
    """Build a StreamingTextDataLoader over a local MDS dir through its Config.

    Pins num_canonical_nodes=1 so sample order does not depend on dp_world_size.
    """
    from torchtitan.datasets.streaming.text.dataloader import StreamingTextDataLoader

    config = StreamingTextDataLoader.Config(
        dataset=StreamingTextDatasetConfig(
            streams=[TextStreamConfig(local=str(local_dir))],
            shuffle=shuffle,
            num_canonical_nodes=1,
        ),
        drop_last=drop_last,
        infinite=infinite,
        num_workers=num_workers,
    )
    return config.build(
        data_world=DataWorld(rank=dp_rank, world_size=dp_world_size),
        tokenizer=tokenizer if tokenizer is not None else StubTokenizer(),
        seq_len=seq_len,
        local_batch_size=local_batch_size,
    )
