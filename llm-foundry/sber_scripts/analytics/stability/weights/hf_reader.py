"""Read tensors from a HuggingFace safetensors export.

Safetensors are tab-indexed and mmap-able. ``get_tensor(name)`` returns
the full tensor straight from disk — no chunked I/O, no slice math.
Working-set RAM is essentially zero (mmap).

Expected layout::

    <hf_dir>/
    ├── model.safetensors.index.json    # mapping {fqn: shard_filename}
    ├── shard-r00-c00000.safetensors    # or model-00001-of-00007.safetensors
    ├── shard-r00-c00001.safetensors
    ├── ...
    └── config.json

A single-file ``<hf_dir>/model.safetensors`` (no index) also works for
small models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import torch
from safetensors import safe_open


class SafetensorsReader:
    """Read tensors from a HF safetensors export.

    Example
    -------
    .. code-block:: python

        reader = SafetensorsReader("/path/to/ep0_ba25000/artifacts/hf_checkpoint")
        print(len(reader.names), "tensors available")
        W = reader.load("model.layers.10.mlp.experts.5.gate_proj.weight")
        for name, tensor in reader.iter_tensors():
            ...  # tensors are bf16 (source dtype) by default
    """

    def __init__(self, hf_dir: Path | str) -> None:
        self.path = Path(hf_dir)
        if not self.path.is_dir():
            raise FileNotFoundError(f"not a directory: {self.path}")
        self._index: dict[str, str] | None = None

    # ------------------------------------------------------------------
    def _load_index(self) -> dict[str, str]:
        if self._index is not None:
            return self._index
        idx_path = self.path / "model.safetensors.index.json"
        if idx_path.exists():
            with idx_path.open() as f:
                self._index = json.load(f)["weight_map"]
            return self._index
        singles = list(self.path.glob("*.safetensors"))
        if len(singles) == 1:
            with safe_open(singles[0], framework="pt") as sf:
                self._index = {k: singles[0].name for k in sf.keys()}
            return self._index
        if not singles:
            raise FileNotFoundError(
                f"no model.safetensors.index.json and no *.safetensors under {self.path}"
            )
        raise FileNotFoundError(
            f"multiple safetensors files in {self.path} but no "
            f"model.safetensors.index.json to map them. Found: "
            f"{[p.name for p in singles[:5]]}"
        )

    @property
    def names(self) -> list[str]:
        """Sorted list of every tensor FQN in this export."""
        return sorted(self._load_index())

    def info(self, name: str) -> tuple[tuple[int, ...], torch.dtype]:
        """Return ``(shape, dtype)`` without materialising the tensor."""
        idx = self._load_index()
        if name not in idx:
            raise KeyError(f"{name!r} not in export. Use .names to list.")
        with safe_open(self.path / idx[name], framework="pt") as f:
            sl = f.get_slice(name)
            shape = tuple(sl.get_shape())
            # safetensors uses str dtype labels ("F32", "BF16", "F16", ...);
            # convert to torch.dtype for consumers.
            dtype_str = sl.get_dtype()
            return shape, _str_to_dtype(dtype_str)

    def load(self, name: str) -> torch.Tensor:
        """Load one tensor in its source dtype."""
        idx = self._load_index()
        if name not in idx:
            raise KeyError(f"{name!r} not in export. Use .names to list.")
        with safe_open(self.path / idx[name], framework="pt") as f:
            return f.get_tensor(name)

    def load_many(self, names: Iterable[str]) -> dict[str, torch.Tensor]:
        """Load many tensors, grouping by shard so each file opens once.

        Per-tensor cost of an mmap-backed ``get_tensor`` is already low
        enough that thread parallelism rarely helps; we just touch each
        shard sequentially.
        """
        idx = self._load_index()
        names = list(names)
        by_shard: dict[str, list[str]] = {}
        for n in names:
            if n not in idx:
                raise KeyError(f"{n!r} not in export. Use .names to list.")
            by_shard.setdefault(idx[n], []).append(n)
        out: dict[str, torch.Tensor] = {}
        for shard, ns in by_shard.items():
            with safe_open(self.path / shard, framework="pt") as f:
                for n in ns:
                    out[n] = f.get_tensor(n)
        return out

    def iter_tensors(
        self,
        names: Iterable[str] | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``(name, tensor)`` pairs grouped by shard (one open per shard)."""
        idx = self._load_index()
        target = list(names) if names is not None else self.names
        # Sort by shard so we touch each safetensors file once.
        target.sort(key=lambda n: (idx[n], n))
        current_shard: str | None = None
        current_handle = None
        try:
            for n in target:
                if idx[n] != current_shard:
                    if current_handle is not None:
                        current_handle.__exit__(None, None, None)
                    current_shard = idx[n]
                    current_handle = safe_open(
                        self.path / current_shard,
                        framework="pt",
                    )
                    current_handle.__enter__()
                yield n, current_handle.get_tensor(n)
        finally:
            if current_handle is not None:
                current_handle.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DTYPE_MAP: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _str_to_dtype(s: str) -> torch.dtype:
    if s in _DTYPE_MAP:
        return _DTYPE_MAP[s]
    raise ValueError(f"unknown safetensors dtype label: {s!r}")
