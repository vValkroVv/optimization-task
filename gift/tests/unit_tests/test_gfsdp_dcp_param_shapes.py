"""A GFSDP2 checkpoint has to carry its own shapes.

Managed parameters are written through ``_FlatCheckpointTensor``, which declares
itself one-dimensional because its chunk is a ragged slice of the flattened
buffer and ``ChunkStorageMetadata`` can only name a rectangle. So the stored
``TensorStorageMetadata.size`` is ``(numel,)`` and the checkpoint is not
self-describing on its own. ``PARAM_SHAPES`` closes that, and these tests pin
the two properties it relies on.
"""

import json

import pytest

pytest.importorskip("torch")

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    TensorStorageMetadata,
)

from torchtitan.distributed.gfsdp._dcp import (
    _param_shapes,
    param_shapes_key,
    PARAM_SHAPES,
)


def test_shapes_are_read_off_the_tensors():
    state_dict = {
        "layers.0.attention.qkv.qkv_proj.weight": torch.zeros(384, 128),
        "layers.0.moe.experts.w1_EFD": torch.zeros(4, 64, 128),
        "norm.weight": torch.zeros(128),
    }

    assert _param_shapes(state_dict) == {
        "layers.0.attention.qkv.qkv_proj.weight": [384, 128],
        "layers.0.moe.experts.w1_EFD": [4, 64, 128],
        "norm.weight": [128],
    }


def test_non_tensor_entries_are_skipped():
    """The state dict also carries the EMA namespace and, on save, this map."""
    assert _param_shapes({"ema_model": {"norm.weight": torch.zeros(8)}}) == {}


def test_shapes_survive_a_dcp_round_trip_as_one_entry(tmp_path):
    """One entry, not one per parameter.

    DCP flattens nested mappings, so storing the dict directly would spread it
    over one checkpoint entry per FQN and leave a reader to reassemble it by
    prefix. A JSON string is a leaf and stays whole.
    """
    shapes = {"layers.0.attention.wq_a.weight": [1536, 2048], "norm.weight": [2048]}
    ckpt = str(tmp_path / "ckpt")
    key = param_shapes_key()

    dcp.save(
        {"weights": torch.arange(6.0), key: json.dumps(shapes)},
        checkpoint_id=ckpt,
    )

    metadata = FileSystemReader(ckpt).read_metadata().state_dict_metadata
    assert [k for k in metadata if k.startswith(PARAM_SHAPES)] == [key]
    # Not a tensor entry: a converter filtering on TensorStorageMetadata, as
    # the PAM mapper does, will not mistake it for a parameter.
    assert isinstance(metadata[key], BytesStorageMetadata)
    assert isinstance(metadata["weights"], TensorStorageMetadata)

    loaded = {"weights": torch.empty(6), key: None}
    dcp.load(loaded, checkpoint_id=ckpt)
    assert json.loads(loaded[key]) == shapes


def test_the_key_is_per_rank_so_stages_do_not_overwrite_each_other(tmp_path):
    """Under pipeline parallelism a rank describes its own stage only.

    One shared key would leave DCP's deduplication to decide which stage's map
    survived, and the converter would then miss every parameter of the others.
    Suffixing by rank makes the entries independent and the union complete.
    """
    stages = {
        0: {
            "tok_embeddings.weight": [2048, 256],
            "layers.0.attention.wq.weight": [256, 256],
        },
        1: {"layers.1.attention.wq.weight": [256, 256], "norm.weight": [256]},
    }
    assert param_shapes_key(0) != param_shapes_key(1)

    ckpt = str(tmp_path / "ckpt")
    dcp.save(
        {
            "weights": torch.arange(6.0),
            **{param_shapes_key(r): json.dumps(m) for r, m in stages.items()},
        },
        checkpoint_id=ckpt,
    )

    metadata = FileSystemReader(ckpt).read_metadata().state_dict_metadata
    keys = sorted(k for k in metadata if k.startswith(PARAM_SHAPES))
    assert keys == [param_shapes_key(0), param_shapes_key(1)]

    loaded = {k: None for k in keys}
    dcp.load(loaded, checkpoint_id=ckpt)
    merged = {}
    for k in keys:
        merged.update(json.loads(loaded[k]))
    assert merged == {**stages[0], **stages[1]}


def test_a_load_target_without_the_entry_ignores_it(tmp_path):
    """Resuming must not care that the checkpoint describes itself.

    ``build_state_dict(load_target=True)`` omits PARAM_SHAPES, so the load asks
    for the weights only; the extra entry has to stay out of the way.
    """
    ckpt = str(tmp_path / "ckpt")
    dcp.save(
        {
            "weights": torch.arange(6.0),
            param_shapes_key(): json.dumps({"weights": [6]}),
        },
        checkpoint_id=ckpt,
    )

    loaded = {"weights": torch.empty(6)}
    dcp.load(loaded, checkpoint_id=ckpt)

    assert torch.equal(loaded["weights"], torch.arange(6.0))
    assert not [k for k in loaded if k.startswith(PARAM_SHAPES)]
