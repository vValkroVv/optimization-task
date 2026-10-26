"""The rescaler reads model structure through named_segments, not a regex.

``_block_indices`` maps every submodule and parameter of a decoder block to
the block's index by walking the model's own structural declaration — the
same channel AC/FSDP/compile/PP read. Pinned here: the index equals the
container key (checkpoint-compatible with the old FQN parse), the MTP head's
blocks are excluded structurally (their group is ``mtp.layers``), and
everything outside the blocks resolves to nothing.
"""

from torch import nn

from torchtitan.callbacks.builtin.dynamic_block_lr_rescaler import _block_indices
from torchtitan.protocols.blocks import Block


class _FakeBlock(Block):
    def __init__(self):
        nn.Module.__init__(self)
        self.attention = nn.Linear(4, 4)
        self.moe = nn.Linear(4, 4)

    def forward(self, x):
        return self.moe(self.attention(x))


def _model() -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleDict({str(i): _FakeBlock() for i in range(3)})
    mtp = nn.Module()
    mtp.layers = nn.ModuleDict({str(i): _FakeBlock() for i in range(2)})
    model.mtp = mtp
    model.lm_head = nn.Linear(4, 4)
    return model


def test_blocks_map_by_identity_with_container_key_indices():
    model = _model()
    owners = _block_indices([model])

    assert owners[id(model.layers["1"].moe)] == 1
    assert owners[id(model.layers["2"])] == 2
    weight = model.layers["0"].attention.weight
    assert owners[id(weight)] == 0


def test_mtp_blocks_and_non_block_modules_are_outside():
    model = _model()
    owners = _block_indices([model])

    assert id(model.mtp.layers["1"]) not in owners
    assert id(model.mtp.layers["1"].moe) not in owners
    assert id(model.lm_head) not in owners
    assert id(model.lm_head.weight) not in owners


def test_a_model_without_blocks_yields_an_empty_map():
    bare = nn.Module()
    bare.layers = nn.ModuleDict()
    assert _block_indices([bare]) == {}
