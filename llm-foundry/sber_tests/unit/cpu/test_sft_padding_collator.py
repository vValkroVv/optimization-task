import pytest
import torch

from llmfoundry.data import text_data
from llmfoundry.models.utils import flash_seqlens

BOS, EOS, PAD = 1, 2, 0


def _wrapper(boundary_mode="eos_bos", pad_token=PAD):
    return text_data.FlashAttnCollatorWrapper(
        base_collator=lambda x: x,
        eos_token=EOS,
        bos_token=BOS,
        pad_token=pad_token,
        boundary_mode=boundary_mode,
    )


@pytest.mark.unit
@pytest.mark.fast
def test_eos_bos_pad_tail_continues_last_document():
    # [BOS a b EOS][BOS c EOS][PAD PAD PAD]
    row = torch.tensor([[BOS, 10, 11, EOS, BOS, 12, EOS, PAD, PAD, PAD]])
    pos = _wrapper().calculate_position_ids(row, pad_token_id=PAD)
    assert pos.tolist() == [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5]]


@pytest.mark.unit
@pytest.mark.fast
def test_cu_seqlens_merges_pad_tail_into_last_document():
    row = torch.tensor([[BOS, 10, 11, EOS, BOS, 12, EOS, PAD, PAD, PAD]])
    pos = _wrapper().calculate_position_ids(row, pad_token_id=PAD)
    cu, max_seqlen = flash_seqlens.get_cu_seqlens_from_pos_ids(pos)
    # pads join the second document: docs [0,4) and [4,10)
    assert cu.tolist() == [0, 4, 10]
    assert max_seqlen == 6


@pytest.mark.unit
@pytest.mark.fast
def test_eos_bos_requires_joint_pattern_not_lone_eos():
    # dialogs joined by EOS only (no BOS after) -> NO boundary is detected;
    # the whole pack becomes one document. This is the silent one-doc no-op
    # mode: real SFT packs must carry [EOS][BOS] at every joint.
    row = torch.tensor([[BOS, 10, 11, EOS, 12, 13, EOS]])
    pos = _wrapper().calculate_position_ids(row, pad_token_id=PAD)
    assert pos.tolist() == [[0, 1, 2, 3, 4, 5, 6]]
    cu, _ = flash_seqlens.get_cu_seqlens_from_pos_ids(pos)
    assert cu.tolist() == [0, 7]


@pytest.mark.unit
@pytest.mark.fast
def test_multirow_batches_reset_documents_per_row():
    rows = torch.tensor([
        [BOS, 10, EOS, BOS, 11, EOS],
        [BOS, 12, 13, 14, EOS, PAD],
    ])
    pos = _wrapper().calculate_position_ids(rows, pad_token_id=PAD)
    assert pos.tolist() == [[0, 1, 2, 0, 1, 2], [0, 1, 2, 3, 4, 5]]
    cu, _ = flash_seqlens.get_cu_seqlens_from_pos_ids(pos)
    # flattening keeps the row boundary because row 2 restarts at 0
    assert cu.tolist() == [0, 3, 6, 12]


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize("tp_sp,scale,sample_len", [(8, 1.0, 131_063), (8, 8.0, 131_063)])
def test_padding_collator_divider_and_dsa_sp2_rule(monkeypatch, tp_sp, scale, sample_len):
    """padding_scale scales the pad-to divider; DSA sp-rebalance needs
    padded length % sp^2 == 0, which tp_sp=8 alone does NOT guarantee."""
    monkeypatch.setattr(text_data.dist, "get_tp_sp_group_size", lambda: tp_sp)
    collator = text_data.PaddingCollatorWrapper(
        pad_token=PAD, max_seq_len=131_072, scale=scale)
    input_ids = torch.full((sample_len,), 10, dtype=torch.long)
    labels = torch.full((sample_len,), 11, dtype=torch.long)
    batch = collator([(input_ids, labels)])
    padded = batch["input_ids"].shape[1]
    divider = int(tp_sp * scale)
    assert padded % divider == 0 and padded >= sample_len
    assert torch.all(batch["labels"][0, sample_len:] == -100)
    assert torch.all(batch["input_ids"][0, sample_len:] == PAD)
    sp = 8
    if scale == sp:
        assert padded % (sp * sp) == 0, "padding_scale=sp must satisfy the rebalance rule"
    else:
        # documents why padding_scale is REQUIRED for DSA sp rebalance:
        # the default divider (tp_sp) does not make S % sp^2 == 0
        assert padded % (sp * sp) != 0


@pytest.mark.unit
@pytest.mark.fast
def test_padding_collator_rejects_fully_masked_sample(monkeypatch):
    monkeypatch.setattr(text_data.dist, "get_tp_sp_group_size", lambda: 1)
    collator = text_data.PaddingCollatorWrapper(pad_token=PAD, max_seq_len=64)
    input_ids = torch.full((8,), 10, dtype=torch.long)
    labels = torch.full((8,), -100, dtype=torch.long)
    with pytest.raises(AssertionError):
        collator([(input_ids, labels)])

