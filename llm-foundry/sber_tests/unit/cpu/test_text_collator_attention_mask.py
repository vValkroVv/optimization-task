import pytest
import torch

from llmfoundry.data.collators import text_collator as text_collator_module
from llmfoundry.data.collators.text_flash_attn_collator import TextFlashAttnCollator


def _build_collator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    group_size: int | None = 1,
    left_padding: bool = False,
    return_attention_mask: bool | None = True,
) -> text_collator_module.TextCollator:
    """Construct a TextCollator after fixing the TP/SP group size."""
    monkeypatch.setattr(
        text_collator_module.dist,
        "get_tp_sp_group_size",
        lambda: group_size,
    )
    return text_collator_module.TextCollator(
        pad_token_id=99,
        left_padding=left_padding,
        return_attention_mask=return_attention_mask,
    )


def _unmasked_examples() -> list[dict[str, torch.Tensor]]:
    return [
        {"input_ids": torch.tensor([11, 12, 13]), "labels": torch.tensor([21, 22, 23])},
        {"input_ids": torch.tensor([14]), "labels": torch.tensor([24])},
    ]


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize("left_padding", [False, True])
def test_default_sends_a_mask_only_for_left_padding(
    monkeypatch: pytest.MonkeyPatch,
    left_padding: bool,
) -> None:
    """End padding needs no mask under causal attention; start padding does."""
    collator = _build_collator(
        monkeypatch,
        left_padding=left_padding,
        return_attention_mask=None,
    )

    batch = collator(_unmasked_examples())

    assert ("attention_mask" in batch) is left_padding
    if left_padding:
        assert batch["attention_mask"].tolist() == [
            [True, True, True],
            [False, False, True],
        ]


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize(
    ("left_padding", "return_attention_mask"),
    [(False, True), (True, False)],
)
def test_explicit_flag_overrides_the_padding_side(
    monkeypatch: pytest.MonkeyPatch,
    left_padding: bool,
    return_attention_mask: bool,
) -> None:
    collator = _build_collator(
        monkeypatch,
        left_padding=left_padding,
        return_attention_mask=return_attention_mask,
    )

    batch = collator(_unmasked_examples())

    assert ("attention_mask" in batch) is return_attention_mask


@pytest.mark.unit
@pytest.mark.fast
def test_sample_mask_is_rejected_when_the_batch_gets_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping a mask a sample brought would silently attend to its holes."""
    collator = _build_collator(monkeypatch, return_attention_mask=None)
    examples = _unmasked_examples()
    examples[0]["attention_mask"] = torch.tensor([True, False, True])

    with pytest.raises(ValueError, match="carries its own attention_mask"):
        collator(examples)


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize("left_padding", [False, True])
def test_flash_attn_collator_has_the_same_default(
    monkeypatch: pytest.MonkeyPatch,
    left_padding: bool,
) -> None:
    monkeypatch.setattr(text_collator_module.dist, "get_tp_sp_group_size", lambda: 1)

    collator = TextFlashAttnCollator(left_padding=left_padding)

    assert collator.base_collator.return_attention_mask is left_padding


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize(
    ("left_padding", "expected_input_ids", "expected_labels", "expected_mask"),
    [
        (
            False,
            [[11, 12, 13, 99], [14, 99, 99, 99]],
            [[21, 22, 23, -100], [24, -100, -100, -100]],
            [[True, False, True, False], [True, False, False, False]],
        ),
        (
            True,
            [[99, 11, 12, 13], [99, 99, 99, 14]],
            [[-100, 21, 22, 23], [-100, -100, -100, 24]],
            [[False, True, False, True], [False, False, False, True]],
        ),
    ],
)
def test_padding_keeps_tokens_labels_and_attention_masks_aligned(
    monkeypatch: pytest.MonkeyPatch,
    left_padding: bool,
    expected_input_ids: list[list[int]],
    expected_labels: list[list[int]],
    expected_mask: list[list[bool]],
) -> None:
    """Padding must preserve internal false positions and synthesize absent masks."""
    collator = _build_collator(
        monkeypatch,
        group_size=4,
        left_padding=left_padding,
    )
    examples = [
        {
            "input_ids": torch.tensor([11, 12, 13]),
            "labels": torch.tensor([21, 22, 23]),
            "attention_mask": torch.tensor([True, False, True]),
        },
        {
            "input_ids": torch.tensor([14]),
            "labels": torch.tensor([24]),
        },
    ]

    batch = collator(examples)

    assert batch["input_ids"].tolist() == expected_input_ids
    assert batch["labels"].tolist() == expected_labels
    assert batch["attention_mask"].tolist() == expected_mask
    assert batch["attention_mask"].dtype is torch.bool


@pytest.mark.unit
@pytest.mark.fast
def test_aligned_input_is_not_padded_and_gets_boolean_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already aligned sequence must retain every token and mask position."""
    collator = _build_collator(monkeypatch, group_size=4)
    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([5, 6, 7, 8]),
            "attention_mask": torch.tensor([1, 0, 1, 1]),
        },
    ]

    batch = collator(examples)

    assert batch["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert batch["labels"].tolist() == [[5, 6, 7, 8]]
    assert batch["attention_mask"].tolist() == [[True, False, True, True]]
    assert batch["attention_mask"].dtype is torch.bool


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize(
    "attention_mask",
    [
        torch.tensor([[True, False]]),
        torch.tensor([True, False, True]),
    ],
    ids=["invalid-rank", "invalid-length"],
)
def test_invalid_attention_mask_shape_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    attention_mask: torch.Tensor,
) -> None:
    """Masks with a non-vector rank or a token-count mismatch must be rejected."""
    collator = _build_collator(monkeypatch)
    examples = [
        {
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([3, 4]),
            "attention_mask": attention_mask,
        },
    ]

    with pytest.raises(
        AssertionError,
        match="attention_mask must be one-dimensional and match input_ids length",
    ):
        collator(examples)


@pytest.mark.unit
@pytest.mark.fast
def test_split_batch_preserves_all_row_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every split must contain the matching token, label, and mask rows."""
    collator = _build_collator(monkeypatch, group_size=None)
    examples = [
        {
            "input_ids": torch.tensor([row, row + 10]),
            "labels": torch.tensor([row + 20, row + 30]),
            "attention_mask": torch.tensor([row % 2 == 0, row % 2 == 1]),
        }
        for row in range(4)
    ]
    batch = collator(examples)

    microbatches = collator.split_batch(batch, microbatch_size=2)

    assert len(microbatches) == 2
    assert all(
        set(part) == {"input_ids", "labels", "attention_mask"} for part in microbatches
    )
    for field in ("input_ids", "labels", "attention_mask"):
        assert torch.equal(
            torch.cat([part[field] for part in microbatches]), batch[field]
        )


@pytest.mark.unit
@pytest.mark.fast
def test_split_batch_rejects_non_divisible_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch must not silently drop rows when its size is not divisible."""
    collator = _build_collator(monkeypatch)
    batch = collator(
        [
            {
                "input_ids": torch.tensor([row]),
                "labels": torch.tensor([row]),
            }
            for row in range(3)
        ],
    )

    with pytest.raises(
        AssertionError,
        match="Batch size 3 is not divisible by microbatch size 2",
    ):
        collator.split_batch(batch, microbatch_size=2)
