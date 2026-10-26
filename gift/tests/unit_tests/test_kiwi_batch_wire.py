import logging

import pytest
import torch

from torchtitan.datasets.kiwi.batch import ModalityData, ModelBatchData
from torchtitan.datasets.kiwi.configs import (
    DatasetConfig,
    PreprocessorConfig,
    TextDataStreamConfig,
)
from torchtitan.datasets.kiwi.magi_ranges import MagiRanges
from torchtitan.datasets.kiwi.kiwi_dataloader import (
    KiwiDataLoader,
    model_batch_from_wire,
    model_batch_to_wire,
)


def _modality(fill: int) -> ModalityData:
    return ModalityData(
        tokens=torch.arange(4) + fill,
        position_ids=torch.arange(4, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        target_mask=torch.tensor([True, False, True, False]),
        span_ids=torch.tensor([0, 0, 1, 1]),
    )


def _batch(*, with_text: bool) -> ModelBatchData:
    ranges = torch.tensor([[0, 4]], dtype=torch.int32)
    return ModelBatchData(
        position_ids=torch.arange(4),
        magi_ranges=MagiRanges(
            seq_len=4,
            q_ranges=ranges,
            k_ranges=ranges.clone(),
            attn_type_map=torch.tensor([0], dtype=torch.int32),
        ),
        span_properties=[{"caption", "user"}, set(), {"ё"}],
        text=_modality(10) if with_text else None,
        gen_visual=_modality(20),
    )


def _assert_batch_equal(actual: ModelBatchData, expected: ModelBatchData) -> None:
    torch.testing.assert_close(actual.position_ids, expected.position_ids)
    assert actual.magi_ranges.seq_len == expected.magi_ranges.seq_len
    assert actual.magi_ranges.dist_attn_key is None
    torch.testing.assert_close(
        actual.magi_ranges.q_ranges, expected.magi_ranges.q_ranges
    )
    torch.testing.assert_close(
        actual.magi_ranges.k_ranges, expected.magi_ranges.k_ranges
    )
    torch.testing.assert_close(
        actual.magi_ranges.attn_type_map, expected.magi_ranges.attn_type_map
    )
    assert actual.span_properties == expected.span_properties
    for name in ("text", "gen_visual", "semantic_visual"):
        got = getattr(actual, name)
        want = getattr(expected, name)
        assert (got is None) == (want is None)
        if got is None:
            continue
        for field_name in (
            "tokens",
            "position_ids",
            "cu_seqlens",
            "target_mask",
            "span_ids",
        ):
            torch.testing.assert_close(
                getattr(got, field_name), getattr(want, field_name)
            )


def test_wire_roundtrip_keeps_a_stable_schema() -> None:
    present = _batch(with_text=True)
    missing = _batch(with_text=False)
    present_inputs, present_labels = model_batch_to_wire(present)
    missing_inputs, missing_labels = model_batch_to_wire(missing)

    assert list(present_inputs) == list(missing_inputs)
    assert "text.tokens" in present_inputs
    assert missing_inputs["text.tokens"] is None
    assert present_labels.numel() == 0
    assert missing_labels.numel() == 0

    _assert_batch_equal(model_batch_from_wire(present_inputs, present_labels), present)
    _assert_batch_equal(model_batch_from_wire(missing_inputs, missing_labels), missing)


def test_wire_rejects_dist_attn_key() -> None:
    batch = _batch(with_text=True)
    batch = ModelBatchData(
        position_ids=batch.position_ids,
        magi_ranges=MagiRanges(
            seq_len=batch.magi_ranges.seq_len,
            q_ranges=batch.magi_ranges.q_ranges,
            k_ranges=batch.magi_ranges.k_ranges,
            attn_type_map=batch.magi_ranges.attn_type_map,
            dist_attn_key=object(),
        ),
        span_properties=batch.span_properties,
        text=batch.text,
        gen_visual=batch.gen_visual,
    )
    with pytest.raises(ValueError, match="dist_attn_key"):
        model_batch_to_wire(batch)


def test_kiwi_config_replicates_over_non_batch_axes() -> None:
    assert KiwiDataLoader.Config.replicate_batch_over == ("pp", "cp", "tp")
    assert KiwiDataLoader.Config.use_data_world is False
    assert KiwiDataLoader.Config.supports_reiteration is False


def _loader_config() -> KiwiDataLoader.Config:
    return KiwiDataLoader.Config(
        datasets=[
            DatasetConfig(
                name="text",
                text_streams=[
                    TextDataStreamConfig(name="train", data_file="shards.txt")
                ],
            )
        ],
        preprocessor=PreprocessorConfig(tokenizer_path="tokenizer"),
        token_budget=8,
    )


def test_kiwi_loader_warns_on_unused_trainer_args(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        loader = KiwiDataLoader(
            _loader_config(),
            dp_rank=0,
            dp_world_size=1,
            tokenizer=object(),
            seq_len=4096,
            local_batch_size=2,
            snapshot_every_n_steps=10,
        )
    loader.terminate()
    message = " ".join(caplog.messages)
    for name in (
        "tokenizer",
        "seq_len",
        "local_batch_size",
        "snapshot_every_n_steps",
    ):
        assert name in message


def test_kiwi_loader_silent_without_trainer_args(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        loader = _loader_config().build(dp_rank=0, dp_world_size=1)
    loader.terminate()
    assert not [record for record in caplog.records if "ignores" in record.getMessage()]
