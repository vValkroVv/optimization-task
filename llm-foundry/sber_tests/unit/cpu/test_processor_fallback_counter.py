import os
import sys
import pytest
import torch

from unittest.mock import MagicMock, patch

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib", "composer"))
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib", "gigafsdp"))
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib", "streaming"))
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib", "flash-linear-attention"))

import llmfoundry.callbacks.processor_fallback_counter as processor_fallback_counter_module  # noqa: E402
from llmfoundry.callbacks.processor_fallback_counter import ProcessorFallbackCounter  # noqa: E402
from llmfoundry.data.processors.base_processor import DataModality  # noqa: E402


class _DummyCollator:
    def __init__(self, data_modality: DataModality):
        self.data_modality = data_modality


class _DummyOmniCollator:
    def __init__(self, modalities):
        self.collators = [_DummyCollator(DataModality.TEXT)] + [
            _DummyCollator(modality) for modality in modalities
        ]


class _DummyDataLoader:
    def __init__(self, modalities):
        self.collate_fn = _DummyOmniCollator(modalities)


class _DummyLogger:
    def __init__(self):
        self.logged_metrics = []

    def log_metrics(self, metrics):
        self.logged_metrics.append(metrics)


def _make_state(modality_inputs, modalities):
    state = MagicMock()
    state.raw_batch = {"modality_inputs": modality_inputs}
    state.dataloader = _DummyDataLoader(modalities)
    state.train_dataloader = state.dataloader
    state.timestamp.batch = 1
    return state


class TestProcessorFallbackCounter:
    def test_extract_non_text_modalities_from_collate_fn_preserves_order(self):
        collate_fn = _DummyOmniCollator(
            [DataModality.IMAGE, DataModality.VIDEO, DataModality.IMAGE]
        )

        ordered_modalities = (
            ProcessorFallbackCounter._extract_non_text_modalities_from_collate_fn(
                collate_fn
            )
        )

        assert ordered_modalities == ["image", "video"]

    def test_extract_non_text_modalities_from_collate_fn_rejects_text_only(self):
        collate_fn = _DummyOmniCollator([])

        with pytest.raises(AssertionError, match="at least one non-text modality"):
            ProcessorFallbackCounter._extract_non_text_modalities_from_collate_fn(
                collate_fn
            )

    def test_build_local_fallback_counts_zero_fills_missing_modalities(self):
        counts = ProcessorFallbackCounter._build_local_fallback_counts(
            modality_inputs={"image": {"processor_fallback_count": 3}},
            ordered_modalities=["image", "video"],
        )

        assert counts == [3, 0]

    def test_build_local_fallback_counts_rejects_unexpected_modality(self):
        with pytest.raises(
            AssertionError, match="outside the active dataloader schema"
        ):
            ProcessorFallbackCounter._build_local_fallback_counts(
                modality_inputs={"audio": {"processor_fallback_count": 1}},
                ordered_modalities=["image", "video"],
            )

    @patch.object(
        processor_fallback_counter_module.dist,
        "get_global_rank",
        return_value=0,
    )
    @patch.object(
        processor_fallback_counter_module.dist,
        "get_world_size",
        return_value=1,
    )
    def test_before_batch_split_dataloader_logs_zero_filled_metrics(
        self,
        _mock_world_size,
        _mock_global_rank,
    ):
        callback = ProcessorFallbackCounter(max_processor_fallback_count=100000)
        logger = _DummyLogger()
        state = _make_state(
            modality_inputs={"image": {"processor_fallback_count": 2}},
            modalities=[DataModality.IMAGE, DataModality.VIDEO],
        )

        callback.before_batch_split_dataloader(state, logger)

        assert logger.logged_metrics[-1] == {
            "trainer/image_processor_fallbacks": 2,
            "trainer/video_processor_fallbacks": 0,
            "trainer/total_processor_fallbacks": 2,
        }

        state.raw_batch = {
            "modality_inputs": {"video": {"processor_fallback_count": 3}}
        }
        callback.before_batch_split_dataloader(state, logger)

        assert logger.logged_metrics[-1] == {
            "trainer/image_processor_fallbacks": 2,
            "trainer/video_processor_fallbacks": 3,
            "trainer/total_processor_fallbacks": 5,
        }

    @patch.object(
        processor_fallback_counter_module.dist,
        "get_global_rank",
        return_value=0,
    )
    @patch.object(
        processor_fallback_counter_module.dist,
        "get_world_size",
        return_value=2,
    )
    def test_before_batch_split_dataloader_reduces_fixed_dense_vector(
        self,
        _mock_world_size,
        _mock_global_rank,
    ):
        def _all_reduce(tensor, reduce_operation):
            assert reduce_operation == "SUM"
            assert tensor.tolist() == [2, 0]
            tensor += torch.tensor([3, 4], dtype=tensor.dtype, device=tensor.device)

        callback = ProcessorFallbackCounter(max_processor_fallback_count=100000)
        logger = _DummyLogger()
        state = _make_state(
            modality_inputs={"image": {"processor_fallback_count": 2}},
            modalities=[DataModality.IMAGE, DataModality.VIDEO],
        )

        with (
            patch.object(
                ProcessorFallbackCounter,
                "_get_reduce_device",
                return_value=torch.device("cpu"),
            ),
            patch.object(
                processor_fallback_counter_module.dist,
                "all_reduce",
                side_effect=_all_reduce,
            ) as mock_all_reduce,
        ):
            callback.before_batch_split_dataloader(state, logger)

        mock_all_reduce.assert_called_once()
        assert logger.logged_metrics[-1] == {
            "trainer/image_processor_fallbacks": 5,
            "trainer/video_processor_fallbacks": 4,
            "trainer/total_processor_fallbacks": 9,
        }

    @patch.object(
        processor_fallback_counter_module.dist,
        "get_global_rank",
        return_value=0,
    )
    @patch.object(
        processor_fallback_counter_module.dist,
        "get_world_size",
        return_value=1,
    )
    def test_before_batch_split_dataloader_triggers_threshold(
        self,
        _mock_world_size,
        _mock_global_rank,
    ):
        callback = ProcessorFallbackCounter(max_processor_fallback_count=0)
        logger = _DummyLogger()
        state = _make_state(
            modality_inputs={"image": {"processor_fallback_count": 1}},
            modalities=[DataModality.IMAGE],
        )

        with patch.object(callback, "_trigger_fail") as mock_trigger_fail:
            callback.before_batch_split_dataloader(state, logger)

        mock_trigger_fail.assert_called_once_with(
            state,
            logger,
            reason="processor_fallback_count_exceeded_threshold",
            step=1,
            total_fallbacks=1,
            max_processor_fallback_count=0,
        )
