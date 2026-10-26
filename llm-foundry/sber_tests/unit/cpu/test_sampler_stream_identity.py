"""Stream identities must follow the sample axis into each microbatch."""

from types import SimpleNamespace

import pytest
import torch

from llmfoundry.callbacks.metric_sampler import MetricSampler
from llmfoundry.data.text_data import SplitTrackingCollatorWrapper

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sample_ids", [False, True])
def test_stream_tracking_rejects_a_collator_that_merges_samples(sample_ids):
    examples = [{"input_ids": torch.arange(4), "_split": "train", "_stream_id": stream}
                for stream in (0, 1)]
    if sample_ids:
        for index, example in enumerate(examples):
            example["_sample_id"] = (0, index)
    collator = SplitTrackingCollatorWrapper(lambda _: {"input_ids": torch.arange(8).view(1, 8)})
    with pytest.raises(ValueError, match="stream identities cannot be preserved"):
        collator(examples)


def test_stream_tracking_preserves_sample_order_and_ids():
    examples = [{"input_ids": torch.tensor([stream, 9]), "_split": "train", "_stream_id": stream,
                 "_sample_id": (2, index)} for index, stream in enumerate((1, 0, 1))]
    collator = SplitTrackingCollatorWrapper(lambda batch: {"input_ids": torch.stack(batch)})
    batch = collator(examples)
    assert batch["input_ids"][:, 0].tolist() == batch["stream_id"].tolist() == [1, 0, 1]
    assert batch["sample_ids"].tolist() == [[2, 0], [2, 1], [2, 2]]


@pytest.mark.parametrize("stream_ids", [torch.tensor([0]), torch.tensor([[0, 1]])])
def test_capture_rejects_a_stream_axis_that_does_not_match_samples(tmp_path, stream_ids):
    sampler = MetricSampler(str(tmp_path))
    state = SimpleNamespace(batch={"input_ids": torch.ones(2, 4, dtype=torch.long), "stream_id": stream_ids})
    with pytest.raises(ValueError, match="one stream_id per microbatch sample"):
        sampler._before_forward(state, None)
