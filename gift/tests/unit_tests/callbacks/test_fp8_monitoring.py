# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 monitoring excludes padding and dequantizes once per capture.

The wrapper records its pre-padding shape in ``orig_shape``, so the padding
comes off after dequantization and the batch-mean weight is the logical count.
Plain tensors carry no such record and are measured as they arrive.
"""

from types import SimpleNamespace

import pytest
import torch

from torchtitan.config.schema.callbacks.metrics.sample import ReduceOp
from torchtitan.callbacks.metrics import stats
from torchtitan.callbacks.metrics.reducers import ParallelReducer
from torchtitan.callbacks.metrics.sample import (
    MetricConfig,
    ReduceConfig,
    ReduceStageConfig,
    Sample,
    StatOpConfig,
)
from torchtitan.callbacks.sources.base import MonitorSource
from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor, InputLayoutType
from tests.unit_tests.callbacks.trainer_stub import snapshot


class _SingleRankDims:
    pp_enabled = False

    def get_optional_mesh(self, name: str):
        del name
        return None


def _wrapper(data: torch.Tensor, orig_shape: tuple[int, ...]):
    """The real wrapper, built on CPU: it needs no CUDA, only torch."""
    wrapper = Float8BlockwiseQTensor(
        data=data,
        scale=torch.ones(data.size(0), 1),
        orig_dtype=torch.bfloat16,
        input_layout_type=InputLayoutType.ROWWISE,
        block_scaling_dim=1,
        scale_block_first=False,
    )
    wrapper.orig_shape = orig_shape
    return wrapper


def _mean_metric(*, accumulable: bool = False) -> MetricConfig:
    """A mean. With ``accumulable``, a reduce op the batch's captures fold with."""
    reduce = (
        ReduceConfig(stages=[ReduceStageConfig(group="dp", op=ReduceOp.mean)])
        if accumulable
        else ReduceConfig()
    )
    return MetricConfig(name="mean", pipeline=[StatOpConfig(op="mean")], reduce=reduce)


def _fake_dequant(result=None):
    """Stands in for the Triton kernel path, honouring its shape contract."""

    def dequantize(wrapper):
        out = wrapper._data if result is None else result
        return out[tuple(slice(0, size) for size in wrapper.orig_shape)]

    return dequantize


def _source() -> MonitorSource:
    source = MonitorSource(MonitorSource.Config())
    source.active = True
    source.reducer = ParallelReducer(parallel_dims=_SingleRankDims())
    return source


def _padded_wrapper(values: list[float], padded_cols: int):
    """An fp8 capture of one row of ``values``, zero-padded to ``padded_cols``."""
    data = torch.zeros(1, padded_cols)
    data[0, : len(values)] = torch.tensor(values)
    return _wrapper(data, (1, len(values)))


# --- the capture path -------------------------------------------------------


def test_metric_of_an_fp8_capture_ignores_the_padding(monkeypatch):
    """The wrapper cannot be sliced, so its padding comes off after dequantization."""
    value = _wrapper(torch.zeros(2, 4), (2, 3))
    dequantized = torch.tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0]])
    monkeypatch.setattr(stats, "dequantize_fp8", _fake_dequant(dequantized))
    source = _source()

    source._record(Sample(name="activation", value=value, metrics=[_mean_metric()]))
    (record,) = source.drain()

    # mean of 1..6, not of 1..6 diluted by the two padding zeros.
    assert record.value.item() == pytest.approx(3.5)


def test_batch_mean_weights_captures_by_their_unpadded_size(monkeypatch):
    """The cross-microbatch fold is weighted by real elements, not by padding."""
    monkeypatch.setattr(stats, "dequantize_fp8", _fake_dequant())
    source = _source()
    source.batch_start(snapshot(SimpleNamespace(step=1, model_parts=[])))

    metric = _mean_metric(accumulable=True)
    source._record(
        Sample(name="act", value=_padded_wrapper([1.0, 1.0], 4), metrics=[metric]),
        accumulate=True,
    )
    source._record(
        Sample(name="act", value=_padded_wrapper([4.0] * 6, 6), metrics=[metric]),
        accumulate=True,
    )
    source.batch_end(snapshot(SimpleNamespace(step=1, model_parts=[])))
    (record,) = source.drain()

    # 1.0 over 2 elements folded with 4.0 over 6, not with the padded 4 and 6.
    assert record.value.item() == pytest.approx(1.0 * 2 / 8 + 4.0 * 6 / 8)


def test_a_capture_is_dequantized_once_for_all_its_metrics(monkeypatch):
    """Every metric re-enters _local_tensor; only the first sees the wrapper."""
    calls = 0
    inner = _fake_dequant()

    def counting(wrapper):
        nonlocal calls
        calls += 1
        return inner(wrapper)

    monkeypatch.setattr(stats, "dequantize_fp8", counting)
    source = _source()
    metrics = [
        MetricConfig(name=op, pipeline=[StatOpConfig(op=op)], reduce=ReduceConfig())
        for op in ("mean", "abs_max", "std", "zero_ratio", "kurtosis")
    ]

    source._record(
        Sample(name="act", value=_padded_wrapper([1.0, 2.0], 4), metrics=metrics)
    )

    assert len(source.drain()) == len(metrics)
    assert calls == 1
