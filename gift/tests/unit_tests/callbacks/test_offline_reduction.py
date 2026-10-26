# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline reduction path and interleaved metric steps.

``reduce.mode != "online"`` means: compute the stat, keep the rank-local
value (no collectives), persist the raw record through the disk store for a
later out-of-band merge — the emitter must not publish the unreduced value.
Interleaved ``steps`` (a stat AFTER a reduce) run through
the inline compute path and are online-only.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from torchtitan.callbacks.metrics.emitters import MetricEmitter
from torchtitan.callbacks.metrics.reducers import (
    finalize_pending_reductions,
    ParallelReducer,
)
from torchtitan.callbacks.metrics.sample import MetricRecord
from torchtitan.config.schema.callbacks.metrics.sample import (
    AxisSpec,
    MetricConfig,
    MetricStepConfig,
    ReduceConfig,
    ReduceStageConfig,
    StatOpConfig,
)
from torchtitan.callbacks.metrics.stats import (
    apply_post_stats,
    MetricPipeline,
)
from torchtitan.callbacks.metrics.stores import MetricStore
from torchtitan.config.schema.callbacks.metrics.stores import MetricStoreConfig
from tests.unit_tests.callbacks.trainer_stub import snapshot


class FakeParallelDims:
    pp_enabled = False

    def get_optional_mesh(self, axis):
        return None


def make_pipeline(metric):
    return MetricPipeline(metric, ParallelReducer(parallel_dims=FakeParallelDims()))


def offline_mean_metric():
    return MetricConfig(
        name="mean",
        pipeline=[StatOpConfig(op="mean")],
        reduce=ReduceConfig(mode="offline", group="dp", op="mean"),
    )


class TestOfflineComputeStats(unittest.TestCase):
    """compute_stats keeps offline metrics local and collective-free."""

    def test_offline_metric_is_rejected_up_front(self):
        """The offline pipeline is untested end to end, so it fails fast."""
        with self.assertRaisesRegex(ValueError, "only ReduceMode.online"):
            make_pipeline(offline_mean_metric()).compute_stats(
                torch.arange(4.0), AxisSpec()
            )

    def test_online_metric_defers_its_reduce_stage(self):
        """The same metric with mode='online' carries a pending stage."""
        metric = MetricConfig(
            name="mean",
            pipeline=[StatOpConfig(op="mean")],
            reduce=ReduceConfig(group="dp", op="mean"),
        )
        _, pending, post = make_pipeline(metric).compute_stats(
            torch.arange(4.0), AxisSpec()
        )
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].op, "mean")

    def test_interleaved_steps_reject_offline_mode(self):
        """A stat after a reduce needs the online collective -> ValueError."""
        metric = MetricConfig(
            name="std",
            steps=[
                MetricStepConfig(
                    kind="reduce",
                    reduce=ReduceStageConfig(group="loss", op="sum"),
                ),
                MetricStepConfig(kind="stat", stat=StatOpConfig(op="std")),
            ],
            reduce=ReduceConfig(mode="offline"),
        )
        with pytest.raises(ValueError, match="interleaves a stat after a reduce"):
            make_pipeline(metric).compute_stats(torch.arange(4.0), AxisSpec())


class TestInterleavedSteps(unittest.TestCase):
    """`steps` runs a stat AFTER a reduce — the reason the form exists.

    The expert-distribution metrics need it: the per-expert token histogram is
    summed over the data ranks first, and only then reduced to mean/std/cv, so
    the statistic describes the global batch instead of one rank's slice.
    """

    def test_stat_after_reduce_defers_the_whole_chain(self):
        # [reduce, stat] used to run its collective inline — one per metric
        # per layer, unfusable. Now the reduce comes back as a pending stage
        # and the stat as post_stats, finished after the fused reductions.
        metric = MetricConfig(
            name="mean_after_sum",
            steps=[
                MetricStepConfig(
                    kind="reduce",
                    reduce=ReduceStageConfig(group="loss", op="sum"),
                ),
                MetricStepConfig(kind="stat", stat=StatOpConfig(op="mean")),
            ],
        )
        value = torch.tensor([1.0, 3.0, 5.0, 7.0])
        result, pending, post = make_pipeline(metric).compute_stats(value, AxisSpec())
        self.assertEqual(pending, (ReduceStageConfig(group="loss", op="sum"),))
        self.assertEqual(len(post), 1)
        self.assertTrue(torch.equal(result, value))

        # Drive the deferred chain to the end on one rank: the collective is a
        # no-op, so the published value is just mean(value).
        records = [
            MetricRecord(
                name="moe/x/mean_after_sum",
                value=result,
                step=10,
                reduce=ReduceConfig(),
                pending_stages=pending,
                post_stats=post,
            )
        ]
        reducer = ParallelReducer(parallel_dims=FakeParallelDims())
        records = apply_post_stats(finalize_pending_reductions(records, reducer))
        self.assertEqual(records[0].value.item(), pytest.approx(4.0))
        self.assertEqual(records[0].post_stats, ())


class TestDiskStoreRoundtrip(unittest.TestCase):
    """Disk store persists raw records for a later out-of-band merge."""

    def test_store_writes_one_file_per_record(self):
        """The disk store keeps every rank's raw record for a later merge."""
        reduce = ReduceConfig(mode="offline", group="dp", op="mean")
        records = [
            MetricRecord(
                name="model/x/mean", value=torch.tensor(1.0), step=10, reduce=reduce
            ),
            MetricRecord(
                name="model/x/mean", value=torch.tensor(3.0), step=10, reduce=reduce
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(MetricStoreConfig(mode="disk", directory=tmp))
            for record in records:
                store.write(record)
            files = list(Path(tmp).glob("step_10/rank_0/*.pt"))
            self.assertEqual(len(files), 2)


class TestEmitterSkipsOfflineRecords(unittest.TestCase):
    """The emitter publishes online records only."""

    def test_offline_record_is_not_logged(self):
        logged = []
        trainer = SimpleNamespace(
            parallel_dims=FakeParallelDims(),
            metrics_processor=SimpleNamespace(
                log_extra_metrics=lambda step, metrics: logged.append(metrics)
            ),
        )
        online = MetricRecord(
            name="online/mean", value=torch.tensor(1.0), step=1, reduce=ReduceConfig()
        )
        offline = MetricRecord(
            name="offline/mean",
            value=torch.tensor(9.0),
            step=1,
            reduce=ReduceConfig(mode="offline"),
        )
        MetricEmitter().emit(
            [online, offline],
            parallel_dims=trainer.parallel_dims,
            publish=snapshot(trainer, step=1).publish,
        )
        (metrics,) = logged
        self.assertIn("online/mean", metrics)
        self.assertNotIn("offline/mean", metrics)


if __name__ == "__main__":
    unittest.main()
