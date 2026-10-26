"""GFSDP2 guards: what ``GFSDP2Backend`` refuses, and the monitoring it keeps.

The guards are reached through ``build_sharding_backend``, which is the only
way a run gets a backend -- so a trainer cannot skip them by forgetting a call.
"""

import logging
from dataclasses import replace

import pytest

pytest.importorskip("streaming")
pytest.importorskip("tree_sitter")

from torchtitan.callbacks import MonitoringCallback
from torchtitan.callbacks.metrics.sample import MetricConfig, StatOpConfig
from torchtitan.callbacks.sources import (
    ModellingSource,
    MoECaptureConfig,
    OptimizerSource,
    TrainerSource,
)
from torchtitan.config.configs import ParallelismConfig
from torchtitan.distributed.backends import build_sharding_backend
from torchtitan.config.decode import decode
from torchtitan.config.loader import load_raw_document
from torchtitan.config.sections import ModelSection


def _model_config(family="gift_moe"):
    document = load_raw_document(f"configs/presets/models/{family}/debugmodel.yaml")
    return decode(ModelSection, document.data["model"]).config


def _validate(parallelism: ParallelismConfig) -> None:
    build_sharding_backend(parallelism, _model_config())


def _filtered(config: MonitoringCallback.Config) -> MonitoringCallback.Config:
    backend = build_sharding_backend(_config(fsdp_backend="gfsdp2"), _model_config())
    return replace(config, sources=backend.filter_monitor_sources(config.sources))


def _config(**parallelism) -> ParallelismConfig:
    return ParallelismConfig(**parallelism)


def _monitoring() -> MonitoringCallback.Config:
    """Whatislove's monitoring shape, cut to the parts the guard touches.

    Deliberately not the conventional source keys: the guard matches on the
    config type, and spelling them differently here is what proves it.
    """
    return MonitoringCallback.Config(
        sources={
            "params": OptimizerSource.Config(
                metrics=[
                    MetricConfig(name="l2norm", pipeline=[StatOpConfig(op="norm")])
                ]
            ),
            "modules": ModellingSource.Config(
                moe=MoECaptureConfig(
                    expert_weights=True,
                    token_metrics=[MetricConfig(name="max_vio")],
                    weight_stats=[
                        MetricConfig(
                            name="weights_l2norm",
                            pipeline=[StatOpConfig(op="norm")],
                        )
                    ],
                    weight_spread=[
                        MetricConfig(name="norm_std", pipeline=[StatOpConfig(op="std")])
                    ],
                ),
            ),
            "loop": TrainerSource.Config(log_lr=True),
        }
    )


def test_gfsdp2_refuses_tensor_parallelism():
    with pytest.raises(ValueError, match="tensor parallelism"):
        _validate(_config(fsdp_backend="gfsdp2", tensor_parallel_degree=2))


def test_a_tensor_parallel_degree_of_one_is_not_tensor_parallelism():
    """The default. Refusing it would ban the backend outright."""
    _validate(_config(fsdp_backend="gfsdp2", tensor_parallel_degree=1))


def test_the_guard_is_inert_for_the_other_backend():
    _validate(_config(fsdp_backend="fsdp2", tensor_parallel_degree=2))


# --- the monitoring that reads sharded parameters ---------------------------


def test_gfsdp2_keeps_the_optimizer_source():
    """Its statistics run through the partial-stats path over ragged shards."""
    sources = _filtered(_monitoring()).sources
    assert "params" in sources


def test_gfsdp2_keeps_expert_weights_and_what_summarises_them():
    """Per-expert weight stats reduce the flat shards by expert segments."""
    moe = _filtered(_monitoring()).sources["modules"].moe
    assert moe.expert_weights is True
    assert moe.weight_stats
    assert moe.weight_spread


def test_gfsdp2_accepts_new_expert_weight_stats():
    config = _monitoring()
    config.sources["modules"].moe.weight_stats = [
        MetricConfig(name=op, pipeline=[StatOpConfig(op=op)])
        for op in (
            "min",
            "max",
            "count",
            "nonzero_count",
            "nan_count",
            "inf_count",
            "zero_ratio",
            "finite_ratio",
            "positive_ratio",
            "std",
            "var",
            "cv",
            "kurtosis",
            "quantile",
            "min_median_ratio",
            "gini",
        )
    ] + [MetricConfig(name="norm0", pipeline=[StatOpConfig(op="norm", p=0)])]
    assert len(_filtered(config).sources["modules"].moe.weight_stats) == 17


@pytest.mark.parametrize("p", [-1.0, -2.0, float("-inf")])
def test_gfsdp2_accepts_negative_expert_norm(p):
    config = _monitoring()
    config.sources["modules"].moe.weight_stats = [
        MetricConfig(name="norm_negative", pipeline=[StatOpConfig(op="norm", p=p)])
    ]
    assert _filtered(config).sources["modules"].moe.weight_stats


def test_gfsdp2_rejects_nan_expert_norm():
    config = _monitoring()
    config.sources["modules"].moe.weight_stats = [
        MetricConfig(
            name="norm_nan", pipeline=[StatOpConfig(op="norm", p=float("nan"))]
        )
    ]
    with pytest.raises(ValueError, match="non-NaN norm p"):
        _filtered(config)


def test_gfsdp2_keeps_the_monitoring_that_reads_activations():
    """Router and trainer statistics are computed from all-gathered tensors."""
    sources = _filtered(_monitoring()).sources
    assert "loop" in sources
    assert sources["modules"].moe.token_metrics


def test_the_users_config_is_not_mutated():
    config = _monitoring()
    _filtered(config)
    assert "params" in config.sources
    assert config.sources["modules"].moe.expert_weights is True


# --- pipeline parallelism and resharding ------------------------------------


def test_gfsdp2_refuses_resharding_under_a_pipeline():
    """Two data buffers per process group cannot cover a pipeline's overlap."""
    with pytest.raises(ValueError, match="cannot run that combination"):
        _validate(
            _config(
                fsdp_backend="gfsdp2",
                pipeline_parallel_degree=2,
                pipeline_parallel_schedule="Interleaved1F1B",
                fsdp_reshard_after_forward="always",
            )
        )


def test_a_pipeline_without_resharding_is_allowed():
    """What the production presets run, and what the acceptance measured."""
    _validate(
        _config(
            fsdp_backend="gfsdp2",
            pipeline_parallel_degree=4,
            pipeline_parallel_schedule="Interleaved1F1B",
            fsdp_reshard_after_forward="never",
        )
    )


def test_the_default_reshard_policy_passes_under_a_pipeline():
    """``default`` resolves to never once a pipeline is on."""
    _validate(
        _config(
            fsdp_backend="gfsdp2",
            pipeline_parallel_degree=4,
            pipeline_parallel_schedule="Interleaved1F1B",
        )
    )


# --- the pipeline schedule --------------------------------------------------


@pytest.mark.parametrize(
    "schedule", ["1F1B", "GPipe", "InterleavedZeroBubble", "ZBVZeroBubble"]
)
def test_gfsdp2_refuses_every_pipeline_schedule_but_interleaved_1f1b(schedule):
    """Including the config default ``1F1B``, which a preset gets by silence."""
    with pytest.raises(ValueError, match="runs only 'Interleaved1F1B'"):
        _validate(
            _config(
                fsdp_backend="gfsdp2",
                pipeline_parallel_degree=2,
                pipeline_parallel_schedule=schedule,
            )
        )


def test_gfsdp2_refuses_a_csv_schedule_under_a_pipeline():
    """The csv path replaces the named schedule, so the name proves nothing."""
    with pytest.raises(ValueError, match="pipeline_parallel_schedule_csv"):
        _validate(
            _config(
                fsdp_backend="gfsdp2",
                pipeline_parallel_degree=2,
                pipeline_parallel_schedule="Interleaved1F1B",
                pipeline_parallel_schedule_csv="/tmp/schedule.csv",
            )
        )


def test_the_schedule_is_not_read_without_a_pipeline():
    """At ``pp=1`` no schedule is built, so the field is dead config."""
    _validate(
        _config(
            fsdp_backend="gfsdp2",
            pipeline_parallel_degree=1,
            pipeline_parallel_schedule="GPipe",
        )
    )


def test_the_other_backend_keeps_every_schedule():
    _validate(
        _config(
            fsdp_backend="fsdp2",
            pipeline_parallel_degree=2,
            pipeline_parallel_schedule="ZBVZeroBubble",
        )
    )


def test_resharding_without_a_pipeline_is_left_alone():
    """The refusal is about the combination, not about resharding."""
    _validate(
        _config(
            fsdp_backend="gfsdp2",
            pipeline_parallel_degree=1,
            fsdp_reshard_after_forward="always",
        )
    )


def test_the_other_backend_keeps_its_resharding():
    _validate(
        _config(
            fsdp_backend="fsdp2",
            pipeline_parallel_degree=4,
            fsdp_reshard_after_forward="always",
        )
    )


def test_gfsdp2_refuses_a_model_it_is_not_implemented_for():
    """The silent case this guard exists for.

    ``apply_gfsdp`` is reached only from gift_moe's parallelize function.
    Every other model calls ``apply_fsdp`` without reading
    ``parallelism.fsdp_backend``, so before this refusal the run sharded with
    fsdp2 and reported gfsdp2 -- in the logs, in the config dump and in the
    checkpoint metadata.
    """
    with pytest.raises(ValueError, match="not supported by model type 'llama3'"):
        build_sharding_backend(_config(fsdp_backend="gfsdp2"), _model_config("llama3"))


def test_gfsdp2_takes_the_model_it_implements():
    backend = build_sharding_backend(_config(fsdp_backend="gfsdp2"), _model_config())
    assert backend.name == "gfsdp2"


@pytest.mark.parametrize("model_name", ["gift_moe", "llama3", "deepseek_v3", "qwen3"])
def test_fsdp2_takes_every_model(model_name):
    """The asymmetry is the point: only gfsdp2 is model-specific."""
    assert build_sharding_backend(_config(), _model_config(model_name)).name == "fsdp2"


@pytest.mark.parametrize("enabled", [False, True])
def test_gfsdp2_matches_fsdp2_init_only_inside_its_weight_init_context(enabled):
    from torchtitan.distributed.gfsdp import _random
    from torchtitan.protocols.module import Module

    backend = build_sharding_backend(
        _config(fsdp_backend="gfsdp2", gfsdp_match_fsdp2_init=enabled),
        _model_config(),
    )
    assert not _random._match_fsdp2_init
    with backend.weight_init_context(Module()):
        assert _random._match_fsdp2_init is enabled
    assert not _random._match_fsdp2_init


@pytest.mark.parametrize(
    ("version", "warns"),
    [("2.15.0.dev20260822+cu130", False), ("2.16.0+cu130", True)],
)
def test_fsdp2_matching_init_warns_on_an_untested_torch(
    monkeypatch, caplog, version, warns
):
    import torch

    from torchtitan.distributed.gfsdp import _random

    monkeypatch.setattr(
        _random, "_FSDP2_MATCHING_INIT_TORCH_VERSION", "2.15.0.dev20260822"
    )
    monkeypatch.setattr(torch, "__version__", version)
    with caplog.at_level(logging.WARNING, logger="gfsdp"):
        with _random.match_fsdp2_init():
            pass
    messages = [r.getMessage() for r in caplog.records]
    assert any("may silently stop matching" in m for m in messages) is warns


def test_fsdp2_warns_that_it_ignores_gfsdp_match_fsdp2_init(caplog):
    with caplog.at_level(logging.WARNING):
        _validate(_config())
        assert not caplog.records, "the default must not warn"
        _validate(_config(gfsdp_match_fsdp2_init=True))
    assert any("gfsdp_match_fsdp2_init" in r.getMessage() for r in caplog.records)
