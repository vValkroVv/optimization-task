import json
import math
import re
import runpy
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, create_autospec

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import save_file

from node_runner.src.config_model import (
    CheckpointStatsEvaluatorConfig,
    EvaluatorsConfig,
)
from node_runner.src.config_model import ModelInfoConfig as ModelInfo
from node_runner.src.metrics import metrics_manager
from node_runner.src.metrics.evaluators import checkpoint_stats as evaluator_module
from node_runner.src.metrics.evaluators.checkpoint_stats import (
    CheckpointStatsEvaluator,
)
from node_runner.src.metrics.evaluators.checkpoint_stats import run as worker_module
from node_runner.src.metrics.evaluators.checkpoint_stats.run import (
    CheckpointStatsAnalyzer,
)

pytestmark = pytest.mark.unit

JD = evaluator_module.JD


def _tensors() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "model.embed_tokens.weight": torch.randn(32, 16, generator=g),
        "model.layers.0.mlp.experts.3.up_proj.weight": torch.randn(
            16, 8, generator=g
        ).to(torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(16),
        "model.layers.1.self_attn.q_a_layernorm.weight": torch.ones(4, 4),
    }


def _write_single(hf_dir: Path, tensors: dict | None = None) -> None:
    hf_dir.mkdir(parents=True)
    save_file(tensors or _tensors(), hf_dir / "model.safetensors")


def _write_sharded(hf_dir: Path, with_index: bool) -> None:
    hf_dir.mkdir(parents=True)
    tensors = _tensors()
    names = sorted(tensors)
    shards = {"a.safetensors": names[:2], "b.safetensors": names[2:]}
    for shard, shard_names in shards.items():
        save_file({n: tensors[n] for n in shard_names}, hf_dir / shard)
    if with_index:
        weight_map = {n: shard for shard, ns in shards.items() for n in ns}
        (hf_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )


def _setup_evaluator(
    tmp_path: Path, cfg: CheckpointStatsEvaluatorConfig | None = None
) -> CheckpointStatsEvaluator:
    checkpoint_dir = tmp_path / "run" / "ep0_ba100"
    evaluator = CheckpointStatsEvaluator()
    evaluator.setup(
        autometric_runs_config=cfg or CheckpointStatsEvaluatorConfig(),
        artifacts_path=checkpoint_dir / "artifacts",
        hf_checkpoint_path=checkpoint_dir / "artifacts" / "hf_checkpoint",
        model_info=ModelInfo(model_class="pretrain", size="lite", username="u"),
    )
    return evaluator


def _analyzer(evaluator: CheckpointStatsEvaluator) -> CheckpointStatsAnalyzer:
    return CheckpointStatsAnalyzer(
        evaluator.hf_checkpoint_path,
        evaluator.report_path,
        Namespace(**evaluator.autometric_runs_config.model_dump()),
    )


def _run(tmp_path: Path, cfg: CheckpointStatsEvaluatorConfig | None = None) -> dict:
    evaluator = _setup_evaluator(tmp_path, cfg)
    _write_single(evaluator.hf_checkpoint_path)
    _analyzer(evaluator).run()
    return json.loads(evaluator.report_path.read_text())


class TestConfig:
    def test_defaults(self):
        cfg = CheckpointStatsEvaluatorConfig()
        assert cfg.model_dump() == {
            "device": "cpu",
            "num_threads": 16,
            "role_overrides": {},
            "topk_sigma": 16,
            "svd_niter": 2,
            "spectral_numel_cap": 20 * 1024**3 // 4,
            "include": None,
            "exclude": None,
            "hist_bins": 80,
            "hist_log_min": -8.0,
            "hist_log_max": 4.0,
        }

    @pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1"])
    def test_accepts_known_devices(self, device: str):
        assert CheckpointStatsEvaluatorConfig(device=device).device == device

    def test_accepts_explicit_null_and_valid_filters(self):
        cfg = CheckpointStatsEvaluatorConfig(include=None, exclude=r"\.bias$")
        assert (cfg.include, cfg.exclude) == (None, r"\.bias$")

    @pytest.mark.parametrize(
        "payload",
        [
            {"device": "tpu"},
            {"topk_sigma": -1},
            {"num_threads": 0},
            {"role_overrides": {"(": "attn"}},
            {"role_overrides": {"foo": "bad/role"}},
            {"svd_niter": 0},
            {"spectral_numel_cap": 0},
            {"hist_bins": 0},
            {"hist_log_min": 4.0, "hist_log_max": 4.0},
            {"include": "("},
            {"unknown": 1},
        ],
    )
    def test_rejects_invalid_values(self, payload: dict):
        with pytest.raises(ValidationError):
            CheckpointStatsEvaluatorConfig.model_validate(payload)

    def test_is_iterated_with_the_other_evaluators(self):
        names = [name for name, _ in EvaluatorsConfig(checkpoint_stats={})]
        assert names == ["checkpoint_stats"]

    def test_is_registered_as_an_evaluator(self):
        assert (
            metrics_manager.EVALUATORS["checkpoint_stats"] is CheckpointStatsEvaluator
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "model.embed_tokens.weight",
            {
                "role": "embedding",
                "proj": "embed_tokens",
                "layer": None,
                "expert": None,
                "block": "main",
            },
        ),
        (
            "model.layers.3.mlp.experts.7.down_proj.weight",
            {
                "role": "moe_expert",
                "proj": "down",
                "layer": 3,
                "expert": 7,
                "block": "main",
            },
        ),
        (
            "model.layers.2.post_attention_layernorm.gate_up_projection.weight",
            {
                "role": "layernorm",
                "proj": "post_attention_layernorm_gate_up",
                "layer": 2,
                "expert": None,
                "block": "main",
            },
        ),
        (
            "model.layers.5.eh_proj.weight",
            {
                "role": "mtp",
                "proj": "eh_proj",
                "layer": 5,
                "expert": None,
                "block": "mtp",
            },
        ),
        (
            "model.layers.5.embed_tokens.weight",
            {
                "role": "mtp",
                "proj": "embed_tokens",
                "layer": 5,
                "expert": None,
                "block": "mtp",
            },
        ),
        (
            "model.something_else",
            {
                "role": "unknown",
                "proj": None,
                "layer": None,
                "expert": None,
                "block": "main",
            },
        ),
    ],
)
def test_classify(name: str, expected: dict):
    assert CheckpointStatsAnalyzer.classify(name) == expected


def test_classify_capture_variants(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        CheckpointStatsAnalyzer,
        "CLASS_PATTERNS",
        [
            (re.compile(r"\.(alpha)\.w$"), "custom", "_capture"),
            (re.compile(r"\.beta\.w$"), "custom", "_capture_tail"),
            (re.compile(r"\.gamma\.w$"), "custom", "_capture"),
            (re.compile(r"\.delta\.w$"), "custom", None),
        ],
    )
    assert CheckpointStatsAnalyzer.classify("x.alpha.w")["proj"] == "alpha"
    assert CheckpointStatsAnalyzer.classify("x.beta.w")["proj"] == "tail"
    assert CheckpointStatsAnalyzer.classify("x.gamma.w")["proj"] is None
    assert CheckpointStatsAnalyzer.classify("x.delta.w")["proj"] is None


class TestRun:
    def test_setup_creates_metrics_dir(self, tmp_path: Path):
        evaluator = _setup_evaluator(tmp_path)

        assert evaluator.metrics_dir == (
            tmp_path
            / "run"
            / "ep0_ba100"
            / "artifacts"
            / "metrics"
            / "checkpoint_stats"
        )
        assert evaluator.metrics_dir.is_dir()
        assert evaluator.report_path == evaluator.metrics_dir / "checkpoint_stats.json"

    @pytest.mark.parametrize(
        "cfg",
        [
            CheckpointStatsEvaluatorConfig(),
            CheckpointStatsEvaluatorConfig(
                device="cuda:1",
                num_threads=2,
                role_overrides={r"\.custom\.": "custom"},
                topk_sigma=3,
                svd_niter=4,
                spectral_numel_cap=123,
                include="-expert.*",
                exclude=r"\.bias$",
                hist_bins=12,
                hist_log_min=-1e-5,
                hist_log_max=2.0,
            ),
        ],
    )
    def test_passes_parameters_to_worker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        cfg: CheckpointStatsEvaluatorConfig,
    ):
        evaluator = _setup_evaluator(tmp_path / "checkpoint with spaces", cfg)
        subprocess_manager = Mock(spec=evaluator_module.SubprocessManager)
        manager_factory = Mock(return_value=subprocess_manager)
        monkeypatch.setattr(evaluator_module, "SubprocessManager", manager_factory)

        evaluator.run()

        command = manager_factory.call_args.args[0]
        assert command[:2] == [sys.executable, "-u"]
        assert Path(command[2]) == Path(worker_module.__file__).resolve()
        assert manager_factory.call_args.args[1] == evaluator.log_file
        assert manager_factory.call_args.kwargs["cwd"] is None
        subprocess_manager.start.assert_called_once_with()
        subprocess_manager.wait_for_completion.assert_called_once_with()
        subprocess_manager.log_error.assert_called_once_with()

        analyzer_factory = Mock()
        monkeypatch.setattr(worker_module, "CheckpointStatsAnalyzer", analyzer_factory)
        monkeypatch.setattr(sys, "argv", command[2:])
        worker_module.main()

        hf_dir, output, parsed_config = analyzer_factory.call_args.args
        assert hf_dir == evaluator.hf_checkpoint_path
        assert output == evaluator.report_path
        assert vars(parsed_config) == cfg.model_dump()
        analyzer_factory.return_value.run.assert_called_once_with()

    @pytest.mark.parametrize("exit_code", [1, -9])
    def test_subprocess_failure_reports_exit_code(
        self, tmp_path, monkeypatch, exit_code
    ):
        evaluator = _setup_evaluator(tmp_path)
        process = Mock(spec=evaluator_module.SubprocessManager)
        process.is_successful.return_value = False
        process.exit_code = exit_code
        monkeypatch.setattr(
            evaluator_module, "SubprocessManager", Mock(return_value=process)
        )
        errors = []
        monkeypatch.setattr(evaluator_module.Logger, "error", errors.append)

        evaluator.run()

        assert errors == [
            f"checkpoint_stats failed (exit={exit_code}), see {evaluator.log_file}"
        ]
        assert evaluator.get_metrics() == {}

    def test_report_covers_every_tensor(self, tmp_path: Path):
        blob = _run(tmp_path)

        assert blob["meta"]["n_tensors"] == 4
        assert blob["meta"]["n_errors"] == 0
        assert blob["meta"]["device"] == "cpu"
        assert blob["meta"]["topk_sigma"] == 16
        assert set(blob["tensors"]) == set(_tensors())

        embed = blob["tensors"]["model.embed_tokens.weight"]
        assert embed["role"] == "embedding" and embed["shape"] == [32, 16]
        assert set(embed) >= {"base", "rowcol", "spectral"}
        assert embed["spectral"]["k"] == 16
        assert len(embed["base"]["abs_hist_counts"]) == 80
        assert sum(embed["base"]["abs_hist_counts"]) == 32 * 16

        expert = blob["tensors"]["model.layers.0.mlp.experts.3.up_proj.weight"]
        assert expert["dtype"] == "torch.bfloat16" and expert["expert"] == 3
        assert expert["spectral"]["k"] == 8

        norm = blob["tensors"]["model.layers.0.input_layernorm.weight"]
        assert norm["base"]["n_zero"] == 0 and norm["base"]["n_pos"] == 16
        assert "rowcol" not in norm and "spectral" not in norm

        small = blob["tensors"]["model.layers.1.self_attn.q_a_layernorm.weight"]
        assert "rowcol" in small and "spectral" not in small

    @pytest.mark.parametrize("run_as_script", [False, True])
    def test_headline_metrics_from_a_constant_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_as_script: bool
    ):
        evaluator = _setup_evaluator(tmp_path)
        _write_single(
            evaluator.hf_checkpoint_path,
            {
                "model.embed_tokens.weight": torch.full((32, 16), 0.5),
                "model.layers.0.input_layernorm.weight": torch.ones(16),
            },
        )

        if run_as_script:
            monkeypatch.setattr(sys, "argv", evaluator.command[2:])
            runpy.run_path(str(worker_module.__file__), run_name="__main__")
        else:
            evaluator.run()

        embed_fro = math.sqrt(32 * 16 * 0.25)
        assert evaluator.get_metrics() == pytest.approx(
            {
                "checkpoint_stats/n_tensors": 2.0,
                "checkpoint_stats/n_errors": 0.0,
                "checkpoint_stats/n_unknown": 0.0,
                "checkpoint_stats/n_tensors_with_nan": 0.0,
                "checkpoint_stats/n_tensors_with_inf": 0.0,
                "checkpoint_stats/max_abs": 1.0,
                "checkpoint_stats/fro_norm_total": math.sqrt(embed_fro**2 + 16),
                "checkpoint_stats/embedding/mean_abs_mean": 0.5,
                "checkpoint_stats/embedding/max_abs_max": 0.5,
                "checkpoint_stats/embedding/fro_mean": embed_fro,
                "checkpoint_stats/embedding/max_over_median_max": 1.0,
                "checkpoint_stats/embedding/stable_rank_min": 1.0,
                "checkpoint_stats/layernorm/mean_abs_mean": 1.0,
                "checkpoint_stats/layernorm/max_abs_max": 1.0,
                "checkpoint_stats/layernorm/fro_mean": 4.0,
            },
            rel=1e-4,
        )

    @pytest.mark.parametrize("with_index", [True, False])
    def test_reads_sharded_checkpoints(self, tmp_path: Path, with_index: bool):
        evaluator = _setup_evaluator(tmp_path)
        _write_sharded(evaluator.hf_checkpoint_path, with_index=with_index)

        _analyzer(evaluator).run()

        blob = json.loads(evaluator.report_path.read_text())
        assert set(blob["tensors"]) == set(_tensors())

    def test_include_and_exclude_filter_tensor_names(self, tmp_path: Path):
        blob = _run(
            tmp_path,
            CheckpointStatsEvaluatorConfig(include=r"layers\.", exclude=r"self_attn"),
        )

        assert set(blob["tensors"]) == {
            "model.layers.0.mlp.experts.3.up_proj.weight",
            "model.layers.0.input_layernorm.weight",
        }

    def test_histogram_settings_are_applied(self, tmp_path: Path):
        blob = _run(
            tmp_path,
            CheckpointStatsEvaluatorConfig(
                hist_bins=4, hist_log_min=-2.0, hist_log_max=2.0
            ),
        )

        counts = blob["tensors"]["model.layers.0.input_layernorm.weight"]["base"][
            "abs_hist_counts"
        ]
        assert counts == [0, 0, 16, 0]

    @pytest.mark.parametrize(
        "cfg",
        [
            CheckpointStatsEvaluatorConfig(topk_sigma=0),
            CheckpointStatsEvaluatorConfig(spectral_numel_cap=100),
        ],
    )
    def test_spectral_stats_can_be_switched_off(
        self, tmp_path: Path, cfg: CheckpointStatsEvaluatorConfig
    ):
        blob = _run(tmp_path, cfg)

        assert all("spectral" not in rec for rec in blob["tensors"].values())

    def test_missing_weights_publish_nothing(self, tmp_path: Path):
        evaluator = _setup_evaluator(tmp_path)
        evaluator.hf_checkpoint_path.mkdir()

        with pytest.raises(FileNotFoundError, match=r"No \*\.safetensors"):
            _analyzer(evaluator).run()

        assert not evaluator.report_path.exists()
        assert evaluator.get_metrics() == {}

    def test_existing_report_is_not_recomputed(self, tmp_path: Path, monkeypatch):
        evaluator = _setup_evaluator(tmp_path)
        evaluator.report_path.write_text(
            json.dumps({"meta": {"n_tensors": 0}, "tensors": {}})
        )

        def _fail(*args, **kwargs):
            raise AssertionError("must not recompute an existing report")

        monkeypatch.setattr(evaluator_module, "SubprocessManager", _fail)

        evaluator.run()

        assert evaluator.get_metrics() == {
            "checkpoint_stats/n_tensors": 0.0,
            "checkpoint_stats/n_errors": 0.0,
            "checkpoint_stats/n_unknown": 0.0,
            "checkpoint_stats/n_tensors_with_nan": 0.0,
            "checkpoint_stats/n_tensors_with_inf": 0.0,
        }

    def test_tensor_errors_are_recorded_and_progress_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _setup_evaluator(tmp_path)
        _write_single(evaluator.hf_checkpoint_path)

        def _boom(tensor, role):
            raise ValueError("bad tensor")

        analyzer = _analyzer(evaluator)
        monkeypatch.setattr(analyzer, "_tensor_stats", _boom)
        monkeypatch.setattr(CheckpointStatsAnalyzer, "PROGRESS_EVERY", 2)
        infos: list[str] = []
        monkeypatch.setattr(worker_module.logging, "info", infos.append)

        analyzer.run()

        blob = json.loads(evaluator.report_path.read_text())
        assert blob["meta"]["n_errors"] == 4
        rec = blob["tensors"]["model.embed_tokens.weight"]
        assert rec["error"] == "ValueError('bad tensor')"
        assert rec["role"] == "embedding" and rec["shape"] == [32, 16]
        assert sum("tensors done" in msg for msg in infos) == 2

    def test_progress_empties_cuda_cache(self, tmp_path: Path, monkeypatch):
        evaluator = _setup_evaluator(
            tmp_path, CheckpointStatsEvaluatorConfig(device="cuda:0")
        )
        _write_single(evaluator.hf_checkpoint_path)
        monkeypatch.setattr(CheckpointStatsAnalyzer, "PROGRESS_EVERY", 1)
        emptied: list[bool] = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
        analyzer = _analyzer(evaluator)
        monkeypatch.setattr(analyzer, "_tensor_stats", lambda tensor, role: {})

        analyzer.run()

        assert len(emptied) == 4

    def test_failed_analysis_publishes_nothing(self, tmp_path: Path, monkeypatch):
        evaluator = _setup_evaluator(tmp_path)
        evaluator.hf_checkpoint_path.mkdir()
        (evaluator.hf_checkpoint_path / "model.safetensors.index.json").write_text(
            "invalid JSON"
        )
        errors: list[str] = []
        monkeypatch.setattr(evaluator_module.Logger, "error", errors.append)

        evaluator.run()

        assert not evaluator.report_path.exists()
        assert evaluator.get_metrics() == {}
        assert "JSONDecodeError" in evaluator.log_file.read_text()
        assert any("checkpoint_stats failed" in message for message in errors)
        assert any("is missing" in message for message in errors)

    def test_report_is_strict_json_even_with_bad_weights(self, tmp_path: Path):
        evaluator = _setup_evaluator(tmp_path)
        bad = torch.zeros(16, 16)
        bad[0, 0] = float("nan")
        bad[1, 1] = float("inf")
        _write_single(
            evaluator.hf_checkpoint_path,
            {"model.embed_tokens.weight": bad, "model.norm.weight": torch.ones(16)},
        )

        _analyzer(evaluator).run()

        text = evaluator.report_path.read_text()
        assert "NaN" not in text and "Infinity" not in text

        def _reject(constant):
            raise AssertionError(f"non-finite constant in report: {constant}")

        blob = json.loads(text, parse_constant=_reject)
        embed = blob["tensors"]["model.embed_tokens.weight"]
        assert embed["base"]["n_nan"] == 1 and embed["base"]["n_inf"] == 1
        assert embed["base"]["mean"] is None
        assert math.isfinite(blob["tensors"]["model.norm.weight"]["base"]["mean"])
        assert not evaluator.report_path.with_name("checkpoint_stats.json.tmp").exists()

    def test_failed_write_leaves_no_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _setup_evaluator(tmp_path)
        _write_single(evaluator.hf_checkpoint_path)

        def _boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(worker_module.os, "replace", _boom)

        with pytest.raises(OSError, match="disk full"):
            _analyzer(evaluator).run()

        assert not evaluator.report_path.exists()


def test_spectral_stats_reports_svd_failures(monkeypatch: pytest.MonkeyPatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("svd")

    monkeypatch.setattr(torch, "svd_lowrank", _boom)

    assert CheckpointStatsAnalyzer.spectral_stats(torch.ones(8, 8), k=4, niter=1) == {
        "error": "svd_lowrank_failed: RuntimeError('svd')"
    }


class TestHeadlineMetrics:
    def test_skips_failed_tensors_and_non_finite_values(self):
        report = {
            "meta": {"n_tensors": 3},
            "tensors": {
                "model.layers.0.mlp.experts.0.up_proj.weight": {
                    "role": "moe_expert",
                    "base": {
                        "n_nan": 4,
                        "n_inf": 0,
                        "max_abs": None,
                        "mean_abs": None,
                        "fro": None,
                    },
                    "rowcol": {
                        "row": {"max_over_median": None},
                        "col": {"max_over_median": 2.5},
                    },
                    "spectral": {"error": "svd_lowrank_failed"},
                },
                "model.layers.0.mlp.experts.1.up_proj.weight": {
                    "role": "moe_expert",
                    "base": {
                        "n_nan": 0,
                        "n_inf": 1,
                        "max_abs": 3.0,
                        "mean_abs": 0.5,
                        "fro": 4.0,
                    },
                    "rowcol": {
                        "row": {"max_over_median": 1.5},
                        "col": {"max_over_median": 1.2},
                    },
                    "spectral": {"stable_rank": 7.0},
                },
                "model.layers.0.mlp.experts.2.up_proj.weight": {
                    "role": "moe_expert",
                    "error": "ValueError('bad tensor')",
                },
            },
        }

        assert CheckpointStatsEvaluator.headline_metrics(report) == {
            "checkpoint_stats/n_tensors": 3.0,
            "checkpoint_stats/n_errors": 2.0,
            "checkpoint_stats/n_unknown": 0.0,
            "checkpoint_stats/n_tensors_with_nan": 1.0,
            "checkpoint_stats/n_tensors_with_inf": 1.0,
            "checkpoint_stats/max_abs": 3.0,
            "checkpoint_stats/fro_norm_total": 4.0,
            "checkpoint_stats/moe_expert/mean_abs_mean": 0.5,
            "checkpoint_stats/moe_expert/max_abs_max": 3.0,
            "checkpoint_stats/moe_expert/fro_mean": 4.0,
            "checkpoint_stats/moe_expert/max_over_median_max": 2.5,
            "checkpoint_stats/moe_expert/stable_rank_min": 7.0,
        }

    def test_all_non_finite_row_and_col_norms_are_skipped(self):
        report = {
            "meta": {"n_tensors": 1},
            "tensors": {
                "w": {
                    "role": "mla",
                    "base": {
                        "n_nan": 1,
                        "n_inf": 0,
                        "max_abs": 1.0,
                        "mean_abs": 1.0,
                        "fro": 1.0,
                    },
                    "rowcol": {
                        "row": {"max_over_median": None},
                        "col": {"max_over_median": None},
                    },
                }
            },
        }

        metrics = CheckpointStatsEvaluator.headline_metrics(report)

        assert "checkpoint_stats/mla/max_over_median_max" not in metrics
        assert "checkpoint_stats/mla/stable_rank_min" not in metrics

    def test_mtp_aggregates_and_failed_unknown_tensors_are_visible(self):
        base = {"n_nan": 0, "n_inf": 0, "max_abs": 1.0, "mean_abs": 0.5, "fro": 2.0}
        report = {
            "meta": {"n_tensors": 3},
            "tensors": {
                "main": {"role": "mla", "block": "main", "base": base},
                "mtp": {
                    "role": "mla",
                    "block": "mtp",
                    "base": {**base, "max_abs": 100.0},
                },
                "failed": {"role": "unknown", "block": "main", "error": "OOM"},
            },
        }

        metrics = CheckpointStatsEvaluator.headline_metrics(report)

        assert metrics["checkpoint_stats/n_errors"] == 1
        assert metrics["checkpoint_stats/n_unknown"] == 1
        assert metrics["checkpoint_stats/mla/max_abs_max"] == 1
        assert metrics["checkpoint_stats/mtp_mla/max_abs_max"] == 100
        assert metrics["checkpoint_stats/n_tensors_with_nan"] == 0

    def test_quantization_errors_are_counted_without_role_aggregates(self):
        good = {"n_nan": 0, "n_inf": 0, "max_abs": 2.0, "mean_abs": 1.0, "fro": 4.0}
        report = {
            "meta": {"n_tensors": 3, "n_errors": 2},
            "tensors": {
                "good": {"role": "attn", "base": good},
                "bad": {"role": "attn", "error": "unsupported quantization"},
                "other_bad": {"role": "moe_expert", "error": "invalid FP8 scales"},
            },
        }
        metrics = CheckpointStatsEvaluator.headline_metrics(report)
        assert metrics["checkpoint_stats/n_errors"] == 2
        assert metrics["checkpoint_stats/attn/mean_abs_mean"] == 1
        assert metrics["checkpoint_stats/attn/max_abs_max"] == 2
        assert metrics["checkpoint_stats/fro_norm_total"] == 4
        assert not any(
            key.startswith("checkpoint_stats/moe_expert/") for key in metrics
        )

    def test_empty_report_has_only_counts(self):
        report = {"meta": {"n_tensors": 0}, "tensors": {}}

        assert CheckpointStatsEvaluator.headline_metrics(report) == {
            "checkpoint_stats/n_tensors": 0.0,
            "checkpoint_stats/n_errors": 0.0,
            "checkpoint_stats/n_unknown": 0.0,
            "checkpoint_stats/n_tensors_with_nan": 0.0,
            "checkpoint_stats/n_tensors_with_inf": 0.0,
        }


class TestAfterRun:
    def test_after_run_sends_and_archives(self, tmp_path: Path, monkeypatch):
        evaluator = _setup_evaluator(tmp_path)
        calls: list = []
        tar_path = tmp_path / "metrics.tar"
        monkeypatch.setattr(
            evaluator, "_send_metrics_to_jd", lambda metrics: calls.append("jd")
        )
        monkeypatch.setattr(
            JD, "tar_metrics_dir", lambda metrics_dir, artifacts_path, _name: tar_path
        )
        monkeypatch.setattr(
            JD,
            "symlink_metrics_for_transfer",
            lambda tar, hf_checkpoint_path, _name: calls.append(("symlink", tar)),
        )

        evaluator.after_run({"checkpoint_stats/n_tensors": 2.0})

        assert calls == ["jd", ("symlink", tar_path)]

    def test_send_metrics_to_jd_builds_payload(self, tmp_path: Path, monkeypatch):
        evaluator = _setup_evaluator(tmp_path)
        send_metrics = create_autospec(JD.send_evaluator_metrics)
        monkeypatch.setattr(JD, "send_evaluator_metrics", send_metrics)

        evaluator._send_metrics_to_jd({"checkpoint_stats/n_tensors": 2.0})

        send_metrics.assert_called_once_with(
            hf_checkpoint_path=evaluator.hf_checkpoint_path,
            metrics_dir=evaluator.metrics_dir,
            jd_metrics={"checkpoint_stats/n_tensors": 2.0},
            shooter_name="checkpoint_stats",
            shooter_version="1.0.0",
            model_info=evaluator.model_info,
        )

    def test_send_metrics_to_jd_skips_empty_metrics(self, tmp_path: Path, monkeypatch):
        evaluator = _setup_evaluator(tmp_path)

        def _fail(**kwargs):
            raise AssertionError("must not contact JD without metrics")

        monkeypatch.setattr(JD, "send_evaluator_metrics", _fail)

        evaluator._send_metrics_to_jd({})
