import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from node_runner.src.config_model import (
    DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE,
    DEFAULT_REQUEST_TIMEOUT,
    AaLcrEvaluatorConfig,
    EvaluatorsConfig,
)
from node_runner.src.config_model import (
    ModelInfoConfig as ModelInfo,
)
from node_runner.src.metrics.evaluators import aa_lcr as aa_lcr_module
from node_runner.src.metrics.evaluators.aa_lcr import AaLcrEvaluator
from node_runner.src.metrics.metrics_manager import EVALUATORS
from sber_tests.test_support import judge_endpoints

pytestmark = pytest.mark.unit

VERSION_SHA = "bddfcacf6aac279b7dbabba57a2c04d85dbaa306"


def _write_dataset(path: Path, question_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["question_id,question,answer"]
    lines += [f"{question_id},q,a" for question_id in question_ids]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_generation_manifest(evaluator: AaLcrEvaluator, records: dict) -> Path:
    outputs_dir = Path(evaluator.metrics_dir) / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = outputs_dir / "manifest.json"
    manifest_path.write_text(json.dumps(records), encoding="utf-8")
    return manifest_path


def _make_config(tmp_path: Path, **raw_config) -> AaLcrEvaluatorConfig:
    raw_config.setdefault("runner_root", str(tmp_path / "bench" / "runner"))
    raw_config.setdefault("dataset_csv", str(tmp_path / "bench" / "data" / "set.csv"))
    raw_config.setdefault("text_dir", str(tmp_path / "bench" / "data" / "texts"))
    raw_config.setdefault("judge", {"model_name": "Qwen3-235B-A22B-Instruct-2507-FP8"})
    cfg = AaLcrEvaluatorConfig(**raw_config)
    _apply_defaults(cfg, 2, "http://localhost:29999", 1, "http://127.0.0.1:31000")
    return cfg


def _apply_defaults(
    cfg: AaLcrEvaluatorConfig,
    inference_instances: int,
    base_url: str,
    judge_instances: int,
    judge_url: str,
) -> None:
    cfg.judge_endpoints = judge_endpoints(
        {cfg.judge.model_name: judge_url}, {cfg.judge.model_name: judge_instances}
    )
    cfg.apply_defaults(
        inference_instances=inference_instances,
        base_url=base_url,
    )


def _build_evaluator(
    tmp_path: Path, checkpoint: str = "hf_ckpt", **raw_config
) -> AaLcrEvaluator:
    cfg = _make_config(tmp_path, **raw_config)
    runner_root = Path(cfg.runner_root)
    runner_root.mkdir(parents=True, exist_ok=True)
    (runner_root / "run_aa_lcr.py").write_text("", encoding="utf-8")
    (runner_root.parent / "aa_lcr_version.txt").write_text(
        VERSION_SHA + "\n", encoding="utf-8"
    )
    _write_dataset(Path(cfg.dataset_csv), ["q1", "q2"])
    text_dir = Path(cfg.text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / "doc.txt").write_text("document\n", encoding="utf-8")
    evaluator = AaLcrEvaluator()
    evaluator.setup(
        autometric_runs_config=cfg,
        artifacts_path=tmp_path / "artifacts",
        hf_checkpoint_path=tmp_path / checkpoint,
        model_info=ModelInfo(model_class="sft", size="lite", username="u"),
    )
    return evaluator


class TestAaLcrConfigDefaults:
    def test_judge_model_name_from_config(self, tmp_path: Path):
        cfg = _make_config(tmp_path, judge={"model_name": "gpt-oss-120b"})
        assert cfg.secondary_model_names == ["gpt-oss-120b"]

    def test_empty_judge_model_name_raises(self, tmp_path: Path):
        with pytest.raises(ValidationError):
            _make_config(tmp_path, judge={"model_name": ""})

    def test_missing_judge_raises(self):
        with pytest.raises(ValidationError):
            AaLcrEvaluatorConfig()

    @pytest.mark.parametrize(
        ("section", "value"),
        [
            ("model", {"num_retries": 0}),
            ("model", {"max_tokens": -1}),
            ("model", {"temperature": "hot"}),
            ("model", {"temperature": True}),
            ("model", {"temperature": "0.5"}),
            ("model", {"top_p": True}),
            ("model", {"top_k": -1}),
            ("model", {"timeout": -1}),
            ("model", {"api_key": "secret"}),
            ("model", {"name": "other-model"}),
            ("model", {"unknown_option": 1}),
            ("judge", {"num_retries": 0}),
            ("judge", {"max_tokens": 0}),
            ("judge", {"temperature": 3}),
            ("judge", {"model_name": 123}),
            ("judge", {"model_name": "  "}),
        ],
    )
    def test_invalid_or_runner_managed_request_fields_are_rejected(
        self, tmp_path: Path, section: str, value: object
    ):
        with pytest.raises(ValidationError):
            _make_config(tmp_path, **{section: value})

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValidationError):
            AaLcrEvaluatorConfig(engine_type="external")

    @pytest.mark.parametrize(
        "config",
        [
            {"n_repeats": True},
            {"llm_timeout": "10"},
            {"parallel": False},
            {"limit": "1"},
            {"use_gold_as_candidate": "yes"},
            {"use_gold_as_candidate": 1},
            {"dataset_csv": "  "},
            {"text_dir": ""},
            {"runner_root": "  "},
            {"python_path": ""},
        ],
    )
    def test_invalid_top_level_values_are_rejected(self, config: dict):
        with pytest.raises(ValidationError):
            AaLcrEvaluatorConfig(**config)

    def test_image_paths_are_the_defaults(self):
        cfg = AaLcrEvaluatorConfig(judge={"model_name": "judge"})
        assert cfg.dataset_csv == "/home/user/aa_lcr/data/AA-LCR_Dataset.csv"
        assert cfg.text_dir == "/home/user/aa_lcr/data/extracted_text"
        assert cfg.runner_root == "/home/user/aa_lcr/runner"

    def test_runner_python_defaults_to_current_interpreter(self):
        assert (
            AaLcrEvaluatorConfig(judge={"model_name": "judge"}).runner_python
            == sys.executable
        )

    def test_explicit_none_python_path_uses_current_interpreter(self):
        cfg = AaLcrEvaluatorConfig(python_path=None, judge={"model_name": "judge"})
        assert cfg.runner_python == sys.executable

    def test_runner_python_prefers_configured_path(self):
        cfg = AaLcrEvaluatorConfig(
            python_path="/opt/py/bin/python", judge={"model_name": "judge"}
        )
        assert cfg.runner_python == "/opt/py/bin/python"


class TestAaLcrConfigApplyDefaults:
    def test_fills_parallelism_from_inference_instances(self):
        cfg = AaLcrEvaluatorConfig(judge={"model_name": "judge"})
        _apply_defaults(cfg, 2, "http://h:1", 3, "http://j:2")
        assert cfg.base_url == "http://h:1/v1"
        judge = cfg.build_overrides(
            out_dir=Path("/tmp/aa_lcr"), checkpoint_model_name="m", phase="evaluation"
        )["judge"]
        assert judge["api_base"] == "http://j:2/v1"
        assert cfg.parallel == 2 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        assert cfg.judge_parallel == 3 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE

    def test_explicit_yaml_parallelism_wins(self):
        cfg = AaLcrEvaluatorConfig(
            parallel=3, judge_parallel=5, judge={"model_name": "judge"}
        )
        _apply_defaults(cfg, 8, "http://h:1", 8, "http://j:2")
        assert cfg.parallel == 3
        assert cfg.judge_parallel == 5


class TestAaLcrConfigBuildOverrides:
    def test_generation_payload_carries_timeout_and_retries(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        overrides = cfg.build_overrides(
            out_dir=tmp_path / "out", checkpoint_model_name="ckpt", phase="generation"
        )
        assert overrides["model"]["timeout"] == DEFAULT_REQUEST_TIMEOUT
        assert overrides["model"]["num_retries"] == 20
        assert overrides["model"]["max_tokens"] == 16384
        assert overrides["model"]["temperature"] == 0.0
        assert overrides["model"]["api_base"] == "http://localhost:29999/v1"
        assert overrides["model"]["name"] == "ckpt"
        assert (
            overrides["model"]["concurrency"]
            == 2 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        )
        assert overrides["n_repeats"] == 3
        assert overrides["use_gold_as_candidate"] is False
        assert "judge" not in overrides
        assert "limit" not in overrides

    def test_evaluation_payload_carries_judge(self, tmp_path: Path):
        cfg = _make_config(tmp_path, limit=5)
        overrides = cfg.build_overrides(
            out_dir=tmp_path / "out", checkpoint_model_name="ckpt", phase="evaluation"
        )
        judge = overrides["judge"]
        assert judge["model_name"] == "Qwen3-235B-A22B-Instruct-2507-FP8"
        assert judge["api_base"] == "http://127.0.0.1:31000/v1"
        assert judge["timeout"] == DEFAULT_REQUEST_TIMEOUT
        assert judge["num_retries"] == 20
        assert judge["max_tokens"] == 4096
        assert judge["concurrency"] == DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        assert overrides["limit"] == 5

    def test_yaml_tuning_overrides_win(self, tmp_path: Path):
        cfg = _make_config(
            tmp_path,
            llm_timeout=123,
            model={"max_tokens": 999},
            judge={"model_name": "judge", "max_tokens": 11, "num_retries": 2},
        )
        overrides = cfg.build_overrides(
            out_dir=tmp_path / "out", checkpoint_model_name="ckpt", phase="evaluation"
        )
        assert overrides["model"]["timeout"] == 123
        assert overrides["model"]["max_tokens"] == 999
        assert overrides["model"]["name"] == "ckpt"
        assert overrides["judge"]["timeout"] == 123
        assert overrides["judge"]["max_tokens"] == 11
        assert overrides["judge"]["num_retries"] == 2

    def test_candidate_name_falls_back_to_model(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        assert cfg.candidate_model_name("") == "model"

    def test_raises_without_apply_defaults(self, tmp_path: Path):
        cfg = AaLcrEvaluatorConfig(parallel=1, judge={"model_name": "judge"})
        with pytest.raises(ValueError, match="base_url must be set"):
            cfg.build_overrides(
                out_dir=tmp_path, checkpoint_model_name="c", phase="generation"
            )

    def test_raises_when_parallel_unset(self, tmp_path: Path):
        cfg = AaLcrEvaluatorConfig(judge={"model_name": "judge"})
        with pytest.raises(ValueError, match="parallel must be set"):
            cfg.build_overrides(
                out_dir=tmp_path, checkpoint_model_name="c", phase="generation"
            )

    def test_evaluation_raises_without_judge_defaults(self, tmp_path: Path):
        cfg = AaLcrEvaluatorConfig(
            judge={"model_name": "judge"}, parallel=1, base_url="http://h:1/v1"
        )
        with pytest.raises(ValueError, match="apply_defaults"):
            cfg.build_overrides(
                out_dir=tmp_path, checkpoint_model_name="c", phase="evaluation"
            )

    def test_rejects_unsupported_phase(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        with pytest.raises(ValueError, match="unsupported phase"):
            cfg.build_overrides(
                out_dir=tmp_path, checkpoint_model_name="c", phase="invalid"
            )


class TestAaLcrEvaluatorSetup:
    def test_setup_creates_metrics_dir(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert evaluator.metrics_dir == tmp_path / "artifacts" / "metrics" / "aa_lcr"
        assert evaluator.metrics_dir.is_dir()

    def test_successful_generation_records_effective_semantic_identity(
        self, tmp_path: Path, monkeypatch
    ):
        evaluator = _build_evaluator(tmp_path, limit=1, n_repeats=2)
        _write_generation_manifest(
            evaluator,
            {f"q1__r{repeat}": {"status": "ok"} for repeat in (1, 2)},
        )
        monkeypatch.setattr(aa_lcr_module, "SubprocessManager", MagicMock())

        evaluator.run_phase("generation")

        setup = json.loads(evaluator._setup_path.read_text(encoding="utf-8"))
        identity = setup["semantic_identity"]
        assert identity["shooter_version"] == VERSION_SHA
        assert len(identity["data_fingerprint"]) == 64
        assert identity["limit"] == 1
        assert identity["n_repeats"] == 2
        assert (
            identity["judge"]["model_name"]
            == evaluator.autometric_runs_config.judge.model_name
        )
        assert setup["data_locations"]["dataset_csv"] == str(
            tmp_path / "bench" / "data" / "set.csv"
        )
        assert "concurrency" not in json.dumps(setup)
        assert "api_base" not in json.dumps(setup)

    def test_data_mount_location_does_not_change_semantic_identity(
        self, tmp_path: Path
    ):
        first = _build_evaluator(tmp_path / "first")
        second = _build_evaluator(tmp_path / "second")
        assert (
            second._setup_identity["data_fingerprint"]
            == first._setup_identity["data_fingerprint"]
        )
        assert second._jd_metric_name("accuracy") == first._jd_metric_name("accuracy")

    def test_invalid_utf8_setup_disables_generation_reuse(self, tmp_path: Path):
        first = _build_evaluator(tmp_path)
        first._setup_path.write_bytes(b"\xff")

        restarted = AaLcrEvaluator()
        restarted.setup(
            first.autometric_runs_config,
            first.artifacts_path,
            first.hf_checkpoint_path,
            first.model_info,
        )

        assert restarted._reuse_generation is False
        assert restarted._setup_path.exists() is False

    def test_stale_setup_unlink_error_does_not_escape_setup(
        self, tmp_path: Path, monkeypatch
    ):
        first = _build_evaluator(tmp_path)
        first._setup_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            Path, "unlink", MagicMock(side_effect=OSError("read-only cache"))
        )

        restarted = AaLcrEvaluator()
        restarted.setup(
            first.autometric_runs_config,
            first.artifacts_path,
            first.hf_checkpoint_path,
            first.model_info,
        )

        assert restarted._reuse_generation is False

    def test_evaluator_config_is_registered(self):
        registered = dict(EvaluatorsConfig(aa_lcr={"judge": {"model_name": "judge"}}))
        assert isinstance(registered["aa_lcr"], AaLcrEvaluatorConfig)
        assert EVALUATORS["aa_lcr"] is AaLcrEvaluator


class TestAaLcrRun:
    @pytest.mark.parametrize("phase", ["generation", "evaluation"])
    def test_missing_runner_fails_the_phase(self, tmp_path: Path, phase):
        evaluator = _build_evaluator(tmp_path)
        (Path(evaluator.autometric_runs_config.runner_root) / "run_aa_lcr.py").unlink()

        with pytest.raises(RuntimeError, match="runner not found"):
            evaluator.run_phase(phase)

    def test_subprocess_start_error_fails_the_phase(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        sm_cls = MagicMock()
        sm_cls.return_value.start.side_effect = FileNotFoundError("python not found")
        monkeypatch.setattr(aa_lcr_module, "SubprocessManager", sm_cls)

        with pytest.raises(FileNotFoundError):
            evaluator.run_phase("evaluation")

    def test_generation_runs_subprocess_and_verifies(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        sm_cls = MagicMock()
        monkeypatch.setattr(aa_lcr_module, "SubprocessManager", sm_cls)
        verified: list[str] = []

        def verify_generation_outputs() -> bool:
            verified.append("gen")
            return True

        monkeypatch.setattr(
            evaluator, "_verify_generation_outputs", verify_generation_outputs
        )
        evaluator.run_phase("generation")
        sm = sm_cls.return_value
        assert sm.start.called
        assert sm.wait_for_completion.called
        assert verified == ["gen"]
        cmd = sm_cls.call_args.args[0]
        overrides = json.loads(cmd[cmd.index("--overrides-json") + 1])
        assert overrides["model"]["timeout"] == DEFAULT_REQUEST_TIMEOUT
        assert overrides["model"]["num_retries"] == 20
        assert "api_key" not in json.dumps(overrides)

    def test_generation_setup_identity_missing_fails_generation(
        self, tmp_path: Path, monkeypatch
    ):
        evaluator = _build_evaluator(tmp_path)
        records = {
            f"q{question}__r{repeat}": {"status": "ok"}
            for question in (1, 2)
            for repeat in (1, 2, 3)
        }
        _write_generation_manifest(evaluator, records)
        evaluator._setup_identity = None
        monkeypatch.setattr(aa_lcr_module, "SubprocessManager", MagicMock())

        with pytest.raises(RuntimeError, match="failed to save generation setup"):
            evaluator.run_phase("generation")

    def test_generation_setup_write_error_fails_generation(
        self, tmp_path: Path, monkeypatch
    ):
        evaluator = _build_evaluator(tmp_path)
        records = {
            f"q{question}__r{repeat}": {"status": "ok"}
            for question in (1, 2)
            for repeat in (1, 2, 3)
        }
        _write_generation_manifest(evaluator, records)
        monkeypatch.setattr(aa_lcr_module, "SubprocessManager", MagicMock())
        monkeypatch.setattr(
            Path, "write_text", MagicMock(side_effect=OSError("read-only"))
        )

        with pytest.raises(RuntimeError, match="failed to save generation setup"):
            evaluator.run_phase("generation")

    def test_generation_reuses_matching_complete_outputs(
        self, tmp_path: Path, monkeypatch
    ):
        first = _build_evaluator(tmp_path)
        records = {
            f"q{question}__r{repeat}": {"status": "ok"}
            for question in (1, 2)
            for repeat in (1, 2, 3)
        }
        _write_generation_manifest(first, records)
        sm_cls = MagicMock()
        monkeypatch.setattr(aa_lcr_module, "SubprocessManager", sm_cls)
        first.run_phase("generation")

        restarted = _build_evaluator(tmp_path)
        restarted.run_phase("generation")

        assert sm_cls.call_count == 1

    def test_generation_reruns_when_dataset_contents_change(
        self, tmp_path: Path, monkeypatch
    ):
        first = _build_evaluator(tmp_path)
        records = {
            f"q{question}__r{repeat}": {"status": "ok"}
            for question in (1, 2)
            for repeat in (1, 2, 3)
        }
        _write_generation_manifest(first, records)
        sm_cls = MagicMock()
        monkeypatch.setattr(aa_lcr_module, "SubprocessManager", sm_cls)
        first.run_phase("generation")

        Path(first.autometric_runs_config.dataset_csv).write_text(
            "question_id,question,answer\nq1,changed,a\nq2,changed,a\n",
            encoding="utf-8",
        )
        restarted = AaLcrEvaluator()
        restarted.setup(
            first.autometric_runs_config,
            first.artifacts_path,
            first.hf_checkpoint_path,
            first.model_info,
        )
        restarted.run_phase("generation")

        assert sm_cls.call_count == 2

    def test_evaluation_verifies_judge_outputs(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        monkeypatch.setattr(aa_lcr_module, "SubprocessManager", MagicMock())
        verified: list[str] = []

        def verify_evaluation_outputs() -> bool:
            verified.append("eval")
            return True

        monkeypatch.setattr(
            evaluator, "_verify_evaluation_outputs", verify_evaluation_outputs
        )
        evaluator.run_phase("evaluation")
        assert verified == ["eval"]

    def test_evaluation_output_error_fails_the_phase(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        monkeypatch.setattr(aa_lcr_module, "SubprocessManager", MagicMock())
        monkeypatch.setattr(
            evaluator,
            "_verify_evaluation_outputs",
            MagicMock(side_effect=ValueError("broken judge output")),
        )

        with pytest.raises(ValueError, match="broken judge output"):
            evaluator.run_phase("evaluation")

    def test_evaluation_invalid_outputs_fail_the_phase(
        self, tmp_path: Path, monkeypatch
    ):
        evaluator = _build_evaluator(tmp_path)
        monkeypatch.setattr(aa_lcr_module, "SubprocessManager", MagicMock())
        monkeypatch.setattr(
            evaluator, "_verify_evaluation_outputs", MagicMock(return_value=False)
        )

        with pytest.raises(RuntimeError, match="evaluation outputs are invalid"):
            evaluator.run_phase("evaluation")

    def test_scoring_exit_failure_fails_before_reading_outputs(
        self, tmp_path: Path, monkeypatch
    ):
        evaluator = _build_evaluator(tmp_path)
        metrics_dir = Path(evaluator.metrics_dir)
        (metrics_dir / "aa_lcr_results.json").write_text(
            json.dumps(
                [
                    {"task_id": f"q{q}__r{r}", "judge_status": "ok"}
                    for q in (1, 2)
                    for r in (1, 2, 3)
                ]
            ),
            encoding="utf-8",
        )
        (metrics_dir / "aa_lcr_metrics.json").write_text(
            json.dumps({"aa_lcr/accuracy": 0.99}), encoding="utf-8"
        )
        sm_cls = MagicMock()
        sm = sm_cls.return_value
        sm.is_successful.return_value = False
        sm.exit_code = 17
        monkeypatch.setattr(aa_lcr_module, "SubprocessManager", sm_cls)
        verify = MagicMock(wraps=evaluator._verify_evaluation_outputs)
        monkeypatch.setattr(evaluator, "_verify_evaluation_outputs", verify)

        with pytest.raises(RuntimeError, match="exit code 17"):
            evaluator.run_phase("evaluation")

        sm.log_error.assert_called_once_with()
        verify.assert_not_called()

    def test_generation_manifest_missing_fails_generation(
        self, tmp_path: Path, monkeypatch
    ):
        evaluator = _build_evaluator(tmp_path)
        monkeypatch.setattr(aa_lcr_module, "SubprocessManager", MagicMock())

        with pytest.raises(RuntimeError, match="generation outputs are invalid"):
            evaluator.run_phase("generation")


class TestAaLcrExpectedTaskCount:
    def test_counts_questions_times_repeats(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert evaluator._expected_task_count() == 6

    def test_skips_rows_without_question_id(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        _write_dataset(
            Path(evaluator.autometric_runs_config.dataset_csv), ["q1", "", "q3"]
        )
        assert evaluator._expected_task_count() == 6

    def test_applies_limit(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path, limit=1)
        assert evaluator._expected_task_count() == 3

    def test_gold_mode_counts_single_repeat(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path, use_gold_as_candidate=True)
        assert evaluator._expected_task_count() == 2

    def test_empty_dataset_has_no_expected_task_count(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        _write_dataset(Path(evaluator.autometric_runs_config.dataset_csv), [])
        assert evaluator._expected_task_count() is None

    def test_missing_dataset_has_no_expected_task_count(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        Path(evaluator.autometric_runs_config.dataset_csv).unlink()
        assert evaluator._expected_task_count() is None

    def test_invalid_utf8_dataset_has_no_expected_task_count(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        Path(evaluator.autometric_runs_config.dataset_csv).write_bytes(b"\xff")
        assert evaluator._expected_task_count() is None


class TestAaLcrRunnerJson:
    def test_missing_runner_json_returns_none(self, tmp_path: Path):
        assert AaLcrEvaluator._load_runner_json(tmp_path / "missing.json") is None


class TestAaLcrVerifyGeneration:
    def test_missing_manifest_fails_verification(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert evaluator._verify_generation_outputs() is False

    def test_corrupt_manifest_fails_verification(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        path = _write_generation_manifest(evaluator, {})
        path.write_text("{broken", encoding="utf-8")
        assert evaluator._verify_generation_outputs() is False

    def test_incomplete_manifest_fails_verification(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        _write_generation_manifest(evaluator, {"q1__r1": {"status": "ok"}})
        assert evaluator._verify_generation_outputs() is False

    def test_failed_task_fails_the_run(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        records = {f"q{q}__r{r}": {"status": "ok"} for q in (1, 2) for r in (1, 2, 3)}
        records["q1__r2"] = {"status": "api_failed"}
        _write_generation_manifest(evaluator, records)
        assert evaluator._verify_generation_outputs() is False

    def test_complete_ok_manifest_passes(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        records = {f"q{q}__r{r}": {"status": "ok"} for q in (1, 2) for r in (1, 2, 3)}
        _write_generation_manifest(evaluator, records)
        assert evaluator._verify_generation_outputs() is True

    def test_unreadable_dataset_fails_verification(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        records = {f"q{q}__r{r}": {"status": "ok"} for q in (1, 2) for r in (1, 2, 3)}
        _write_generation_manifest(evaluator, records)
        Path(evaluator.autometric_runs_config.dataset_csv).write_bytes(b"\xff")
        assert evaluator._verify_generation_outputs() is False


class TestAaLcrVerifyEvaluation:
    @staticmethod
    def _write_results(evaluator: AaLcrEvaluator, rows: list) -> Path:
        results_path = Path(evaluator.metrics_dir) / "aa_lcr_results.json"
        results_path.write_text(json.dumps(rows), encoding="utf-8")
        return results_path

    def test_missing_results_fail_verification(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert evaluator._verify_evaluation_outputs() is False

    def test_incomplete_results_fail_verification(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path, n_repeats=1)
        self._write_results(evaluator, [{"task_id": "q1__r1", "judge_status": "ok"}])
        assert evaluator._verify_evaluation_outputs() is False

    def test_judge_failure_fails_the_run(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path, n_repeats=1)
        self._write_results(
            evaluator,
            [
                {"task_id": "q1__r1", "judge_status": "ok"},
                {"task_id": "q2__r1", "judge_status": "parse_failed"},
            ],
        )
        assert evaluator._verify_evaluation_outputs() is False

    def test_ok_and_empty_answer_statuses_pass(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path, n_repeats=1)
        self._write_results(
            evaluator,
            [
                {"task_id": "q1__r1", "judge_status": "ok"},
                {"task_id": "q2__r1", "judge_status": "skipped_empty_answer"},
            ],
        )
        assert evaluator._verify_evaluation_outputs() is True

    def test_non_list_results_fail_verification(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        results_path = Path(evaluator.metrics_dir) / "aa_lcr_results.json"
        results_path.write_text("{}", encoding="utf-8")
        assert evaluator._verify_evaluation_outputs() is False

    def test_unreadable_dataset_fails_verification(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path, n_repeats=1)
        self._write_results(
            evaluator,
            [
                {"task_id": "q1__r1", "judge_status": "ok"},
                {"task_id": "q2__r1", "judge_status": "ok"},
            ],
        )
        Path(evaluator.autometric_runs_config.dataset_csv).write_bytes(b"\xff")
        assert evaluator._verify_evaluation_outputs() is False


class TestAaLcrGetMetrics:
    def test_returns_empty_when_metrics_dir_unset(self):
        evaluator = AaLcrEvaluator()
        assert evaluator.get_metrics() == {}

    def test_missing_metrics_file_returns_no_metrics(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)

        assert evaluator.get_metrics() == {}

    def test_reads_metrics_and_drops_nulls(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        metrics_path = Path(evaluator.metrics_dir) / "aa_lcr_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {"aa_lcr/accuracy": 0.75, "aa_lcr/n_questions": 2, "aa_lcr/skip": None}
            ),
            encoding="utf-8",
        )
        assert evaluator.get_metrics() == {
            "aa_lcr/accuracy": pytest.approx(0.75),
            "aa_lcr/n_questions": pytest.approx(2.0),
        }

    def test_invalid_utf8_metrics_returns_no_metrics(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        metrics_path = Path(evaluator.metrics_dir) / "aa_lcr_metrics.json"
        metrics_path.write_bytes(b"\xff")

        assert evaluator.get_metrics() == {}

    def test_non_object_metrics_returns_no_metrics(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        metrics_path = Path(evaluator.metrics_dir) / "aa_lcr_metrics.json"
        metrics_path.write_text("[]", encoding="utf-8")

        assert evaluator.get_metrics() == {}

    def test_metric_collection_error_returns_no_metrics(
        self, tmp_path: Path, monkeypatch
    ):
        evaluator = _build_evaluator(tmp_path)
        monkeypatch.setattr(
            evaluator, "_collect_metrics", MagicMock(side_effect=RuntimeError("broken"))
        )

        assert evaluator.get_metrics() == {}

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"aa_lcr/accuracy": None},
            {"aa_lcr/n_questions": 2},
            {"aa_lcr/accuracy": "0.5"},
            {"aa_lcr/accuracy": float("nan")},
            {"aa_lcr/accuracy": -0.1},
            {"aa_lcr/accuracy": 1.1},
        ],
    )
    def test_missing_or_invalid_headline_metric_returns_no_metrics(
        self, tmp_path: Path, payload: dict
    ):
        evaluator = _build_evaluator(tmp_path)
        metrics_path = Path(evaluator.metrics_dir) / "aa_lcr_metrics.json"
        metrics_path.write_text(json.dumps(payload), encoding="utf-8")

        assert evaluator.get_metrics() == {}


class TestAaLcrShooterVersion:
    def test_reads_pinned_commit_from_image(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert evaluator._shooter_version() == f"vendored-{VERSION_SHA[:12]}"


class TestAaLcrSetupIdentity:
    def test_missing_dataset_has_no_effective_setup(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        Path(evaluator.autometric_runs_config.dataset_csv).unlink()

        assert evaluator._data_fingerprint() is None
        assert evaluator._effective_setup() is None

    def test_missing_text_directory_has_no_data_fingerprint(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        text_dir = Path(evaluator.autometric_runs_config.text_dir)
        (text_dir / "doc.txt").unlink()
        text_dir.rmdir()
        assert evaluator._data_fingerprint() is None

    def test_data_read_error_has_no_data_fingerprint(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        monkeypatch.setattr(
            Path, "read_bytes", MagicMock(side_effect=OSError("unreadable"))
        )
        assert evaluator._data_fingerprint() is None


class TestAaLcrJdKeys:
    def test_filters_and_names_by_setup_only(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        converted = evaluator._convert_metrics_to_jd_keys(
            {
                "aa_lcr/accuracy": 0.8,
                "aa_lcr/accuracy_repeat_1": 0.7,
                "aa_lcr/n_questions": 2.0,
            }
        )
        assert converted == {
            evaluator._jd_metric_name("accuracy"): pytest.approx(0.8),
            evaluator._jd_metric_name("accuracy_repeat_1"): pytest.approx(0.7),
        }
        assert evaluator._jd_metric_name("accuracy") == (
            "AA_LCR_ACCURACY_JUDGE_QWEN3_235B_A22B_INSTRUCT_2507_FP8_"
            "MODE_MODEL_REPEATS_3"
        )

    def test_two_candidates_one_setup_share_identical_keys(self, tmp_path: Path):
        metrics = {"aa_lcr/accuracy": 0.5, "aa_lcr/accuracy_repeat_1": 0.4}
        key_sets = []
        for checkpoint in (
            "candidate-a/ep0_ba100/artifacts/hf_checkpoint",
            "candidate-b/ep0_ba200/artifacts/hf_checkpoint",
        ):
            evaluator = _build_evaluator(tmp_path, checkpoint=checkpoint)
            key_sets.append(sorted(evaluator._convert_metrics_to_jd_keys(metrics)))
        assert key_sets[0] == key_sets[1]

    @pytest.mark.parametrize(
        "changed_config",
        [
            {"n_repeats": 7},
            {"use_gold_as_candidate": True},
            {"judge": {"model_name": "j"}},
        ],
    )
    def test_judge_mode_or_repeats_change_metric_identity(
        self, tmp_path: Path, changed_config: dict
    ):
        baseline = _build_evaluator(tmp_path)
        baseline_key = baseline._jd_metric_name("accuracy")
        changed = _build_evaluator(tmp_path, **changed_config)
        assert changed._jd_metric_name("accuracy") != baseline_key

    @pytest.mark.parametrize(
        "changed_config",
        [
            {"limit": 1},
            {"llm_timeout": 123},
            {"model": {"max_tokens": 64}},
            {"model": {"temperature": 0.5}},
            {"model": {"top_p": 0.5}},
            {"model": {"top_k": 20}},
            {
                "judge": {
                    "model_name": "Qwen3-235B-A22B-Instruct-2507-FP8",
                    "max_tokens": 1,
                }
            },
            {
                "judge": {
                    "model_name": "Qwen3-235B-A22B-Instruct-2507-FP8",
                    "temperature": 0.7,
                }
            },
            {
                "judge": {
                    "model_name": "Qwen3-235B-A22B-Instruct-2507-FP8",
                    "top_p": 0.5,
                }
            },
            {
                "judge": {
                    "model_name": "Qwen3-235B-A22B-Instruct-2507-FP8",
                    "top_k": 20,
                }
            },
        ],
    )
    def test_other_config_does_not_change_metric_identity(
        self, tmp_path: Path, changed_config: dict
    ):
        baseline = _build_evaluator(tmp_path)
        baseline_key = baseline._jd_metric_name("accuracy")
        changed = _build_evaluator(tmp_path, **changed_config)
        assert changed._jd_metric_name("accuracy") == baseline_key

    def test_gold_mode_uses_one_repeat(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path, use_gold_as_candidate=True)
        assert "_MODE_GOLD_REPEATS_1" in evaluator._jd_metric_name("accuracy")

    def test_parallelism_does_not_change_metric_identity(self, tmp_path: Path):
        first = _build_evaluator(tmp_path, parallel=1, judge_parallel=2)
        first_key = first._jd_metric_name("accuracy")
        second = _build_evaluator(tmp_path, parallel=99, judge_parallel=100)
        assert second._jd_metric_name("accuracy") == first_key


class TestAaLcrSendMetricsToJd:
    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_skips_when_metrics_dir_missing(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        evaluator.metrics_dir = None
        send = MagicMock()
        monkeypatch.setattr(aa_lcr_module.JD, "send_evaluator_metrics", send)
        evaluator._send_metrics_to_jd({"aa_lcr/accuracy": 0.5})
        send.assert_not_called()

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_sends_setup_keyed_metrics_with_real_version(
        self, tmp_path: Path, monkeypatch
    ):
        evaluator = _build_evaluator(tmp_path)
        send = MagicMock()
        monkeypatch.setattr(aa_lcr_module.JD, "send_evaluator_metrics", send)
        evaluator._send_metrics_to_jd(
            {"aa_lcr/accuracy": 0.8, "aa_lcr/n_questions": 2.0}
        )
        kwargs = send.call_args.kwargs
        assert kwargs["shooter_name"] == "aa_lcr"
        assert kwargs["shooter_version"] == f"vendored-{VERSION_SHA[:12]}"
        assert kwargs["jd_metrics"] == {evaluator._jd_metric_name("accuracy"): 0.8}


class TestAaLcrAfterRun:
    def test_after_run_sends_tars_and_symlinks(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        send = MagicMock()
        tar = MagicMock(return_value="TAR")
        symlink = MagicMock()
        monkeypatch.setattr(evaluator, "_send_metrics_to_jd", send)
        monkeypatch.setattr(aa_lcr_module.JD, "tar_metrics_dir", tar)
        monkeypatch.setattr(aa_lcr_module.JD, "symlink_metrics_for_transfer", symlink)
        metrics = {"aa_lcr/accuracy": 0.9}
        evaluator.after_run(metrics)
        send.assert_called_once_with(metrics)
        tar.assert_called_once_with(
            evaluator.metrics_dir, evaluator.artifacts_path, "AaLcrEvaluator"
        )
        symlink.assert_called_once_with(
            "TAR", evaluator.hf_checkpoint_path, "AaLcrEvaluator"
        )

    def test_archive_error_does_not_escape_after_run(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        monkeypatch.setattr(evaluator, "_send_metrics_to_jd", MagicMock())
        monkeypatch.setattr(
            aa_lcr_module.JD,
            "tar_metrics_dir",
            MagicMock(side_effect=RuntimeError("tar failed")),
        )

        evaluator.after_run({"aa_lcr/accuracy": 0.9})

    def test_symlink_error_does_not_escape_after_run(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        monkeypatch.setattr(evaluator, "_send_metrics_to_jd", MagicMock())
        monkeypatch.setattr(
            aa_lcr_module.JD, "tar_metrics_dir", MagicMock(return_value="TAR")
        )
        monkeypatch.setattr(
            aa_lcr_module.JD,
            "symlink_metrics_for_transfer",
            MagicMock(side_effect=OSError("symlink failed")),
        )

        evaluator.after_run({"aa_lcr/accuracy": 0.9})


class TestAaLcrFlatMetrics:
    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_accuracy_and_repeats_have_separate_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path)
        JD = aa_lcr_module.JD
        sent: dict[str, object] = {}
        monkeypatch.setattr(
            JD, "save_metrics_json", lambda body, step, metrics_dir: None
        )
        monkeypatch.setattr(
            JD, "send_metrics", lambda body, _artifacts: sent.update(body=body)
        )

        evaluator._send_metrics_to_jd(
            {
                "aa_lcr/accuracy": 0.8,
                "aa_lcr/accuracy_repeat_1": 0.81,
                "aa_lcr/accuracy_repeat_2": 0.79,
                "aa_lcr/n_questions": 2.0,
            }
        )

        assert sent["body"]["metrics"] == {
            0: {
                evaluator._jd_metric_name("accuracy"): {
                    "value": 0.8,
                    "submetrics": {
                        evaluator._jd_metric_name("accuracy_repeat_1"): 0.81,
                        evaluator._jd_metric_name("accuracy_repeat_2"): 0.79,
                    },
                },
            }
        }


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "gold",
        "duplicate",
        "invalid_verdict",
        "missing_repeat",
        "malformed",
        "unverified",
    ],
)
def test_ci_averages_complete_question_repeats(tmp_path, monkeypatch, case):
    evaluator = _build_evaluator(
        tmp_path, n_repeats=2, use_gold_as_candidate=case == "gold"
    )
    monkeypatch.setattr(
        evaluator, "_verify_evaluation_outputs", lambda: case != "unverified"
    )
    repeats = [1] if case == "gold" else [1, 2]
    rows = [
        {
            "question_id": uid,
            "repeat": repeat,
            "verdict": "CORRECT" if uid == "q1" else "INCORRECT",
        }
        for uid in ("q1", "q2")
        for repeat in repeats
    ]
    if case == "duplicate":
        rows.append(rows[0])
    elif case == "invalid_verdict":
        rows[0]["verdict"] = "ERROR"
    elif case == "missing_repeat":
        rows.pop()
    elif case == "malformed":
        rows = [{}]
    (evaluator.metrics_dir / "aa_lcr_results.json").write_text(json.dumps(rows))
    scores = evaluator._cache_scores({"ACCURACY": 0.5, "ERRORS": 0}, "ACCURACY")
    assert set(scores.records) == (
        {"ACCURACY"} if case in {"complete", "gold"} else set()
    )
    if scores.records:
        assert [row["score"] for row in scores.records["ACCURACY"]] == [1, 0]
